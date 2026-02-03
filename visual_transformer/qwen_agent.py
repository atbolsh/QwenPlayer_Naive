# QwenAgent - Extended Qwen3 model with vision capabilities
# Implements qwen_extension (extended Qwen3Model) and qwen_agent_pipe (full pipeline)

import torch
import torch.nn as nn
from typing import Optional, List, Union
from PIL import Image
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# Import image encoder/decoder from model.py (with 1024 embed_dim, 8 heads as per negative_example.py)
from .model import ImageTransformerEncoder, ImageTransformerDecoder


class QwenExtension(nn.Module):
    """
    Extension of Qwen3Model with image encoder/decoder.
    
    This class wraps a Qwen3 model and adds:
    - Image encoder (1024 embed_dim, 8 heads)
    - Image decoder (1024 embed_dim, 8 heads)
    - Per-layer scaling factors for image context in KV cache
    """
    
    # Measured per-layer INPUT magnitudes from Qwen3-0.6B (layers 0-27)
    # These represent typical magnitudes of vectors entering each transformer layer
    LAYER_INPUT_MAGNITUDES = [
        0.8584,    # Layer 0
        8.8750,    # Layer 1
        12.2422,   # Layer 2
        703.5000,  # Layer 3
        706.0000,  # Layer 4
        707.5000,  # Layer 5
        710.5000,  # Layer 6
        712.0000,  # Layer 7
        715.5000,  # Layer 8
        718.0000,  # Layer 9
        721.5000,  # Layer 10
        728.0000,  # Layer 11
        735.5000,  # Layer 12
        739.0000,  # Layer 13
        740.5000,  # Layer 14
        740.5000,  # Layer 15
        748.5000,  # Layer 16
        763.5000,  # Layer 17
        780.0000,  # Layer 18
        800.0000,  # Layer 19
        827.0000,  # Layer 20
        860.0000,  # Layer 21
        904.5000,  # Layer 22
        952.5000,  # Layer 23
        1003.0000, # Layer 24
        1071.0000, # Layer 25
        1146.0000, # Layer 26
        1174.0000, # Layer 27
    ]
    
    # Measured magnitudes for reference
    EMBEDDING_MAGNITUDE = 0.8584
    IMG_ENC_MAGNITUDE = 31.73
    
    def __init__(self, qwen_model, embed_dim: int = 1024, num_heads: int = 8):
        """
        Args:
            qwen_model: A Qwen3 model (e.g., from AutoModelForCausalLM.from_pretrained)
            embed_dim: Embedding dimension for image encoder/decoder (default: 1024)
            num_heads: Number of attention heads for image encoder/decoder (default: 8)
        """
        super().__init__()
        
        self.qwen_model = qwen_model
        self.embed_dim = embed_dim
        
        # Get the hidden size from the Qwen model config
        qwen_hidden_size = qwen_model.config.hidden_size
        self.num_hidden_layers = qwen_model.config.num_hidden_layers
        
        # Verify that embed_dim matches Qwen's hidden_size
        assert embed_dim == qwen_hidden_size, (
            f"embed_dim ({embed_dim}) must equal Qwen model's hidden_size ({qwen_hidden_size})"
        )
        
        # Image encoder/decoder with parameters from negative_example.py
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads)
        
        # Per-layer scaling factors for image context
        # Initialized to layer_magnitude / img_enc_magnitude so image activations 
        # appear "typical" for each transformer layer
        initial_scales = torch.tensor([
            mag / self.IMG_ENC_MAGNITUDE for mag in self.LAYER_INPUT_MAGNITUDES
        ], dtype=torch.float32)
        self.layer_scale_factors = nn.Parameter(initial_scales)
    
    def get_device(self):
        """Get the device of the model."""
        return next(self.qwen_model.parameters()).device
    
    def encode_image(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            img_batch: Tensor of shape (batch, channels, height, width)
            
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        return self.img_enc(img_batch)
    
    def decode_image(self, img_encoding: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode image embeddings back to images.
        
        Args:
            img_encoding: Tensor of shape (batch, seq_len, embed_dim)
            context: Optional context tensor
            
        Returns:
            Tensor of reconstructed images
        """
        return self.img_dec(img_encoding, context)
    
    def img_autoencoder(self, img_batch: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Image autoencoder: encode then decode an image.
        
        Args:
            img_batch: Tensor of shape (batch, channels, height, width)
            context: Optional context tensor for decoder (defaults to image encoding)
            
        Returns:
            Tensor of reconstructed images
        """
        img_encoding = self.img_enc(img_batch)
        if context is None:
            context = img_encoding
        return self.img_dec(img_encoding, context)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input for rotary embeddings."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _get_base_model(self):
        """Get the base Qwen3Model (the transformer layers).
        
        Handles both regular Qwen models and PEFT-wrapped models (LoRA).
        """
        qwen = self.qwen_model
        
        # Check for PEFT wrapper (LoRA) - has base_model attribute
        if hasattr(qwen, 'base_model'):
            # PEFT wraps: qwen_model.base_model.model.model
            base = qwen.base_model
            if hasattr(base, 'model') and hasattr(base.model, 'model'):
                return base.model.model
            elif hasattr(base, 'model'):
                return base.model
        
        # Regular Qwen3ForCausalLM: qwen_model.model
        if hasattr(qwen, 'model'):
            return qwen.model
        
        # Qwen3Model directly
        return qwen
    
    def load_bases(self, image_context: torch.Tensor) -> DynamicCache:
        """
        Compute keys and values for every decoder layer from image context embeddings
        and populate the KV cache. This allows the full Qwen model to only process
        text tokens while still attending to image information via the cache.
        
        The image context is scaled per-layer to match typical activation magnitudes
        for each transformer layer, using trainable scaling factors.
        
        Args:
            image_context: Tensor of shape (batch, seq_len, embed_dim) containing 
                          the full image context (vision_start + image embeddings + vision_end)
                          All tokens should be normalized to img_enc output magnitude (~32)
                          
        Returns:
            DynamicCache: A cache containing pre-computed keys and values for all layers
        """
        batch_size, seq_len, _ = image_context.shape
        device = image_context.device
        
        # Create a new cache
        cache = DynamicCache()
        
        # Get the base model (handle Qwen3ForCausalLM, Qwen3Model, and PEFT wrappers)
        base_model = self._get_base_model()
        
        # Generate position embeddings for the image sequence
        # Use the base image_context for position embeddings (scaling doesn't affect positions)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = base_model.rotary_emb(image_context, position_ids)
        cos, sin = position_embeddings
        
        # For each decoder layer, compute and cache the keys and values
        for layer_idx, decoder_layer in enumerate(base_model.layers):
            attn = decoder_layer.self_attn
            
            # Scale image context to match typical activation magnitude for this layer
            # layer_scale_factors are initialized to layer_mag / img_enc_mag
            scale = self.layer_scale_factors[layer_idx]
            scaled_hidden = image_context * scale.to(image_context.dtype)
            
            # Apply input layer norm
            normed_hidden = decoder_layer.input_layernorm(scaled_hidden)
            
            # Compute key and value projections
            head_dim = attn.head_dim
            num_key_value_heads = self.qwen_model.config.num_key_value_heads
            
            # Shape: (batch, seq_len, num_kv_heads * head_dim)
            key_states = attn.k_proj(normed_hidden)
            value_states = attn.v_proj(normed_hidden)
            
            # Reshape to (batch, seq_len, num_kv_heads, head_dim)
            hidden_shape = (batch_size, seq_len, num_key_value_heads, head_dim)
            key_states = key_states.view(*hidden_shape)
            value_states = value_states.view(*hidden_shape)
            
            # Apply key normalization and transpose to (batch, num_kv_heads, seq_len, head_dim)
            key_states = attn.k_norm(key_states).transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            
            # Apply rotary position embeddings to keys
            # cos and sin have shape (batch, seq_len, head_dim)
            cos_unsqueezed = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
            sin_unsqueezed = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
            
            # Rotate keys
            key_states_rotated = (key_states * cos_unsqueezed) + (self._rotate_half(key_states) * sin_unsqueezed)
            
            # Update the cache for this layer
            cache.update(key_states_rotated, value_states, layer_idx)
        
        return cache
    
    def text_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Text-only forward pass (no images).
        
        Returns logits in format (batch, vocab, seq_len) for backward compatibility
        with loss functions that expect this shape.
        
        Args:
            input_ids: Token tensor (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            logits: Tensor of shape (batch, vocab_size, seq_len)
        """
        if attention_mask is None:
            # Create attention mask (1 for non-pad tokens)
            pad_token_id = self.qwen_model.config.pad_token_id or 0
            attention_mask = (input_ids != pad_token_id).long()
        
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # logits: (batch, seq_len, vocab) -> (batch, vocab, seq_len)
        logits = outputs.logits.permute(0, 2, 1)
        return logits
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,  # bf16 embeddings
        past_key_values: Optional[DynamicCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through the extended Qwen model.
        
        Supports passing pre-computed KV cache from load_bases.
        
        Args:
            input_ids: Token tensor (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, total_seq_len including cache)
            inputs_embeds: Input embeddings (batch_size, seq_len, embed_dim)
            past_key_values: Pre-computed KV cache from load_bases
            position_ids: Position IDs for the input (accounts for cache offset)
            use_cache: Whether to use/return cache
            cache_position: Positions in the cache
            **kwargs: Additional arguments passed to qwen_model
        """
        return self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )


class QwenAgentPipe(nn.Module):
    """
    Full pipeline for Qwen with visual capabilities.
    
    Consumes string text and a list of images (arbitrary length, possibly empty).
    - Prepends image tokens to text
    - Encodes images to embeddings
    - Replaces image_pad tokens with actual image embeddings
    - Loads image bases into the KV cache
    """
    
    # Special tokens for vision (already built into Qwen tokenizer)
    BEGIN_VISION = "<|vision_start|>"
    END_VISION = "<|vision_end|>"
    IMAGE_PAD = "<|image_pad|>"
    
    # Game control tokens (from general_framework_lightweight.py)
    GAME_CONTROL_TOKENS = ['<forward>', '<clock>', '<anticlock>']
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        embed_dim: int = 1024,
        num_heads: int = 8,
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        image_patch_tokens: int = 256,  # Number of patches per image (16x16 grid)
    ):
        """
        Args:
            model_name: Name of the Qwen model to load
            embed_dim: Embedding dimension for image encoder/decoder
            num_heads: Number of attention heads for image encoder/decoder
            device: Device to load model on
            torch_dtype: Data type for the model
            image_patch_tokens: Number of tokens per image (default: 256 for 16x16 patches)
        """
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.image_patch_tokens = image_patch_tokens
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add game control tokens (vision tokens are already in the tokenizer)
        # Note: We don't resize embeddings - Qwen3 already has room for additional tokens
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.GAME_CONTROL_TOKENS})
        
        # Get token IDs for special tokens (these are already in the tokenizer)
        self.begin_vision_id = self.tokenizer.convert_tokens_to_ids(self.BEGIN_VISION)
        self.end_vision_id = self.tokenizer.convert_tokens_to_ids(self.END_VISION)
        
        # Load base Qwen model (no need to resize - model already has capacity for added tokens)
        # Use default tie_word_embeddings=True to share weights between embed_tokens and lm_head
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        
        # Wrap in QwenExtension (this will verify embed_dim == hidden_size)
        self.model = QwenExtension(qwen_model, embed_dim=embed_dim, num_heads=num_heads)
        
        # Move to device
        self.to(self.device)
    
    def get_device(self):
        """Get the device of the model."""
        return self.device
    
    def _preprocess_images(self, images: List[Union[torch.Tensor, np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Preprocess a list of images to tensor format (bf16).
        
        Args:
            images: List of images (can be tensors, numpy arrays, or PIL images)
            
        Returns:
            Tensor of shape (num_images, channels, height, width) in bf16
        """
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                # Convert PIL to tensor
                img = img.resize((224, 224))
                img = np.array(img) / 255.0
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                img = torch.tensor(img, dtype=self.torch_dtype).permute(2, 0, 1)
            elif isinstance(img, np.ndarray):
                if img.max() > 1.0:
                    img = img / 255.0
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                img = torch.tensor(img, dtype=self.torch_dtype)
                if img.shape[-1] == 3:  # HWC -> CHW
                    img = img.permute(2, 0, 1)
            elif isinstance(img, torch.Tensor):
                if img.shape[-1] == 3:  # HWC -> CHW
                    img = img.permute(2, 0, 1)
                # Convert to bf16 if not already
                if img.dtype != self.torch_dtype:
                    img = img.to(self.torch_dtype)
            processed.append(img)
        
        if len(processed) > 0:
            return torch.stack(processed).to(self.device)
        else:
            return None
    
    def _build_image_prefix_string(self, num_images: int) -> str:
        """
        Build the image prefix string with <|image_pad|> placeholders.
        
        Format: "Image 0:<|vision_start|><|image_pad|>...<|vision_end|>\nImage 1:..."
        """
        parts = []
        image_pads = self.IMAGE_PAD * self.image_patch_tokens
        for i in range(num_images):
            prefix = "" if i == 0 else "\n"
            parts.append(f"{prefix}Image {i}:{self.BEGIN_VISION}{image_pads}{self.END_VISION}")
        return "".join(parts)
    
    def _get_embed_tokens(self):
        """Get the embedding layer from the model.
        
        Handles both regular Qwen models and PEFT-wrapped models (LoRA).
        """
        qwen = self.model.qwen_model
        
        # Check for PEFT wrapper (LoRA) - has base_model attribute
        if hasattr(qwen, 'base_model'):
            # PEFT wraps: qwen_model.base_model.model.model.embed_tokens
            base = qwen.base_model
            if hasattr(base, 'model') and hasattr(base.model, 'model'):
                return base.model.model.embed_tokens
            elif hasattr(base, 'model'):
                return base.model.embed_tokens
        
        # Regular Qwen3ForCausalLM: qwen_model.model.embed_tokens
        if hasattr(qwen, 'model') and hasattr(qwen.model, 'embed_tokens'):
            return qwen.model.embed_tokens
        
        # Qwen3Model directly: qwen_model.embed_tokens
        if hasattr(qwen, 'embed_tokens'):
            return qwen.embed_tokens
        
        raise AttributeError("Could not find embed_tokens in model structure")
    
    def _build_image_context_and_cache(
        self,
        images: Optional[List[torch.Tensor]],
        batch_size: int,
        shared_images: bool = False,
    ):
        """
        Build image context tensor and pre-compute KV cache.
        
        This is a shared subroutine used by batch_forward and generate to avoid code duplication.
        
        Args:
            images: List of image tensors. 
                   If shared_images=False: Each tensor is (batch_size, 3, 224, 224)
                   If shared_images=True: Each tensor is (3, 224, 224) - shared across batch
            batch_size: Batch size
            shared_images: If True, images are shared across batch (expand them)
            
        Returns:
            Tuple of (past_key_values, image_seq_len, image_encodings) or (None, 0, None) if no images
        """
        if images is None or len(images) == 0:
            return None, 0, None
        
        num_images = len(images)
        embed_tokens = self._get_embed_tokens()
        
        # ===== Encode images =====
        if shared_images:
            # Images are (3, 224, 224) each - stack and encode
            stacked = torch.stack([img.to(self.device) for img in images], dim=0)  # (num_images, 3, 224, 224)
            flat_encodings = self.model.encode_image(stacked)  # (num_images, 256, embed_dim)
            # Expand for batch: (num_images, batch_size, 256, embed_dim)
            image_encodings = flat_encodings.unsqueeze(1).expand(-1, batch_size, -1, -1)
        else:
            # Images are (batch_size, 3, 224, 224) each - stack and encode
            stacked_images = torch.stack([img.to(self.device) for img in images], dim=0)  # (num_images, batch_size, 3, 224, 224)
            flat_images = stacked_images.view(-1, 3, 224, 224)  # (num_images * batch_size, 3, 224, 224)
            flat_encodings = self.model.encode_image(flat_images)  # (num_images * batch_size, 256, embed_dim)
            image_encodings = flat_encodings.view(num_images, batch_size, 256, self.embed_dim)
        
        # ===== Scale factor for non-image tokens =====
        embed_to_img_scale = QwenExtension.IMG_ENC_MAGNITUDE / QwenExtension.EMBEDDING_MAGNITUDE
        
        # ===== Get vision start/end embeddings (scaled) =====
        vision_start_ids = torch.tensor([[self.begin_vision_id]], device=self.device)
        vision_end_ids = torch.tensor([[self.end_vision_id]], device=self.device)
        vision_start_embed = embed_tokens(vision_start_ids) * embed_to_img_scale
        vision_end_embed = embed_tokens(vision_end_ids) * embed_to_img_scale
        
        # Expand for batch: (batch_size, 1, embed_dim)
        vision_start_batch = vision_start_embed.expand(batch_size, -1, -1)
        vision_end_batch = vision_end_embed.expand(batch_size, -1, -1)
        
        # ===== Get label embeddings (scaled) =====
        label_embeds = []
        for img_idx in range(num_images):
            label_text = f"Image {img_idx}:" if img_idx == 0 else f"\nImage {img_idx}:"
            label_ids = self.tokenizer(label_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
            label_embed = embed_tokens(label_ids) * embed_to_img_scale
            label_embed_batch = label_embed.expand(batch_size, -1, -1)
            label_embeds.append(label_embed_batch)
        
        # ===== Build image context =====
        # All tokens have consistent magnitude (~31.73, matching img_enc output)
        context_parts = []
        for img_idx in range(num_images):
            img_context = torch.cat([
                label_embeds[img_idx],
                vision_start_batch,
                image_encodings[img_idx],  # (batch_size, 256, embed_dim)
                vision_end_batch,
            ], dim=1)
            context_parts.append(img_context)
        
        image_context = torch.cat(context_parts, dim=1)  # (batch_size, total_image_seq_len, embed_dim)
        image_seq_len = image_context.shape[1]
        
        # ===== Pre-compute KV cache =====
        # load_bases will scale per-layer to match typical activation magnitudes
        past_key_values = self.model.load_bases(image_context)
        
        return past_key_values, image_seq_len, image_encodings
    
    def batch_forward(
        self,
        input_ids: torch.LongTensor,
        images: Optional[List[torch.Tensor]] = None,  # bf16 tensors: (batch_size, 3, 224, 224)
        attention_mask: Optional[torch.LongTensor] = None,
        generate_image: bool = True,
    ):
        """
        Batch forward pass working directly with tensors.
        
        Uses KV-cache pre-computation for efficiency: image embeddings are processed
        via load_bases() to compute K/V values directly, then only text is passed
        through the full transformer layers.
        
        Each batch item has its OWN images (not shared across batch).
        
        Args:
            input_ids: Tokenized text inputs (batch_size, text_seq_len)
            images: List of image tensors, where each tensor is (batch_size, 3, 224, 224).
                    len(images) = num_images. Each batch item gets one image from each tensor.
                    None for no images.
            attention_mask: Optional attention mask for text (batch_size, text_seq_len)
            generate_image: Whether to generate output images (default: True)
            
        Returns:
            Dictionary with:
                - outputs: Model outputs (includes logits for TEXT ONLY)
                - image_encodings: Encoded images (num_images, batch_size, 256, embed_dim) or None
                - inputs_embeds: Text embeddings only (batch_size, text_seq_len, embed_dim)
                - attention_mask: Full attention mask (including cached image tokens)
                - image_seq_len: Length of image context (in cache, not in outputs)
                - generated_images: Generated images (batch_size, ...) if generate_image=True
        """
        batch_size = input_ids.shape[0]
        input_ids = input_ids.to(self.device)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        
        embed_tokens = self._get_embed_tokens()
        
        # ===== Build image context and KV cache using shared helper =====
        # Images are per-batch-item (not shared), so shared_images=False
        past_key_values, image_seq_len, image_encodings = self._build_image_context_and_cache(
            images=images,
            batch_size=batch_size,
            shared_images=False,
        )
        
        # ===== Get text embeddings (this is ALL we pass through the model) =====
        text_embeds = embed_tokens(input_ids)  # (batch_size, text_seq_len, embed_dim)
        text_seq_len = text_embeds.shape[1]
        
        # ===== Build attention mask and position IDs =====
        if image_seq_len > 0:
            # Extend attention mask to include cached image tokens
            cache_attention = torch.ones(batch_size, image_seq_len, device=self.device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([cache_attention, attention_mask], dim=1)
            
            # Position IDs start after the cached sequence
            position_ids = torch.arange(
                image_seq_len, 
                image_seq_len + text_seq_len, 
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # Cache position for proper cache handling
            cache_position = torch.arange(
                image_seq_len,
                image_seq_len + text_seq_len,
                device=self.device
            )
        else:
            full_attention_mask = attention_mask
            position_ids = None
            cache_position = None
        
        # ===== Forward ONLY text through model (images are in KV cache) =====
        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=generate_image,  # Only need hidden states if generating images
        )
        
        # ===== Generate images =====
        generated_images = None
        if generate_image:
            # Get text encoding from output (no slicing needed - output is text only)
            # CausalLMOutputWithPast has hidden_states tuple
            last_hidden_state = outputs.hidden_states[-1]
            text_context = last_hidden_state  # Already text only, keep in bf16
            
            # Determine decoder input
            if image_encodings is not None:
                # Use the encoding of the last image for each batch item
                # image_encodings: (num_images, batch_size, 256, embed_dim)
                decoder_input = image_encodings[-1]  # (batch_size, 256, embed_dim)
            else:
                # Random tensor in bf16
                decoder_input = torch.randn(
                    batch_size, 256, self.embed_dim,
                    device=self.device,
                    dtype=self.torch_dtype
                ) / 32.0
            
            generated_images = self.model.decode_image(
                decoder_input,
                context=text_context
            )
        
        return {
            'outputs': outputs,
            'image_encodings': image_encodings,
            'inputs_embeds': text_embeds,  # Now text only
            'attention_mask': full_attention_mask,
            'image_seq_len': image_seq_len,  # Still tracked for reference
            'generated_images': generated_images,
        }
    
    def forward(
        self,
        text: Union[str, List[str]],
        images: Optional[List[Union[torch.Tensor, np.ndarray, Image.Image]]] = None,
        max_length: int = 512,
        return_dict: bool = True,
        generate_image: bool = True,
    ):
        """
        User-friendly forward pass that handles strings and various image formats.
        
        Note: In this function, images are SHARED across all texts in the batch.
        For per-batch-item images, use batch_forward() directly.
        
        Args:
            text: Input text or list of texts
            images: Optional list of images (same images used for all texts in batch)
            max_length: Maximum sequence length
            return_dict: Whether to return a dictionary of outputs
            generate_image: Whether to generate an output image (default: True)
            
        Returns:
            Model outputs with image embeddings integrated, and optionally generated image
        """
        # Handle single text input
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        
        # Handle empty or None images
        if images is None:
            images = []
        num_images = len(images)
        
        # Note: With KV-cache approach, images are processed via load_bases
        # and are not prepended to text. input_strings just contains the text.
        input_strings = list(text)
        
        # ===== Tokenize text =====
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # ===== Prepare images as list of tensors =====
        # In forward(), images are shared across batch
        # batch_forward expects List[Tensor] where each Tensor is (batch_size, 3, 224, 224)
        images_list = None
        if num_images > 0:
            # Preprocess images: (num_images, 3, 224, 224)
            preprocessed = self._preprocess_images(images)
            # Convert to list, expanding each image for batch
            # Each image becomes (batch_size, 3, 224, 224) by repeating
            images_list = [
                preprocessed[i].unsqueeze(0).expand(batch_size, -1, -1, -1)
                for i in range(num_images)
            ]
        
        # ===== Call batch_forward =====
        batch_result = self.batch_forward(
            input_ids=input_ids,
            images=images_list,
            attention_mask=attention_mask,
            generate_image=generate_image,
        )
        
        # ===== Format output =====
        if return_dict:
            result = {
                'outputs': batch_result['outputs'],
                'input_ids': input_ids,
                'inputs_embeds': batch_result['inputs_embeds'],
                'image_encodings': batch_result['image_encodings'],
                'attention_mask': batch_result['attention_mask'],
                'image_seq_len': batch_result['image_seq_len'],
                'input_strings': input_strings,
            }
            if generate_image:
                result['generated_image'] = batch_result['generated_images']
            return result
        
        if generate_image:
            return batch_result['outputs'], batch_result['generated_images']
        return batch_result['outputs']
    
    def generate(
        self,
        text: Union[str, List[str]],
        images: Optional[List[Union[torch.Tensor, np.ndarray, Image.Image]]] = None,
        max_new_tokens: int = 100,
        generate_image: bool = True,
        **generate_kwargs
    ):
        """
        Generate text and optionally an image given input text and optional images.
        
        Uses KV-cache pre-computation for efficiency: image embeddings are processed
        via load_bases() to compute K/V values directly, then only text is passed
        through the full transformer layers.
        
        Note: In this function, images are SHARED across all texts in the batch.
        
        Args:
            text: Input text or list of texts
            images: Optional list of images (same images used for all texts)
            max_new_tokens: Maximum number of new tokens to generate
            generate_image: Whether to generate an output image (default: True)
            **generate_kwargs: Additional arguments passed to model.generate()
            
        Returns:
            Dictionary with:
                - generated_texts: List of generated text strings
                - generated_image: Generated image tensor (if generate_image=True)
        """
        # Handle single text input
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        
        # Handle empty or None images
        if images is None:
            images = []
        num_images = len(images)
        
        embed_tokens = self._get_embed_tokens()
        
        # ===== Build image context and KV cache using shared helper =====
        # Preprocess images first (convert PIL/numpy to tensor)
        images_list = None
        if num_images > 0:
            preprocessed = self._preprocess_images(images)  # (num_images, 3, 224, 224)
            # Convert to list for the helper - images are shared across batch
            images_list = [preprocessed[i] for i in range(num_images)]
        
        past_key_values, image_seq_len, image_encodings = self._build_image_context_and_cache(
            images=images_list,
            batch_size=batch_size,
            shared_images=True,  # Images are shared across batch in generate()
        )
        
        # ===== Tokenize and embed text =====
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        text_attention_mask = encoded['attention_mask'].to(self.device)
        text_seq_len = input_ids.shape[1]
        
        text_embeds = embed_tokens(input_ids)
        
        # ===== Build attention mask and position IDs =====
        if image_seq_len > 0:
            # Extend attention mask to include cached image tokens
            cache_attention = torch.ones(batch_size, image_seq_len, device=self.device, dtype=text_attention_mask.dtype)
            attention_mask = torch.cat([cache_attention, text_attention_mask], dim=1)
            
            # Position IDs start after the cached sequence
            position_ids = torch.arange(
                image_seq_len, 
                image_seq_len + text_seq_len, 
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            attention_mask = text_attention_mask
            position_ids = None
        
        # Generate text tokens (only text embeds, images in cache)
        generated_ids = self.model.qwen_model.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs
        )
        
        # Decode generated tokens
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # ===== Generate image =====
        generated_image = None
        if generate_image:
            # We need hidden states for image decoding
            # Re-run forward pass with generated tokens using KV cache approach
            generated_embeds = embed_tokens(generated_ids)  # (batch_size, gen_seq_len, embed_dim)
            gen_seq_len = generated_embeds.shape[1]
            
            # Rebuild KV cache for image (need fresh cache for this forward pass)
            if num_images > 0:
                # Use helper to rebuild fresh cache (re-uses images_list from above)
                past_key_values_fresh, _, _ = self._build_image_context_and_cache(
                    images=images_list,
                    batch_size=batch_size,
                    shared_images=True,
                )
                
                # Attention mask for generated tokens + cached image
                cache_attention = torch.ones(batch_size, image_seq_len, device=self.device, dtype=attention_mask.dtype)
                gen_attention = torch.ones(batch_size, gen_seq_len, device=self.device, dtype=attention_mask.dtype)
                full_attention = torch.cat([cache_attention, gen_attention], dim=1)
                
                # Position IDs for generated tokens
                gen_position_ids = torch.arange(
                    image_seq_len,
                    image_seq_len + gen_seq_len,
                    device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
                
                # Cache position
                cache_position = torch.arange(
                    image_seq_len,
                    image_seq_len + gen_seq_len,
                    device=self.device
                )
            else:
                past_key_values_fresh = None
                full_attention = torch.ones(batch_size, gen_seq_len, device=self.device, dtype=torch.long)
                gen_position_ids = None
                cache_position = None
            
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=generated_embeds,
                    attention_mask=full_attention,
                    past_key_values=past_key_values_fresh,
                    position_ids=gen_position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    output_hidden_states=True,
                )
            
            # Get hidden states (already text only - no slicing needed)
            last_hidden_state = outputs.hidden_states[-1]
            text_context = last_hidden_state  # Keep in bf16
            
            # Decoder input: last image encoding or random
            if image_encodings is not None:
                # image_encodings: (num_images, 256, embed_dim) - use last, expand for batch
                decoder_input = image_encodings[-1].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                decoder_input = torch.randn(
                    batch_size, 256, self.embed_dim,
                    device=self.device,
                    dtype=self.torch_dtype
                ) / 32.0
            
            generated_image = self.model.decode_image(
                decoder_input,
                context=text_context
            )
        
        return {
            'generated_texts': generated_texts,
            'generated_image': generated_image,
        }


class QwenAgentPlayer:
    """
    A stateful agent that maintains a rolling window of canvas images.
    
    Each forward/generate/batch_forward call:
    1. Receives one input image (NOT stored in canvases)
    2. Uses canvases + input_image as the image list for the underlying QwenAgentPipe
    3. Generates a new image (appended to canvases)
    4. Keeps only the last 3 images in canvases
    
    Attributes:
        pipe: The underlying QwenAgentPipe
        canvases: List of torch tensors (max 3), representing generated image history
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        embed_dim: int = 1024,
        num_heads: int = 8,
        device: str = None,
    ):
        """
        Initialize the QwenAgentPlayer.
        
        Args:
            model_name: Qwen model name/path
            embed_dim: Embedding dimension for image encoder/decoder
            num_heads: Number of attention heads
            device: Device to use (defaults to cuda if available)
        """
        self.pipe = QwenAgentPipe(
            model_name=model_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
        )
        self.canvases: List[torch.Tensor] = []
        self.device = self.pipe.device
    
    def _to_tensor(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """Convert an image to a torch tensor (3, 224, 224) in bf16."""
        if isinstance(image, torch.Tensor):
            img_tensor = image
        elif isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).to(self.pipe.torch_dtype)
            if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        elif isinstance(image, Image.Image):
            img_tensor = torch.from_numpy(np.array(image)).to(self.pipe.torch_dtype)
            if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Normalize to [0, 1] if needed
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Resize if needed
        if img_tensor.shape[-2:] != (224, 224):
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        # Ensure bf16
        if img_tensor.dtype != self.pipe.torch_dtype:
            img_tensor = img_tensor.to(self.pipe.torch_dtype)
        
        return img_tensor.to(self.device)
    
    def _trim_canvases(self):
        """Keep only the last 3 canvases."""
        if len(self.canvases) > 3:
            self.canvases = self.canvases[-3:]
    
    def reset(self):
        """Clear all canvases."""
        self.canvases = []
    
    def soft_reset(self):
        """Detach all canvas tensors from the computation graph."""
        self.canvases = [canvas.detach() for canvas in self.canvases]
    
    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        max_length: int = 512,
        return_dict: bool = True,
        generate_image: bool = True,
    ):
        """
        Forward pass with a new input image.
        
        The input image + canvases are used as context, and the generated 
        output image is appended to canvases (input image is NOT stored).
        
        Args:
            text: Input text or list of texts
            image: Single input image (used as context, not stored in canvases)
            max_length: Maximum sequence length
            return_dict: Whether to return a dictionary
            generate_image: Whether to generate output image (default: True)
            
        Returns:
            Model outputs (same as QwenAgentPipe.forward)
        """
        # Convert input image to tensor
        img_tensor = self._to_tensor(image)
        
        # Call pipe.forward with canvases + input image (input NOT stored)
        result = self.pipe.forward(
            text=text,
            images=self.canvases + [img_tensor],
            max_length=max_length,
            return_dict=return_dict,
            generate_image=generate_image,
        )
        
        # Only the generated image gets stored in canvases
        if generate_image and result.get('generated_image') is not None:
            gen_img = result['generated_image']
            if gen_img.dim() == 4:
                gen_img = gen_img[0]  # Take first batch item: (C, H, W)
            self.canvases.append(gen_img)  # Don't detach - may need gradients for training
        
        # Trim to last 3
        self._trim_canvases()
        
        return result
    
    def generate(
        self,
        text: Union[str, List[str]],
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        max_new_tokens: int = 100,
        generate_image: bool = True,
        **generate_kwargs
    ):
        """
        Generate text and image with a new input image.
        
        The input image + canvases are used as context, and the generated
        output image is appended to canvases (input image is NOT stored).
        
        Args:
            text: Input text or list of texts
            image: Single input image (used as context, not stored in canvases)
            max_new_tokens: Maximum new tokens to generate
            generate_image: Whether to generate output image (default: True)
            **generate_kwargs: Additional generation arguments
            
        Returns:
            Dictionary with generated_texts and generated_image
        """
        # Convert input image to tensor
        img_tensor = self._to_tensor(image)
        
        # Call pipe.generate with canvases + input image (input NOT stored)
        result = self.pipe.generate(
            text=text,
            images=self.canvases + [img_tensor],
            max_new_tokens=max_new_tokens,
            generate_image=generate_image,
            **generate_kwargs
        )
        
        # Only the generated image gets stored in canvases
        if generate_image and result.get('generated_image') is not None:
            gen_img = result['generated_image']
            if gen_img.dim() == 4:
                gen_img = gen_img[0]  # Take first batch item: (C, H, W)
            self.canvases.append(gen_img)  # Don't detach - may need gradients for training
        
        # Trim to last 3
        self._trim_canvases()
        
        return result
    
    def batch_forward(
        self,
        input_ids: torch.LongTensor,
        image: torch.Tensor,  # bf16 tensor: (batch_size, 3, 224, 224)
        attention_mask: Optional[torch.LongTensor] = None,
        generate_image: bool = True,
    ):
        """
        Batch forward pass with a new batch of input images.
        
        The input image + canvases are used as context, and the generated
        output images are appended to canvases (input image is NOT stored).
        
        Args:
            input_ids: Tokenized text inputs (batch_size, text_seq_len)
            image: Batch of images (batch_size, 3, 224, 224) - used as context, not stored
            attention_mask: Optional attention mask
            generate_image: Whether to generate output images
            
        Returns:
            Model outputs (same as QwenAgentPipe.batch_forward)
        """
        # Ensure image is on device
        image = image.to(self.device)
        
        # Call pipe.batch_forward with canvases + input image (input NOT stored)
        result = self.pipe.batch_forward(
            input_ids=input_ids,
            images=self.canvases + [image],
            attention_mask=attention_mask,
            generate_image=generate_image,
        )
        
        # Only the generated images get stored in canvases
        if generate_image and result.get('generated_images') is not None:
            gen_imgs = result['generated_images']  # Don't detach - may need gradients
            if gen_imgs.dim() == 4:
                self.canvases.append(gen_imgs)
        
        # Trim to last 3
        self._trim_canvases()
        
        return result


# Convenience exports
__all__ = ['QwenExtension', 'QwenAgentPipe', 'QwenAgentPlayer']
