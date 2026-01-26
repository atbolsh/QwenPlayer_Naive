# QwenAgent - Extended Qwen3 model with vision capabilities
# Implements qwen_extension (extended Qwen3Model) and qwen_agent_pipe (full pipeline)

import torch
import torch.nn as nn
from typing import Optional, List, Union
from PIL import Image
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import image encoder/decoder from model.py (with 1024 embed_dim, 8 heads as per negative_example.py)
from .model import ImageTransformerEncoder, ImageTransformerDecoder


class QwenExtension(nn.Module):
    """
    Extension of Qwen3Model with image encoder/decoder.
    
    This class wraps a Qwen3 model and adds:
    - Image encoder (1024 embed_dim, 8 heads)
    - Image decoder (1024 embed_dim, 8 heads)
    """
    
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
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        Forward pass through the extended Qwen model.
        """
        return self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
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
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
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
        Preprocess a list of images to tensor format.
        
        Args:
            images: List of images (can be tensors, numpy arrays, or PIL images)
            
        Returns:
            Tensor of shape (num_images, channels, height, width)
        """
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                # Convert PIL to tensor
                img = img.resize((224, 224))
                img = np.array(img) / 255.0
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            elif isinstance(img, np.ndarray):
                if img.max() > 1.0:
                    img = img / 255.0
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                img = torch.tensor(img, dtype=torch.float32)
                if img.shape[-1] == 3:  # HWC -> CHW
                    img = img.permute(2, 0, 1)
            elif isinstance(img, torch.Tensor):
                if img.shape[-1] == 3:  # HWC -> CHW
                    img = img.permute(2, 0, 1)
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
        """Get the embedding layer from the model."""
        if hasattr(self.model.qwen_model, 'model'):
            return self.model.qwen_model.model.embed_tokens
        else:
            return self.model.qwen_model.embed_tokens
    
    def batch_forward(
        self,
        input_ids: torch.LongTensor,
        images: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generate_image: bool = True,
    ):
        """
        Batch forward pass working directly with tensors.
        
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
                - outputs: Model outputs (includes last_hidden_state, logits if applicable)
                - image_encodings: Encoded images (num_images, batch_size, 256, embed_dim) or None
                - inputs_embeds: Full input embeddings (batch_size, total_seq_len, embed_dim)
                - attention_mask: Full attention mask
                - image_seq_len: Length of image context per batch item
                - generated_images: Generated images (batch_size, ...) if generate_image=True
        """
        batch_size = input_ids.shape[0]
        input_ids = input_ids.to(self.device)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        
        embed_tokens = self._get_embed_tokens()
        
        # ===== Encode images =====
        image_encodings = None
        num_images = 0
        
        if images is not None and len(images) > 0:
            num_images = len(images)
            
            # Stack all images: (num_images, batch_size, 3, 224, 224)
            stacked_images = torch.stack([img.to(self.device) for img in images], dim=0)
            
            # Reshape to encode all at once: (num_images * batch_size, 3, 224, 224)
            flat_images = stacked_images.view(-1, 3, 224, 224)
            
            # Encode: (num_images * batch_size, 256, embed_dim)
            flat_encodings = self.model.encode_image(flat_images)
            
            # Convert to model dtype (img_enc is float32, but Qwen model may be bfloat16)
            flat_encodings = flat_encodings.to(self.torch_dtype)
            
            # Reshape back: (num_images, batch_size, 256, embed_dim)
            image_encodings = flat_encodings.view(num_images, batch_size, 256, self.embed_dim)
        
        # ===== Build image context (no for-loop over batch_size) =====
        image_context = None
        image_seq_len = 0
        
        if image_encodings is not None:
            # Get vision start/end embeddings (same for all)
            vision_start_ids = torch.tensor([[self.begin_vision_id]], device=self.device)
            vision_end_ids = torch.tensor([[self.end_vision_id]], device=self.device)
            vision_start_embed = embed_tokens(vision_start_ids)  # (1, 1, embed_dim)
            vision_end_embed = embed_tokens(vision_end_ids)      # (1, 1, embed_dim)
            
            # Expand for batch: (batch_size, 1, embed_dim)
            vision_start_batch = vision_start_embed.expand(batch_size, -1, -1)
            vision_end_batch = vision_end_embed.expand(batch_size, -1, -1)
            
            # Tokenize all labels ONCE (they're the same for all batch items)
            label_embeds = []
            for img_idx in range(num_images):
                label_text = f"Image {img_idx}:" if img_idx == 0 else f"\nImage {img_idx}:"
                label_ids = self.tokenizer(label_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
                label_embed = embed_tokens(label_ids)  # (1, label_len, embed_dim)
                # Expand for batch: (batch_size, label_len, embed_dim)
                label_embed_batch = label_embed.expand(batch_size, -1, -1)
                label_embeds.append(label_embed_batch)
            
            # Build image context by concatenating for each image
            # For image i: [label[i], vision_start, image_enc[i], vision_end]
            context_parts = []
            for img_idx in range(num_images):
                # label_embeds[img_idx]: (batch_size, label_len, embed_dim)
                # vision_start_batch: (batch_size, 1, embed_dim)
                # image_encodings[img_idx]: (batch_size, 256, embed_dim)
                # vision_end_batch: (batch_size, 1, embed_dim)
                img_context = torch.cat([
                    label_embeds[img_idx],
                    vision_start_batch,
                    image_encodings[img_idx],  # (batch_size, 256, embed_dim)
                    vision_end_batch,
                ], dim=1)
                context_parts.append(img_context)
            
            # Concatenate all image contexts: (batch_size, total_image_seq_len, embed_dim)
            image_context = torch.cat(context_parts, dim=1)
            image_seq_len = image_context.shape[1]
        
        # ===== Get text embeddings =====
        text_embeds = embed_tokens(input_ids)  # (batch_size, text_seq_len, embed_dim)
        
        # ===== Concatenate image context + text =====
        if image_context is not None:
            inputs_embeds = torch.cat([image_context, text_embeds], dim=1)
            
            # Build attention mask
            image_attention = torch.ones(batch_size, image_seq_len, device=self.device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds
            full_attention_mask = attention_mask
        
        # ===== Forward through model =====
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=generate_image,  # Only need hidden states if generating images
        )
        
        # ===== Generate images =====
        generated_images = None
        if generate_image:
            # Get the text portion of the output
            # CausalLMOutputWithPast has hidden_states tuple, not last_hidden_state
            last_hidden_state = outputs.hidden_states[-1]
            if image_seq_len > 0:
                text_encoding = last_hidden_state[:, image_seq_len:, :]
            else:
                text_encoding = last_hidden_state
            text_context = text_encoding.float()
            
            # Determine decoder input
            if image_encodings is not None:
                # Use the encoding of the last image for each batch item
                # image_encodings: (num_images, batch_size, 256, embed_dim)
                decoder_input = image_encodings[-1]  # (batch_size, 256, embed_dim)
            else:
                # Random tensor
                decoder_input = torch.randn(
                    batch_size, 256, self.embed_dim,
                    device=self.device,
                    dtype=torch.float32
                ) / 32.0
            
            generated_images = self.model.decode_image(
                decoder_input.float(),
                context=text_context
            )
        
        return {
            'outputs': outputs,
            'image_encodings': image_encodings,
            'inputs_embeds': inputs_embeds,
            'attention_mask': full_attention_mask,
            'image_seq_len': image_seq_len,
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
        
        # ===== Build input string with <|image_pad|> placeholders =====
        image_prefix_string = self._build_image_prefix_string(num_images) if num_images > 0 else ""
        input_strings = [image_prefix_string + t for t in text] if num_images > 0 else list(text)
        
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
        
        # ===== Encode images and build image context =====
        image_context = None
        image_seq_len = 0
        image_encodings = None
        
        if num_images > 0:
            image_tensor = self._preprocess_images(images)
            # image_tensor: (num_images, 3, 224, 224)
            image_encodings = self.model.encode_image(image_tensor)  # (num_images, 256, embed_dim)
            # Convert to model dtype (img_enc is float32, but Qwen model may be bfloat16)
            image_encodings = image_encodings.to(self.torch_dtype)
            
            # Get vision start/end embeddings
            vision_start_ids = torch.tensor([[self.begin_vision_id]], device=self.device)
            vision_end_ids = torch.tensor([[self.end_vision_id]], device=self.device)
            vision_start_embed = embed_tokens(vision_start_ids)  # (1, 1, embed_dim)
            vision_end_embed = embed_tokens(vision_end_ids)      # (1, 1, embed_dim)
            
            # Expand for batch
            vision_start_batch = vision_start_embed.expand(batch_size, -1, -1)
            vision_end_batch = vision_end_embed.expand(batch_size, -1, -1)
            
            # Tokenize labels once, expand for batch
            label_embeds = []
            for img_idx in range(num_images):
                label_text = f"Image {img_idx}:" if img_idx == 0 else f"\nImage {img_idx}:"
                label_ids = self.tokenizer(label_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
                label_embed = embed_tokens(label_ids).expand(batch_size, -1, -1)
                label_embeds.append(label_embed)
            
            # Build image context (images are shared, so expand each encoding for batch)
            context_parts = []
            for img_idx in range(num_images):
                # image_encodings[img_idx]: (256, embed_dim) - expand for batch
                img_enc_batch = image_encodings[img_idx].unsqueeze(0).expand(batch_size, -1, -1)
                img_context = torch.cat([
                    label_embeds[img_idx],
                    vision_start_batch,
                    img_enc_batch,
                    vision_end_batch,
                ], dim=1)
                context_parts.append(img_context)
            
            image_context = torch.cat(context_parts, dim=1)
            image_seq_len = image_context.shape[1]
        
        # ===== Tokenize and embed text =====
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        text_attention_mask = encoded['attention_mask'].to(self.device)
        
        text_embeds = embed_tokens(input_ids)
        
        # ===== Concatenate and generate text =====
        if image_context is not None:
            inputs_embeds = torch.cat([image_context, text_embeds], dim=1)
            image_attention = torch.ones(batch_size, image_seq_len, device=self.device, dtype=text_attention_mask.dtype)
            attention_mask = torch.cat([image_attention, text_attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds
            attention_mask = text_attention_mask
        
        # Generate text tokens
        generated_ids = self.model.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs
        )
        
        # Decode generated tokens
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # ===== Generate image =====
        generated_image = None
        if generate_image:
            # Do a forward pass with generated tokens to get hidden states for image decoding
            # Embed the generated tokens
            generated_embeds = embed_tokens(generated_ids)  # (batch_size, gen_seq_len, embed_dim)
            
            # Build full sequence: image_context + generated_embeds
            if image_context is not None:
                full_embeds = torch.cat([image_context, generated_embeds], dim=1)
                gen_attention = torch.ones(batch_size, generated_ids.shape[1], device=self.device, dtype=attention_mask.dtype)
                full_attention = torch.cat([image_attention, gen_attention], dim=1)
            else:
                full_embeds = generated_embeds
                full_attention = torch.ones_like(generated_ids)
            
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=full_embeds,
                    attention_mask=full_attention,
                    output_hidden_states=True,
                )
            
            # Get text portion of hidden states (after image context)
            # CausalLMOutputWithPast has hidden_states tuple, not last_hidden_state
            last_hidden_state = outputs.hidden_states[-1]
            if image_seq_len > 0:
                text_hidden = last_hidden_state[:, image_seq_len:, :]
            else:
                text_hidden = last_hidden_state
            text_context = text_hidden.float()
            
            # Decoder input: last image encoding or random
            if image_encodings is not None:
                # image_encodings: (num_images, 256, embed_dim) - use last, expand for batch
                decoder_input = image_encodings[-1].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                decoder_input = torch.randn(
                    batch_size, 256, self.embed_dim,
                    device=self.device,
                    dtype=torch.float32
                ) / 32.0
            
            generated_image = self.model.decode_image(
                decoder_input.float(),
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
        """Convert an image to a torch tensor (3, 224, 224)."""
        if isinstance(image, torch.Tensor):
            img_tensor = image
        elif isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).float()
            if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        elif isinstance(image, Image.Image):
            img_tensor = torch.from_numpy(np.array(image)).float()
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
        image: torch.FloatTensor,
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
