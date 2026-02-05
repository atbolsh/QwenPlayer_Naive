"""
Widget Interface for QwenAgentPlayer

This module provides helper functions for interactive exploration of the
QwenAgentPlayer model using Jupyter widgets.

Usage in notebook:
    from widget_interface import WidgetInterface
    interface = WidgetInterface(model, game)
    interface.display()
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets

from game import discreteGame, BIG_tool_use_advanced_2_5


# Action token mappings - will be populated dynamically from tokenizer
# These are placeholders, actual IDs set in __init__ based on tokenizer
SPECIAL_SYMBOLS = set()  # Token IDs for special actions
SYMBOL_ACTION_MAP = {}  # Token ID -> game action index


class WidgetInterface:
    """
    Interactive widget interface for QwenAgentPlayer.
    
    Provides buttons for:
    - Running forward passes
    - Extending text generation
    - Generating complete responses
    - Resetting model state
    - Viewing internal canvases
    """
    
    def __init__(
        self, 
        model, 
        device,
        tokenizer,
        game_settings=None,
        max_len: int = 32,
        temp: float = 1.0,
        temp_eps: float = 1e-4,
    ):
        """
        Initialize the widget interface.
        
        Args:
            model: QwenAgentPlayer instance
            device: Torch device
            tokenizer: Tokenizer for text encoding/decoding
            game_settings: Optional game settings (uses default if None)
            max_len: Maximum text length for generation
            temp: Temperature for sampling
            temp_eps: Epsilon for temperature sampling
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.temp = temp
        self.temp_eps = temp_eps
        
        # Build action token mappings from tokenizer
        self._build_action_mappings()
        
        # Initialize game
        if game_settings is None:
            game_settings = BIG_tool_use_advanced_2_5
        game_settings.gameSize = 224
        self.game_settings = game_settings
        self.game = discreteGame(game_settings)
        
        # Optional override for input tensor
        self.inp_tensor = None
        
        # Create widgets
        self._create_widgets()
    
    def _build_action_mappings(self):
        """Build action token ID mappings from tokenizer."""
        # Get token IDs for game action tokens
        forward_id = self.tokenizer.convert_tokens_to_ids('<forward>')
        clock_id = self.tokenizer.convert_tokens_to_ids('<clock>')
        anticlock_id = self.tokenizer.convert_tokens_to_ids('<anticlock>')
        
        # Set of action token IDs
        self.action_token_set = {forward_id, clock_id, anticlock_id}
        
        # Map token IDs to game action indices (based on game.actions order)
        self.token_to_action = {
            forward_id: 1,     # forward
            clock_id: 3,       # clockwise
            anticlock_id: 4,   # anticlockwise
        }
    
    def _create_widgets(self):
        """Create all UI widgets."""
        self.output = widgets.Output(layout={'border': '1px solid black'})
        self.text_input = widgets.Textarea(
            value='Hello World',
            placeholder='Type a prompt...',
            description='Input:',
            disabled=False,
            layout=widgets.Layout(width='80%', height='80px')
        )
        
        self.btn_game_reset = widgets.Button(description="Reset Game")
        self.btn_forward = widgets.Button(description="Forward Pass")
        self.btn_soft_reset = widgets.Button(description="Soft Reset")
        self.btn_reset = widgets.Button(description="Hard Reset")
        self.btn_extend = widgets.Button(description="Extend (One Token)")
        self.btn_generate = widgets.Button(description="Generate Full")
        self.btn_canvases = widgets.Button(description="Show Canvases")
        self.btn_clear = widgets.Button(description="Clear Output")
        
        # Bind callbacks
        self.btn_game_reset.on_click(self._reset_game)
        self.btn_forward.on_click(self._forward)
        self.btn_soft_reset.on_click(self._soft_reset)
        self.btn_reset.on_click(self._reset)
        self.btn_extend.on_click(self._extend)
        self.btn_generate.on_click(self._generate)
        self.btn_canvases.on_click(self._show_canvases)
        self.btn_clear.on_click(self._clear_output)
    
    def display(self):
        """Display the widget interface."""
        # Create centered button rows
        button_row1 = widgets.HBox(
            [
                self.btn_game_reset,
                self.btn_forward,
                self.btn_soft_reset,
                self.btn_reset,
            ],
            layout=widgets.Layout(
                justify_content='center',
                margin='10px 0px',
            )
        )
        button_row2 = widgets.HBox(
            [
                self.btn_extend,
                self.btn_generate,
                self.btn_canvases,
                self.btn_clear,
            ],
            layout=widgets.Layout(
                justify_content='center',
                margin='10px 0px',
            )
        )
        
        # Container for everything, centered
        container = widgets.VBox(
            [button_row1, button_row2, self.text_input, self.output],
            layout=widgets.Layout(
                align_items='center',
                width='100%',
            )
        )
        
        display(container)
        
        # Initial display
        self._show_canvases(None)
    
    def reset_game(self):
        """Reset the game to a new random state."""
        self.game = discreteGame(
            self.game.random_bare_settings(gameSize=224, max_agent_offset=0.5)
        )
    
    def get_image(self):
        """Get current game image as tensor."""
        img = torch.FloatTensor(self.game.getData()).unsqueeze(0)
        img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(self.device)
        return img
    
    def pre_imshow_numpy(self, torch_img):
        """Convert torch image to numpy for display."""
        clean = torch_img.detach()[0].cpu()
        right_order = torch.permute(clean, (1, 2, 0))
        array = right_order.numpy()
        return array
    
    # Widget callbacks
    
    def _reset_game(self, b):
        with self.output:
            self.output.clear_output()
            self.reset_game()
            print("Game reset!")
            plt.imshow(self.game.getData())
            plt.show()
    
    @property
    def _input_image(self):
        """Get input image (override or from game)."""
        if self.inp_tensor is not None:
            return self.inp_tensor
        return self.get_image()
    
    def _forward(self, b):
        with self.output:
            self.output.clear_output()
            
            if self.inp_tensor is None:
                print("Using game image as input\n")
            else:
                print("Using custom inp_tensor as input\n")
            
            local_tensor = self._input_image
            text = self.text_input.value
            
            # Squeeze batch dimension if present - forward expects (C, H, W) per image
            img_for_forward = local_tensor.squeeze(0) if local_tensor.dim() == 4 else local_tensor
            
            # Run model forward (QwenAgentPlayer.forward handles canvas storage)
            result = self.model.forward(
                text=[text],
                image=img_for_forward.to(torch.bfloat16),
                generate_image=True,
                return_dict=True,
            )
            
            # Display output
            print("Output image:\n")
            if result.get('generated_image') is not None:
                # Convert from bf16 to float for display
                img = result['generated_image'].float()
                plt.imshow(self.pre_imshow_numpy(img))
                plt.show()
            else:
                print("No image generated")
            
            self._show_canvases(b)
    
    def _soft_reset(self, b):
        with self.output:
            self.output.clear_output()
            print("Soft reset (clearing internal gradients)\n")
            self.model.soft_reset()
    
    def _reset(self, b):
        with self.output:
            self.output.clear_output()
            print("Hard reset (clearing canvases and history)\n")
            self.model.reset()
    
    def _extend(self, b):
        """Extend text by one token."""
        with self.output:
            self.output.clear_output()
            
            print("Game status:\n")
            plt.imshow(self.game.getData())
            plt.show()
            
            local_tensor = self._input_image
            text = self.text_input.value
            
            # Squeeze batch dimension if present - forward expects (C, H, W) per image
            img_for_forward = local_tensor.squeeze(0) if local_tensor.dim() == 4 else local_tensor
            
            # Run model forward (QwenAgentPlayer.forward handles canvas storage)
            result = self.model.forward(
                text=[text],
                image=img_for_forward.to(torch.bfloat16),
                generate_image=True,
                return_dict=True,
            )
            
            # Get next token prediction from outputs
            outputs = result.get('outputs')
            if outputs is not None and hasattr(outputs, 'logits'):
                logits = outputs.logits  # (batch, seq_len, vocab_size)
                # Get last token logits
                last_logits = logits[0, -1]  # (vocab_size,)
                next_token = torch.argmax(last_logits).item()
                
                # Check for special action tokens
                if next_token in self.action_token_set:
                    action = self.token_to_action[next_token]
                    token_name = self.tokenizer.decode([next_token])
                    print(f"Detected action token '{token_name}' (ID {next_token}) -> action {action}")
                    reward = self.game.actions[action]()
                    print(f"Reward: {reward}")
                
                # Decode and append token
                next_text = self.tokenizer.decode([next_token])
                self.text_input.value = text + next_text
            
            print(f"Updated text: {self.text_input.value}\n")
            
            print("Output image:\n")
            if result.get('generated_image') is not None:
                # Convert from bf16 to float for display
                img = result['generated_image'].float()
                plt.imshow(self.pre_imshow_numpy(img))
                plt.show()
            
            self._show_canvases(b)
    
    def _generate(self, b):
        """Generate text until EOS token or max length."""
        with self.output:
            self.output.clear_output()
            print("Generating...\n")
            
            eos_token = self.tokenizer.eos_token  # e.g., '<|im_end|>' for Qwen
            generated_tokens = 0
            
            while generated_tokens < self.max_len:
                if eos_token in self.text_input.value:
                    print(f"\nGeneration complete (found {eos_token})")
                    break
                self._extend(b)
                generated_tokens += 1
                time.sleep(0.5)
            
            if generated_tokens >= self.max_len:
                print(f"\nGeneration stopped (max {self.max_len} tokens)")
    
    def _show_canvases(self, b):
        """Display internal canvas state."""
        with self.output:
            print("\nGame status:\n")
            plt.imshow(self.game.getData())
            plt.show()
            
            print("\nCanvas history:")
            if hasattr(self.model, 'canvases') and len(self.model.canvases) > 0:
                for i, canvas in enumerate(self.model.canvases):
                    print(f"\nCanvas {i}:")
                    # Handle 3D (C,H,W) or 4D (B,C,H,W) tensors, convert bf16 to float
                    img = canvas.float()
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    plt.imshow(self.pre_imshow_numpy(img))
                    plt.show()
            else:
                print("No canvas history available")
    
    def _clear_output(self, b):
        self.output.clear_output()


def create_interface(model, device, tokenizer):
    """
    Convenience function to create a widget interface.
    
    Args:
        model: QwenAgentPlayer instance
        device: Torch device
        tokenizer: Tokenizer instance
        
    Returns:
        WidgetInterface instance
    """
    return WidgetInterface(model, device, tokenizer)
