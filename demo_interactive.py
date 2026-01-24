"""
Demo Script: Interactive Model Exploration

This script provides an interactive session for exploring the QwenAgentPlayer
model without Jupyter widgets. Useful for terminal-based exploration.

Run with: python demo_interactive.py
"""

import torch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

from frameworks import device, create_model, tokenizer
from frameworks import G, discreteGame


class InteractiveDemo:
    """Interactive demo for QwenAgentPlayer."""
    
    def __init__(self):
        print("Initializing QwenAgentPlayer...")
        self.model = create_model(device=device, use_lora=False)
        self.device = device
        self.tokenizer = tokenizer
        self.reset_game()
        print("Ready!")
    
    def reset_game(self):
        """Reset to a new random game state."""
        settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.5)
        self.game = discreteGame(settings)
        print("Game reset!")
    
    def show_game(self):
        """Display current game state."""
        plt.figure(figsize=(6, 6))
        plt.imshow(self.game.getData())
        plt.title("Current Game State")
        plt.axis('off')
        plt.show()
    
    def get_image_tensor(self):
        """Get current game as tensor."""
        img = torch.FloatTensor(self.game.getData()).unsqueeze(0)
        img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(self.device)
        return img
    
    def forward(self, text: str, show_output: bool = True):
        """
        Run a forward pass with given text.
        
        Args:
            text: Input text prompt
            show_output: Whether to display output image
            
        Returns:
            Result object with logits and generated image
        """
        img_tensor = self.get_image_tensor()
        
        result = self.model.pipe.forward(
            texts=[text],
            images=img_tensor,
            generate_image=True,
        )
        
        if show_output and result.generated_image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Input
            axes[0].imshow(self.game.getData())
            axes[0].set_title("Input")
            axes[0].axis('off')
            
            # Output
            output_img = result.generated_image[0].detach().cpu()
            output_img = torch.permute(output_img, (1, 2, 0)).numpy()
            axes[1].imshow(output_img)
            axes[1].set_title("Output")
            axes[1].axis('off')
            
            plt.suptitle(f"Prompt: {text}")
            plt.tight_layout()
            plt.show()
        
        return result
    
    def take_action(self, action: int):
        """
        Execute an action in the game.
        
        Args:
            action: 1=forward, 3=clockwise, 4=counter-clockwise
        """
        action_names = {1: "forward", 3: "clockwise", 4: "counter-clockwise"}
        if action in self.game.actions:
            reward = self.game.actions[action]()
            print(f"Action: {action_names.get(action, action)}, Reward: {reward}")
        else:
            print(f"Invalid action: {action}")
    
    def reset_model(self):
        """Reset model state (canvases, etc.)."""
        self.model.reset()
        print("Model state reset!")
    
    def show_canvases(self):
        """Display model's canvas history."""
        if hasattr(self.model, 'canvases') and len(self.model.canvases) > 0:
            n = len(self.model.canvases)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            if n == 1:
                axes = [axes]
            
            for i, canvas in enumerate(self.model.canvases):
                img = canvas[0].detach().cpu()
                img = torch.permute(img, (1, 2, 0)).numpy()
                axes[i].imshow(img)
                axes[i].set_title(f"Canvas {i}")
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No canvas history available")
    
    def interactive_loop(self):
        """Run an interactive command loop."""
        print("\n" + "=" * 50)
        print("Interactive QwenAgentPlayer Demo")
        print("=" * 50)
        print("\nCommands:")
        print("  show      - Show current game state")
        print("  reset     - Reset game to new random state")
        print("  forward   - Move agent forward")
        print("  cw        - Turn clockwise")
        print("  ccw       - Turn counter-clockwise")
        print("  prompt    - Enter a text prompt for the model")
        print("  canvases  - Show canvas history")
        print("  clear     - Reset model state")
        print("  quit      - Exit")
        print("=" * 50 + "\n")
        
        while True:
            try:
                cmd = input(">>> ").strip().lower()
                
                if cmd == "quit" or cmd == "exit" or cmd == "q":
                    print("Goodbye!")
                    break
                elif cmd == "show":
                    self.show_game()
                elif cmd == "reset":
                    self.reset_game()
                elif cmd == "forward":
                    self.take_action(1)
                elif cmd == "cw":
                    self.take_action(3)
                elif cmd == "ccw":
                    self.take_action(4)
                elif cmd == "prompt":
                    text = input("Enter prompt: ")
                    self.forward(text)
                elif cmd == "canvases":
                    self.show_canvases()
                elif cmd == "clear":
                    self.reset_model()
                elif cmd == "":
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    demo = InteractiveDemo()
    demo.interactive_loop()


if __name__ == "__main__":
    main()
