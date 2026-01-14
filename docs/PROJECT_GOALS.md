# Project Goals

## Vision

Create an AI agent that can:

1. **Play the game** defined in `game/` — navigating, making decisions, achieving objectives
2. **Communicate intelligently** about the game — explaining what it sees, describing strategies, answering questions about game state
3. **Learn from experience** using reinforcement learning signals (dopamine system)
4. **Maintain context** across multiple interactions (memory system)

## Why QwenBastardBrain?

The architecture combines:

- **Qwen3's language understanding** (pretrained on massive text corpora)
- **Vision transformers** for visual perception
- **Custom memory and reward systems** for game-specific learning

This allows the agent to leverage Qwen's language capabilities while learning game-specific visual and strategic skills.

## The Game

Located in `game/`, the game environment:
- Renders as 224×224 pixel images
- Has discrete actions: `<forward>`, `<clock>` (clockwise rotation), `<anticlock>` (counter-clockwise rotation)
- Contains agents, goals (gold), and obstacles (walls)
- Supports various difficulty levels and configurations

Visual assets are stored in `game_images_and_modifications/`.

## Training Pipeline

1. **Text pretraining** on ProcessBench (mathematical reasoning)
2. **Multimodal training** combining game images with text
3. **Reinforcement learning** using dopamine signals for game objectives
4. **QA fine-tuning** for conversational abilities about game state

## Key Files

| File | Purpose |
|------|---------|
| `visual_transformer/qwen_player.py` | Main model class |
| `general_framework.py` | Shared utilities, tokenizer, datasets |
| `control_framework.py` | Training batches for control/reconstruction |
| `generalQA_framework.py` | Question-answering utilities |
| `game/` | Game environment (do not modify) |

## Future Directions

- Expand memory capacity for longer game sessions
- Add more sophisticated reward shaping
- Multi-game transfer learning
- Real-time gameplay with streaming input

