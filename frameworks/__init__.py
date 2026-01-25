# Frameworks package for QwenAgentPlayer training
# Each framework provides batch training functions for different tasks

# Core utilities
from .general_framework import (
    device, model, tokenizer, create_model, apply_lora_to_text,
    encode_text, encode_batch, decode_text, decode_batch,
    get_text_loss, model_forward, model_forward_with_tokens,
    sdt, sdv, num_controls, img_criterion,
    MAX_SEQ_LENGTH, vocab_size
)

from .general_framework_lightweight import (
    get_settings_batch, get_images, G, game_settings
)

from .general_qa import (
    tensorify_list, get_lens, simple_sample,
    text_generator, text_generator_simple, text_generator_simple_GENERAL
)

from .game_logic_solver import (
    gold_direction_angle, will_intersect_forward, true_angle_difference_magnitude,
    should_turn_anticlockwise_forward, should_turn_anticlockwise_forward_ENGINE,
    best_move_forward, trace_forward, best_move, trace_any, get_trace
)

# Framework batch functions
from .control import control_batch
from .arrow_to_gold import arrow_task_batch
from .position_qa import qa_task_batch
from .mem_canvas_use import mem_canvas_batch
from .blue_line_qa import blue_line_direction_batch
from .gold_direction_qa import gold_direction_batch
from .near_gold_qa import gold_proximity_batch
from .please_turn_qa import please_turn_batch
from .relposition_qa import relposition_qa_batch
from .direction_names import direction_names_batch
from .zoom import zoom_task_batch, zoomAgent_task_batch, zoomGold_task_batch, zoomHalfway_task_batch
from .comparison_v1 import comparisonv1_task_batch
from .complex_loss_v1 import complex_loss_batch

# Imagination frameworks
from .imagine_without_you import imagineWithoutYou_task_batch
from .imagine_without_gold import imagineWithoutGold_task_batch
from .imagine_without_walls import imagineWithoutWalls_task_batch
from .imagine_walls_only import imagineWallsOnly_task_batch
from .imagine_facing_gold import imagineFacingGold_task_batch
from .imagine_closer_to_gold import imagineCloser2Gold_task_batch
from .imagine_after_move import imagineAfterMove_task_batch

# All exported batch functions
__all__ = [
    # Core utilities
    'device', 'model', 'tokenizer', 'create_model', 'apply_lora_to_text',
    'encode_text', 'encode_batch', 'decode_text', 'decode_batch',
    'get_text_loss', 'model_forward', 'model_forward_with_tokens',
    'sdt', 'sdv', 'num_controls', 'img_criterion',
    'get_settings_batch', 'get_images', 'G', 'game_settings',
    'tensorify_list', 'get_lens', 'simple_sample',
    'text_generator', 'text_generator_simple', 'text_generator_simple_GENERAL',
    
    # Game logic
    'gold_direction_angle', 'will_intersect_forward', 'true_angle_difference_magnitude',
    'should_turn_anticlockwise_forward', 'best_move_forward', 'trace_forward',
    'best_move', 'trace_any', 'get_trace',
    
    # Batch training functions
    'control_batch',
    'arrow_task_batch',
    'qa_task_batch',
    'mem_canvas_batch',
    'blue_line_direction_batch',
    'gold_direction_batch',
    'gold_proximity_batch',
    'please_turn_batch',
    'relposition_qa_batch',
    'direction_names_batch',
    'zoom_task_batch', 'zoomAgent_task_batch', 'zoomGold_task_batch', 'zoomHalfway_task_batch',
    'comparisonv1_task_batch',
    'complex_loss_batch',
    'imagineWithoutYou_task_batch',
    'imagineWithoutGold_task_batch',
    'imagineWithoutWalls_task_batch',
    'imagineWallsOnly_task_batch',
    'imagineFacingGold_task_batch',
    'imagineCloser2Gold_task_batch',
    'imagineAfterMove_task_batch',
]
