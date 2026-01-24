# Parallel Frameworks Package
# Provides data-parallel training infrastructure for QwenAgentPlayer
#
# Data Parallel Approach:
# - Multiple frameworks run on different batches simultaneously
# - Single model instance shared across all frameworks
# - Gradients accumulated and applied together

from .data_parallel import (
    ParallelFrameworkRunner,
    create_parallel_batches,
    run_parallel_training_step,
)

__all__ = [
    'ParallelFrameworkRunner',
    'create_parallel_batches',
    'run_parallel_training_step',
]
