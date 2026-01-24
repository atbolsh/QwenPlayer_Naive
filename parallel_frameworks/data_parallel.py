"""
Data Parallel Training for QwenAgentPlayer

This module provides data-parallel training where multiple frameworks
process different batches simultaneously using the same model instance.

For single-step frameworks, they are run 'dry' first to load any
necessary state before the actual training begins.
"""

import random
from typing import List, Tuple, Callable, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn as nn


@dataclass
class FrameworkConfig:
    """Configuration for a training framework."""
    batch_func: Callable
    name: str
    weight: int = 1
    is_multi_step: bool = False
    dry_run_steps: int = 0  # For single-step frameworks that need initialization


class ParallelFrameworkRunner:
    """
    Runs multiple frameworks in data-parallel mode.
    
    In data-parallel mode, each framework processes a different batch
    of data simultaneously, with gradients accumulated before the
    optimizer step.
    """
    
    def __init__(
        self,
        model: nn.Module,
        frameworks: List[FrameworkConfig],
        device: torch.device,
    ):
        self.model = model
        self.frameworks = frameworks
        self.device = device
        self.framework_batch_counts = {f.name: 0 for f in frameworks}
        self._lock = threading.Lock()
        
    def _run_single_framework(
        self,
        framework: FrameworkConfig,
        batch_size: int,
        optimizer: Optional[torch.optim.Optimizer],
        training: bool,
        use_lora: bool,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Run a single framework and return results."""
        batch_num = self.framework_batch_counts[framework.name]
        
        try:
            results = framework.batch_func(
                batch_size,
                self.model,
                optimizer=None,  # Don't step optimizer per-framework
                batch_num=batch_num,
                compute_grad=training,
                random_order=True,
                model_eval=not training,
                reset_model=False,  # Don't reset between parallel batches
                printing=False,
                training=False,  # Accumulate gradients, don't step
                use_lora=use_lora,
            )
            
            loss = results[0] if isinstance(results, tuple) else results
            
            with self._lock:
                self.framework_batch_counts[framework.name] += 1
            
            return framework.name, loss, {"results": results}
            
        except Exception as e:
            return framework.name, float('inf'), {"error": str(e)}
    
    def run_parallel_step(
        self,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        num_parallel: int = 4,
        training: bool = True,
        use_lora: bool = False,
    ) -> Dict[str, Any]:
        """
        Run multiple frameworks in parallel, accumulate gradients, then step.
        
        Args:
            batch_size: Batch size per framework
            optimizer: Optimizer for parameter updates
            num_parallel: Number of frameworks to run in parallel
            training: Whether to compute gradients
            use_lora: Whether using LoRA adapters
            
        Returns:
            Dictionary with loss info for each framework
        """
        # Sample frameworks to run
        sampled = self._sample_frameworks(num_parallel)
        
        # Set model to train mode
        if training:
            self.model.pipe.model.train()
            optimizer.zero_grad()
        else:
            self.model.pipe.model.eval()
        
        results = {}
        total_loss = 0.0
        
        # Run frameworks sequentially (GPU operations are already parallel internally)
        # For true data parallelism, you'd need multiple GPUs
        for framework in sampled:
            name, loss, info = self._run_single_framework(
                framework, batch_size, optimizer, training, use_lora
            )
            results[name] = {"loss": loss, **info}
            if loss != float('inf'):
                total_loss += loss
        
        # Step optimizer after all frameworks
        if training and total_loss > 0:
            optimizer.step()
            self.model.soft_reset()
        
        results["total_loss"] = total_loss
        return results
    
    def _sample_frameworks(self, n: int) -> List[FrameworkConfig]:
        """Sample n frameworks weighted by their weights."""
        weighted_pool = []
        for f in self.frameworks:
            weighted_pool.extend([f] * f.weight)
        
        if n >= len(weighted_pool):
            return list(self.frameworks)
        
        sampled = random.sample(weighted_pool, n)
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for f in sampled:
            if f.name not in seen:
                seen.add(f.name)
                result.append(f)
        return result
    
    def dry_run_frameworks(self, batch_size: int = 4):
        """
        Run all frameworks once without training to initialize state.
        
        Useful for multi-step frameworks that need to build up context.
        """
        print("Running dry initialization for all frameworks...")
        self.model.pipe.model.eval()
        
        with torch.no_grad():
            for framework in self.frameworks:
                for step in range(framework.dry_run_steps + 1):
                    try:
                        framework.batch_func(
                            batch_size,
                            self.model,
                            optimizer=None,
                            batch_num=0,
                            compute_grad=False,
                            random_order=False,
                            model_eval=True,
                            reset_model=True,
                            printing=False,
                            training=False,
                        )
                        print(f"  Initialized: {framework.name}")
                    except Exception as e:
                        print(f"  Warning: {framework.name} failed: {e}")
        
        self.model.reset()
        print("Dry initialization complete.")


def create_parallel_batches(
    frameworks: List[Tuple[Callable, int]],
    multi_step_frameworks: Optional[List[str]] = None,
) -> List[FrameworkConfig]:
    """
    Convert framework tuples to FrameworkConfig objects.
    
    Args:
        frameworks: List of (batch_func, weight) tuples
        multi_step_frameworks: Names of frameworks that are multi-step
        
    Returns:
        List of FrameworkConfig objects
    """
    multi_step = set(multi_step_frameworks or [])
    configs = []
    
    for func, weight in frameworks:
        name = func.__name__
        is_multi = name in multi_step
        configs.append(FrameworkConfig(
            batch_func=func,
            name=name,
            weight=weight,
            is_multi_step=is_multi,
            dry_run_steps=2 if is_multi else 0,
        ))
    
    return configs


def run_parallel_training_step(
    runner: ParallelFrameworkRunner,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    num_parallel: int = 4,
    use_lora: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for running a parallel training step.
    
    Args:
        runner: ParallelFrameworkRunner instance
        batch_size: Batch size per framework
        optimizer: Optimizer for updates
        num_parallel: Number of frameworks per step
        use_lora: Whether using LoRA
        
    Returns:
        Results dictionary
    """
    return runner.run_parallel_step(
        batch_size=batch_size,
        optimizer=optimizer,
        num_parallel=num_parallel,
        training=True,
        use_lora=use_lora,
    )
