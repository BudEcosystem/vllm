"""
vLLM Engine Integration for Monitoring System.

This module provides deep integration with vLLM's core engine components
to enable real-time monitoring, predictive failure detection, and automatic
mitigation during runtime.
"""

import time
import asyncio
import functools
import threading
import weakref
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass
import logging

# vLLM imports
if TYPE_CHECKING:
    from vllm import LLMEngine, AsyncLLMEngine
    from vllm.core.scheduler import SchedulerOutputs
    from vllm.engine.metrics import Stats
    from vllm.sequence import SequenceGroup
    from vllm.outputs import RequestOutput
    from vllm.config import VllmConfig
    from vllm.worker.worker import Worker

# Monitoring imports
from .core import VLLMMonitor, get_logger
from .lifecycle_tracker import LifecycleState, StateTransition, StateCheckpoint
from .predictive_failure_detection import PredictiveFailureDetector
from .continuous_learning import ContinuousLearningSystem, MitigationAttempt, MitigationOutcome
from .vllm_integration_plugins import register_all_vllm_plugins
from .vllm_mitigation_strategies import register_all_mitigation_strategies


class VLLMEngineMonitor:
    """
    Deep integration monitor for vLLM engine components.
    
    This class integrates directly with LLMEngine/AsyncLLMEngine to provide:
    - Real-time state tracking
    - Performance monitoring
    - Predictive failure detection
    - Automatic mitigation execution
    - Guardrail enforcement
    """
    
    def __init__(self, 
                 enable_predictive: bool = True,
                 enable_learning: bool = True,
                 enable_auto_mitigation: bool = True):
        self.logger = get_logger()
        
        # Core monitoring components
        self.monitor = VLLMMonitor(enable_history=True)
        self.monitor.setup_lifecycle_tracking()
        self.monitor.setup_plugin_system()
        
        # Advanced components
        self.predictive_detector = None
        self.continuous_learner = None
        
        if enable_predictive:
            self.predictive_detector = PredictiveFailureDetector()
            self.predictive_detector.start_background_analysis()
        
        if enable_learning:
            self.continuous_learner = ContinuousLearningSystem()
            self.continuous_learner.start_background_learning()
            register_all_mitigation_strategies(self.continuous_learner)
        
        # Configuration
        self.enable_auto_mitigation = enable_auto_mitigation
        self.mitigation_in_progress = False
        self._mitigation_lock = threading.Lock()
        
        # Engine references (weak to avoid circular refs)
        self._engine_ref = None
        self._async_engine_ref = None
        self._workers: List[weakref.ref] = []
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_start_times: Dict[str, float] = {}
        
        # Performance tracking
        self.last_step_time = time.time()
        self.step_durations = []
        self.tokens_generated = 0
        
        # Register all vLLM plugins
        register_all_vllm_plugins(self.monitor.plugin_manager)
        
        self.logger.info("vLLM Engine Monitor initialized")
    
    def attach_to_engine(self, engine: Union['LLMEngine', 'AsyncLLMEngine']) -> None:
        """
        Attach monitor to a vLLM engine instance.
        
        Args:
            engine: LLMEngine or AsyncLLMEngine instance
        """
        from vllm import LLMEngine, AsyncLLMEngine
        
        if isinstance(engine, AsyncLLMEngine):
            self._async_engine_ref = weakref.ref(engine)
            self._attach_async_hooks(engine)
        else:
            self._engine_ref = weakref.ref(engine)
            self._attach_sync_hooks(engine)
        
        # Track engine initialization
        self._track_engine_init(engine)
        
        self.logger.info(f"Attached monitor to {type(engine).__name__}")
    
    def _attach_sync_hooks(self, engine: 'LLMEngine') -> None:
        """Attach monitoring hooks to synchronous LLMEngine"""
        # Wrap key methods
        engine._original_add_request = engine.add_request
        engine._original_step = engine.step
        engine._original_abort_request = engine.abort_request
        
        # Replace with monitored versions
        engine.add_request = self._create_monitored_add_request(engine)
        engine.step = self._create_monitored_step(engine)
        engine.abort_request = self._create_monitored_abort_request(engine)
        
        # Monitor scheduler
        if hasattr(engine, 'scheduler'):
            for scheduler in engine.scheduler:
                self._attach_scheduler_hooks(scheduler)
        
        # Monitor workers
        if hasattr(engine, 'model_executor') and hasattr(engine.model_executor, 'workers'):
            for worker in engine.model_executor.workers:
                self._attach_worker_hooks(worker)
    
    def _attach_async_hooks(self, engine: 'AsyncLLMEngine') -> None:
        """Attach monitoring hooks to AsyncLLMEngine"""
        # Wrap async methods
        engine._original_add_request = engine.add_request
        engine._original_engine_step = engine.engine_step
        engine._original_abort = engine.abort
        
        # Replace with monitored versions
        engine.add_request = self._create_async_monitored_add_request(engine)
        engine.engine_step = self._create_async_monitored_step(engine)
        engine.abort = self._create_async_monitored_abort(engine)
    
    def _create_monitored_add_request(self, engine: 'LLMEngine'):
        """Create monitored version of add_request"""
        def monitored_add_request(request_id: str, *args, **kwargs):
            # Pre-execution monitoring
            self._on_request_start(request_id, args, kwargs)
            
            try:
                # Execute original method
                result = engine._original_add_request(request_id, *args, **kwargs)
                
                # Post-execution monitoring
                self._on_request_added(request_id, result)
                
                return result
                
            except Exception as e:
                # Monitor exceptions
                self._on_request_error(request_id, e)
                raise
        
        return monitored_add_request
    
    def _create_monitored_step(self, engine: 'LLMEngine'):
        """Create monitored version of step"""
        def monitored_step(*args, **kwargs):
            # Pre-execution monitoring
            step_start_time = time.time()
            self._on_step_start()
            
            try:
                # Check for mitigation needs before step
                self._check_and_execute_mitigations(engine)
                
                # Execute original method
                outputs = engine._original_step(*args, **kwargs)
                
                # Post-execution monitoring
                step_duration = time.time() - step_start_time
                self._on_step_complete(outputs, step_duration)
                
                # Analyze state for predictions
                self._analyze_engine_state(engine)
                
                return outputs
                
            except Exception as e:
                # Monitor exceptions
                self._on_step_error(e)
                
                # Attempt automatic recovery
                if self.enable_auto_mitigation:
                    recovery_success = self._attempt_error_recovery(engine, e)
                    if recovery_success:
                        # Retry step after recovery
                        return engine._original_step(*args, **kwargs)
                
                raise
        
        return monitored_step
    
    def _create_async_monitored_add_request(self, engine: 'AsyncLLMEngine'):
        """Create async monitored version of add_request"""
        async def monitored_add_request(request_id: str, *args, **kwargs):
            self._on_request_start(request_id, args, kwargs)
            
            try:
                result = await engine._original_add_request(request_id, *args, **kwargs)
                self._on_request_added(request_id, result)
                return result
                
            except Exception as e:
                self._on_request_error(request_id, e)
                raise
        
        return monitored_add_request
    
    def _create_async_monitored_step(self, engine: 'AsyncLLMEngine'):
        """Create async monitored version of engine_step"""
        async def monitored_engine_step(*args, **kwargs):
            step_start_time = time.time()
            self._on_step_start()
            
            try:
                # Check for mitigation needs
                await self._async_check_and_execute_mitigations(engine)
                
                # Execute original method
                outputs = await engine._original_engine_step(*args, **kwargs)
                
                # Post-execution monitoring
                step_duration = time.time() - step_start_time
                self._on_step_complete(outputs, step_duration)
                
                # Analyze state
                await self._async_analyze_engine_state(engine)
                
                return outputs
                
            except Exception as e:
                self._on_step_error(e)
                
                if self.enable_auto_mitigation:
                    recovery_success = await self._async_attempt_error_recovery(engine, e)
                    if recovery_success:
                        return await engine._original_engine_step(*args, **kwargs)
                
                raise
        
        return monitored_engine_step
    
    def _create_monitored_abort_request(self, engine: 'LLMEngine'):
        """Create monitored version of abort_request"""
        def monitored_abort_request(request_id: str, *args, **kwargs):
            self._on_request_abort(request_id)
            engine._original_abort_request(request_id, *args, **kwargs)
        
        return monitored_abort_request
    
    def _create_async_monitored_abort(self, engine: 'AsyncLLMEngine'):
        """Create async monitored version of abort"""
        async def monitored_abort(request_id: str, *args, **kwargs):
            self._on_request_abort(request_id)
            await engine._original_abort(request_id, *args, **kwargs)
        
        return monitored_abort
    
    def _attach_scheduler_hooks(self, scheduler):
        """Attach hooks to scheduler"""
        # Monitor scheduling decisions
        original_schedule = scheduler.schedule
        
        def monitored_schedule(*args, **kwargs):
            schedule_start = time.time()
            
            # Execute original scheduling
            seq_group_metadata_list, scheduler_outputs, allow_async = original_schedule(*args, **kwargs)
            
            # Monitor scheduling decisions
            self._on_scheduler_decision(scheduler_outputs, time.time() - schedule_start)
            
            return seq_group_metadata_list, scheduler_outputs, allow_async
        
        scheduler.schedule = monitored_schedule
    
    def _attach_worker_hooks(self, worker):
        """Attach hooks to worker"""
        self._workers.append(weakref.ref(worker))
        
        # Monitor model execution
        if hasattr(worker, 'execute_model'):
            original_execute = worker.execute_model
            
            def monitored_execute(*args, **kwargs):
                exec_start = time.time()
                
                try:
                    outputs = original_execute(*args, **kwargs)
                    self._on_model_execution(outputs, time.time() - exec_start)
                    return outputs
                    
                except Exception as e:
                    self._on_worker_error(worker, e)
                    raise
            
            worker.execute_model = monitored_execute
    
    # Monitoring event handlers
    
    def _track_engine_init(self, engine) -> None:
        """Track engine initialization"""
        # Extract configuration
        config_data = {}
        
        if hasattr(engine, 'model_config'):
            config_data['model'] = {
                'model': getattr(engine.model_config, 'model', 'unknown'),
                'dtype': str(getattr(engine.model_config, 'dtype', 'unknown')),
                'max_model_len': getattr(engine.model_config, 'max_model_len', 0)
            }
        
        if hasattr(engine, 'parallel_config'):
            config_data['parallel'] = {
                'tensor_parallel_size': getattr(engine.parallel_config, 'tensor_parallel_size', 1),
                'pipeline_parallel_size': getattr(engine.parallel_config, 'pipeline_parallel_size', 1)
            }
        
        if hasattr(engine, 'scheduler_config'):
            config_data['scheduler'] = {
                'max_num_seqs': getattr(engine.scheduler_config, 'max_num_seqs', 0),
                'max_num_batched_tokens': getattr(engine.scheduler_config, 'max_num_batched_tokens', 0)
            }
        
        # Track initialization
        checkpoint = self.monitor.track_lifecycle_state(
            LifecycleState.INITIALIZING,
            StateTransition.STARTUP,
            config_data
        )
        
        # Validate configuration
        if self.predictive_detector:
            predictions = self.predictive_detector.analyze_checkpoint(checkpoint)
            if predictions:
                self.logger.warning(f"Initialization concerns: {len(predictions)} potential issues detected")
    
    def _on_request_start(self, request_id: str, args: tuple, kwargs: dict) -> None:
        """Called when a request is about to be added"""
        self.request_start_times[request_id] = time.time()
        self.active_requests[request_id] = {
            'start_time': time.time(),
            'status': 'pending'
        }
    
    def _on_request_added(self, request_id: str, result: Any) -> None:
        """Called after a request is successfully added"""
        if request_id in self.active_requests:
            self.active_requests[request_id]['status'] = 'active'
        
        # Update metrics
        self.monitor.collect_state({
            'active_requests': len(self.active_requests),
            'queue_size': self._get_queue_size()
        })
    
    def _on_request_error(self, request_id: str, error: Exception) -> None:
        """Called when request addition fails"""
        if request_id in self.active_requests:
            self.active_requests[request_id]['status'] = 'error'
            self.active_requests[request_id]['error'] = str(error)
        
        # Track error
        self.monitor.lifecycle_tracker.current_state = LifecycleState.ERROR
        checkpoint = self.monitor.lifecycle_tracker._create_checkpoint(
            LifecycleState.ERROR,
            StateTransition.ERROR_RECOVERY,
            error_context={'errors': [str(error)], 'request_id': request_id}
        )
        
        # Analyze for mitigation
        if self.predictive_detector:
            self.predictive_detector.analyze_checkpoint(checkpoint)
    
    def _on_request_abort(self, request_id: str) -> None:
        """Called when a request is aborted"""
        if request_id in self.active_requests:
            self.active_requests[request_id]['status'] = 'aborted'
            self.active_requests[request_id]['end_time'] = time.time()
    
    def _on_step_start(self) -> None:
        """Called at the start of each engine step"""
        self.last_step_time = time.time()
    
    def _on_step_complete(self, outputs: List['RequestOutput'], duration: float) -> None:
        """Called after each engine step completes"""
        self.step_durations.append(duration)
        
        # Track completed requests
        completed_requests = []
        for output in outputs:
            if output.finished:
                request_id = output.request_id
                if request_id in self.active_requests:
                    self.active_requests[request_id]['status'] = 'completed'
                    self.active_requests[request_id]['end_time'] = time.time()
                    
                    # Calculate request latency
                    if request_id in self.request_start_times:
                        latency = time.time() - self.request_start_times[request_id]
                        completed_requests.append(latency)
                
                # Count tokens
                if hasattr(output, 'outputs'):
                    for completion in output.outputs:
                        self.tokens_generated += len(completion.token_ids)
        
        # Update metrics
        metrics = {
            'step_duration_ms': duration * 1000,
            'active_requests': len([r for r in self.active_requests.values() if r['status'] == 'active']),
            'tokens_generated': self.tokens_generated
        }
        
        if completed_requests:
            metrics['avg_request_latency_ms'] = sum(completed_requests) / len(completed_requests) * 1000
        
        self.monitor.collect_state(metrics)
    
    def _on_step_error(self, error: Exception) -> None:
        """Called when engine step fails"""
        self.logger.error(f"Engine step error: {error}")
        
        # Track error state
        checkpoint = self.monitor.lifecycle_tracker._create_checkpoint(
            LifecycleState.ERROR,
            StateTransition.ERROR_RECOVERY,
            error_context={'errors': [str(error)], 'type': type(error).__name__}
        )
        
        # Get predictions
        if self.predictive_detector:
            predictions = self.predictive_detector.analyze_checkpoint(checkpoint)
            for pred in predictions:
                self.logger.warning(
                    f"Failure prediction: {pred.failure_type} "
                    f"(probability: {pred.probability:.2%})"
                )
    
    def _on_scheduler_decision(self, scheduler_outputs: 'SchedulerOutputs', duration: float) -> None:
        """Called after scheduler makes decisions"""
        # Track scheduling metrics
        metrics = {
            'scheduled_seqs': len(scheduler_outputs.scheduled_seq_groups),
            'preempted_seqs': len(scheduler_outputs.preemption_mode_dict),
            'swapped_in': len(scheduler_outputs.blocks_to_swap_in),
            'swapped_out': len(scheduler_outputs.blocks_to_swap_out),
            'scheduling_time_ms': duration * 1000
        }
        
        self.monitor.collect_state(metrics)
    
    def _on_model_execution(self, outputs: Any, duration: float) -> None:
        """Called after model execution"""
        metrics = {
            'model_execution_time_ms': duration * 1000
        }
        
        self.monitor.collect_state(metrics)
    
    def _on_worker_error(self, worker: 'Worker', error: Exception) -> None:
        """Called when a worker encounters an error"""
        self.logger.error(f"Worker error: {error}")
        
        # Track worker failure
        checkpoint = self.monitor.lifecycle_tracker._create_checkpoint(
            LifecycleState.ERROR,
            StateTransition.ERROR_RECOVERY,
            error_context={
                'errors': [str(error)],
                'type': 'worker_error',
                'worker_id': id(worker)
            }
        )
        
        if self.predictive_detector:
            self.predictive_detector.analyze_checkpoint(checkpoint)
    
    # Analysis and mitigation
    
    def _analyze_engine_state(self, engine) -> None:
        """Analyze current engine state for issues"""
        # Collect comprehensive metrics
        metrics = self._collect_engine_metrics(engine)
        
        # Create checkpoint
        checkpoint = StateCheckpoint(
            timestamp=time.time(),
            state=self._determine_engine_state(engine),
            previous_state=self.monitor.lifecycle_tracker.current_state,
            transition_type=None,
            arguments={},
            environment={},
            hardware_state={},
            metrics=metrics
        )
        
        # Analyze for predictions
        if self.predictive_detector:
            predictions = self.predictive_detector.analyze_checkpoint(checkpoint)
            
            # Log high-probability predictions
            for pred in predictions:
                if pred.probability > 0.7:
                    self.logger.warning(
                        f"High probability failure predicted: {pred.failure_type} "
                        f"({pred.probability:.1%} in {pred.time_to_failure:.0f}s)"
                    )
    
    async def _async_analyze_engine_state(self, engine) -> None:
        """Async version of analyze engine state"""
        # Run analysis in background to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._analyze_engine_state, engine)
    
    def _collect_engine_metrics(self, engine) -> Dict[str, float]:
        """Collect current engine metrics"""
        metrics = {}
        
        # Memory metrics
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_info = torch.cuda.mem_get_info(i)
                    metrics[f'gpu_{i}_memory_percent'] = (1 - mem_info[0] / mem_info[1]) * 100
                    metrics[f'gpu_{i}_memory_used_gb'] = (mem_info[1] - mem_info[0]) / (1024**3)
        except:
            pass
        
        # Request metrics
        metrics['active_requests'] = len([r for r in self.active_requests.values() if r['status'] == 'active'])
        metrics['error_rate'] = len([r for r in self.active_requests.values() if r['status'] == 'error']) / max(1, len(self.active_requests))
        
        # Performance metrics
        if self.step_durations:
            metrics['avg_step_duration_ms'] = sum(self.step_durations[-10:]) / len(self.step_durations[-10:]) * 1000
        
        # Queue metrics
        metrics['queue_size'] = self._get_queue_size(engine)
        
        return metrics
    
    def _determine_engine_state(self, engine) -> LifecycleState:
        """Determine current engine lifecycle state"""
        # Check for errors
        error_count = len([r for r in self.active_requests.values() if r['status'] == 'error'])
        if error_count > 5:
            return LifecycleState.ERROR
        
        # Check if serving
        if self.active_requests:
            return LifecycleState.SERVING
        
        # Default to ready
        return LifecycleState.READY
    
    def _get_queue_size(self, engine=None) -> int:
        """Get current request queue size"""
        if engine and hasattr(engine, 'scheduler'):
            total_queue_size = 0
            for scheduler in engine.scheduler:
                if hasattr(scheduler, 'waiting'):
                    total_queue_size += len(scheduler.waiting)
            return total_queue_size
        return 0
    
    def _check_and_execute_mitigations(self, engine) -> None:
        """Check if mitigations are needed and execute them"""
        if not self.enable_auto_mitigation or self.mitigation_in_progress:
            return
        
        with self._mitigation_lock:
            if self.mitigation_in_progress:
                return
            
            # Get current metrics
            metrics = self._collect_engine_metrics(engine)
            
            # Check critical conditions
            critical_conditions = [
                ('gpu_memory_critical', any(
                    metrics.get(f'gpu_{i}_memory_percent', 0) > 95
                    for i in range(8)  # Check up to 8 GPUs
                )),
                ('high_error_rate', metrics.get('error_rate', 0) > 0.1),
                ('queue_overflow', metrics.get('queue_size', 0) > 1000)
            ]
            
            for condition_name, condition_met in critical_conditions:
                if condition_met:
                    self.logger.warning(f"Critical condition detected: {condition_name}")
                    self._execute_mitigation(engine, condition_name, metrics)
                    break
    
    async def _async_check_and_execute_mitigations(self, engine) -> None:
        """Async version of check and execute mitigations"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._check_and_execute_mitigations, engine)
    
    def _execute_mitigation(self, engine, condition: str, metrics: Dict[str, float]) -> None:
        """Execute appropriate mitigation strategy"""
        if not self.continuous_learner:
            return
        
        self.mitigation_in_progress = True
        
        try:
            # Get recommended strategies
            current_state = self._determine_engine_state(engine)
            recommendations = self.continuous_learner.get_learning_recommendations(
                current_state,
                [condition],
                metrics
            )
            
            if recommendations:
                # Execute top recommendation
                top_strategy = recommendations[0]['strategy']
                self.logger.info(f"Executing mitigation: {top_strategy}")
                
                # Create context for mitigation
                context = {
                    'engine': engine,
                    'monitor': self.monitor,
                    'logger': self.logger,
                    'metrics': metrics
                }
                
                # Execute strategy
                strategy = self.continuous_learner.strategy_registry.get(top_strategy)
                if strategy:
                    start_time = time.time()
                    outcome = strategy.execute(context)
                    execution_time = time.time() - start_time
                    
                    # Record attempt
                    attempt = MitigationAttempt(
                        attempt_id=f"auto_{int(time.time())}",
                        timestamp=time.time(),
                        initial_state=current_state,
                        initial_metrics=metrics,
                        error_context=[condition],
                        strategy_name=top_strategy,
                        interventions_executed=strategy.interventions,
                        execution_time=execution_time,
                        outcome=outcome,
                        final_state=self._determine_engine_state(engine),
                        final_metrics=self._collect_engine_metrics(engine)
                    )
                    
                    self.continuous_learner.record_attempt(attempt)
                    
                    self.logger.info(f"Mitigation completed with outcome: {outcome.name}")
        
        finally:
            self.mitigation_in_progress = False
    
    def _attempt_error_recovery(self, engine, error: Exception) -> bool:
        """Attempt to recover from an error"""
        if not self.continuous_learner:
            return False
        
        self.logger.info(f"Attempting automatic recovery from {type(error).__name__}")
        
        # Map error to mitigation
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Determine appropriate strategy
        strategy_name = None
        if "out of memory" in error_msg or "oom" in error_msg:
            strategy_name = "emergency_gpu_memory_cleanup"
        elif "nccl" in error_msg:
            strategy_name = "restart_nccl"
        elif "timeout" in error_msg:
            strategy_name = "handle_timeout"
        
        if strategy_name and strategy_name in self.continuous_learner.strategy_registry:
            context = {
                'engine': engine,
                'error': error,
                'logger': self.logger
            }
            
            strategy = self.continuous_learner.strategy_registry[strategy_name]
            outcome = strategy.execute(context)
            
            return outcome in [MitigationOutcome.SUCCESS, MitigationOutcome.PARTIAL_SUCCESS]
        
        return False
    
    async def _async_attempt_error_recovery(self, engine, error: Exception) -> bool:
        """Async version of attempt error recovery"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._attempt_error_recovery, engine, error)
    
    # Public interface
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine monitoring status"""
        engine = self._engine_ref() if self._engine_ref else None
        
        return {
            'monitor_active': True,
            'engine_attached': engine is not None,
            'active_requests': len(self.active_requests),
            'completed_requests': len([r for r in self.active_requests.values() if r['status'] == 'completed']),
            'error_requests': len([r for r in self.active_requests.values() if r['status'] == 'error']),
            'tokens_generated': self.tokens_generated,
            'avg_step_duration_ms': sum(self.step_durations[-10:]) / len(self.step_durations[-10:]) * 1000 if self.step_durations else 0,
            'mitigation_in_progress': self.mitigation_in_progress,
            'predictive_detection_enabled': self.predictive_detector is not None,
            'continuous_learning_enabled': self.continuous_learner is not None
        }
    
    def force_mitigation(self, strategy_name: str) -> bool:
        """Manually trigger a mitigation strategy"""
        if not self.continuous_learner or strategy_name not in self.continuous_learner.strategy_registry:
            return False
        
        engine = self._engine_ref() if self._engine_ref else None
        if not engine:
            return False
        
        self._execute_mitigation(engine, "manual_trigger", self._collect_engine_metrics(engine))
        return True


# Convenience functions for integration

def create_monitored_engine(engine_args, **monitor_kwargs) -> tuple:
    """
    Create a vLLM engine with integrated monitoring.
    
    Args:
        engine_args: EngineArgs for vLLM
        **monitor_kwargs: Arguments for VLLMEngineMonitor
        
    Returns:
        Tuple of (engine, monitor)
    """
    from vllm import LLMEngine
    
    # Create monitor
    monitor = VLLMEngineMonitor(**monitor_kwargs)
    
    # Create engine
    engine = LLMEngine.from_engine_args(engine_args)
    
    # Attach monitor
    monitor.attach_to_engine(engine)
    
    return engine, monitor


def create_monitored_async_engine(engine_args, **monitor_kwargs) -> tuple:
    """
    Create an async vLLM engine with integrated monitoring.
    
    Args:
        engine_args: AsyncEngineArgs for vLLM
        **monitor_kwargs: Arguments for VLLMEngineMonitor
        
    Returns:
        Tuple of (engine, monitor)
    """
    from vllm import AsyncLLMEngine
    
    # Create monitor
    monitor = VLLMEngineMonitor(**monitor_kwargs)
    
    # Create engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Attach monitor
    monitor.attach_to_engine(engine)
    
    return engine, monitor


# Patch function to integrate with existing vLLM code

def patch_vllm_with_monitoring():
    """
    Patch vLLM classes to automatically integrate monitoring.
    
    This can be called at startup to transparently add monitoring
    to all vLLM engine instances.
    """
    from vllm import LLMEngine, AsyncLLMEngine
    
    # Store original constructors
    LLMEngine._original_init = LLMEngine.__init__
    AsyncLLMEngine._original_init = AsyncLLMEngine.__init__
    
    # Create patched constructors
    def monitored_init(self, *args, **kwargs):
        # Call original constructor
        self._original_init(*args, **kwargs)
        
        # Attach monitor if enabled
        if os.environ.get('VLLM_ENABLE_MONITORING', 'false').lower() == 'true':
            self._monitor = VLLMEngineMonitor()
            self._monitor.attach_to_engine(self)
    
    # Apply patches
    LLMEngine.__init__ = monitored_init
    AsyncLLMEngine.__init__ = monitored_init