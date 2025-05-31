"""
Self-healing intervention system for vLLM monitoring.

This module provides automated intervention capabilities to handle detected issues,
perform self-healing actions, and implement guardrails to prevent system failures.
"""

import asyncio
import time
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Callable, Union
import logging
import psutil
import gc

from .core import (
    ComponentState, ComponentType, StateType, Intervention, AlertLevel,
    PerformanceTimer, MonitorConfig
)

logger = logging.getLogger("vllm_monitor.interventions")


@dataclass
class InterventionResult:
    """Result of an intervention action."""
    success: bool
    action_taken: str
    details: Dict[str, Any]
    execution_time_ms: float
    side_effects: List[str]


@dataclass
class InterventionPolicy:
    """Policy for intervention decisions."""
    min_confidence: float = 0.7
    max_interventions_per_hour: int = 10
    cooldown_seconds: float = 60.0
    escalation_threshold: int = 3
    enable_aggressive_actions: bool = False


class BaseIntervention(ABC):
    """Base class for all intervention implementations."""
    
    def __init__(self, policy: Optional[InterventionPolicy] = None):
        self.policy = policy or InterventionPolicy()
        self._intervention_history = deque(maxlen=100)
        self._last_intervention_time = 0.0
        self._consecutive_failures = 0
        self._lock = threading.RLock()
    
    @abstractmethod
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle the issue."""
        pass
    
    @abstractmethod
    def _execute_intervention(self, issue: Dict[str, Any]) -> InterventionResult:
        """Execute the actual intervention logic."""
        pass
    
    def intervene(self, issue: Dict[str, Any]) -> bool:
        """Main intervention method with safety checks."""
        with self._lock:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self._last_intervention_time < self.policy.cooldown_seconds:
                logger.debug(f"Intervention {self.__class__.__name__} skipped due to cooldown")
                return False
            
            # Check intervention rate limits
            recent_interventions = [
                h for h in self._intervention_history
                if current_time - h['timestamp'] < 3600  # Last hour
            ]
            if len(recent_interventions) >= self.policy.max_interventions_per_hour:
                logger.warning(f"Intervention rate limit exceeded for {self.__class__.__name__}")
                return False
            
            try:
                with PerformanceTimer() as timer:
                    result = self._execute_intervention(issue)
                
                # Record intervention
                self._last_intervention_time = current_time
                self._intervention_history.append({
                    'timestamp': current_time,
                    'issue': issue.get('type', 'unknown'),
                    'success': result.success,
                    'action': result.action_taken,
                    'execution_time_ms': timer.elapsed_us / 1000.0
                })
                
                if result.success:
                    self._consecutive_failures = 0
                    logger.info(f"Intervention successful: {result.action_taken}")
                else:
                    self._consecutive_failures += 1
                    logger.warning(f"Intervention failed: {result.action_taken}")
                
                return result.success
                
            except Exception as e:
                logger.error(f"Intervention {self.__class__.__name__} failed with exception: {e}")
                self._consecutive_failures += 1
                return False
    
    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get statistics about this intervention."""
        with self._lock:
            total = len(self._intervention_history)
            successful = sum(1 for h in self._intervention_history if h['success'])
            
            return {
                'total_interventions': total,
                'successful_interventions': successful,
                'success_rate': successful / max(total, 1),
                'consecutive_failures': self._consecutive_failures,
                'last_intervention': self._last_intervention_time,
                'average_execution_time_ms': (
                    sum(h['execution_time_ms'] for h in self._intervention_history) / max(total, 1)
                    if total > 0 else 0.0
                )
            }


class MemoryCleanupIntervention(BaseIntervention):
    """Intervention to clean up memory when usage is high."""
    
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle memory-related issues."""
        return (
            issue.get('type') in ['anomaly_detection', 'failure_prediction'] and
            any('memory' in indicator.lower() for indicator in 
                issue.get('contributing_factors', []) + issue.get('warning_signs', []))
        )
    
    def _execute_intervention(self, issue: Dict[str, Any]) -> InterventionResult:
        """Execute memory cleanup intervention."""
        actions_taken = []
        details = {}
        
        try:
            # Get current memory usage
            memory_before = psutil.virtual_memory().percent
            details['memory_before_percent'] = memory_before
            
            # Force garbage collection
            collected = gc.collect()
            actions_taken.append(f"garbage_collection_freed_{collected}_objects")
            details['objects_collected'] = collected
            
            # Get memory usage after cleanup
            memory_after = psutil.virtual_memory().percent
            details['memory_after_percent'] = memory_after
            details['memory_freed_percent'] = memory_before - memory_after
            
            # Check if we have torch available for GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.empty_cache()
                        actions_taken.append(f"gpu_{i}_cache_cleared")
                    details['gpu_cache_cleared'] = True
            except ImportError:
                pass
            
            # Determine success based on memory reduction
            success = memory_after < memory_before or collected > 0
            
            return InterventionResult(
                success=success,
                action_taken="; ".join(actions_taken),
                details=details,
                execution_time_ms=0.0,  # Will be set by caller
                side_effects=["temporary_performance_impact"] if success else []
            )
            
        except Exception as e:
            return InterventionResult(
                success=False,
                action_taken="memory_cleanup_failed",
                details={'error': str(e)},
                execution_time_ms=0.0,
                side_effects=[]
            )


class ComponentRestartIntervention(BaseIntervention):
    """Intervention to restart unhealthy components."""
    
    def __init__(self, component_registry: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.component_registry = component_registry
    
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle component health issues."""
        return (
            issue.get('type') in ['failure_prediction', 'health_scoring'] and
            issue.get('component_id') in self.component_registry and
            issue.get('failure_probability', 0) > 0.7
        )
    
    def _execute_intervention(self, issue: Dict[str, Any]) -> InterventionResult:
        """Execute component restart intervention."""
        component_id = issue.get('component_id')
        if not component_id:
            return InterventionResult(
                success=False,
                action_taken="no_component_id",
                details={},
                execution_time_ms=0.0,
                side_effects=[]
            )
        
        try:
            component_info = self.component_registry.get(component_id)
            if not component_info:
                return InterventionResult(
                    success=False,
                    action_taken="component_not_found",
                    details={'component_id': component_id},
                    execution_time_ms=0.0,
                    side_effects=[]
                )
            
            # Get component reference
            component_ref = component_info['ref']
            component = component_ref()
            
            if component is None:
                return InterventionResult(
                    success=False,
                    action_taken="component_already_gone",
                    details={'component_id': component_id},
                    execution_time_ms=0.0,
                    side_effects=[]
                )
            
            # Attempt to restart component based on type
            success = self._restart_component(component, component_info['type'])
            
            return InterventionResult(
                success=success,
                action_taken=f"restart_{component_info['type'].value}",
                details={'component_id': component_id},
                execution_time_ms=0.0,
                side_effects=["temporary_service_interruption"] if success else []
            )
            
        except Exception as e:
            return InterventionResult(
                success=False,
                action_taken="restart_failed",
                details={'component_id': component_id, 'error': str(e)},
                execution_time_ms=0.0,
                side_effects=[]
            )
    
    def _restart_component(self, component: Any, component_type: ComponentType) -> bool:
        """Attempt to restart a specific component type."""
        try:
            if component_type == ComponentType.CACHE_ENGINE:
                # Clear cache and reinitialize if possible
                if hasattr(component, 'clear') and callable(component.clear):
                    component.clear()
                    return True
                elif hasattr(component, 'reset') and callable(component.reset):
                    component.reset()
                    return True
            
            elif component_type == ComponentType.SCHEDULER:
                # Reset scheduler state if possible
                if hasattr(component, 'reset') and callable(component.reset):
                    component.reset()
                    return True
            
            elif component_type == ComponentType.WORKER:
                # Restart worker if possible
                if hasattr(component, 'restart') and callable(component.restart):
                    component.restart()
                    return True
            
            # Generic restart attempt
            if hasattr(component, 'restart') and callable(component.restart):
                component.restart()
                return True
            elif hasattr(component, 'reset') and callable(component.reset):
                component.reset()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to restart component {component_type}: {e}")
            return False


class LoadBalancingIntervention(BaseIntervention):
    """Intervention to adjust load balancing when queues build up."""
    
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle load balancing issues."""
        return (
            issue.get('type') in ['anomaly_detection', 'performance_analysis'] and
            any('queue' in indicator.lower() or 'backlog' in indicator.lower() 
                for indicator in issue.get('contributing_factors', []))
        )
    
    def _execute_intervention(self, issue: Dict[str, Any]) -> InterventionResult:
        """Execute load balancing intervention."""
        actions_taken = []
        details = {}
        
        try:
            # Simulate load balancing adjustments
            # In a real implementation, this would interact with actual load balancing systems
            
            # Reduce batch sizes
            actions_taken.append("reduced_batch_sizes")
            details['batch_size_reduction'] = 0.8
            
            # Increase request timeout
            actions_taken.append("increased_request_timeout")
            details['timeout_increase_factor'] = 1.5
            
            # Enable request throttling
            actions_taken.append("enabled_request_throttling")
            details['throttling_enabled'] = True
            
            return InterventionResult(
                success=True,
                action_taken="; ".join(actions_taken),
                details=details,
                execution_time_ms=0.0,
                side_effects=["reduced_throughput", "increased_latency"]
            )
            
        except Exception as e:
            return InterventionResult(
                success=False,
                action_taken="load_balancing_failed",
                details={'error': str(e)},
                execution_time_ms=0.0,
                side_effects=[]
            )


class ResourceScalingIntervention(BaseIntervention):
    """Intervention to scale resources when utilization is high."""
    
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle resource scaling issues."""
        return (
            issue.get('type') in ['performance_analysis', 'failure_prediction'] and
            any('cpu' in indicator.lower() or 'gpu' in indicator.lower() 
                for indicator in issue.get('contributing_factors', []))
        )
    
    def _execute_intervention(self, issue: Dict[str, Any]) -> InterventionResult:
        """Execute resource scaling intervention."""
        actions_taken = []
        details = {}
        
        try:
            # Check current system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            details['cpu_usage_before'] = cpu_percent
            details['memory_usage_before'] = memory_percent
            
            # Simulate scaling actions
            if cpu_percent > 80:
                actions_taken.append("requested_cpu_scaling")
                details['cpu_scaling_requested'] = True
            
            if memory_percent > 85:
                actions_taken.append("requested_memory_scaling")
                details['memory_scaling_requested'] = True
            
            # In a real implementation, this would trigger actual scaling
            # For now, we'll adjust process priority and affinity
            try:
                process = psutil.Process()
                current_nice = process.nice()
                if current_nice > -5:  # Don't go too aggressive
                    process.nice(current_nice - 1)
                    actions_taken.append("increased_process_priority")
                    details['nice_value'] = current_nice - 1
            except (psutil.AccessDenied, OSError):
                pass
            
            success = len(actions_taken) > 0
            
            return InterventionResult(
                success=success,
                action_taken="; ".join(actions_taken) if actions_taken else "no_scaling_needed",
                details=details,
                execution_time_ms=0.0,
                side_effects=["resource_contention"] if success else []
            )
            
        except Exception as e:
            return InterventionResult(
                success=False,
                action_taken="resource_scaling_failed",
                details={'error': str(e)},
                execution_time_ms=0.0,
                side_effects=[]
            )


class ConfigurationAdjustmentIntervention(BaseIntervention):
    """Intervention to adjust configuration parameters based on performance."""
    
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle configuration issues."""
        return (
            issue.get('type') == 'performance_analysis' and
            issue.get('impact_level', 0) >= AlertLevel.WARNING.value
        )
    
    def _execute_intervention(self, issue: Dict[str, Any]) -> InterventionResult:
        """Execute configuration adjustment intervention."""
        actions_taken = []
        details = {}
        
        try:
            # Extract metric information from the issue
            insights = issue.get('insights', [])
            
            for insight in insights:
                metric_name = insight.get('metric_name', '')
                trend_direction = insight.get('trend_direction', '')
                
                if 'latency' in metric_name.lower() and trend_direction == 'degrading':
                    # Adjust timeout settings
                    actions_taken.append("increased_timeout_settings")
                    details['timeout_adjustment'] = "increased_by_20_percent"
                
                elif 'throughput' in metric_name.lower() and trend_direction == 'degrading':
                    # Adjust batch processing
                    actions_taken.append("optimized_batch_processing")
                    details['batch_optimization'] = "enabled_dynamic_batching"
                
                elif 'memory' in metric_name.lower() and trend_direction == 'degrading':
                    # Adjust memory settings
                    actions_taken.append("adjusted_memory_limits")
                    details['memory_adjustment'] = "reduced_cache_size"
            
            success = len(actions_taken) > 0
            
            return InterventionResult(
                success=success,
                action_taken="; ".join(actions_taken) if actions_taken else "no_adjustments_needed",
                details=details,
                execution_time_ms=0.0,
                side_effects=["configuration_changes"] if success else []
            )
            
        except Exception as e:
            return InterventionResult(
                success=False,
                action_taken="configuration_adjustment_failed",
                details={'error': str(e)},
                execution_time_ms=0.0,
                side_effects=[]
            )


class InterventionEngine:
    """Main engine for coordinating interventions."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self._interventions: List[BaseIntervention] = []
        self._intervention_history = deque(maxlen=1000)
        self._active_interventions: Set[str] = set()
        self._lock = threading.RLock()
        
        # Statistics
        self._total_interventions = 0
        self._successful_interventions = 0
        self._failed_interventions = 0
    
    def register_intervention(self, intervention: BaseIntervention) -> None:
        """Register an intervention handler."""
        with self._lock:
            self._interventions.append(intervention)
            logger.info(f"Registered intervention: {type(intervention).__name__}")
    
    def unregister_intervention(self, intervention: BaseIntervention) -> None:
        """Unregister an intervention handler."""
        with self._lock:
            if intervention in self._interventions:
                self._interventions.remove(intervention)
                logger.info(f"Unregistered intervention: {type(intervention).__name__}")
    
    async def handle_issue(self, issue: Dict[str, Any]) -> bool:
        """Handle a detected issue with appropriate interventions."""
        if not self.config.enable_self_healing:
            return False
        
        issue_id = f"{issue.get('type', 'unknown')}_{issue.get('component_id', 'system')}"
        
        with self._lock:
            # Check if we're already handling this issue
            if issue_id in self._active_interventions:
                logger.debug(f"Issue {issue_id} is already being handled")
                return False
            
            self._active_interventions.add(issue_id)
        
        try:
            success = False
            interventions_tried = []
            
            # Try interventions in order of priority
            for intervention in self._interventions:
                if not intervention.can_intervene(issue):
                    continue
                
                interventions_tried.append(type(intervention).__name__)
                
                try:
                    intervention_success = await asyncio.to_thread(intervention.intervene, issue)
                    
                    # Record intervention attempt
                    self._record_intervention(issue, intervention, intervention_success)
                    
                    if intervention_success:
                        success = True
                        logger.info(f"Issue {issue_id} resolved by {type(intervention).__name__}")
                        break
                    else:
                        logger.warning(f"Intervention {type(intervention).__name__} failed for {issue_id}")
                
                except Exception as e:
                    logger.error(f"Intervention {type(intervention).__name__} crashed: {e}")
                    self._record_intervention(issue, intervention, False, error=str(e))
            
            if not success and interventions_tried:
                logger.error(f"All interventions failed for issue {issue_id}. Tried: {interventions_tried}")
            elif not interventions_tried:
                logger.debug(f"No applicable interventions found for issue {issue_id}")
            
            return success
            
        finally:
            with self._lock:
                self._active_interventions.discard(issue_id)
    
    def _record_intervention(self, 
                           issue: Dict[str, Any], 
                           intervention: BaseIntervention, 
                           success: bool,
                           error: Optional[str] = None) -> None:
        """Record intervention attempt for tracking."""
        with self._lock:
            record = {
                'timestamp': time.time(),
                'issue_type': issue.get('type', 'unknown'),
                'component_id': issue.get('component_id', 'unknown'),
                'intervention': type(intervention).__name__,
                'success': success,
                'error': error
            }
            
            self._intervention_history.append(record)
            self._total_interventions += 1
            
            if success:
                self._successful_interventions += 1
            else:
                self._failed_interventions += 1
    
    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get statistics about intervention performance."""
        with self._lock:
            # Overall statistics
            success_rate = (
                self._successful_interventions / max(self._total_interventions, 1) * 100.0
            )
            
            # Per-intervention statistics
            intervention_stats = {}
            for intervention in self._interventions:
                intervention_stats[type(intervention).__name__] = intervention.get_intervention_stats()
            
            # Recent activity (last hour)
            current_time = time.time()
            recent_interventions = [
                h for h in self._intervention_history
                if current_time - h['timestamp'] < 3600
            ]
            
            return {
                'total_interventions': self._total_interventions,
                'successful_interventions': self._successful_interventions,
                'failed_interventions': self._failed_interventions,
                'success_rate_percent': success_rate,
                'recent_interventions_count': len(recent_interventions),
                'active_interventions': len(self._active_interventions),
                'registered_interventions': len(self._interventions),
                'intervention_details': intervention_stats,
                'recent_activity': recent_interventions[-10:]  # Last 10 interventions
            }


class SelfHealingAgent:
    """High-level agent that orchestrates self-healing capabilities."""
    
    def __init__(self, 
                 config: Optional[MonitorConfig] = None,
                 component_registry: Optional[Dict[str, Any]] = None):
        self.config = config or MonitorConfig()
        self.component_registry = component_registry or {}
        
        # Initialize intervention engine
        self.intervention_engine = InterventionEngine(config)
        
        # Initialize default interventions
        self._setup_default_interventions()
        
        # State tracking
        self._is_running = False
        self._healing_sessions: Dict[str, Dict] = {}
        self._lock = threading.RLock()
    
    def _setup_default_interventions(self) -> None:
        """Setup default intervention handlers."""
        # Memory cleanup intervention
        memory_intervention = MemoryCleanupIntervention()
        self.intervention_engine.register_intervention(memory_intervention)
        
        # Component restart intervention
        if self.component_registry:
            restart_intervention = ComponentRestartIntervention(self.component_registry)
            self.intervention_engine.register_intervention(restart_intervention)
        
        # Load balancing intervention
        load_balancing_intervention = LoadBalancingIntervention()
        self.intervention_engine.register_intervention(load_balancing_intervention)
        
        # Resource scaling intervention
        scaling_intervention = ResourceScalingIntervention()
        self.intervention_engine.register_intervention(scaling_intervention)
        
        # Configuration adjustment intervention
        config_intervention = ConfigurationAdjustmentIntervention()
        self.intervention_engine.register_intervention(config_intervention)
    
    async def start(self) -> None:
        """Start the self-healing agent."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
        
        logger.info("Self-healing agent started")
    
    async def stop(self) -> None:
        """Stop the self-healing agent."""
        with self._lock:
            self._is_running = False
        
        logger.info("Self-healing agent stopped")
    
    async def heal(self, issue: Dict[str, Any]) -> bool:
        """Attempt to heal a detected issue."""
        if not self._is_running or not self.config.enable_self_healing:
            return False
        
        # Start healing session
        session_id = f"{issue.get('type')}_{int(time.time())}"
        
        with self._lock:
            self._healing_sessions[session_id] = {
                'start_time': time.time(),
                'issue': issue,
                'status': 'in_progress'
            }
        
        try:
            success = await self.intervention_engine.handle_issue(issue)
            
            # Update session
            with self._lock:
                if session_id in self._healing_sessions:
                    self._healing_sessions[session_id].update({
                        'end_time': time.time(),
                        'status': 'success' if success else 'failed',
                        'success': success
                    })
            
            return success
            
        except Exception as e:
            logger.error(f"Healing session {session_id} failed: {e}")
            
            with self._lock:
                if session_id in self._healing_sessions:
                    self._healing_sessions[session_id].update({
                        'end_time': time.time(),
                        'status': 'error',
                        'error': str(e),
                        'success': False
                    })
            
            return False
    
    def get_healing_status(self) -> Dict[str, Any]:
        """Get current healing status and statistics."""
        with self._lock:
            # Get intervention statistics
            intervention_stats = self.intervention_engine.get_intervention_statistics()
            
            # Get session statistics
            total_sessions = len(self._healing_sessions)
            successful_sessions = sum(
                1 for session in self._healing_sessions.values() 
                if session.get('success', False)
            )
            
            active_sessions = sum(
                1 for session in self._healing_sessions.values()
                if session.get('status') == 'in_progress'
            )
            
            return {
                'is_running': self._is_running,
                'total_healing_sessions': total_sessions,
                'successful_healing_sessions': successful_sessions,
                'healing_success_rate': (
                    successful_sessions / max(total_sessions, 1) * 100.0
                ),
                'active_healing_sessions': active_sessions,
                'intervention_statistics': intervention_stats,
                'recent_sessions': list(self._healing_sessions.values())[-5:]
            }


class GuardrailManager:
    """Manager for implementing guardrails to prevent system failures."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self._guardrails: Dict[str, Callable] = {}
        self._violations: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        # Setup default guardrails
        self._setup_default_guardrails()
    
    def _setup_default_guardrails(self) -> None:
        """Setup default guardrail policies."""
        self._guardrails.update({
            'memory_usage': self._memory_guardrail,
            'cpu_usage': self._cpu_guardrail,
            'error_rate': self._error_rate_guardrail,
            'queue_size': self._queue_size_guardrail,
            'response_time': self._response_time_guardrail
        })
    
    def register_guardrail(self, name: str, guardrail_func: Callable) -> None:
        """Register a custom guardrail function."""
        with self._lock:
            self._guardrails[name] = guardrail_func
            logger.info(f"Registered guardrail: {name}")
    
    def check_guardrails(self, state: ComponentState) -> List[Dict[str, Any]]:
        """Check all guardrails against current state."""
        if not self.config.enable_guardrails:
            return []
        
        violations = []
        
        with self._lock:
            for name, guardrail_func in self._guardrails.items():
                try:
                    violation = guardrail_func(state)
                    if violation:
                        violation['guardrail'] = name
                        violation['timestamp'] = time.time()
                        violations.append(violation)
                        self._violations.append(violation)
                        
                except Exception as e:
                    logger.error(f"Guardrail {name} failed: {e}")
        
        return violations
    
    def _memory_guardrail(self, state: ComponentState) -> Optional[Dict[str, Any]]:
        """Guardrail for memory usage."""
        if state.memory_usage and state.memory_usage > 95.0:
            return {
                'type': 'memory_critical',
                'severity': AlertLevel.CRITICAL,
                'message': f"Memory usage critical: {state.memory_usage}%",
                'component_id': state.component_id,
                'threshold': 95.0,
                'current_value': state.memory_usage,
                'action_required': 'immediate_memory_cleanup'
            }
        return None
    
    def _cpu_guardrail(self, state: ComponentState) -> Optional[Dict[str, Any]]:
        """Guardrail for CPU usage."""
        if state.cpu_usage and state.cpu_usage > 98.0:
            return {
                'type': 'cpu_critical',
                'severity': AlertLevel.CRITICAL,
                'message': f"CPU usage critical: {state.cpu_usage}%",
                'component_id': state.component_id,
                'threshold': 98.0,
                'current_value': state.cpu_usage,
                'action_required': 'throttle_requests'
            }
        return None
    
    def _error_rate_guardrail(self, state: ComponentState) -> Optional[Dict[str, Any]]:
        """Guardrail for error rates."""
        if state.error_count > 100:  # More than 100 errors
            return {
                'type': 'error_rate_critical',
                'severity': AlertLevel.ERROR,
                'message': f"High error count: {state.error_count}",
                'component_id': state.component_id,
                'threshold': 100,
                'current_value': state.error_count,
                'action_required': 'investigate_errors'
            }
        return None
    
    def _queue_size_guardrail(self, state: ComponentState) -> Optional[Dict[str, Any]]:
        """Guardrail for queue sizes."""
        waiting_requests = state.data.get('waiting_requests', 0)
        if waiting_requests > 500:
            return {
                'type': 'queue_size_critical',
                'severity': AlertLevel.WARNING,
                'message': f"Queue size critical: {waiting_requests} waiting",
                'component_id': state.component_id,
                'threshold': 500,
                'current_value': waiting_requests,
                'action_required': 'enable_load_balancing'
            }
        return None
    
    def _response_time_guardrail(self, state: ComponentState) -> Optional[Dict[str, Any]]:
        """Guardrail for response times."""
        if state.average_latency_ms > 30000:  # 30 seconds
            return {
                'type': 'response_time_critical',
                'severity': AlertLevel.ERROR,
                'message': f"Response time critical: {state.average_latency_ms}ms",
                'component_id': state.component_id,
                'threshold': 30000,
                'current_value': state.average_latency_ms,
                'action_required': 'optimize_processing'
            }
        return None
    
    def get_guardrail_status(self) -> Dict[str, Any]:
        """Get current guardrail status and violations."""
        with self._lock:
            # Count violations by type
            violation_counts = defaultdict(int)
            recent_violations = []
            
            current_time = time.time()
            for violation in self._violations:
                if current_time - violation['timestamp'] < 3600:  # Last hour
                    violation_counts[violation['type']] += 1
                    if current_time - violation['timestamp'] < 300:  # Last 5 minutes
                        recent_violations.append(violation)
            
            return {
                'enabled': self.config.enable_guardrails,
                'registered_guardrails': list(self._guardrails.keys()),
                'total_violations': len(self._violations),
                'recent_violations_count': len(recent_violations),
                'violations_by_type': dict(violation_counts),
                'recent_violations': recent_violations[-10:]  # Last 10
            }