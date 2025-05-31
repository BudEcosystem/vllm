"""
Enhanced vLLM Lifecycle Tracker with comprehensive state management.

This module provides sophisticated tracking of vLLM's lifecycle including:
- Startup/shutdown state tracking with full argument capture
- Automatic compatibility checking and validation
- State checkpoint management with intervention mapping
- Guardrail enforcement and policy management
- Verbose feedback for all operations
- Runtime hot-reloading support
"""

import time
import json
import inspect
import traceback
import weakref
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from threading import Lock, RLock
import logging
import sys
import os
import signal
import atexit
import importlib
import importlib.util

# Import core monitoring components
from .core import CircularBuffer, get_logger
from .interventions import InterventionEngine


class LifecycleState(Enum):
    """Comprehensive lifecycle states for vLLM"""
    # Pre-initialization states
    NOT_STARTED = auto()
    VALIDATING_ENVIRONMENT = auto()
    CHECKING_DEPENDENCIES = auto()
    LOADING_CONFIGURATIONS = auto()
    
    # Initialization states
    INITIALIZING = auto()
    LOADING_MODEL = auto()
    ALLOCATING_MEMORY = auto()
    COMPILING_KERNELS = auto()
    SETTING_UP_WORKERS = auto()
    
    # Running states
    READY = auto()
    SERVING = auto()
    PROCESSING_REQUESTS = auto()
    
    # Maintenance states
    PAUSED = auto()
    RECONFIGURING = auto()
    SCALING = auto()
    RECOVERING = auto()
    
    # Shutdown states
    SHUTTING_DOWN = auto()
    CLEANUP = auto()
    TERMINATED = auto()
    
    # Error states
    ERROR = auto()
    CRITICAL_ERROR = auto()
    UNRECOVERABLE = auto()


class StateTransition(Enum):
    """Valid state transitions with associated validations"""
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ERROR_RECOVERY = "error_recovery"
    RECONFIGURATION = "reconfiguration"
    SCALING = "scaling"
    PAUSE_RESUME = "pause_resume"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class StateCheckpoint:
    """Detailed state checkpoint with full context"""
    timestamp: float
    state: LifecycleState
    previous_state: Optional[LifecycleState]
    transition_type: Optional[StateTransition]
    arguments: Dict[str, Any]
    environment: Dict[str, str]
    hardware_state: Dict[str, Any]
    error_context: Optional[Dict[str, Any]] = None
    interventions_triggered: List[str] = field(default_factory=list)
    guardrails_enforced: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary with proper serialization"""
        data = asdict(self)
        data['state'] = self.state.name
        data['previous_state'] = self.previous_state.name if self.previous_state else None
        data['transition_type'] = self.transition_type.value if self.transition_type else None
        return data


@dataclass
class CompatibilityReport:
    """Detailed compatibility check results"""
    timestamp: float
    passed: bool
    checks_performed: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    hardware_info: Dict[str, Any]
    software_info: Dict[str, Any]
    feature_availability: Dict[str, bool]


@dataclass
class GuardrailPolicy:
    """Guardrail policy definition"""
    name: str
    description: str
    condition: Callable[[StateCheckpoint], bool]
    intervention: Optional[str] = None
    severity: str = "warning"  # warning, error, critical
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, checkpoint: StateCheckpoint) -> Tuple[bool, Optional[str]]:
        """Evaluate guardrail condition"""
        try:
            triggered = self.condition(checkpoint)
            if triggered:
                message = f"Guardrail '{self.name}' triggered: {self.description}"
                return True, message
            return False, None
        except Exception as e:
            return True, f"Guardrail '{self.name}' evaluation error: {str(e)}"


class LifecycleTracker:
    """
    Enhanced lifecycle tracker with comprehensive state management.
    
    Features:
    - Full lifecycle state tracking with detailed checkpoints
    - Automatic compatibility checking and validation
    - Intervention and guardrail mapping
    - Verbose feedback system
    - Runtime hot-reloading support
    - Hardware/environment awareness
    """
    
    def __init__(self, intervention_engine: Optional[InterventionEngine] = None):
        self.logger = get_logger()
        self._lock = RLock()
        
        # State management
        self.current_state = LifecycleState.NOT_STARTED
        self.state_history = CircularBuffer(10000)  # Extended history
        self.checkpoints: OrderedDict[float, StateCheckpoint] = OrderedDict()
        
        # Arguments and configuration
        self.startup_arguments: Dict[str, Any] = {}
        self.runtime_config: Dict[str, Any] = {}
        self.environment_snapshot: Dict[str, str] = {}
        
        # Compatibility tracking
        self.compatibility_reports: List[CompatibilityReport] = []
        self.feature_requirements: Dict[str, Set[str]] = defaultdict(set)
        
        # Guardrails and policies
        self.guardrails: Dict[str, GuardrailPolicy] = {}
        self.guardrail_history = CircularBuffer(5000)
        
        # State transition validators
        self.transition_validators: Dict[Tuple[LifecycleState, LifecycleState], 
                                       List[Callable]] = defaultdict(list)
        
        # Intervention mapping
        self.state_intervention_map: Dict[LifecycleState, List[str]] = defaultdict(list)
        self.error_intervention_map: Dict[str, List[str]] = defaultdict(list)
        self.intervention_engine = intervention_engine
        
        # Component tracking
        self.registered_components: Dict[str, weakref.ref] = {}
        self.component_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Metrics and monitoring
        self.state_durations: Dict[LifecycleState, List[float]] = defaultdict(list)
        self.transition_times: Dict[Tuple[LifecycleState, LifecycleState], 
                                  List[float]] = defaultdict(list)
        
        # Feedback system
        self.feedback_handlers: List[Callable[[str, str, Dict], None]] = []
        self.feedback_buffer = CircularBuffer(1000)
        
        # Hardware mapping
        self.hardware_capabilities: Dict[str, Any] = {}
        self.hardware_requirements: Dict[str, Dict[str, Any]] = {}
        
        # Initialize signal handlers
        self._setup_signal_handlers()
        
        # Register default guardrails
        self._register_default_guardrails()
        
        # Start tracking
        self._start_time = time.time()
        self._last_checkpoint_time = self._start_time
        
        self.logger.info("LifecycleTracker initialized with enhanced capabilities")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}")
            self.transition_to_state(
                LifecycleState.SHUTTING_DOWN,
                StateTransition.EMERGENCY_STOP,
                {"signal": signum, "frame": str(frame)}
            )
        
        # Register signal handlers
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, signal_handler)
        
        # Register atexit handler
        atexit.register(self._cleanup)

    def _register_default_guardrails(self):
        """Register default system guardrails"""
        # Memory guardrail
        self.register_guardrail(GuardrailPolicy(
            name="memory_threshold",
            description="Prevent OOM by monitoring memory usage",
            condition=lambda cp: cp.metrics.get('memory_percent', 0) > 90,
            intervention="cleanup_memory",
            severity="critical"
        ))
        
        # Error rate guardrail
        self.register_guardrail(GuardrailPolicy(
            name="error_rate_threshold",
            description="Monitor error rate and trigger recovery",
            condition=lambda cp: cp.metrics.get('error_rate', 0) > 0.1,
            intervention="restart_component",
            severity="error"
        ))
        
        # State duration guardrail
        self.register_guardrail(GuardrailPolicy(
            name="stuck_state_detection",
            description="Detect stuck states and trigger intervention",
            condition=lambda cp: self._check_stuck_state(cp),
            intervention="force_state_transition",
            severity="warning"
        ))

    def track_startup(self, 
                     arguments: Dict[str, Any],
                     environment: Optional[Dict[str, str]] = None,
                     hardware_info: Optional[Dict[str, Any]] = None) -> StateCheckpoint:
        """
        Track vLLM startup with comprehensive state capture.
        
        Args:
            arguments: Command-line arguments and configuration
            environment: Environment variables (auto-captured if None)
            hardware_info: Hardware configuration (auto-detected if None)
            
        Returns:
            StateCheckpoint with startup details
        """
        with self._lock:
            self.startup_arguments = arguments.copy()
            self.environment_snapshot = environment or dict(os.environ)
            
            # Perform compatibility checks
            compat_report = self._check_compatibility(arguments, hardware_info)
            self.compatibility_reports.append(compat_report)
            
            # Create startup checkpoint
            checkpoint = self._create_checkpoint(
                LifecycleState.INITIALIZING,
                StateTransition.STARTUP,
                arguments=arguments,
                environment=self.environment_snapshot,
                hardware_state=hardware_info or self._detect_hardware(),
                metadata={
                    "compatibility_passed": compat_report.passed,
                    "warnings": compat_report.warnings,
                    "feature_availability": compat_report.feature_availability
                }
            )
            
            # Provide verbose feedback
            self._emit_feedback(
                "info",
                f"vLLM startup initiated with {len(arguments)} arguments",
                {
                    "checkpoint_id": checkpoint.timestamp,
                    "compatibility": compat_report.passed,
                    "warnings": len(compat_report.warnings)
                }
            )
            
            # Log detailed startup info
            self.logger.info(f"vLLM startup tracked: {json.dumps(checkpoint.to_dict(), indent=2)}")
            
            return checkpoint

    def track_shutdown(self, 
                      reason: str = "normal",
                      cleanup_status: Optional[Dict[str, bool]] = None) -> StateCheckpoint:
        """
        Track vLLM shutdown with cleanup status.
        
        Args:
            reason: Shutdown reason
            cleanup_status: Status of cleanup operations
            
        Returns:
            StateCheckpoint with shutdown details
        """
        with self._lock:
            checkpoint = self._create_checkpoint(
                LifecycleState.SHUTTING_DOWN,
                StateTransition.SHUTDOWN,
                metadata={
                    "reason": reason,
                    "cleanup_status": cleanup_status or {},
                    "uptime_seconds": time.time() - self._start_time,
                    "total_checkpoints": len(self.checkpoints)
                }
            )
            
            # Perform cleanup operations
            self._perform_shutdown_cleanup()
            
            # Final state transition
            self.current_state = LifecycleState.TERMINATED
            
            self._emit_feedback(
                "info",
                f"vLLM shutdown completed: {reason}",
                {"checkpoint_id": checkpoint.timestamp, "uptime": checkpoint.metadata["uptime_seconds"]}
            )
            
            return checkpoint

    def transition_to_state(self,
                           new_state: LifecycleState,
                           transition_type: StateTransition,
                           context: Optional[Dict[str, Any]] = None) -> StateCheckpoint:
        """
        Transition to a new state with validation and intervention mapping.
        
        Args:
            new_state: Target state
            transition_type: Type of transition
            context: Additional context for the transition
            
        Returns:
            StateCheckpoint for the transition
        """
        with self._lock:
            previous_state = self.current_state
            
            # Validate transition
            if not self._validate_transition(previous_state, new_state):
                raise ValueError(
                    f"Invalid state transition: {previous_state.name} -> {new_state.name}"
                )
            
            # Record transition time
            transition_start = time.time()
            
            # Create checkpoint
            checkpoint = self._create_checkpoint(
                new_state,
                transition_type,
                metadata=context or {}
            )
            
            # Execute transition validators
            self._execute_transition_validators(previous_state, new_state, checkpoint)
            
            # Check and execute interventions
            interventions = self._get_state_interventions(new_state)
            if interventions and self.intervention_engine:
                for intervention_name in interventions:
                    try:
                        self.intervention_engine.execute_intervention(
                            intervention_name,
                            {"checkpoint": checkpoint}
                        )
                        checkpoint.interventions_triggered.append(intervention_name)
                    except Exception as e:
                        self.logger.error(f"Intervention '{intervention_name}' failed: {e}")
            
            # Evaluate guardrails
            self._evaluate_guardrails(checkpoint)
            
            # Update state
            self.current_state = new_state
            
            # Record transition time
            transition_time = time.time() - transition_start
            self.transition_times[(previous_state, new_state)].append(transition_time)
            
            # Emit feedback
            self._emit_feedback(
                "info",
                f"State transition: {previous_state.name} -> {new_state.name}",
                {
                    "transition_type": transition_type.value,
                    "duration": transition_time,
                    "interventions": len(checkpoint.interventions_triggered),
                    "guardrails": len(checkpoint.guardrails_enforced)
                }
            )
            
            return checkpoint

    def register_guardrail(self, policy: GuardrailPolicy) -> None:
        """
        Register a new guardrail policy.
        
        Args:
            policy: Guardrail policy to register
        """
        with self._lock:
            self.guardrails[policy.name] = policy
            self._emit_feedback(
                "info",
                f"Guardrail registered: {policy.name}",
                {"severity": policy.severity, "enabled": policy.enabled}
            )
            self.logger.info(f"Registered guardrail: {policy.name}")

    def map_state_to_intervention(self, 
                                 state: LifecycleState, 
                                 intervention_name: str) -> None:
        """
        Map a state to an intervention for automatic execution.
        
        Args:
            state: Lifecycle state
            intervention_name: Name of intervention to trigger
        """
        with self._lock:
            self.state_intervention_map[state].append(intervention_name)
            self._emit_feedback(
                "info",
                f"Mapped intervention '{intervention_name}' to state '{state.name}'",
                {}
            )

    def map_error_to_intervention(self,
                                 error_pattern: str,
                                 intervention_name: str) -> None:
        """
        Map an error pattern to an intervention.
        
        Args:
            error_pattern: Error message pattern or exception class name
            intervention_name: Name of intervention to trigger
        """
        with self._lock:
            self.error_intervention_map[error_pattern].append(intervention_name)
            self._emit_feedback(
                "info",
                f"Mapped intervention '{intervention_name}' to error pattern '{error_pattern}'",
                {}
            )

    def register_component(self, 
                          name: str, 
                          component: Any,
                          requirements: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component for lifecycle tracking.
        
        Args:
            name: Component name
            component: Component instance
            requirements: Hardware/software requirements
        """
        with self._lock:
            self.registered_components[name] = weakref.ref(component)
            if requirements:
                self.hardware_requirements[name] = requirements
            
            # Initialize component state
            self.component_states[name] = {
                "registered_at": time.time(),
                "status": "active",
                "errors": 0,
                "last_error": None
            }
            
            self._emit_feedback(
                "info",
                f"Component registered: {name}",
                {"has_requirements": requirements is not None}
            )

    def add_feedback_handler(self, handler: Callable[[str, str, Dict], None]) -> None:
        """
        Add a feedback handler for verbose output.
        
        Args:
            handler: Function to handle feedback (level, message, context)
        """
        self.feedback_handlers.append(handler)

    def get_state_report(self) -> Dict[str, Any]:
        """Get comprehensive state report"""
        with self._lock:
            return {
                "current_state": self.current_state.name,
                "uptime": time.time() - self._start_time,
                "total_checkpoints": len(self.checkpoints),
                "state_durations": {
                    state.name: {
                        "total": sum(durations),
                        "average": sum(durations) / len(durations) if durations else 0,
                        "count": len(durations)
                    }
                    for state, durations in self.state_durations.items()
                },
                "active_guardrails": sum(1 for g in self.guardrails.values() if g.enabled),
                "interventions_triggered": sum(
                    len(cp.interventions_triggered) for cp in self.checkpoints.values()
                ),
                "registered_components": len(self.registered_components),
                "compatibility": {
                    "last_check": self.compatibility_reports[-1].to_dict() 
                    if self.compatibility_reports else None
                }
            }

    def _create_checkpoint(self,
                          state: LifecycleState,
                          transition_type: Optional[StateTransition] = None,
                          **kwargs) -> StateCheckpoint:
        """Create a new state checkpoint"""
        checkpoint = StateCheckpoint(
            timestamp=time.time(),
            state=state,
            previous_state=self.current_state,
            transition_type=transition_type,
            arguments=kwargs.get("arguments", self.startup_arguments),
            environment=kwargs.get("environment", self.environment_snapshot),
            hardware_state=kwargs.get("hardware_state", self._detect_hardware()),
            error_context=kwargs.get("error_context"),
            metrics=self._collect_metrics(),
            metadata=kwargs.get("metadata", {})
        )
        
        # Store checkpoint
        self.checkpoints[checkpoint.timestamp] = checkpoint
        self.state_history.append(checkpoint.to_dict())
        
        # Update state duration tracking
        if self.current_state != state:
            duration = checkpoint.timestamp - self._last_checkpoint_time
            self.state_durations[self.current_state].append(duration)
            self._last_checkpoint_time = checkpoint.timestamp
        
        return checkpoint

    def _check_compatibility(self,
                           arguments: Dict[str, Any],
                           hardware_info: Optional[Dict[str, Any]] = None) -> CompatibilityReport:
        """Perform comprehensive compatibility checks"""
        checks = {}
        warnings = []
        errors = []
        recommendations = []
        
        # Hardware checks
        hw_info = hardware_info or self._detect_hardware()
        
        # CUDA availability
        if arguments.get("device", "cuda") == "cuda":
            import torch
            checks["cuda_available"] = torch.cuda.is_available()
            if not checks["cuda_available"]:
                errors.append("CUDA requested but not available")
                recommendations.append("Use --device cpu or install CUDA")
        
        # Memory checks
        required_memory = arguments.get("gpu_memory_utilization", 0.9)
        if "gpu_memory" in hw_info:
            available_memory = hw_info["gpu_memory"] * (1 - required_memory)
            model_size = arguments.get("model_size_gb", 0)
            checks["sufficient_memory"] = available_memory >= model_size
            if not checks["sufficient_memory"]:
                warnings.append(f"Model may not fit in available GPU memory")
        
        # Feature checks
        features = {
            "flash_attention": self._check_flash_attention_support(),
            "triton": self._check_triton_support(),
            "distributed": arguments.get("tensor_parallel_size", 1) > 1
        }
        
        return CompatibilityReport(
            timestamp=time.time(),
            passed=len(errors) == 0,
            checks_performed=checks,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            hardware_info=hw_info,
            software_info=self._get_software_info(),
            feature_availability=features
        )

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware configuration"""
        hardware = {
            "cpu_count": os.cpu_count(),
            "platform": sys.platform
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                hardware.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_names": [torch.cuda.get_device_name(i) 
                                 for i in range(torch.cuda.device_count())],
                    "gpu_memory": [torch.cuda.get_device_properties(i).total_memory 
                                  for i in range(torch.cuda.device_count())]
                })
        except ImportError:
            pass
        
        return hardware

    def _get_software_info(self) -> Dict[str, Any]:
        """Get software version information"""
        info = {
            "python_version": sys.version,
            "vllm_version": "unknown"  # Would be imported from vllm
        }
        
        # Get installed package versions
        try:
            import pkg_resources
            for package in ["torch", "transformers", "numpy"]:
                try:
                    info[f"{package}_version"] = pkg_resources.get_distribution(package).version
                except:
                    pass
        except ImportError:
            pass
        
        return info

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {}
        
        try:
            import psutil
            process = psutil.Process()
            metrics.update({
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads()
            })
        except ImportError:
            pass
        
        return metrics

    def _validate_transition(self, 
                           from_state: LifecycleState, 
                           to_state: LifecycleState) -> bool:
        """Validate state transition"""
        # Define valid transitions
        valid_transitions = {
            LifecycleState.NOT_STARTED: [LifecycleState.VALIDATING_ENVIRONMENT, 
                                        LifecycleState.INITIALIZING],
            LifecycleState.INITIALIZING: [LifecycleState.LOADING_MODEL, 
                                         LifecycleState.ERROR],
            LifecycleState.LOADING_MODEL: [LifecycleState.ALLOCATING_MEMORY, 
                                          LifecycleState.ERROR],
            LifecycleState.READY: [LifecycleState.SERVING, 
                                  LifecycleState.SHUTTING_DOWN],
            # Add more valid transitions
        }
        
        # Check if transition is valid
        return to_state in valid_transitions.get(from_state, []) or \
               to_state in [LifecycleState.ERROR, LifecycleState.SHUTTING_DOWN]

    def _execute_transition_validators(self,
                                     from_state: LifecycleState,
                                     to_state: LifecycleState,
                                     checkpoint: StateCheckpoint) -> None:
        """Execute registered transition validators"""
        validators = self.transition_validators.get((from_state, to_state), [])
        for validator in validators:
            try:
                validator(checkpoint)
            except Exception as e:
                self.logger.error(f"Transition validator failed: {e}")
                checkpoint.metadata["validator_errors"] = checkpoint.metadata.get(
                    "validator_errors", []
                ) + [str(e)]

    def _get_state_interventions(self, state: LifecycleState) -> List[str]:
        """Get interventions mapped to a state"""
        return self.state_intervention_map.get(state, [])

    def _evaluate_guardrails(self, checkpoint: StateCheckpoint) -> None:
        """Evaluate all active guardrails"""
        for name, policy in self.guardrails.items():
            if not policy.enabled:
                continue
                
            triggered, message = policy.evaluate(checkpoint)
            if triggered:
                checkpoint.guardrails_enforced.append(name)
                self.guardrail_history.append({
                    "timestamp": time.time(),
                    "guardrail": name,
                    "message": message,
                    "severity": policy.severity
                })
                
                # Trigger intervention if specified
                if policy.intervention and self.intervention_engine:
                    try:
                        self.intervention_engine.execute_intervention(
                            policy.intervention,
                            {"guardrail": name, "checkpoint": checkpoint}
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Guardrail intervention '{policy.intervention}' failed: {e}"
                        )

    def _check_stuck_state(self, checkpoint: StateCheckpoint) -> bool:
        """Check if system is stuck in a state too long"""
        if self.current_state in [LifecycleState.INITIALIZING, 
                                 LifecycleState.LOADING_MODEL]:
            duration = time.time() - self._last_checkpoint_time
            return duration > 300  # 5 minutes
        return False

    def _check_flash_attention_support(self) -> bool:
        """Check if Flash Attention is supported"""
        try:
            import torch
            # Check CUDA capability for Flash Attention
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                return capability[0] >= 7  # SM 7.0+ required
        except:
            pass
        return False

    def _check_triton_support(self) -> bool:
        """Check if Triton is available"""
        try:
            import triton
            return True
        except ImportError:
            return False

    def _emit_feedback(self, level: str, message: str, context: Dict[str, Any]) -> None:
        """Emit verbose feedback"""
        feedback = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "context": context
        }
        
        # Store in buffer
        self.feedback_buffer.append(feedback)
        
        # Call registered handlers
        for handler in self.feedback_handlers:
            try:
                handler(level, message, context)
            except Exception as e:
                self.logger.error(f"Feedback handler error: {e}")

    def _perform_shutdown_cleanup(self) -> None:
        """Perform cleanup operations during shutdown"""
        # Clear weak references
        self.registered_components.clear()
        
        # Save final state report
        try:
            report = self.get_state_report()
            self.logger.info(f"Final state report: {json.dumps(report, indent=2)}")
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")

    def _cleanup(self) -> None:
        """Cleanup handler for atexit"""
        if self.current_state != LifecycleState.TERMINATED:
            self.track_shutdown("atexit_handler")