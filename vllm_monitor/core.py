"""
Core monitoring system for vLLM with minimal overhead design.

This module provides the fundamental monitoring infrastructure that can track
all aspects of vLLM operation while maintaining sub-microsecond overhead.
"""

import asyncio
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    TypeVar, Generic, Protocol, runtime_checkable
)
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# Custom logger with minimal overhead
logger = logging.getLogger("vllm_monitor.core")


def get_logger(name: str = "vllm_monitor") -> logging.Logger:
    """Get a logger instance for the monitoring system."""
    return logging.getLogger(name)


class ComponentType(Enum):
    """Types of vLLM components that can be monitored."""
    ENGINE = "engine"
    SCHEDULER = "scheduler"
    WORKER = "worker"
    MODEL_RUNNER = "model_runner"
    CACHE_ENGINE = "cache_engine"
    TOKENIZER = "tokenizer"
    API_SERVER = "api_server"
    REQUEST_HANDLER = "request_handler"
    SEQUENCE_GROUP = "sequence_group"
    SEQUENCE = "sequence"
    BLOCK_MANAGER = "block_manager"
    ATTENTION = "attention"
    SAMPLER = "sampler"


class StateType(Enum):
    """Types of states that can be monitored."""
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR = "error"
    REQUEST = "request"
    CONFIGURATION = "configuration"
    HEALTH = "health"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5


@dataclass
class MonitorConfig:
    """Configuration for the monitoring system."""
    
    # Core settings
    enabled: bool = True
    collection_interval_ms: float = 100.0  # Collection interval in milliseconds
    max_history_size: int = 10000  # Maximum number of historical records
    enable_async_collection: bool = True
    
    # Performance settings
    max_collection_time_us: float = 50.0  # Maximum collection time in microseconds
    enable_adaptive_sampling: bool = True
    performance_impact_threshold: float = 0.001  # 0.1% maximum performance impact
    
    # Component monitoring
    monitor_engine: bool = True
    monitor_scheduler: bool = True
    monitor_workers: bool = True
    monitor_cache: bool = True
    monitor_requests: bool = True
    monitor_performance: bool = True
    monitor_resources: bool = True
    monitor_errors: bool = True
    
    # Analysis settings
    enable_anomaly_detection: bool = True
    enable_failure_prediction: bool = True
    enable_performance_analysis: bool = True
    anomaly_threshold: float = 2.0  # Standard deviations for anomaly detection
    
    # Intervention settings
    enable_self_healing: bool = True
    enable_guardrails: bool = True
    intervention_cooldown_s: float = 60.0  # Minimum time between interventions
    max_interventions_per_hour: int = 10
    
    # Export settings
    enable_metrics_export: bool = True
    enable_logging: bool = True
    enable_alerts: bool = True
    log_level: AlertLevel = AlertLevel.INFO
    
    # Storage settings
    metrics_retention_hours: int = 24
    logs_retention_hours: int = 72
    enable_compression: bool = True
    
    # Advanced settings
    enable_ml_analysis: bool = False  # Enable ML-based analysis (higher overhead)
    sampling_rate: float = 1.0  # Fraction of events to sample (1.0 = all)
    thread_pool_size: int = 2  # Number of background threads


@dataclass
class ComponentState:
    """State information for a monitored component."""
    
    component_id: str
    component_type: ComponentType
    state_type: StateType
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    
    # Health indicators
    is_healthy: bool = True
    health_score: float = 1.0
    last_error: Optional[str] = None
    error_count: int = 0
    
    # Operational metrics
    requests_processed: int = 0
    average_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "state_type": self.state_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "is_healthy": self.is_healthy,
            "health_score": self.health_score,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "requests_processed": self.requests_processed,
            "average_latency_ms": self.average_latency_ms,
            "throughput_rps": self.throughput_rps,
        }


@dataclass
class MonitorState:
    """Overall state of the monitoring system."""
    
    # System status
    is_running: bool = False
    start_time: float = 0.0
    uptime_seconds: float = 0.0
    
    # Collection statistics
    total_collections: int = 0
    failed_collections: int = 0
    average_collection_time_us: float = 0.0
    max_collection_time_us: float = 0.0
    
    # Component tracking
    monitored_components: Set[str] = field(default_factory=set)
    component_count: int = 0
    healthy_components: int = 0
    unhealthy_components: int = 0
    
    # Performance impact
    cpu_overhead_percent: float = 0.0
    memory_overhead_mb: float = 0.0
    performance_impact: float = 0.0
    
    # Alert statistics
    alerts_generated: int = 0
    interventions_performed: int = 0
    predictions_made: int = 0
    
    # Data statistics
    metrics_stored: int = 0
    logs_stored: int = 0
    storage_size_mb: float = 0.0


@runtime_checkable
class Collector(Protocol):
    """Protocol for data collectors."""
    
    def collect(self) -> List[ComponentState]:
        """Collect current state data."""
        ...
    
    def get_overhead_us(self) -> float:
        """Get collection overhead in microseconds."""
        ...


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for data analyzers."""
    
    def analyze(self, states: List[ComponentState]) -> Dict[str, Any]:
        """Analyze collected states."""
        ...


@runtime_checkable
class Intervention(Protocol):
    """Protocol for intervention actions."""
    
    def can_intervene(self, issue: Dict[str, Any]) -> bool:
        """Check if this intervention can handle the issue."""
        ...
    
    def intervene(self, issue: Dict[str, Any]) -> bool:
        """Perform intervention. Returns success status."""
        ...


class CircularBuffer:
    """High-performance circular buffer for time-series data."""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = np.empty(maxsize, dtype=object)
        self.head = 0
        self.size = 0
        self._lock = threading.RLock()
    
    def append(self, item: Any) -> None:
        """Add item to buffer with O(1) complexity."""
        with self._lock:
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.maxsize
            if self.size < self.maxsize:
                self.size += 1
    
    def get_latest(self, n: int = None) -> List[Any]:
        """Get latest n items (or all if n is None)."""
        with self._lock:
            if n is None:
                n = self.size
            n = min(n, self.size)
            
            if n == 0:
                return []
            
            items = []
            start = (self.head - n) % self.maxsize
            for i in range(n):
                idx = (start + i) % self.maxsize
                items.append(self.buffer[idx])
            return items
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.head = 0
            self.size = 0


class PerformanceTimer:
    """Ultra-low-overhead performance timer."""
    
    __slots__ = ['start_time', 'end_time']
    
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return (self.end_time - self.start_time) * 1_000_000


class VLLMMonitor:
    """
    Main monitoring system for vLLM with minimal performance overhead.
    
    This class orchestrates all monitoring activities while ensuring that
    the performance impact on vLLM remains below the configured threshold.
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.state = MonitorState()
        
        # Component registries
        self._components: Dict[str, Any] = {}  # Weak references to monitored components
        self._collectors: List[Collector] = []
        self._analyzers: List[Analyzer] = []
        self._interventions: List[Intervention] = []
        
        # Data storage
        self._state_history = CircularBuffer(self.config.max_history_size)
        self._metrics_cache: Dict[str, CircularBuffer] = defaultdict(
            lambda: CircularBuffer(1000)
        )
        self._alerts = CircularBuffer(1000)
        
        # Async infrastructure
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._collection_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive tasks
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="vllm_monitor"
        )
        
        # Performance tracking
        self._collection_times = deque(maxlen=100)
        self._last_performance_check = 0.0
        self._adaptive_sampling_rate = 1.0
        
        # Thread safety
        self._lock = threading.RLock()
        self._running = False
        
        # Initialize subsystems
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging with minimal overhead."""
        if self.config.enable_logging:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] vLLM Monitor: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.config.log_level.value * 10)
    
    def register_component(self, 
                          component: Any, 
                          component_id: str, 
                          component_type: ComponentType) -> None:
        """Register a vLLM component for monitoring."""
        with self._lock:
            # Use weak reference to avoid preventing garbage collection
            self._components[component_id] = {
                'ref': weakref.ref(component),
                'type': component_type,
                'registered_at': time.time()
            }
            self.state.monitored_components.add(component_id)
            self.state.component_count = len(self._components)
            
            logger.debug(f"Registered component {component_id} of type {component_type}")
    
    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from monitoring."""
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                self.state.monitored_components.discard(component_id)
                self.state.component_count = len(self._components)
                
                logger.debug(f"Unregistered component {component_id}")
    
    def add_collector(self, collector: Collector) -> None:
        """Add a data collector."""
        with self._lock:
            self._collectors.append(collector)
            logger.debug(f"Added collector {type(collector).__name__}")
    
    def add_analyzer(self, analyzer: Analyzer) -> None:
        """Add a data analyzer."""
        with self._lock:
            self._analyzers.append(analyzer)
            logger.debug(f"Added analyzer {type(analyzer).__name__}")
    
    def add_intervention(self, intervention: Intervention) -> None:
        """Add an intervention handler."""
        with self._lock:
            self._interventions.append(intervention)
            logger.debug(f"Added intervention {type(intervention).__name__}")
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return
        
        logger.info("Starting vLLM Monitor")
        
        with self._lock:
            self._running = True
            self.state.is_running = True
            self.state.start_time = time.time()
        
        # Start event loop if needed
        if self.config.enable_async_collection:
            self._event_loop = asyncio.get_running_loop()
            
            # Start collection task
            self._collection_task = asyncio.create_task(self._collection_loop())
            
            # Start analysis task
            self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("vLLM Monitor started successfully")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self._running:
            return
        
        logger.info("Stopping vLLM Monitor")
        
        with self._lock:
            self._running = False
            self.state.is_running = False
        
        # Cancel async tasks
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("vLLM Monitor stopped")
    
    async def _collection_loop(self) -> None:
        """Main collection loop with adaptive sampling."""
        while self._running:
            try:
                await self._collect_data()
                
                # Adaptive sleep based on performance impact
                sleep_time = self.config.collection_interval_ms / 1000.0
                if self.config.enable_adaptive_sampling:
                    sleep_time *= (2.0 - self._adaptive_sampling_rate)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.state.failed_collections += 1
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _analysis_loop(self) -> None:
        """Main analysis loop."""
        while self._running:
            try:
                await self._analyze_data()
                await asyncio.sleep(1.0)  # Analyze every second
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _collect_data(self) -> None:
        """Collect data from all registered collectors."""
        if not self._collectors:
            return
        
        with PerformanceTimer() as timer:
            try:
                # Collect from all collectors in parallel
                collection_tasks = []
                for collector in self._collectors:
                    if self.config.enable_async_collection:
                        task = asyncio.create_task(
                            asyncio.to_thread(collector.collect)
                        )
                        collection_tasks.append(task)
                
                if collection_tasks:
                    results = await asyncio.gather(*collection_tasks, return_exceptions=True)
                    
                    # Process results
                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning(f"Collection error: {result}")
                            continue
                        
                        if isinstance(result, list):
                            for state in result:
                                self._store_state(state)
                
                self.state.total_collections += 1
                
            except Exception as e:
                logger.error(f"Data collection failed: {e}")
                self.state.failed_collections += 1
        
        # Track performance impact
        collection_time_us = timer.elapsed_us
        self._collection_times.append(collection_time_us)
        
        # Update performance metrics
        self.state.average_collection_time_us = np.mean(self._collection_times)
        self.state.max_collection_time_us = max(
            self.state.max_collection_time_us, collection_time_us
        )
        
        # Adaptive sampling based on performance impact
        if self.config.enable_adaptive_sampling:
            self._update_adaptive_sampling(collection_time_us)
    
    def _update_adaptive_sampling(self, collection_time_us: float) -> None:
        """Update adaptive sampling rate based on performance impact."""
        target_time_us = self.config.max_collection_time_us
        
        if collection_time_us > target_time_us:
            # Reduce sampling rate
            self._adaptive_sampling_rate *= 0.9
        else:
            # Increase sampling rate (but not above 1.0)
            self._adaptive_sampling_rate = min(1.0, self._adaptive_sampling_rate * 1.05)
        
        # Apply sampling rate to collectors if they support it
        for collector in self._collectors:
            if hasattr(collector, 'set_sampling_rate'):
                collector.set_sampling_rate(self._adaptive_sampling_rate)
    
    def _store_state(self, state: ComponentState) -> None:
        """Store component state with efficient indexing."""
        self._state_history.append(state)
        
        # Update component health tracking
        if state.is_healthy:
            if state.component_id not in getattr(self, '_healthy_components', set()):
                self.state.healthy_components += 1
                getattr(self, '_healthy_components', set()).add(state.component_id)
        else:
            if state.component_id not in getattr(self, '_unhealthy_components', set()):
                self.state.unhealthy_components += 1
                getattr(self, '_unhealthy_components', set()).add(state.component_id)
        
        # Cache metrics for quick access
        metric_key = f"{state.component_id}:{state.state_type.value}"
        self._metrics_cache[metric_key].append(state)
    
    async def _analyze_data(self) -> None:
        """Run analysis on collected data."""
        if not self._analyzers:
            return
        
        # Get recent states for analysis
        recent_states = self._state_history.get_latest(1000)
        if not recent_states:
            return
        
        # Run analyzers in thread pool to avoid blocking
        analysis_tasks = []
        for analyzer in self._analyzers:
            task = asyncio.create_task(
                asyncio.to_thread(analyzer.analyze, recent_states)
            )
            analysis_tasks.append(task)
        
        if analysis_tasks:
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process analysis results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Analysis error: {result}")
                    continue
                
                if isinstance(result, dict):
                    await self._handle_analysis_result(result)
    
    async def _handle_analysis_result(self, result: Dict[str, Any]) -> None:
        """Handle analysis results and trigger interventions if needed."""
        # Check for issues that require intervention
        if result.get('requires_intervention', False):
            await self._trigger_intervention(result)
        
        # Generate alerts if needed
        if result.get('alert_level', 0) >= self.config.log_level.value:
            self._generate_alert(result)
    
    async def _trigger_intervention(self, issue: Dict[str, Any]) -> None:
        """Trigger appropriate intervention for detected issue."""
        if not self.config.enable_self_healing:
            return
        
        for intervention in self._interventions:
            if intervention.can_intervene(issue):
                try:
                    success = await asyncio.to_thread(intervention.intervene, issue)
                    if success:
                        self.state.interventions_performed += 1
                        logger.info(f"Intervention successful: {type(intervention).__name__}")
                        break
                except Exception as e:
                    logger.error(f"Intervention failed: {e}")
    
    def _generate_alert(self, result: Dict[str, Any]) -> None:
        """Generate alert for significant events."""
        alert = {
            'timestamp': time.time(),
            'level': result.get('alert_level', AlertLevel.INFO),
            'message': result.get('message', 'Unknown issue'),
            'component': result.get('component', 'unknown'),
            'data': result.get('data', {})
        }
        
        self._alerts.append(alert)
        self.state.alerts_generated += 1
        
        # Log alert
        level = AlertLevel(alert['level'])
        if level.value >= self.config.log_level.value:
            log_func = getattr(logger, level.name.lower(), logger.info)
            log_func(f"Alert: {alert['message']} (Component: {alert['component']})")
    
    def get_component_state(self, component_id: str) -> Optional[ComponentState]:
        """Get current state of a specific component."""
        metric_keys = [k for k in self._metrics_cache.keys() if k.startswith(f"{component_id}:")]
        if not metric_keys:
            return None
        
        # Return most recent state
        latest_states = []
        for key in metric_keys:
            states = self._metrics_cache[key].get_latest(1)
            if states:
                latest_states.extend(states)
        
        if latest_states:
            return max(latest_states, key=lambda s: s.timestamp)
        return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        self.state.uptime_seconds = time.time() - self.state.start_time
        
        total_components = self.state.healthy_components + self.state.unhealthy_components
        health_percentage = (
            self.state.healthy_components / total_components * 100.0
            if total_components > 0 else 100.0
        )
        
        return {
            'overall_health': health_percentage,
            'monitored_components': self.state.component_count,
            'healthy_components': self.state.healthy_components,
            'unhealthy_components': self.state.unhealthy_components,
            'uptime_seconds': self.state.uptime_seconds,
            'performance_impact': self.state.performance_impact,
            'collections_success_rate': (
                (self.state.total_collections - self.state.failed_collections) /
                max(self.state.total_collections, 1) * 100.0
            ),
            'average_collection_time_us': self.state.average_collection_time_us,
            'alerts_count': self.state.alerts_generated,
            'interventions_count': self.state.interventions_performed,
        }
    
    def get_metrics(self, 
                   component_id: Optional[str] = None,
                   state_type: Optional[StateType] = None,
                   limit: int = 100) -> List[ComponentState]:
        """Get metrics with optional filtering."""
        if component_id and state_type:
            key = f"{component_id}:{state_type.value}"
            return self._metrics_cache[key].get_latest(limit)
        elif component_id:
            states = []
            for key in self._metrics_cache.keys():
                if key.startswith(f"{component_id}:"):
                    states.extend(self._metrics_cache[key].get_latest(limit))
            return sorted(states, key=lambda s: s.timestamp)[-limit:]
        else:
            return self._state_history.get_latest(limit)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._running:
            # Can't use async in __exit__, so we'll just stop synchronously
            self._running = False
    
    # Integration with new components
    
    def setup_lifecycle_tracking(self, intervention_engine: Optional[Any] = None) -> 'LifecycleTracker':
        """
        Setup and integrate lifecycle tracking.
        
        Args:
            intervention_engine: Optional intervention engine to use
            
        Returns:
            Configured LifecycleTracker instance
        """
        from .lifecycle_tracker import LifecycleTracker
        
        if not hasattr(self, '_lifecycle_tracker'):
            self._lifecycle_tracker = LifecycleTracker(intervention_engine)
            
            # Add feedback handler that logs to our logger
            def feedback_handler(level: str, message: str, context: Dict[str, Any]):
                log_func = getattr(logger, level, logger.info)
                log_func(f"Lifecycle: {message} - {context}")
            
            self._lifecycle_tracker.add_feedback_handler(feedback_handler)
            
            # Register lifecycle tracker as a component
            self.register_component(
                self._lifecycle_tracker,
                "lifecycle_tracker",
                ComponentType.ENGINE
            )
        
        return self._lifecycle_tracker
    
    def setup_plugin_system(self) -> 'PluginManager':
        """
        Setup and integrate the plugin system.
        
        Returns:
            Configured PluginManager instance
        """
        from .plugin_system import PluginManager, PluginType
        
        if not hasattr(self, '_plugin_manager'):
            self._plugin_manager = PluginManager()
            
            # Add feedback handler
            def feedback_handler(level: str, message: str, context: Dict[str, Any]):
                log_func = getattr(logger, level, logger.info)
                log_func(f"Plugin: {message} - {context}")
            
            self._plugin_manager.add_feedback_handler(feedback_handler)
            
            # Auto-register plugins as collectors/analyzers/interventions
            def on_plugin_registered(plugin_name: str):
                plugin = self._plugin_manager.registry.get_plugin(plugin_name)
                if not plugin:
                    return
                    
                metadata = plugin.get_metadata()
                
                # Register based on plugin type
                if metadata.type == PluginType.COLLECTOR:
                    self.add_collector(plugin)
                elif metadata.type == PluginType.ANALYZER:
                    self.add_analyzer(plugin)
                elif metadata.type == PluginType.INTERVENTION:
                    self.add_intervention(plugin)
                elif metadata.type == PluginType.GUARDRAIL and hasattr(self, '_lifecycle_tracker'):
                    # Create guardrail policy
                    from .lifecycle_tracker import GuardrailPolicy
                    policy = GuardrailPolicy(
                        name=metadata.name,
                        description=metadata.description,
                        condition=lambda cp: plugin.execute(cp)[0],
                        intervention=metadata.configuration.get('intervention'),
                        severity=metadata.configuration.get('severity', 'warning')
                    )
                    self._lifecycle_tracker.register_guardrail(policy)
            
            # Hook into plugin registration
            self._plugin_manager._on_plugin_registered = on_plugin_registered
        
        return self._plugin_manager
    
    def register_plugin(self, 
                       name: str,
                       plugin_type: str,
                       execute_code: str,
                       description: str = "",
                       **kwargs) -> bool:
        """
        Register a new plugin easily.
        
        Args:
            name: Plugin name
            plugin_type: Type of plugin (collector, analyzer, etc.)
            execute_code: Python code for the execute function
            description: Plugin description
            **kwargs: Additional metadata
            
        Returns:
            True if plugin registered successfully
        """
        if not hasattr(self, '_plugin_manager'):
            self.setup_plugin_system()
        
        return self._plugin_manager.create_plugin(
            name, plugin_type, execute_code, description, **kwargs
        )
    
    def track_lifecycle_state(self,
                             new_state: 'LifecycleState',
                             transition_type: 'StateTransition',
                             context: Optional[Dict[str, Any]] = None) -> 'StateCheckpoint':
        """
        Track a lifecycle state transition.
        
        Args:
            new_state: Target state
            transition_type: Type of transition
            context: Additional context
            
        Returns:
            StateCheckpoint for the transition
        """
        if not hasattr(self, '_lifecycle_tracker'):
            self.setup_lifecycle_tracking()
        
        return self._lifecycle_tracker.transition_to_state(
            new_state, transition_type, context
        )
    
    def register_guardrail(self,
                          name: str,
                          description: str,
                          condition_code: str,
                          intervention: Optional[str] = None,
                          severity: str = "warning") -> bool:
        """
        Register a new guardrail.
        
        Args:
            name: Guardrail name
            description: What the guardrail checks
            condition_code: Python code that returns True if guardrail triggered
            intervention: Name of intervention to execute
            severity: warning, error, or critical
            
        Returns:
            True if guardrail registered successfully
        """
        if not hasattr(self, '_plugin_manager'):
            self.setup_plugin_system()
        
        return self._plugin_manager.create_guardrail(
            name, description, condition_code, intervention, severity
        )
    
    def get_lifecycle_report(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle tracking report."""
        if not hasattr(self, '_lifecycle_tracker'):
            return {"error": "Lifecycle tracking not initialized"}
        
        return self._lifecycle_tracker.get_state_report()
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        if not hasattr(self, '_plugin_manager'):
            return []
        
        return self._plugin_manager.list_plugins(plugin_type)