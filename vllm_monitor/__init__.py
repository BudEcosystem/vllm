"""
vLLM Monitor - Comprehensive monitoring and self-healing system for vLLM

A standalone library for monitoring, tracking, and managing vLLM internal states,
performance metrics, errors, and providing self-healing capabilities with minimal overhead.

Features:
- Real-time state tracking for all vLLM components
- Performance and resource monitoring
- Error detection and exception handling
- Request lifecycle tracking
- Configuration monitoring
- Predictive maintenance capabilities
- Self-healing interventions
- Minimal performance overhead
"""

from .core import (
    VLLMMonitor,
    MonitorConfig,
    MonitorState,
    ComponentState,
)

from .lifecycle_tracker import (
    LifecycleTracker,
    LifecycleState,
    StateTransition,
    StateCheckpoint,
    CompatibilityReport,
    GuardrailPolicy,
)

from .plugin_system import (
    PluginManager,
    PluginRegistry,
    PluginLoader,
    PluginInterface,
    SimplePlugin,
    PluginType,
    PluginStatus,
    PluginMetadata,
    PluginValidationResult,
)

from .collectors import (
    StateCollector,
    PerformanceCollector,
    ErrorCollector,
    RequestCollector,
    ResourceCollector,
)

from .analyzers import (
    AnomalyDetector,
    PerformanceAnalyzer,
    FailurePredictor,
    HealthScorer,
)

try:
    from .interventions import (
        InterventionEngine,
        SelfHealingAgent,
        GuardrailManager,
    )
except ImportError:
    # Interventions module may have dependencies not available
    InterventionEngine = None
    SelfHealingAgent = None
    GuardrailManager = None

try:
    from .exporters import (
        MetricsExporter,
        AlertManager,
        LogExporter,
        ExportManager,
    )
except ImportError:
    # Exporters module may have dependencies not available
    MetricsExporter = None
    AlertManager = None
    LogExporter = None
    ExportManager = None

__version__ = "1.0.0"
__author__ = "vLLM Monitor Team"

__all__ = [
    # Core components
    "VLLMMonitor",
    "MonitorConfig", 
    "MonitorState",
    "ComponentState",
    
    # Lifecycle tracking
    "LifecycleTracker",
    "LifecycleState",
    "StateTransition",
    "StateCheckpoint",
    "CompatibilityReport",
    "GuardrailPolicy",
    
    # Plugin system
    "PluginManager",
    "PluginRegistry",
    "PluginLoader",
    "PluginInterface",
    "SimplePlugin",
    "PluginType",
    "PluginStatus",
    "PluginMetadata",
    "PluginValidationResult",
    
    # Collectors
    "StateCollector",
    "PerformanceCollector", 
    "ErrorCollector",
    "RequestCollector",
    "ResourceCollector",
    
    # Analyzers
    "AnomalyDetector",
    "PerformanceAnalyzer",
    "FailurePredictor", 
    "HealthScorer",
    
    # Interventions
    "InterventionEngine",
    "SelfHealingAgent",
    "GuardrailManager",
    
    # Exporters
    "MetricsExporter",
    "AlertManager",
    "LogExporter",
    "ExportManager",
]


def create_monitor_with_plugins(config: Optional[MonitorConfig] = None) -> VLLMMonitor:
    """
    Create a VLLMMonitor instance with lifecycle tracking and plugin system pre-configured.
    
    Args:
        config: Optional monitor configuration
        
    Returns:
        Configured VLLMMonitor instance with plugins and lifecycle tracking ready
    """
    monitor = VLLMMonitor(config)
    monitor.setup_lifecycle_tracking()
    monitor.setup_plugin_system()
    return monitor


# Export the factory function
__all__.append("create_monitor_with_plugins")