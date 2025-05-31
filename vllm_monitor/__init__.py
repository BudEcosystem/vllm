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