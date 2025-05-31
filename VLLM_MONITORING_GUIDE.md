# vLLM Monitoring System Guide

A comprehensive monitoring, tracking, and self-healing system for vLLM with minimal performance overhead.

## Overview

The vLLM Monitoring System is a standalone library designed to monitor every aspect of vLLM operation while maintaining sub-microsecond overhead. It provides:

- **Real-time state tracking** for all vLLM components
- **Performance and resource monitoring**
- **Error detection and exception handling**
- **Request lifecycle tracking**
- **Predictive maintenance capabilities**
- **Self-healing interventions**
- **Guardrail system** to prevent failures
- **Comprehensive alerting and export**

## Key Features

### 🔍 Comprehensive Monitoring
- **Component State Tracking**: Engine, Scheduler, Workers, Cache, Model Runner
- **Performance Metrics**: CPU, Memory, GPU utilization, Latency, Throughput
- **Error Tracking**: Exception handling, Error rates, Stack trace analysis
- **Request Monitoring**: Lifecycle tracking, Success rates, Token generation

### 🤖 Intelligent Analysis
- **Anomaly Detection**: Statistical and ML-based detection using Z-score, IQR, Isolation Forest
- **Performance Analysis**: Trend analysis, Optimization insights, Resource utilization
- **Failure Prediction**: Predictive maintenance, Time-to-failure estimation, Risk assessment
- **Health Scoring**: Weighted component health, Overall system health

### 🏥 Self-Healing Capabilities
- **Memory Cleanup**: Automatic garbage collection, GPU cache clearing
- **Component Restart**: Intelligent component recovery
- **Load Balancing**: Dynamic request throttling, Batch size adjustment
- **Resource Scaling**: Automatic resource allocation, Priority adjustment
- **Configuration Tuning**: Performance-based parameter adjustment

### 🛡️ Guardrail System
- **Memory Guardrails**: Prevent memory exhaustion
- **CPU Guardrails**: Avoid CPU overload
- **Error Rate Guardrails**: Detect error cascades
- **Queue Guardrails**: Prevent request backlog
- **Response Time Guardrails**: Ensure timely responses

### 📊 Export & Alerting
- **Multiple Formats**: JSON, CSV, Prometheus
- **Real-time Alerts**: Webhook, Email, Slack integration
- **Structured Logging**: Comprehensive log export
- **Metrics Retention**: Configurable data retention

## Installation

```bash
# Install the monitoring system
pip install psutil GPUtil numpy scipy scikit-learn numba

# Optional dependencies for full functionality
pip install aiohttp  # For webhook alerts
pip install requests  # For HTTP integrations
```

## Quick Start

### Basic Monitoring Setup

```python
import asyncio
from vllm_monitor import (
    VLLMMonitor, MonitorConfig, ComponentType,
    StateCollector, PerformanceCollector, ErrorCollector,
    AnomalyDetector, FailurePredictor, SelfHealingAgent
)

async def setup_monitoring():
    # 1. Configure monitoring
    config = MonitorConfig(
        enabled=True,
        collection_interval_ms=1000,
        enable_anomaly_detection=True,
        enable_failure_prediction=True,
        enable_self_healing=True
    )
    
    # 2. Create monitor
    monitor = VLLMMonitor(config)
    
    # 3. Register your vLLM components
    # Assuming you have vllm_engine, scheduler, workers, etc.
    monitor.register_component(vllm_engine, "engine_main", ComponentType.ENGINE)
    monitor.register_component(scheduler, "scheduler_0", ComponentType.SCHEDULER)
    monitor.register_component(worker, "worker_0", ComponentType.WORKER)
    
    # 4. Add collectors
    monitor.add_collector(StateCollector(monitor._components))
    monitor.add_collector(PerformanceCollector())
    monitor.add_collector(ErrorCollector())
    
    # 5. Add analyzers
    monitor.add_analyzer(AnomalyDetector())
    monitor.add_analyzer(FailurePredictor())
    
    # 6. Setup self-healing
    healer = SelfHealingAgent(config, monitor._components)
    
    # 7. Start monitoring
    await monitor.start()
    await healer.start()
    
    return monitor, healer

# Usage
monitor, healer = await setup_monitoring()
```

### Advanced Configuration

```python
from vllm_monitor import (
    MonitorConfig, ExportConfig, AlertConfig,
    ExportManager, GuardrailManager
)

# Detailed monitoring configuration
monitor_config = MonitorConfig(
    # Core settings
    enabled=True,
    collection_interval_ms=500,  # Collect every 500ms
    max_history_size=50000,
    enable_async_collection=True,
    
    # Performance settings
    max_collection_time_us=100.0,  # Max 100μs overhead
    enable_adaptive_sampling=True,
    performance_impact_threshold=0.001,  # 0.1% max impact
    
    # Analysis settings
    enable_anomaly_detection=True,
    enable_failure_prediction=True,
    anomaly_threshold=2.5,  # 2.5 standard deviations
    
    # Self-healing settings
    enable_self_healing=True,
    enable_guardrails=True,
    intervention_cooldown_s=30.0,
    max_interventions_per_hour=20,
    
    # Export settings
    enable_metrics_export=True,
    enable_logging=True,
    enable_alerts=True
)

# Export configuration
export_config = ExportConfig(
    export_interval_seconds=60,
    export_format="json",
    enable_compression=True,
    retention_hours=24
)

# Alert configuration
alert_config = AlertConfig(
    enabled=True,
    min_alert_level=AlertLevel.WARNING,
    webhook_url="https://your-webhook.com/alerts",
    email_recipients=["admin@yourcompany.com"],
    slack_webhook="https://hooks.slack.com/your-webhook"
)
```

## Component Integration

### Integrating with vLLM Components

```python
# Example: Monitoring LLM Engine
from vllm import LLM

# Create your vLLM instance
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Register with monitor
monitor.register_component(
    llm.llm_engine, 
    "llm_engine", 
    ComponentType.ENGINE
)

# Register scheduler
if hasattr(llm.llm_engine, 'scheduler'):
    monitor.register_component(
        llm.llm_engine.scheduler,
        "scheduler",
        ComponentType.SCHEDULER
    )

# Register workers
if hasattr(llm.llm_engine, 'workers'):
    for i, worker in enumerate(llm.llm_engine.workers):
        monitor.register_component(
            worker,
            f"worker_{i}",
            ComponentType.WORKER
        )
```

### Custom Component Monitoring

```python
class CustomVLLMComponent:
    def __init__(self):
        self.requests_processed = 0
        self.error_count = 0
        self.is_healthy = True
    
    def process_request(self):
        try:
            # Your processing logic
            self.requests_processed += 1
        except Exception:
            self.error_count += 1
            self.is_healthy = False

# Register custom component
component = CustomVLLMComponent()
monitor.register_component(
    component,
    "custom_processor",
    ComponentType.MODEL_RUNNER
)
```

## Monitoring Capabilities

### 1. Anomaly Detection

The system detects anomalies using multiple methods:

```python
# Statistical anomaly detection
anomaly_detector = AnomalyDetector(
    sensitivity=2.0,  # Z-score threshold
    min_samples=20    # Minimum samples for detection
)

# Results include:
# - Component ID
# - Anomaly score
# - Confidence level
# - Contributing factors
# - Severity assessment
```

### 2. Performance Analysis

Analyze performance trends and generate optimization insights:

```python
perf_analyzer = PerformanceAnalyzer(
    lookback_hours=1.0  # Analyze last hour
)

# Provides insights on:
# - CPU/Memory/GPU trends
# - Latency degradation
# - Throughput optimization
# - Resource utilization
```

### 3. Failure Prediction

Predict potential failures before they occur:

```python
failure_predictor = FailurePredictor(
    prediction_horizon_hours=4.0  # Predict 4 hours ahead
)

# Predictions include:
# - Failure probability
# - Time to failure
# - Failure type
# - Warning signs
# - Recommended actions
```

### 4. Health Scoring

Calculate comprehensive health scores:

```python
health_scorer = HealthScorer()

# Provides:
# - Overall system health percentage
# - Component-level health
# - Health insights
# - Trending analysis
```

## Self-Healing System

### Automatic Interventions

The system includes several built-in interventions:

```python
# Memory cleanup intervention
class MemoryCleanupIntervention:
    def can_intervene(self, issue):
        return 'memory' in issue.get('contributing_factors', [])
    
    def intervene(self, issue):
        # Force garbage collection
        # Clear GPU cache
        # Return success status

# Component restart intervention
class ComponentRestartIntervention:
    def can_intervene(self, issue):
        return issue.get('failure_probability', 0) > 0.7
    
    def intervene(self, issue):
        # Restart unhealthy component
        # Verify recovery
        # Return success status
```

### Custom Interventions

```python
from vllm_monitor.interventions import BaseIntervention

class CustomIntervention(BaseIntervention):
    def can_intervene(self, issue):
        # Define when this intervention applies
        return issue.get('component_type') == 'custom'
    
    def _execute_intervention(self, issue):
        # Implement your intervention logic
        try:
            # Fix the issue
            success = True
            action = "custom_fix_applied"
        except Exception as e:
            success = False
            action = f"custom_fix_failed: {e}"
        
        return InterventionResult(
            success=success,
            action_taken=action,
            details={},
            execution_time_ms=0.0,
            side_effects=[]
        )

# Register custom intervention
self_healing_agent.intervention_engine.register_intervention(
    CustomIntervention()
)
```

## Guardrail System

### Built-in Guardrails

```python
guardrail_manager = GuardrailManager(config)

# Built-in guardrails check for:
# - Memory usage > 95%
# - CPU usage > 98%  
# - Error count > 100
# - Queue size > 500
# - Response time > 30s
```

### Custom Guardrails

```python
def custom_guardrail(state):
    """Custom guardrail for specific conditions."""
    if state.component_type == ComponentType.WORKER:
        gpu_util = state.data.get('gpu_utilization', 0)
        if gpu_util > 0.99:  # 99% GPU utilization
            return {
                'type': 'gpu_exhaustion',
                'severity': AlertLevel.CRITICAL,
                'message': f"GPU utilization critical: {gpu_util:.1%}",
                'component_id': state.component_id,
                'action_required': 'scale_gpu_resources'
            }
    return None

# Register custom guardrail
guardrail_manager.register_guardrail('gpu_utilization', custom_guardrail)
```

## Export and Alerting

### Metrics Export

```python
# Export to multiple formats
export_manager = ExportManager(
    export_config=ExportConfig(
        export_format="json",  # or "csv", "prometheus"
        export_interval_seconds=60,
        enable_compression=True
    )
)

# Exported files include:
# - metrics_timestamp.json(.gz)
# - logs_timestamp.json(.gz)  
# - prometheus metrics
```

### Real-time Alerts

```python
# Webhook alerts
alert_config = AlertConfig(
    webhook_url="https://your-webhook.com/alerts"
)

# Slack integration
alert_config = AlertConfig(
    slack_webhook="https://hooks.slack.com/your-webhook"
)

# Email alerts (requires SMTP configuration)
alert_config = AlertConfig(
    email_recipients=["admin@company.com"]
)
```

## Performance Impact

The monitoring system is designed for minimal overhead:

- **Collection Time**: < 100μs per collection cycle
- **Memory Overhead**: < 50MB for typical deployments
- **CPU Impact**: < 0.1% additional CPU usage
- **Adaptive Sampling**: Automatically reduces sampling under load
- **JIT Optimization**: NumPy/Numba acceleration for calculations

### Performance Monitoring

```python
# Monitor the monitor's performance
health_status = monitor.get_system_health()
print(f"Average collection time: {health_status['average_collection_time_us']:.1f}μs")
print(f"Performance impact: {health_status['performance_impact']:.3f}%")
```

## Configuration Reference

### MonitorConfig

```python
@dataclass
class MonitorConfig:
    # Core settings
    enabled: bool = True
    collection_interval_ms: float = 100.0
    max_history_size: int = 10000
    enable_async_collection: bool = True
    
    # Performance settings  
    max_collection_time_us: float = 50.0
    enable_adaptive_sampling: bool = True
    performance_impact_threshold: float = 0.001
    
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
    anomaly_threshold: float = 2.0
    
    # Intervention settings
    enable_self_healing: bool = True
    enable_guardrails: bool = True
    intervention_cooldown_s: float = 60.0
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
```

## API Reference

### VLLMMonitor

```python
class VLLMMonitor:
    def __init__(self, config: MonitorConfig)
    
    async def start(self) -> None
    async def stop(self) -> None
    
    def register_component(self, component, component_id: str, component_type: ComponentType)
    def unregister_component(self, component_id: str)
    
    def add_collector(self, collector: Collector)
    def add_analyzer(self, analyzer: Analyzer)
    
    def get_system_health(self) -> Dict[str, Any]
    def get_component_state(self, component_id: str) -> Optional[ComponentState]
    def get_metrics(self, component_id: str = None, limit: int = 100) -> List[ComponentState]
```

### SelfHealingAgent

```python
class SelfHealingAgent:
    def __init__(self, config: MonitorConfig, component_registry: Dict[str, Any])
    
    async def start(self) -> None
    async def stop(self) -> None
    async def heal(self, issue: Dict[str, Any]) -> bool
    
    def get_healing_status(self) -> Dict[str, Any]
```

## Examples

### Complete Integration Example

See `/app/vllm_monitor/examples/complete_monitoring_example.py` for a comprehensive demonstration.

### Simple Health Check

```python
async def health_check():
    """Simple health monitoring."""
    monitor = VLLMMonitor()
    
    # Add basic collectors
    monitor.add_collector(PerformanceCollector())
    monitor.add_collector(ResourceCollector())
    
    await monitor.start()
    
    # Check health periodically
    while True:
        health = monitor.get_system_health()
        print(f"System Health: {health['overall_health']:.1f}%")
        await asyncio.sleep(10)
```

### Custom Analysis

```python
class CustomAnalyzer:
    def analyze(self, states: List[ComponentState]) -> Dict[str, Any]:
        """Custom analysis logic."""
        # Implement your analysis
        issues_found = []
        
        for state in states:
            if state.health_score < 0.5:
                issues_found.append(state.component_id)
        
        return {
            'type': 'custom_analysis',
            'timestamp': time.time(),
            'issues_found': issues_found,
            'requires_intervention': len(issues_found) > 0,
            'message': f"Found {len(issues_found)} unhealthy components"
        }

# Add to monitor
monitor.add_analyzer(CustomAnalyzer())
```

## Best Practices

### 1. Configuration Tuning

```python
# For production environments
production_config = MonitorConfig(
    collection_interval_ms=1000,    # 1 second intervals
    max_collection_time_us=50.0,    # 50μs max overhead
    performance_impact_threshold=0.0005,  # 0.05% max impact
    enable_adaptive_sampling=True,
    anomaly_threshold=2.5,          # Less sensitive
    intervention_cooldown_s=120.0,  # 2 minute cooldown
)

# For development/testing
dev_config = MonitorConfig(
    collection_interval_ms=500,     # 500ms intervals
    enable_ml_analysis=True,        # Enable ML features
    anomaly_threshold=1.5,          # More sensitive
    intervention_cooldown_s=30.0,   # Short cooldown
)
```

### 2. Component Registration

```python
# Register components early in initialization
def setup_vllm_with_monitoring():
    # Create vLLM components
    llm = LLM(model="your-model")
    
    # Setup monitoring
    monitor = VLLMMonitor()
    
    # Register in order of importance
    monitor.register_component(llm.llm_engine, "engine", ComponentType.ENGINE)
    monitor.register_component(llm.llm_engine.scheduler, "scheduler", ComponentType.SCHEDULER)
    
    # Register workers
    for i, worker in enumerate(llm.llm_engine.workers):
        monitor.register_component(worker, f"worker_{i}", ComponentType.WORKER)
    
    return llm, monitor
```

### 3. Error Handling

```python
try:
    await monitor.start()
except Exception as e:
    logger.error(f"Failed to start monitoring: {e}")
    # Continue without monitoring rather than failing
    monitor = None

# Check if monitoring is available
if monitor:
    health = monitor.get_system_health()
else:
    # Fallback health check
    health = {"overall_health": 100.0}
```

### 4. Resource Management

```python
# Use context manager for proper cleanup
async def main():
    config = MonitorConfig(enable_self_healing=True)
    
    async with VLLMMonitor(config) as monitor:
        # Add collectors and analyzers
        await setup_monitoring(monitor)
        
        # Your application logic
        await run_application()
        
    # Monitor automatically cleaned up
```

## Troubleshooting

### Common Issues

1. **High Collection Overhead**
   ```python
   # Reduce collection frequency
   config.collection_interval_ms = 2000  # 2 seconds
   
   # Enable adaptive sampling
   config.enable_adaptive_sampling = True
   
   # Reduce analyzers
   config.enable_ml_analysis = False
   ```

2. **Memory Usage**
   ```python
   # Reduce history size
   config.max_history_size = 1000
   
   # Enable compression
   config.enable_compression = True
   
   # Reduce retention
   config.metrics_retention_hours = 12
   ```

3. **False Positives**
   ```python
   # Increase anomaly threshold
   config.anomaly_threshold = 3.0
   
   # Increase minimum samples
   anomaly_detector = AnomalyDetector(min_samples=50)
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger("vllm_monitor").setLevel(logging.DEBUG)

# Check system status
status = monitor.get_system_health()
print(f"Monitoring overhead: {status['performance_impact']:.4f}%")
print(f"Collection success rate: {status['collections_success_rate']:.1f}%")
```

## Contributing

The vLLM Monitoring System is designed to be extensible. You can contribute by:

1. **Adding Custom Collectors** for new metrics
2. **Creating Analyzers** for specific use cases  
3. **Implementing Interventions** for custom scenarios
4. **Adding Export Backends** for different systems
5. **Improving Performance** optimization

## License

This monitoring system is designed to work with vLLM and follows the same licensing terms.

## Support

For issues and questions:
- Review this documentation
- Check the example code
- Examine debug logs
- File issues with detailed error information