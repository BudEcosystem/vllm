# vLLM Monitoring System Integration Guide

This guide explains how to integrate the comprehensive monitoring system into vLLM's core engine for runtime monitoring, predictive failure detection, and automatic mitigation.

## Overview

The vLLM monitoring system provides:

1. **Lifecycle Tracking**: Complete tracking of vLLM states from startup to shutdown
2. **Predictive Failure Detection**: AI-powered analysis to predict failures before they occur
3. **Continuous Learning**: Self-improving mitigation selection based on outcomes
4. **Automatic Mitigation**: Real-time execution of corrective actions
5. **Deep Engine Integration**: Non-invasive monitoring through method wrapping
6. **Hot-Reload Plugins**: Add new monitoring capabilities without restart
7. **Distributed Support**: Multi-worker and heterogeneous hardware scenarios

## Quick Start

### 1. Pre-Startup Validation

Before starting vLLM, run the pre-startup validation:

```bash
python -m vllm_monitor.prestartup_check --auto-fix --model meta-llama/Llama-2-7b-hf
```

### 2. Environment Variables

Set these environment variables to enable monitoring:

```bash
export VLLM_ENABLE_MONITORING=true
export VLLM_MONITOR_PREDICTIVE=true
export VLLM_MONITOR_LEARNING=true
export VLLM_MONITOR_AUTO_MITIGATE=true
```

### 3. Using Monitored vLLM

#### Option A: Using LLM Class
```python
from vllm import LLM, SamplingParams

# Create LLM with monitoring enabled
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_monitoring=True  # This enables the monitoring system
)

# Use normally
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.8))

# Check monitoring status
status = llm.get_monitoring_status()
print(f"Monitoring active: {status['monitor_active']}")
```

#### Option B: Using Engine Directly
```python
from vllm_monitor.vllm_engine_integration import create_monitored_engine
from vllm.engine.arg_utils import EngineArgs

# Create engine args
engine_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    enable_monitoring=True,
    enable_auto_mitigation=True
)

# Create monitored engine
engine, monitor = create_monitored_engine(engine_args)

# Use the engine
engine.add_request("req_1", "Hello world", SamplingParams())
outputs = engine.step()
```

## Integration with vLLM Source Code

To integrate the monitoring system into vLLM's source code, the following modifications are needed:

### 1. Modify `vllm/engine/llm_engine.py`

Add monitoring parameters to `LLMEngine.__init__`:

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: Type[ExecutorBase],
    log_stats: bool,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    # ADD: New parameters for monitoring
    enable_monitoring: bool = True,
    monitor_config: Optional[Dict[str, Any]] = None,
) -> None:
    # ... existing initialization code ...
    
    # ADD: Initialize monitoring system (after line 400)
    self._monitor: Optional[VLLMEngineMonitor] = None
    if enable_monitoring:
        from vllm_monitor.vllm_engine_integration import VLLMEngineMonitor
        monitor_kwargs = monitor_config or {
            'enable_predictive': True,
            'enable_learning': True,
            'enable_auto_mitigation': True
        }
        self._monitor = VLLMEngineMonitor(**monitor_kwargs)
        self._monitor.attach_to_engine(self)
        logger.info("vLLM monitoring system enabled")
```

### 2. Modify `vllm/engine/arg_utils.py`

Add monitoring configuration to `EngineArgs`:

```python
@dataclass
class EngineArgs:
    # ... existing fields ...
    
    # ADD: Monitoring configuration
    enable_monitoring: bool = False
    enable_predictive_detection: bool = True
    enable_continuous_learning: bool = True
    enable_auto_mitigation: bool = True
```

Add CLI arguments:

```python
parser.add_argument(
    '--enable-monitoring',
    action='store_true',
    help='Enable comprehensive monitoring system with predictive failure detection'
)
```

### 3. Create Custom StatLogger

Create `vllm_monitor/engine_stat_logger.py`:

```python
from vllm.engine.metrics import StatLoggerBase, Stats

class MonitorStatLogger(StatLoggerBase):
    """StatLogger that forwards vLLM statistics to the monitoring system."""
    
    def __init__(self, monitor: 'VLLMEngineMonitor'):
        self.monitor = monitor
    
    def log(self, stats: Stats) -> None:
        """Forward stats to monitoring system"""
        metrics = {
            'num_running_requests': stats.num_running,
            'num_waiting_requests': stats.num_waiting,
            'gpu_cache_usage': stats.gpu_cache_usage,
            # ... extract other metrics ...
        }
        self.monitor.monitor.collect_state(metrics)
```

## Monitoring Features

### 1. Lifecycle States

The system tracks these vLLM lifecycle states:

- `NOT_STARTED`: Initial state
- `INITIALIZING`: Setting up engine
- `LOADING_MODEL`: Loading model weights
- `READY`: Ready to serve
- `SERVING`: Processing requests
- `DEGRADED`: Performance issues detected
- `ERROR`: Error state
- `SHUTTING_DOWN`: Graceful shutdown

### 2. Predictive Failure Detection

The system analyzes patterns to predict:

- Out of Memory (OOM) errors
- Queue overflow
- Performance degradation
- Hardware failures
- Network issues

Example prediction:
```python
predictions = monitor.predictive_detector.analyze_checkpoint(checkpoint)
for pred in predictions:
    print(f"Failure type: {pred.failure_type}")
    print(f"Probability: {pred.probability:.1%}")
    print(f"Time to failure: {pred.time_to_failure}s")
```

### 3. Automatic Mitigation Strategies

Built-in mitigation strategies include:

- **EmergencyGPUMemoryCleanup**: Clear GPU memory when OOM risk detected
- **ReduceBatchSize**: Dynamically reduce batch size under memory pressure
- **RestartNCCL**: Fix distributed communication issues
- **RebalanceLoad**: Redistribute requests across workers
- **ScaleResources**: Request additional compute resources

### 4. Continuous Learning

The system learns from mitigation outcomes:

- Tracks success/failure of each mitigation
- Updates confidence scores
- Extracts new rules from patterns
- Improves strategy selection over time

### 5. Plugin System

Add custom monitoring capabilities:

```python
from vllm_monitor.plugin_system import PluginInterface

class CustomMonitor(PluginInterface):
    def execute(self) -> Dict[str, Any]:
        # Your monitoring logic
        return {"status": "ok", "custom_metric": 42}

monitor.plugin_manager.register_plugin(CustomMonitor())
```

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/vllm-monitored.service`:

```ini
[Unit]
Description=vLLM Inference Server with Monitoring
After=network.target

[Service]
Type=simple
User=vllm
Group=vllm

# Environment variables
Environment="VLLM_ENABLE_MONITORING=true"
Environment="VLLM_MONITOR_PREDICTIVE=true"
Environment="VLLM_MONITOR_LEARNING=true"
Environment="VLLM_MONITOR_AUTO_MITIGATE=true"

# Pre-startup validation
ExecStartPre=/usr/bin/python -m vllm_monitor.prestartup_check

# Main service
ExecStart=/usr/bin/python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --enable-monitoring \
    --host 0.0.0.0 \
    --port 8000

Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

### Docker Integration

Add to your Dockerfile:

```dockerfile
# Install monitoring system
COPY vllm_monitor /app/vllm_monitor

# Set monitoring environment variables
ENV VLLM_ENABLE_MONITORING=true
ENV VLLM_MONITOR_PREDICTIVE=true
ENV VLLM_MONITOR_LEARNING=true
ENV VLLM_MONITOR_AUTO_MITIGATE=true

# Run pre-startup check before starting vLLM
ENTRYPOINT ["sh", "-c", "python -m vllm_monitor.prestartup_check && exec $@"]
```

## Monitoring Dashboard

Access monitoring metrics:

### Prometheus Metrics

The system exposes metrics at `/metrics`:

- `vllm:monitor_active_requests`: Number of active requests
- `vllm:monitor_tokens_generated`: Total tokens generated
- `vllm:monitor_mitigation_triggered`: Mitigation events
- `vllm:monitor_prediction_accuracy`: Prediction accuracy

### Custom Metrics

Get real-time status:

```python
status = monitor.get_engine_status()
print(f"""
Active Requests: {status['active_requests']}
Tokens Generated: {status['tokens_generated']}
Avg Step Duration: {status['avg_step_duration_ms']}ms
Mitigation Active: {status['mitigation_in_progress']}
""")
```

## Best Practices

1. **Enable Monitoring in Production**: Always run with monitoring enabled for production deployments

2. **Configure Thresholds**: Adjust mitigation thresholds based on your hardware:
   ```bash
   export VLLM_MONITOR_MITIGATION_THRESHOLD=0.8  # Trigger at 80% resource usage
   ```

3. **Review Learning Data**: Periodically review what the system has learned:
   ```python
   recommendations = monitor.continuous_learner.get_learning_recommendations(
       state, errors, metrics
   )
   ```

4. **Custom Plugins**: Create organization-specific monitoring plugins for your use cases

5. **Integration Testing**: Test monitoring integration with your specific models and workloads

## Troubleshooting

### Monitoring Not Active

Check:
1. Environment variable `VLLM_ENABLE_MONITORING` is set to `true`
2. No import errors for `vllm_monitor` package
3. Engine was created with `enable_monitoring=True`

### Predictions Not Working

Verify:
1. `VLLM_MONITOR_PREDICTIVE` is set to `true`
2. Sufficient data has been collected (predictions improve over time)
3. Check logs for prediction analysis results

### Mitigations Not Executing

Ensure:
1. `VLLM_MONITOR_AUTO_MITIGATE` is set to `true`
2. Mitigation threshold is appropriate for your system
3. Required permissions for mitigation actions (e.g., GPU access)

## Advanced Usage

### Custom Mitigation Strategies

```python
from vllm_monitor.vllm_mitigation_strategies import MitigationStrategy

class CustomMitigation(MitigationStrategy):
    def execute(self, context: Dict[str, Any]) -> MitigationOutcome:
        # Your mitigation logic
        return MitigationOutcome.SUCCESS

monitor.continuous_learner.register_strategy(
    "custom_mitigation",
    CustomMitigation()
)
```

### Distributed Monitoring

For multi-node deployments:

```python
from vllm_monitor.distributed_metrics import DistributedMetricsCollector

distributed_collector = DistributedMetricsCollector()
monitor.monitor.register_collector(distributed_collector)
```

### Export Monitoring Data

```python
# Export to Prometheus
monitor.monitor.export_prometheus("/path/to/metrics")

# Export to JSON
monitoring_data = monitor.monitor.export_data()
with open("monitoring_data.json", "w") as f:
    json.dump(monitoring_data, f)
```

## Conclusion

The vLLM monitoring system provides comprehensive runtime monitoring, predictive failure detection, and automatic mitigation capabilities. By integrating it into your vLLM deployment, you can:

- Prevent failures before they occur
- Automatically recover from errors
- Continuously improve system reliability
- Gain deep insights into engine performance

For more examples, see the `examples/` directory in the monitoring package.