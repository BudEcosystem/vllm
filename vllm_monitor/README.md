# vLLM Monitoring System

A comprehensive monitoring, predictive failure detection, and automatic mitigation system for vLLM (Large Language Model serving engine).

## Features

### Core Capabilities

1. **🔍 Lifecycle Tracking**
   - Complete state tracking from startup to shutdown
   - 20+ lifecycle states with transition validation
   - Hardware and environment state capture
   - Intervention and guardrail tracking

2. **🔮 Predictive Failure Detection**
   - AI-powered failure prediction using state graphs
   - Pattern recognition for common failure modes
   - Time-to-failure estimation
   - Preventive action recommendations

3. **🧠 Continuous Learning**
   - Self-improving mitigation selection
   - Multi-method learning (RL, Bayesian, rule-based)
   - Outcome tracking and strategy optimization
   - Automatic rule extraction from patterns

4. **🛡️ Automatic Mitigation**
   - Real-time execution of corrective actions
   - 20+ built-in mitigation strategies
   - Context-aware strategy selection
   - Non-disruptive intervention execution

5. **🔌 Plugin System**
   - Hot-reload capability for new components
   - Automatic dependency resolution
   - AI-agent friendly SDK
   - Simple plugin creation interface

6. **📊 Deep vLLM Integration**
   - Non-invasive engine monitoring
   - Request lifecycle tracking
   - Performance metrics collection
   - Distributed deployment support

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd vllm_monitor

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Enable monitoring via environment variables
import os
os.environ['VLLM_ENABLE_MONITORING'] = 'true'

# Create LLM with monitoring
llm = LLM(
    model="gpt2",
    enable_monitoring=True
)

# Use normally - monitoring happens automatically
outputs = llm.generate(
    ["Hello, world!"], 
    SamplingParams(temperature=0.8)
)

# Check monitoring status
status = llm.get_monitoring_status()
print(f"Active requests: {status['active_requests']}")
```

### Pre-Startup Validation

Run validation before starting vLLM:

```bash
python -m vllm_monitor.prestartup_check --auto-fix --model <model_name>
```

## Architecture

### Component Overview

```
vllm_monitor/
├── core.py                          # Core monitoring framework
├── lifecycle_tracker.py             # State tracking system
├── plugin_system.py                 # Plugin architecture
├── distributed_metrics.py           # Multi-worker metrics
├── predictive_failure_detection.py  # AI failure prediction
├── continuous_learning.py           # Self-improving system
├── vllm_integration_plugins.py      # vLLM-specific plugins
├── vllm_mitigation_strategies.py    # Mitigation actions
├── vllm_engine_integration.py       # Engine integration
├── prestartup_check.py             # Pre-startup validation
└── engine_modifications.py          # Source code patches
```

### Integration Flow

1. **Pre-Startup**: Validation and system preparation
2. **Engine Creation**: Monitor attachment to LLMEngine
3. **Request Processing**: Real-time monitoring and analysis
4. **Prediction**: Continuous failure risk assessment
5. **Mitigation**: Automatic corrective actions
6. **Learning**: Outcome tracking and improvement

## Key Components

### 1. Lifecycle Tracker

Tracks vLLM states and transitions:

```python
from vllm_monitor.lifecycle_tracker import LifecycleTracker, LifecycleState

tracker = LifecycleTracker()
checkpoint = tracker.track_startup(
    arguments={'model': 'gpt2'},
    environment=dict(os.environ),
    hardware_info={'gpus': 4}
)
```

### 2. Predictive Failure Detector

Analyzes patterns to predict failures:

```python
from vllm_monitor.predictive_failure_detection import PredictiveFailureDetector

detector = PredictiveFailureDetector()
predictions = detector.analyze_checkpoint(checkpoint)

for pred in predictions:
    print(f"{pred.failure_type}: {pred.probability:.1%} in {pred.time_to_failure}s")
```

### 3. Continuous Learning System

Learns from mitigation outcomes:

```python
from vllm_monitor.continuous_learning import ContinuousLearningSystem

learner = ContinuousLearningSystem()
strategy, confidence = learner.select_best_strategy(
    state, errors, metrics, available_strategies
)
```

### 4. Plugin System

Add custom monitoring capabilities:

```python
from vllm_monitor.plugin_system import PluginManager

manager = PluginManager()
manager.create_plugin(
    name="custom_monitor",
    plugin_type="monitor",
    execute_code="""
def execute(self):
    return {"custom_metric": 42}
"""
)
```

## Mitigation Strategies

Built-in strategies include:

- **Memory Management**
  - `EmergencyGPUMemoryCleanup`: Clear GPU memory
  - `ReduceBatchSize`: Lower batch size
  - `EnableCPUOffloading`: Move data to CPU

- **Performance**
  - `DisableCUDAGraphs`: Fix graph issues
  - `ReduceModelParallelism`: Adjust parallelism
  - `OptimizeScheduler`: Tune scheduling

- **Error Recovery**
  - `RestartWorkers`: Restart failed workers
  - `RestartNCCL`: Fix communication issues
  - `HandleTimeouts`: Manage request timeouts

- **Load Management**
  - `RebalanceLoad`: Redistribute requests
  - `EnableBackpressure`: Control request flow
  - `ScaleResources`: Request more resources

## Configuration

### Environment Variables

```bash
# Core monitoring
export VLLM_ENABLE_MONITORING=true

# Feature flags
export VLLM_MONITOR_PREDICTIVE=true
export VLLM_MONITOR_LEARNING=true
export VLLM_MONITOR_AUTO_MITIGATE=true

# Configuration
export VLLM_MONITOR_LOG_LEVEL=INFO
export VLLM_MONITOR_MITIGATION_THRESHOLD=0.7
export VLLM_MONITOR_PRESTARTUP_CHECK=true
```

### Configuration File

Create `monitor_config.json`:

```json
{
  "enable_predictive": true,
  "enable_learning": true,
  "enable_auto_mitigation": true,
  "mitigation_threshold": 0.7,
  "learning_rate": 0.1,
  "plugin_directory": "/path/to/plugins"
}
```

## Examples

### Complete Monitoring Example

```python
# See examples/complete_monitoring_example.py
from vllm_monitor.vllm_engine_integration import create_monitored_engine

# Create monitored engine
engine, monitor = create_monitored_engine(engine_args)

# Process requests with monitoring
engine.add_request("req_1", "Hello world", SamplingParams())
outputs = engine.step()

# Check predictions
if monitor.predictive_detector:
    # Predictions are made automatically
    pass
```

### Custom Plugin Example

```python
from vllm_monitor import VLLMMonitor
from vllm_monitor.plugin_system import PluginInterface

class GPUTemperatureMonitor(PluginInterface):
    def initialize(self, config):
        self.threshold = config.get('temp_threshold', 80)
    
    def execute(self):
        # Read GPU temperature
        temp = read_gpu_temperature()
        
        if temp > self.threshold:
            return {
                "status": "warning",
                "temperature": temp,
                "message": "GPU running hot"
            }
        return {"status": "ok", "temperature": temp}

# Register plugin
monitor = VLLMMonitor()
monitor.plugin_manager.register_plugin(GPUTemperatureMonitor())
```

## Deployment

### Production Setup

1. **Pre-startup validation**
   ```bash
   python -m vllm_monitor.prestartup_check --auto-fix
   ```

2. **Systemd service**
   ```ini
   [Service]
   Environment="VLLM_ENABLE_MONITORING=true"
   ExecStartPre=/usr/bin/python -m vllm_monitor.prestartup_check
   ExecStart=/usr/bin/python -m vllm.entrypoints.openai.api_server --enable-monitoring
   ```

3. **Docker integration**
   ```dockerfile
   ENV VLLM_ENABLE_MONITORING=true
   ENTRYPOINT ["python", "-m", "vllm_monitor.prestartup_check", "&&", "python", "-m", "vllm.entrypoints.openai.api_server"]
   ```

### Monitoring Metrics

Access metrics via:
- Prometheus endpoint: `/metrics`
- Direct API: `monitor.get_engine_status()`
- Export: `monitor.export_data()`

## Development

### Adding New Strategies

```python
from vllm_monitor.vllm_mitigation_strategies import MitigationStrategy, MitigationOutcome

class MyStrategy(MitigationStrategy):
    def execute(self, context):
        # Implementation
        return MitigationOutcome.SUCCESS

# Register
learner.register_strategy("my_strategy", MyStrategy())
```

### Creating Plugins

```python
# Create plugin file: my_plugin.py
from vllm_monitor.plugin_system import PluginInterface

class MyPlugin(PluginInterface):
    plugin_type = "analyzer"
    
    def execute(self):
        return {"analysis": "complete"}

# Hot-reload
monitor.plugin_manager.load_plugin_file("my_plugin.py")
```

## Testing

Run tests:
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Full system test
python examples/complete_monitoring_example.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[License information]

## Support

- Documentation: See `VLLM_MONITORING_INTEGRATION_GUIDE.md`
- Issues: GitHub Issues
- Examples: `examples/` directory

---

Built with ❤️ for reliable LLM serving