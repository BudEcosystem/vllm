# vLLM Monitoring System - Persistence Guide

## Overview

The vLLM monitoring system now includes a comprehensive persistence layer that automatically saves and restores:
- Metrics collection history
- Predictive strategies and patterns
- Dynamically added modules and plugins
- Guardrails and policies
- Mitigation strategies and outcomes
- Happy path recordings
- Learning data and models

## Features

### 1. Automatic Data Persistence

All monitoring data is automatically persisted to ensure continuity across restarts:

```python
# Enable persistence when creating the monitor
monitor = VLLMMonitor(
    enable_persistence=True,
    persistence_config=PersistenceConfig(
        storage_dir=Path("./monitor_data"),
        metrics_retention_days=30,
        checkpoint_retention_days=90
    )
)
```

### 2. Persisted Data Types

#### Metrics
- Performance metrics (throughput, latency, GPU usage)
- Resource utilization (memory, CPU, GPU)
- Error rates and health scores
- Custom metrics from plugins

#### State Checkpoints
- Lifecycle state transitions
- System configuration at each state
- Hardware and environment snapshots
- Guardrail violations and interventions

#### Predictions
- Failure predictions with confidence scores
- Contributing factors and time-to-failure estimates
- Recommended mitigations
- Actual outcomes for learning

#### Mitigation Attempts
- Strategy execution history
- Success/failure outcomes
- Execution times and side effects
- Initial and final system states

#### Happy Paths
- Successful execution sequences
- Optimal metrics profiles
- Required guardrails
- Success/failure statistics

#### Dynamic Components
- Custom plugins with source code
- Guardrail policies and conditions
- Mitigation strategies
- Learning models (Q-tables, neural networks)

### 3. Storage Backends

The persistence layer supports multiple storage backends:

```python
class StorageBackend(Enum):
    SQLITE = "sqlite"      # Structured data
    JSON = "json"          # Configuration files
    HYBRID = "hybrid"      # Default: SQLite + JSON
```

### 4. Performance Optimization

The persistence layer is optimized for minimal overhead:
- Batch writing to reduce I/O operations
- Background flush thread
- Data compression (optional)
- Configurable retention policies

```python
persistence_config = PersistenceConfig(
    batch_size=1000,                    # Write in batches
    flush_interval_seconds=60,          # Background flush interval
    compression_enabled=True,           # Enable compression
    metrics_retention_days=30           # Auto-cleanup old data
)
```

### 5. Data Export/Import

Export all monitoring data for backup or analysis:

```python
# Export data
monitor.export_monitoring_data("./export_dir")

# Import data (future feature)
monitor.import_monitoring_data("./import_dir", merge=True)
```

### 6. Querying Historical Data

Access historical data with flexible queries:

```python
# Get metrics from last 24 hours
metrics = monitor.get_historical_metrics(
    metric_type="performance",
    hours_back=24.0
)

# Get recent checkpoints
checkpoints = persistence_manager.get_checkpoints(
    state=LifecycleState.ERROR,
    start_time=time.time() - 3600,
    limit=100
)

# Get mitigation history
attempts = persistence_manager.get_mitigation_history(
    strategy_name="reduce_batch_size",
    outcome=MitigationOutcome.SUCCESS
)
```

## Integration with Core Components

### Lifecycle Tracker
The lifecycle tracker automatically persists all state checkpoints:

```python
lifecycle_tracker = LifecycleTracker(
    intervention_engine=intervention_engine,
    persistence_manager=persistence_manager
)
```

### Predictive Failure Detector
Failure patterns and predictions are automatically saved:

```python
detector = PredictiveFailureDetector(
    persistence_manager=persistence_manager
)
```

### Continuous Learning System
Learning data and models are persisted for continuous improvement:

```python
learner = ContinuousLearningSystem(
    persistence_manager=persistence_manager
)
```

## Best Practices

1. **Regular Exports**: Schedule regular exports of monitoring data for backup
2. **Retention Policies**: Configure appropriate retention periods based on your needs
3. **Batch Sizes**: Adjust batch sizes based on your system's I/O capabilities
4. **Compression**: Enable compression for long-term storage efficiency
5. **Cleanup**: Run periodic cleanup to maintain database performance

## Example Usage

See [persistence_example.py](examples/persistence_example.py) for a complete example demonstrating all persistence features.

## Database Schema

The persistence layer uses the following main tables:
- `metrics`: Time-series metrics data
- `checkpoints`: Lifecycle state checkpoints
- `predictions`: Failure predictions
- `mitigation_attempts`: Mitigation execution history
- `happy_paths`: Successful execution patterns
- `learning_data`: Training data for ML models

## Configuration Options

```python
@dataclass
class PersistenceConfig:
    storage_dir: Path                    # Storage directory
    backend: StorageBackend              # Storage backend type
    
    # Retention policies
    metrics_retention_days: int = 30
    checkpoint_retention_days: int = 90
    prediction_retention_days: int = 60
    
    # Performance settings
    batch_size: int = 1000
    flush_interval_seconds: int = 60
    compression_enabled: bool = True
    
    # File paths
    db_file: str = "monitor.db"
    plugins_file: str = "plugins.json"
    guardrails_file: str = "guardrails.json"
    strategies_file: str = "strategies.json"
    happy_paths_file: str = "happy_paths.json"
    models_dir: str = "models"
```

## Future Enhancements

1. **Cloud Storage**: Support for S3, GCS, Azure Blob Storage
2. **Distributed Storage**: Support for distributed databases
3. **Real-time Sync**: Multi-instance synchronization
4. **Advanced Queries**: GraphQL or SQL query interface
5. **Data Visualization**: Built-in visualization tools