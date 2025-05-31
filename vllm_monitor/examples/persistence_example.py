#!/usr/bin/env python3
"""
Example demonstrating the persistence capabilities of the vLLM monitoring system.

This example shows how to:
- Enable persistence for all monitoring components
- Save and load metrics, checkpoints, predictions, and strategies
- Persist dynamically added plugins and guardrails
- Export and import monitoring data
"""

import time
import asyncio
from pathlib import Path

# Import monitoring components
from vllm_monitor.core import VLLMMonitor, MonitorConfig, ComponentType
from vllm_monitor.lifecycle_tracker import LifecycleState, StateTransition
from vllm_monitor.persistence import PersistenceConfig, PersistenceManager
from vllm_monitor.continuous_learning import (
    MitigationAttempt, MitigationOutcome, MitigationStrategy
)
from vllm_monitor.predictive_failure_detection import FailurePrediction, PredictionConfidence


async def main():
    """Main example function."""
    print("=== vLLM Monitoring System - Persistence Example ===\n")
    
    # Configure persistence
    persistence_config = PersistenceConfig(
        storage_dir=Path("./monitor_data"),
        metrics_retention_days=30,
        checkpoint_retention_days=90,
        batch_size=10,  # Small batch for demo
        flush_interval_seconds=5
    )
    
    # Create monitor with persistence enabled
    monitor = VLLMMonitor(
        config=MonitorConfig(
            collection_interval_ms=100,
            enable_ml_analysis=True
        ),
        enable_persistence=True,
        persistence_config=persistence_config
    )
    
    # Setup components
    lifecycle_tracker = monitor.setup_lifecycle_tracking()
    plugin_manager = monitor.setup_plugin_system()
    
    # Start monitoring
    await monitor.start()
    
    print("1. Demonstrating metric persistence...")
    # Simulate some metrics
    for i in range(5):
        monitor.persistence_manager.save_metric(
            metric_type="performance",
            metric_name="throughput",
            value=100 + i * 10,
            tags={"model": "llama-7b", "gpu": "0"},
            metadata={"batch_size": 32}
        )
        await asyncio.sleep(0.1)
    
    print("   - Saved 5 performance metrics")
    
    print("\n2. Demonstrating checkpoint persistence...")
    # Create lifecycle checkpoints
    checkpoint1 = lifecycle_tracker.track_startup(
        arguments={"model": "llama-7b", "gpu_memory_utilization": 0.9},
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        hardware_info={"gpu_count": 1, "gpu_memory": 24576}
    )
    
    checkpoint2 = lifecycle_tracker.transition_to_state(
        LifecycleState.READY,
        StateTransition.STARTUP,
        {"initialization_time": 15.2}
    )
    
    print("   - Saved 2 lifecycle checkpoints")
    
    print("\n3. Demonstrating dynamic plugin persistence...")
    # Register a custom plugin
    monitor.register_plugin(
        name="custom_gpu_monitor",
        plugin_type="collector",
        execute_code="""
def execute(context):
    import random
    gpu_usage = random.uniform(0.5, 0.9)
    return [{'metric': 'gpu_usage', 'value': gpu_usage}]
""",
        description="Monitor GPU usage",
        interval_seconds=5
    )
    
    # Persist plugins
    monitor.persist_plugins()
    print("   - Saved custom plugin")
    
    print("\n4. Demonstrating guardrail persistence...")
    # Register a custom guardrail
    monitor.register_guardrail(
        name="memory_overflow_guard",
        description="Prevent memory overflow",
        condition_code="checkpoint.metrics.get('memory_percent', 0) > 90",
        intervention="reduce_batch_size",
        severity="critical"
    )
    
    # Persist guardrails
    monitor.persist_guardrails()
    print("   - Saved custom guardrail")
    
    print("\n5. Demonstrating prediction persistence...")
    # Create a mock failure prediction
    if hasattr(monitor, '_predictive_detector'):
        prediction = FailurePrediction(
            prediction_id="pred_001",
            timestamp=time.time(),
            current_state=LifecycleState.SERVING,
            predicted_failure_state=LifecycleState.ERROR,
            failure_type="OOM",
            time_to_failure=300,
            probability=0.75,
            confidence=PredictionConfidence.HIGH,
            contributing_factors=["High memory usage", "Large batch size"],
            recommended_mitigations=[]
        )
        monitor.persistence_manager.save_prediction(prediction)
        print("   - Saved failure prediction")
    
    print("\n6. Demonstrating mitigation attempt persistence...")
    # Create a mock mitigation attempt
    attempt = MitigationAttempt(
        attempt_id="attempt_001",
        timestamp=time.time(),
        initial_state=LifecycleState.SERVING,
        initial_metrics={"memory_percent": 92, "gpu_usage": 0.85},
        error_context=["High memory usage detected"],
        strategy_name="reduce_batch_size",
        interventions_executed=["batch_size_reduction"],
        execution_time=2.5,
        outcome=MitigationOutcome.SUCCESS,
        final_state=LifecycleState.SERVING,
        final_metrics={"memory_percent": 78, "gpu_usage": 0.82},
        side_effects_observed=[]
    )
    monitor.persistence_manager.save_mitigation_attempt(attempt)
    print("   - Saved mitigation attempt")
    
    print("\n7. Demonstrating happy path persistence...")
    # Save current execution as a happy path
    monitor.save_happy_path(
        path_id="startup_path_001",
        name="Standard Startup Sequence"
    )
    print("   - Saved happy path")
    
    # Flush all pending writes
    monitor.persistence_manager.flush_all()
    
    await asyncio.sleep(1)
    
    print("\n8. Loading persisted data...")
    # Load metrics
    metrics = monitor.persistence_manager.get_metrics(
        metric_type="performance",
        limit=10
    )
    print(f"   - Loaded {len(metrics)} metrics")
    
    # Load checkpoints
    checkpoints = monitor.persistence_manager.get_checkpoints(limit=10)
    print(f"   - Loaded {len(checkpoints)} checkpoints")
    
    # Load predictions
    predictions = monitor.persistence_manager.get_predictions(limit=10)
    print(f"   - Loaded {len(predictions)} predictions")
    
    # Load mitigation history
    mitigations = monitor.persistence_manager.get_mitigation_history(limit=10)
    print(f"   - Loaded {len(mitigations)} mitigation attempts")
    
    # Load happy paths
    happy_paths = monitor.persistence_manager.get_happy_paths()
    print(f"   - Loaded {len(happy_paths)} happy paths")
    
    print("\n9. Checking persistence statistics...")
    stats = monitor.get_persistence_stats()
    print(f"   - Database size: {stats.get('db_size_mb', 0):.2f} MB")
    print(f"   - Total metrics: {stats.get('metrics_count', 0)}")
    print(f"   - Total checkpoints: {stats.get('checkpoints_count', 0)}")
    print(f"   - Total predictions: {stats.get('predictions_count', 0)}")
    
    print("\n10. Exporting monitoring data...")
    export_dir = "./monitor_export"
    monitor.export_monitoring_data(export_dir)
    print(f"   - Exported all data to {export_dir}")
    
    # Stop monitoring
    await monitor.stop()
    
    print("\n11. Creating new monitor instance to test loading...")
    # Create a new monitor instance that will load persisted data
    monitor2 = VLLMMonitor(
        enable_persistence=True,
        persistence_config=persistence_config
    )
    
    # The monitor should automatically load persisted plugins and guardrails
    loaded_plugins = monitor2.list_plugins()
    print(f"   - Loaded {len(loaded_plugins)} plugins from persistence")
    
    if loaded_plugins:
        print("   - Plugins loaded:")
        for plugin in loaded_plugins:
            print(f"     * {plugin['name']}: {plugin['description']}")
    
    print("\n=== Persistence Example Complete ===")
    
    # Cleanup retention policy
    print("\n12. Testing data retention cleanup...")
    monitor2.persistence_manager.cleanup_old_data()
    print("   - Cleaned up old data based on retention policies")


if __name__ == "__main__":
    asyncio.run(main())