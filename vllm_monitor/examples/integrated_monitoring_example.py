#!/usr/bin/env python3
"""
Example demonstrating the integrated vLLM monitoring system with 
lifecycle tracking and plugin system.
"""

import asyncio
import time
from vllm_monitor import (
    create_monitor_with_plugins,
    MonitorConfig,
    LifecycleState,
    StateTransition,
    ComponentType
)


async def main():
    # Create monitor with plugins and lifecycle tracking
    config = MonitorConfig(
        collection_interval_ms=100,
        enable_self_healing=True,
        enable_guardrails=True
    )
    
    monitor = create_monitor_with_plugins(config)
    
    # Start the monitor
    await monitor.start()
    
    # Track startup
    startup_checkpoint = monitor._lifecycle_tracker.track_startup(
        arguments={
            "model": "llama-2-7b",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9
        }
    )
    
    print(f"Startup tracked: {startup_checkpoint.state.name}")
    
    # Create a memory monitoring guardrail
    monitor.register_guardrail(
        name="high_memory_usage",
        description="Triggers when memory usage exceeds 80%",
        condition_code="return checkpoint.metrics.get('memory_percent', 0) > 80",
        intervention="reduce_batch_size",
        severity="warning"
    )
    
    # Create an intervention plugin
    monitor.register_plugin(
        name="reduce_batch_size",
        plugin_type="intervention",
        execute_code="""
# Simulate reducing batch size
print("Reducing batch size to free memory...")
# In real implementation, this would adjust vLLM's batch size
return True
""",
        description="Reduces batch size when memory is high"
    )
    
    # Create a custom collector plugin
    monitor.register_plugin(
        name="custom_metrics_collector",
        plugin_type="collector",
        execute_code="""
import time
import random

# Simulate collecting custom metrics
return [{
    'component_id': 'custom_component',
    'component_type': 'ENGINE',
    'state_type': 'PERFORMANCE',
    'timestamp': time.time(),
    'data': {
        'custom_metric': random.randint(0, 100),
        'memory_percent': random.randint(70, 90)  # Simulate memory usage
    },
    'is_healthy': True,
    'health_score': 0.95
}]
""",
        description="Collects custom performance metrics"
    )
    
    # Simulate lifecycle transitions
    print("\nSimulating lifecycle transitions...")
    
    # Transition to loading model
    monitor.track_lifecycle_state(
        LifecycleState.LOADING_MODEL,
        StateTransition.STARTUP,
        {"model_size_gb": 13.5}
    )
    
    await asyncio.sleep(1)
    
    # Transition to ready
    monitor.track_lifecycle_state(
        LifecycleState.READY,
        StateTransition.STARTUP,
        {"load_time_seconds": 45.2}
    )
    
    # Register a mock vLLM component
    class MockEngine:
        def __init__(self):
            self.name = "mock_engine"
    
    engine = MockEngine()
    monitor.register_component(engine, "main_engine", ComponentType.ENGINE)
    
    # Let the system run for a bit
    print("\nMonitoring system running...")
    await asyncio.sleep(5)
    
    # Get system health
    health = monitor.get_system_health()
    print(f"\nSystem Health Report:")
    print(f"  Overall Health: {health['overall_health']:.1f}%")
    print(f"  Components: {health['monitored_components']}")
    print(f"  Uptime: {health['uptime_seconds']:.1f}s")
    print(f"  Collection Success Rate: {health['collections_success_rate']:.1f}%")
    
    # Get lifecycle report
    lifecycle_report = monitor.get_lifecycle_report()
    print(f"\nLifecycle Report:")
    print(f"  Current State: {lifecycle_report['current_state']}")
    print(f"  Uptime: {lifecycle_report['uptime']:.1f}s")
    print(f"  Total Checkpoints: {lifecycle_report['total_checkpoints']}")
    print(f"  Active Guardrails: {lifecycle_report['active_guardrails']}")
    
    # List plugins
    plugins = monitor.list_plugins()
    print(f"\nRegistered Plugins ({len(plugins)}):")
    for plugin in plugins:
        print(f"  - {plugin['name']} ({plugin['type']}): {plugin['description']}")
    
    # Simulate shutdown
    print("\nShutting down...")
    shutdown_checkpoint = monitor._lifecycle_tracker.track_shutdown(
        reason="example_complete",
        cleanup_status={"cache_cleared": True, "logs_saved": True}
    )
    
    # Stop the monitor
    await monitor.stop()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())