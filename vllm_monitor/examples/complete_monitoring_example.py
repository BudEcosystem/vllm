"""
Complete example demonstrating all vLLM monitoring features.

This example shows:
1. Lifecycle tracking with startup/shutdown monitoring
2. Plugin system with custom plugins
3. Distributed metrics collection
4. Guardrails and interventions
5. Hardware compatibility checking
6. Runtime hot-reloading
"""

import time
import json
import random
from pathlib import Path

# Import monitoring components
from vllm_monitor import (
    VLLMMonitor,
    create_monitor_with_plugins,
    LifecycleState,
    StateTransition,
    GuardrailPolicy,
    PluginManager,
    PluginType,
    DistributedMetricsCollector,
    WorkerRole,
    ParallelismType
)


def main():
    print("=== vLLM Comprehensive Monitoring Example ===\n")
    
    # 1. Create monitor with plugins and lifecycle tracking
    print("1. Initializing monitoring system...")
    monitor = create_monitor_with_plugins()
    
    # Add verbose feedback handler
    def feedback_handler(level: str, message: str, context: dict):
        print(f"[{level.upper()}] {message}")
        if context:
            print(f"  Context: {json.dumps(context, indent=2)}")
    
    monitor.lifecycle_tracker.add_feedback_handler(feedback_handler)
    monitor.plugin_manager.add_feedback_handler(feedback_handler)
    
    # 2. Track vLLM startup
    print("\n2. Tracking vLLM startup...")
    startup_args = {
        "model": "meta-llama/Llama-2-7b-hf",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
        "device": "cuda"
    }
    
    checkpoint = monitor.track_lifecycle_state(
        LifecycleState.INITIALIZING,
        StateTransition.STARTUP,
        startup_args
    )
    print(f"Startup checkpoint created: {checkpoint.timestamp}")
    
    # 3. Create and register custom plugins
    print("\n3. Creating custom plugins...")
    
    # Create a custom collector plugin
    monitor.plugin_manager.create_plugin(
        name="custom_latency_collector",
        plugin_type="collector",
        execute_code="""
import random
latency = random.uniform(10, 100)
return {
    'latency_ms': latency,
    'timestamp': time.time(),
    'status': 'healthy' if latency < 50 else 'degraded'
}
""",
        description="Collects latency metrics"
    )
    
    # Create a guardrail for high latency
    monitor.plugin_manager.create_guardrail(
        name="high_latency_detector",
        description="Detects high latency and triggers optimization",
        condition_code="""
# Check if latency exceeds threshold
latency_data = checkpoint.metrics.get('custom_latency', {})
return latency_data.get('latency_ms', 0) > 50
""",
        intervention="optimize_performance",
        severity="warning"
    )
    
    # Create an intervention
    monitor.plugin_manager.create_intervention(
        name="optimize_performance",
        description="Optimize performance when high latency detected",
        action_code="""
print("Optimizing performance...")
# In real scenario, this would adjust batch sizes, enable optimizations, etc.
context['optimized'] = True
""",
        required_context=[]
    )
    
    # 4. Set up distributed metrics collection
    print("\n4. Setting up distributed metrics...")
    dist_collector = DistributedMetricsCollector(
        node_id="main-node",
        role=WorkerRole.MASTER
    )
    
    # Register distributed workers
    dist_collector.register_worker("worker-0", WorkerRole.PREFILL, "main-node", 12345, gpu_id=0)
    dist_collector.register_worker("worker-1", WorkerRole.DECODE, "main-node", 12346, gpu_id=1)
    
    # Register distributed collector as a plugin
    dist_plugin = dist_collector.create_plugin()
    monitor.register_plugin(dist_plugin)
    
    # Start distributed collection
    dist_collector.start_collection(interval=2.0)
    
    # 5. Demonstrate lifecycle state transitions
    print("\n5. Demonstrating lifecycle transitions...")
    
    # Transition through states
    states = [
        (LifecycleState.LOADING_MODEL, StateTransition.STARTUP),
        (LifecycleState.ALLOCATING_MEMORY, StateTransition.STARTUP),
        (LifecycleState.READY, StateTransition.STARTUP),
        (LifecycleState.SERVING, StateTransition.STARTUP)
    ]
    
    for state, transition in states:
        time.sleep(0.5)
        checkpoint = monitor.track_lifecycle_state(state, transition)
        print(f"Transitioned to: {state.name}")
    
    # 6. Simulate monitoring with guardrails
    print("\n6. Running monitoring simulation...")
    
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")
        
        # Update distributed metrics
        dist_collector.update_worker_metrics("worker-0", {
            "throughput": random.uniform(100, 200),
            "queue_size": random.randint(0, 20),
            "active_requests": random.randint(1, 10),
            "gpu_utilization": random.uniform(60, 95)
        })
        
        dist_collector.update_worker_metrics("worker-1", {
            "throughput": random.uniform(80, 180),
            "queue_size": random.randint(0, 15),
            "active_requests": random.randint(1, 8),
            "gpu_utilization": random.uniform(50, 90)
        })
        
        # Update parallelism metrics
        dist_collector.update_parallelism_metrics(
            ParallelismType.TENSOR_PARALLEL,
            {
                "world_size": 2,
                "sync_time_ms": random.uniform(1, 5),
                "compute_time_ms": random.uniform(40, 60),
                "worker_times": {
                    "worker-0": random.uniform(40, 60),
                    "worker-1": random.uniform(42, 58)
                }
            }
        )
        
        # Execute custom latency collector
        success, latency_data = monitor.plugin_manager.execute_plugin("custom_latency_collector")
        if success:
            print(f"Latency: {latency_data['latency_ms']:.2f}ms - Status: {latency_data['status']}")
            
            # Update monitor metrics
            monitor.collect_state({})  # This will trigger guardrails
        
        # Get monitoring state
        health = monitor.get_system_health()
        print(f"System Health Score: {health['overall_score']:.2f}")
        
        time.sleep(1)
    
    # 7. Demonstrate plugin hot-reload capability
    print("\n7. Demonstrating plugin hot-reload...")
    
    # Create a plugin file
    plugin_dir = Path("./test_plugins")
    plugin_dir.mkdir(exist_ok=True)
    
    plugin_file = plugin_dir / "dynamic_plugin.py"
    plugin_code = '''
from vllm_monitor.plugin_system import PluginInterface, PluginMetadata, PluginType

class DynamicAnalyzer(PluginInterface):
    def get_metadata(self):
        return PluginMetadata(
            name="dynamic_analyzer",
            version="1.0.0",
            type=PluginType.ANALYZER,
            description="Dynamically loaded analyzer"
        )
    
    def initialize(self, context):
        self.context = context
        return True
    
    def execute(self, data):
        return {
            "analysis": "Dynamic analysis complete",
            "data_points": len(data) if isinstance(data, (list, dict)) else 1
        }
    
    def cleanup(self):
        pass
'''
    
    plugin_file.write_text(plugin_code)
    monitor.plugin_manager.loader.add_plugin_directory(plugin_dir)
    
    # Load the plugin
    if monitor.plugin_manager.loader.load_plugin_from_file(plugin_file):
        print("Dynamic plugin loaded successfully!")
        
        # Execute the dynamic plugin
        success, result = monitor.plugin_manager.execute_plugin("dynamic_analyzer", {"test": "data"})
        if success:
            print(f"Dynamic analyzer result: {result}")
    
    # 8. Get comprehensive reports
    print("\n8. Generating comprehensive reports...")
    
    # Lifecycle report
    lifecycle_report = monitor.get_lifecycle_report()
    print("\n--- Lifecycle Report ---")
    print(f"Current State: {lifecycle_report['current_state']}")
    print(f"Uptime: {lifecycle_report['uptime']:.2f} seconds")
    print(f"Total Checkpoints: {lifecycle_report['total_checkpoints']}")
    print(f"Active Guardrails: {lifecycle_report['active_guardrails']}")
    
    # Distributed metrics report
    dist_summary = dist_collector.get_distributed_summary()
    print("\n--- Distributed Metrics Report ---")
    print(f"Active Workers: {dist_summary['workers']['active']}/{dist_summary['workers']['total']}")
    print(f"Average GPU Utilization: {dist_summary['resources']['avg_gpu_utilization']:.1f}%")
    print(f"Total GPU Memory Used: {dist_summary['resources']['total_gpu_memory_used_gb']:.2f} GB")
    
    # Plugin report
    plugins = monitor.list_plugins()
    print("\n--- Registered Plugins ---")
    for plugin in plugins:
        print(f"- {plugin['name']} ({plugin['type']}): {plugin['status']}")
    
    # 9. Test error handling and recovery
    print("\n9. Testing error handling...")
    
    # Create a failing plugin
    monitor.plugin_manager.create_plugin(
        name="failing_plugin",
        plugin_type="component",
        execute_code="raise Exception('Simulated error')",
        description="Plugin that fails"
    )
    
    success, error = monitor.plugin_manager.execute_plugin("failing_plugin")
    print(f"Failed plugin result: Success={success}, Error={error}")
    
    # 10. Shutdown monitoring
    print("\n10. Shutting down monitoring system...")
    
    # Stop distributed collection
    dist_collector.stop_collection()
    
    # Track shutdown
    shutdown_checkpoint = monitor.lifecycle_tracker.track_shutdown(
        reason="example_complete",
        cleanup_status={
            "collectors_stopped": True,
            "plugins_unloaded": True,
            "resources_freed": True
        }
    )
    
    print(f"\nShutdown complete. Total runtime: {shutdown_checkpoint.metadata['uptime_seconds']:.2f} seconds")
    
    # Final statistics
    print("\n=== Final Statistics ===")
    final_report = monitor.get_monitoring_report()
    print(f"Total events collected: {sum(len(v) for v in final_report['history'].values())}")
    print(f"Plugin executions: {len(plugins)} plugins registered")
    print(f"Lifecycle transitions: {lifecycle_report['total_checkpoints']}")
    
    # Cleanup
    if plugin_dir.exists():
        import shutil
        shutil.rmtree(plugin_dir)


if __name__ == "__main__":
    main()