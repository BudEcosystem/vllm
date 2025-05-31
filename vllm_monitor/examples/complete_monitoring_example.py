#!/usr/bin/env python3
"""
Complete vLLM Monitoring System Example

This example demonstrates how to set up and use the comprehensive vLLM monitoring
system with all features including:
- Real-time state tracking
- Performance monitoring
- Anomaly detection
- Failure prediction
- Self-healing interventions
- Alerting and export
"""

import asyncio
import time
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import vLLM monitoring system
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    VLLMMonitor, MonitorConfig, ComponentType, ComponentState, StateType, AlertLevel
)
from collectors import (
    StateCollector, PerformanceCollector, ErrorCollector, 
    RequestCollector, ResourceCollector
)
from analyzers import (
    AnomalyDetector, PerformanceAnalyzer, FailurePredictor, HealthScorer
)
from interventions import (
    InterventionEngine, SelfHealingAgent, GuardrailManager
)
from exporters import (
    ExportManager, ExportConfig, AlertConfig
)


class MockVLLMComponent:
    """Mock vLLM component for demonstration."""
    
    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
        self.is_healthy = True
        self.error_count = 0
        self.requests_processed = 0
        
    def process_request(self):
        """Simulate request processing."""
        self.requests_processed += 1
        # Simulate occasional errors
        if self.requests_processed % 50 == 0:
            self.error_count += 1
        
    def simulate_load(self):
        """Simulate varying load conditions."""
        import random
        # Randomly process requests
        for _ in range(random.randint(1, 10)):
            self.process_request()


async def main():
    """Main demonstration function."""
    print("🚀 Starting vLLM Monitoring System Demo")
    print("=" * 50)
    
    # 1. Configure monitoring system
    monitor_config = MonitorConfig(
        enabled=True,
        collection_interval_ms=1000,  # Collect every second for demo
        enable_async_collection=True,
        enable_anomaly_detection=True,
        enable_failure_prediction=True,
        enable_self_healing=True,
        enable_guardrails=True,
        enable_metrics_export=True,
        enable_logging=True,
        enable_alerts=True,
        max_collection_time_us=100.0,
        sampling_rate=1.0
    )
    
    # 2. Configure export system
    export_config = ExportConfig(
        export_interval_seconds=30,  # Export every 30 seconds for demo
        export_format="json",
        enable_compression=False,  # Disable for easier inspection
        retention_hours=1
    )
    
    alert_config = AlertConfig(
        enabled=True,
        min_alert_level=AlertLevel.WARNING,
        alert_cooldown_seconds=10.0,  # Short cooldown for demo
        max_alerts_per_hour=20
    )
    
    # 3. Create mock vLLM components
    print("📦 Creating mock vLLM components...")
    components = {
        "engine_main": MockVLLMComponent("engine_main", ComponentType.ENGINE),
        "scheduler_0": MockVLLMComponent("scheduler_0", ComponentType.SCHEDULER),
        "worker_0": MockVLLMComponent("worker_0", ComponentType.WORKER),
        "worker_1": MockVLLMComponent("worker_1", ComponentType.WORKER),
        "cache_engine": MockVLLMComponent("cache_engine", ComponentType.CACHE_ENGINE),
        "model_runner": MockVLLMComponent("model_runner", ComponentType.MODEL_RUNNER),
    }
    
    # 4. Initialize monitoring system
    print("🔧 Initializing monitoring system...")
    monitor = VLLMMonitor(monitor_config)
    
    # 5. Register components
    print("📋 Registering components...")
    for comp_id, component in components.items():
        monitor.register_component(component, comp_id, component.component_type)
    
    # 6. Setup collectors
    print("📊 Setting up data collectors...")
    
    # State collector
    state_collector = StateCollector(monitor._components)
    monitor.add_collector(state_collector)
    
    # Performance collector
    perf_collector = PerformanceCollector()
    monitor.add_collector(perf_collector)
    
    # Error collector
    error_collector = ErrorCollector()
    monitor.add_collector(error_collector)
    
    # Request collector
    request_collector = RequestCollector()
    monitor.add_collector(request_collector)
    
    # Resource collector
    resource_collector = ResourceCollector()
    monitor.add_collector(resource_collector)
    
    # 7. Setup analyzers
    print("🔍 Setting up analyzers...")
    
    # Anomaly detector
    anomaly_detector = AnomalyDetector(sensitivity=2.0, min_samples=5)
    monitor.add_analyzer(anomaly_detector)
    
    # Performance analyzer
    perf_analyzer = PerformanceAnalyzer(lookback_hours=0.1)  # 6 minutes for demo
    monitor.add_analyzer(perf_analyzer)
    
    # Failure predictor
    failure_predictor = FailurePredictor(prediction_horizon_hours=0.5)  # 30 minutes for demo
    monitor.add_analyzer(failure_predictor)
    
    # Health scorer
    health_scorer = HealthScorer()
    monitor.add_analyzer(health_scorer)
    
    # 8. Setup self-healing system
    print("🏥 Setting up self-healing system...")
    
    self_healing_agent = SelfHealingAgent(monitor_config, monitor._components)
    
    # Setup guardrails
    guardrail_manager = GuardrailManager(monitor_config)
    
    # 9. Setup export system
    print("📤 Setting up export system...")
    
    # Create export directories
    os.makedirs("./demo_metrics", exist_ok=True)
    os.makedirs("./demo_logs", exist_ok=True)
    
    export_manager = ExportManager(monitor_config, export_config, alert_config)
    
    # 10. Connect monitoring to self-healing
    async def handle_analysis_result(result):
        """Handle analysis results and trigger healing if needed."""
        if result.get('requires_intervention', False):
            print(f"🚨 Issue detected: {result.get('message', 'Unknown issue')}")
            success = await self_healing_agent.heal(result)
            if success:
                print("✅ Self-healing intervention successful")
            else:
                print("❌ Self-healing intervention failed")
        
        # Check guardrails
        if 'states' in result:
            for state in result.get('states', []):
                violations = guardrail_manager.check_guardrails(state)
                for violation in violations:
                    print(f"⚠️  Guardrail violation: {violation['message']}")
                    await export_manager.queue_alert(violation)
        
        # Queue alert if significant
        if result.get('alert_level', 0) >= AlertLevel.WARNING.value:
            await export_manager.queue_alert(result)
    
    # Override the analysis result handler
    original_handle_result = monitor._handle_analysis_result
    
    async def enhanced_handle_result(result):
        await original_handle_result(result)
        await handle_analysis_result(result)
    
    monitor._handle_analysis_result = enhanced_handle_result
    
    # 11. Start all systems
    print("🚀 Starting monitoring systems...")
    
    await monitor.start()
    await self_healing_agent.start()
    await export_manager.start()
    
    print("✅ All systems started successfully!")
    print("\n" + "=" * 50)
    print("📈 Monitoring Dashboard")
    print("=" * 50)
    
    # 12. Simulation loop
    try:
        demo_duration = 120  # Run for 2 minutes
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < demo_duration:
            iteration += 1
            current_time = time.time() - start_time
            
            # Simulate component activity
            for component in components.values():
                component.simulate_load()
            
            # Simulate request tracking
            request_collector.record_request_start(f"req_{iteration}", {
                "prompt_length": 100,
                "max_tokens": 50
            })
            
            # Complete some requests
            if iteration > 5:
                request_collector.record_request_completion(
                    f"req_{iteration-5}", 
                    tokens_generated=45, 
                    success=True
                )
            
            # Simulate occasional issues
            if iteration % 20 == 0:
                # Simulate memory pressure
                print(f"🔴 Simulating memory pressure at {current_time:.1f}s")
                components["cache_engine"].is_healthy = False
                components["cache_engine"].error_count += 5
            
            if iteration % 30 == 0:
                # Simulate recovery
                print(f"🟢 Simulating recovery at {current_time:.1f}s")
                for component in components.values():
                    component.is_healthy = True
            
            # Print status every 10 seconds
            if iteration % 10 == 0:
                health_status = monitor.get_system_health()
                print(f"\n⏱️  Time: {current_time:.1f}s")
                print(f"💚 System Health: {health_status['overall_health']:.1f}%")
                print(f"📊 Components: {health_status['healthy_components']}/{health_status['monitored_components']}")
                print(f"🔄 Collections: {health_status['collections_success_rate']:.1f}% success")
                print(f"⚡ Avg Collection Time: {health_status['average_collection_time_us']:.1f}μs")
                print(f"🚨 Alerts: {health_status['alerts_count']}")
                print(f"🏥 Interventions: {health_status['interventions_count']}")
                
                # Export status
                export_status = export_manager.get_export_status()
                print(f"📤 Export Queue: M:{export_status['queue_sizes']['metrics']} L:{export_status['queue_sizes']['logs']} A:{export_status['queue_sizes']['alerts']}")
            
            await asyncio.sleep(1)  # Check every second
        
        print(f"\n🎯 Demo completed after {demo_duration} seconds")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    # 13. Show final statistics
    print("\n" + "=" * 50)
    print("📊 Final Statistics")
    print("=" * 50)
    
    # System health
    final_health = monitor.get_system_health()
    print(f"Final System Health: {final_health['overall_health']:.1f}%")
    print(f"Total Collections: {final_health['collections_success_rate']:.1f}% success rate")
    print(f"Total Alerts: {final_health['alerts_count']}")
    print(f"Total Interventions: {final_health['interventions_count']}")
    
    # Self-healing statistics
    healing_status = self_healing_agent.get_healing_status()
    print(f"Healing Success Rate: {healing_status['healing_success_rate']:.1f}%")
    print(f"Total Healing Sessions: {healing_status['total_healing_sessions']}")
    
    # Export statistics
    export_status = export_manager.get_export_status()
    print(f"Metrics Export Success: {export_status['metrics_exporter']['success_rate_percent']:.1f}%")
    print(f"Alert Success Rate: {export_status['alert_manager']['success_rate_percent']:.1f}%")
    
    # Guardrail statistics
    guardrail_status = guardrail_manager.get_guardrail_status()
    print(f"Guardrail Violations: {guardrail_status['total_violations']}")
    
    # 14. Cleanup
    print("\n🧹 Cleaning up...")
    await monitor.stop()
    await self_healing_agent.stop()
    await export_manager.stop()
    
    print("✅ vLLM Monitoring System Demo completed successfully!")
    print("\n📁 Check the following directories for exported data:")
    print("   - ./demo_metrics/ - Exported metrics in JSON format")
    print("   - ./demo_logs/ - Structured logs")
    print("\n🔍 Review the console output above for real-time monitoring insights.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())