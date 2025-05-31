#!/usr/bin/env python3
"""
vLLM Monitoring System Integration Example

This example demonstrates how to integrate the comprehensive monitoring system
with a vLLM deployment for production use.
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

# Set up monitoring system path
import sys
sys.path.append(str(Path(__file__).parent / "vllm_monitor"))

# Import monitoring components
from vllm_monitor.core import (
    VLLMMonitor, MonitorConfig, ComponentType, ComponentState, 
    StateType, AlertLevel
)

# For this example, we'll simulate vLLM components
class MockVLLMEngine:
    """Mock vLLM Engine for demonstration."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_running = True
        self.requests_processed = 0
        self.error_count = 0
        self.scheduler = MockScheduler()
        self.workers = [MockWorker(i) for i in range(2)]
        self.cache_engine = MockCacheEngine()
        
    def generate(self, prompt: str):
        """Simulate text generation."""
        self.requests_processed += 1
        # Simulate processing time
        time.sleep(0.1)
        return f"Generated response for: {prompt[:20]}..."


class MockScheduler:
    """Mock Scheduler component."""
    
    def __init__(self):
        self.waiting = []
        self.running = []
        self.completed = []
        
    def get_num_unfinished_seq_groups(self):
        return len(self.waiting) + len(self.running)


class MockWorker:
    """Mock Worker component."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.rank = worker_id
        self.local_rank = worker_id
        self.model_loaded = True


class MockCacheEngine:
    """Mock Cache Engine component."""
    
    def __init__(self):
        self.total_blocks = 1000
        self.used_blocks = 0
        
    def get_num_free_gpu_blocks(self):
        return self.total_blocks - self.used_blocks


async def setup_production_monitoring(vllm_engine):
    """Set up comprehensive monitoring for production vLLM deployment."""
    
    print("🔧 Setting up production monitoring system...")
    
    # 1. Configure monitoring with production settings
    monitor_config = MonitorConfig(
        # Core settings - optimized for production
        enabled=True,
        collection_interval_ms=5000,  # 5 second intervals for production
        max_history_size=50000,       # Store more history for analysis
        enable_async_collection=True,
        
        # Performance settings - minimal overhead
        max_collection_time_us=100.0,    # Max 100μs collection time
        enable_adaptive_sampling=True,    # Auto-adjust under load
        performance_impact_threshold=0.0005,  # 0.05% max performance impact
        
        # Enable all monitoring features
        monitor_engine=True,
        monitor_scheduler=True,
        monitor_workers=True,
        monitor_cache=True,
        monitor_requests=True,
        monitor_performance=True,
        monitor_resources=True,
        monitor_errors=True,
        
        # Analysis settings
        enable_anomaly_detection=True,
        enable_failure_prediction=True,
        enable_performance_analysis=True,
        anomaly_threshold=2.5,  # Conservative threshold for production
        
        # Self-healing settings
        enable_self_healing=True,
        enable_guardrails=True,
        intervention_cooldown_s=300.0,  # 5 minute cooldown
        max_interventions_per_hour=10,
        
        # Export and alerting
        enable_metrics_export=True,
        enable_logging=True,
        enable_alerts=True,
        log_level=AlertLevel.WARNING,  # Only warn level and above in production
        
        # Data retention
        metrics_retention_hours=72,  # 3 days
        logs_retention_hours=168,    # 1 week
        enable_compression=True
    )
    
    # 2. Create monitor instance
    monitor = VLLMMonitor(monitor_config)
    
    # 3. Register vLLM components for monitoring
    print("📋 Registering vLLM components...")
    
    # Register main engine
    monitor.register_component(
        vllm_engine, 
        "vllm_engine_main", 
        ComponentType.ENGINE
    )
    
    # Register scheduler
    monitor.register_component(
        vllm_engine.scheduler,
        "scheduler_main",
        ComponentType.SCHEDULER
    )
    
    # Register workers
    for i, worker in enumerate(vllm_engine.workers):
        monitor.register_component(
            worker,
            f"worker_{i}",
            ComponentType.WORKER
        )
    
    # Register cache engine
    monitor.register_component(
        vllm_engine.cache_engine,
        "cache_engine_main",
        ComponentType.CACHE_ENGINE
    )
    
    # 4. Add data collectors (with error handling for missing dependencies)
    print("📊 Setting up data collectors...")
    
    try:
        from vllm_monitor.collectors import StateCollector
        state_collector = StateCollector(monitor._components)
        monitor.add_collector(state_collector)
        print("  ✅ State collector added")
    except ImportError:
        print("  ⚠️ State collector not available")
    
    try:
        from vllm_monitor.collectors import PerformanceCollector
        perf_collector = PerformanceCollector()
        monitor.add_collector(perf_collector)
        print("  ✅ Performance collector added")
    except ImportError:
        print("  ⚠️ Performance collector not available")
    
    try:
        from vllm_monitor.collectors import ErrorCollector
        error_collector = ErrorCollector()
        monitor.add_collector(error_collector)
        print("  ✅ Error collector added")
    except ImportError:
        print("  ⚠️ Error collector not available")
    
    # 5. Add analyzers
    print("🔍 Setting up analyzers...")
    
    try:
        from vllm_monitor.analyzers import AnomalyDetector
        anomaly_detector = AnomalyDetector(
            sensitivity=2.5,    # Conservative for production
            min_samples=20      # Need more samples for reliable detection
        )
        monitor.add_analyzer(anomaly_detector)
        print("  ✅ Anomaly detector added")
    except ImportError:
        print("  ⚠️ Anomaly detector not available")
    
    try:
        from vllm_monitor.analyzers import FailurePredictor
        failure_predictor = FailurePredictor(
            prediction_horizon_hours=6.0  # Predict 6 hours ahead
        )
        monitor.add_analyzer(failure_predictor)
        print("  ✅ Failure predictor added")
    except ImportError:
        print("  ⚠️ Failure predictor not available")
    
    try:
        from vllm_monitor.analyzers import HealthScorer
        health_scorer = HealthScorer()
        monitor.add_analyzer(health_scorer)
        print("  ✅ Health scorer added")
    except ImportError:
        print("  ⚠️ Health scorer not available")
    
    # 6. Setup self-healing (if available)
    self_healing_agent = None
    try:
        from vllm_monitor.interventions import SelfHealingAgent
        self_healing_agent = SelfHealingAgent(
            config=monitor_config,
            component_registry=monitor._components
        )
        print("  ✅ Self-healing agent configured")
    except ImportError:
        print("  ⚠️ Self-healing agent not available")
    
    # 7. Setup guardrails (if available)
    guardrail_manager = None
    try:
        from vllm_monitor.interventions import GuardrailManager
        guardrail_manager = GuardrailManager(monitor_config)
        print("  ✅ Guardrail manager configured")
    except ImportError:
        print("  ⚠️ Guardrail manager not available")
    
    # 8. Setup export system (if available)
    export_manager = None
    try:
        from vllm_monitor.exporters import (
            ExportManager, ExportConfig, AlertConfig
        )
        
        # Configure export system
        export_config = ExportConfig(
            export_interval_seconds=300,  # Export every 5 minutes
            export_format="json",
            enable_compression=True,
            retention_hours=72
        )
        
        alert_config = AlertConfig(
            enabled=True,
            min_alert_level=AlertLevel.ERROR,  # Only errors and critical
            alert_cooldown_seconds=600.0,      # 10 minute cooldown
            max_alerts_per_hour=20,
            # Configure your webhook/email/slack here:
            # webhook_url="https://your-monitoring.com/webhook",
            # email_recipients=["admin@yourcompany.com"],
            # slack_webhook="https://hooks.slack.com/your-webhook"
        )
        
        export_manager = ExportManager(
            monitor_config=monitor_config,
            export_config=export_config,
            alert_config=alert_config
        )
        print("  ✅ Export manager configured")
        
    except ImportError:
        print("  ⚠️ Export manager not available")
    
    return monitor, self_healing_agent, guardrail_manager, export_manager


async def production_monitoring_loop(monitor, vllm_engine, 
                                   self_healing_agent=None, 
                                   guardrail_manager=None,
                                   export_manager=None):
    """Main monitoring loop for production deployment."""
    
    print("🚀 Starting production monitoring loop...")
    
    # Start all systems
    systems_started = []
    
    try:
        await monitor.start()
        systems_started.append(("Monitor", monitor))
        print("✅ Monitor started")
    except Exception as e:
        print(f"❌ Failed to start monitor: {e}")
        return
    
    if self_healing_agent:
        try:
            await self_healing_agent.start()
            systems_started.append(("Self-Healing Agent", self_healing_agent))
            print("✅ Self-healing agent started")
        except Exception as e:
            print(f"⚠️ Failed to start self-healing agent: {e}")
    
    if export_manager:
        try:
            await export_manager.start()
            systems_started.append(("Export Manager", export_manager))
            print("✅ Export manager started")
        except Exception as e:
            print(f"⚠️ Failed to start export manager: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 vLLM Production Monitoring Dashboard")
    print("=" * 60)
    
    # Monitoring loop
    try:
        runtime_minutes = 2  # Run for 2 minutes in demo
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < runtime_minutes * 60:
            iteration += 1
            current_time = time.time() - start_time
            
            # Simulate vLLM processing requests
            if iteration % 3 == 0:  # Every 3rd iteration
                prompt = f"Tell me about AI technology #{iteration}"
                response = vllm_engine.generate(prompt)
                
            # Simulate occasional load spikes
            if iteration % 20 == 0:
                print(f"📈 Simulating load spike at {current_time:.1f}s")
                vllm_engine.cache_engine.used_blocks += 100
            
            # Simulate occasional issues
            if iteration % 30 == 0:
                print(f"⚠️ Simulating error condition at {current_time:.1f}s")
                vllm_engine.error_count += 5
            
            # Check guardrails if available
            if guardrail_manager and iteration % 5 == 0:
                # Create a test state for guardrail checking
                test_state = ComponentState(
                    component_id="vllm_engine_main",
                    component_type=ComponentType.ENGINE,
                    state_type=StateType.OPERATIONAL,
                    timestamp=time.time(),
                    memory_usage=min(95.0, 60.0 + iteration * 0.5),  # Gradually increase
                    health_score=max(0.1, 1.0 - iteration * 0.01)   # Gradually decrease
                )
                
                violations = guardrail_manager.check_guardrails(test_state)
                if violations:
                    for violation in violations:
                        print(f"🚨 Guardrail violation: {violation['message']}")
                        
                        # Queue alert if export manager available
                        if export_manager:
                            await export_manager.queue_alert(violation)
            
            # Display status every 10 seconds
            if iteration % 10 == 0:
                health_status = monitor.get_system_health()
                
                print(f"\n⏱️  Runtime: {current_time:.1f}s | Iteration: {iteration}")
                print(f"💚 System Health: {health_status['overall_health']:.1f}%")
                print(f"📊 Components: {health_status['healthy_components']}/{health_status['monitored_components']} healthy")
                print(f"🔄 Collections: {health_status['collections_success_rate']:.1f}% success")
                print(f"⚡ Avg Collection Time: {health_status['average_collection_time_us']:.1f}μs")
                print(f"🎯 Performance Impact: {health_status.get('performance_impact', 0):.4f}%")
                print(f"🚨 Total Alerts: {health_status['alerts_count']}")
                print(f"🏥 Interventions: {health_status['interventions_count']}")
                print(f"📈 Requests Processed: {vllm_engine.requests_processed}")
                print(f"❌ Error Count: {vllm_engine.error_count}")
                
                # Self-healing status
                if self_healing_agent:
                    healing_status = self_healing_agent.get_healing_status()
                    print(f"🔧 Healing Success Rate: {healing_status['healing_success_rate']:.1f}%")
                
                # Export status
                if export_manager:
                    export_status = export_manager.get_export_status()
                    queues = export_status['queue_sizes']
                    print(f"📤 Export Queues: M:{queues['metrics']} L:{queues['logs']} A:{queues['alerts']}")
            
            await asyncio.sleep(1)  # Check every second
    
    except KeyboardInterrupt:
        print("\n🛑 Monitoring interrupted by user")
    
    # Shutdown sequence
    print("\n🧹 Shutting down monitoring systems...")
    
    for system_name, system in reversed(systems_started):
        try:
            if hasattr(system, 'stop'):
                await system.stop()
                print(f"✅ {system_name} stopped")
        except Exception as e:
            print(f"⚠️ Error stopping {system_name}: {e}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 Final Production Monitoring Statistics")
    print("=" * 60)
    
    final_health = monitor.get_system_health()
    print(f"Final System Health: {final_health['overall_health']:.1f}%")
    print(f"Total Collections: {final_health['total_collections']} ({final_health['collections_success_rate']:.1f}% success)")
    print(f"Average Collection Time: {final_health['average_collection_time_us']:.1f}μs")
    print(f"Total Alerts Generated: {final_health['alerts_count']}")
    print(f"Total Interventions: {final_health['interventions_count']}")
    print(f"Total Requests Processed: {vllm_engine.requests_processed}")
    
    if self_healing_agent:
        healing_status = self_healing_agent.get_healing_status()
        print(f"Self-Healing Success Rate: {healing_status['healing_success_rate']:.1f}%")
        print(f"Total Healing Sessions: {healing_status['total_healing_sessions']}")
    
    if export_manager:
        export_status = export_manager.get_export_status()
        print(f"Metrics Export Success: {export_status['metrics_exporter']['success_rate_percent']:.1f}%")
        print(f"Alert Success Rate: {export_status['alert_manager']['success_rate_percent']:.1f}%")
    
    print("\n✅ Production monitoring demonstration completed successfully!")


async def main():
    """Main function demonstrating production vLLM monitoring setup."""
    
    print("🏭 vLLM Production Monitoring Integration Demo")
    print("=" * 50)
    
    # 1. Create mock vLLM engine (replace with your actual vLLM setup)
    print("🤖 Initializing vLLM engine...")
    vllm_engine = MockVLLMEngine("meta-llama/Llama-2-7b-chat-hf")
    print(f"✅ vLLM engine initialized with model: {vllm_engine.model_name}")
    
    # 2. Setup comprehensive monitoring
    monitor, self_healing_agent, guardrail_manager, export_manager = await setup_production_monitoring(vllm_engine)
    
    # 3. Run production monitoring loop
    await production_monitoring_loop(
        monitor, 
        vllm_engine,
        self_healing_agent, 
        guardrail_manager,
        export_manager
    )
    
    print("\n📋 Integration Notes:")
    print("=" * 20)
    print("1. Replace MockVLLMEngine with your actual vLLM instance")
    print("2. Configure webhook URLs, email recipients, and Slack webhooks in AlertConfig")
    print("3. Adjust collection intervals and thresholds based on your requirements")
    print("4. Set up proper log rotation and metric storage for production")
    print("5. Consider adding custom collectors for your specific metrics")
    print("6. Implement custom interventions for your deployment-specific issues")
    print("7. Test the monitoring system thoroughly before production deployment")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)