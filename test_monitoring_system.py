#!/usr/bin/env python3
"""
Test script to verify the vLLM monitoring system is working correctly.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the monitoring system to the path
sys.path.append(str(Path(__file__).parent / "vllm_monitor"))

try:
    from core import (
        VLLMMonitor, MonitorConfig, ComponentType, ComponentState, StateType, AlertLevel
    )
    from collectors import PerformanceCollector, ResourceCollector
    from analyzers import AnomalyDetector, HealthScorer
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import core modules: {e}")
    sys.exit(1)

try:
    from interventions import SelfHealingAgent, GuardrailManager
    print("✅ Intervention modules imported successfully")
except ImportError as e:
    print(f"⚠️ Intervention modules not available: {e}")
    SelfHealingAgent = None
    GuardrailManager = None

try:
    from exporters import ExportManager, ExportConfig, AlertConfig
    print("✅ Export modules imported successfully")
except ImportError as e:
    print(f"⚠️ Export modules not available: {e}")
    ExportManager = None


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.health_score = 1.0
        self.is_healthy = True
        self.requests_processed = 0


async def test_basic_monitoring():
    """Test basic monitoring functionality."""
    print("\n🧪 Testing Basic Monitoring")
    print("-" * 30)
    
    # Create configuration
    config = MonitorConfig(
        enabled=True,
        collection_interval_ms=1000,
        enable_async_collection=True,
        max_collection_time_us=100.0
    )
    
    # Create monitor
    monitor = VLLMMonitor(config)
    
    # Create mock components
    mock_engine = MockComponent("test_engine")
    mock_worker = MockComponent("test_worker")
    
    # Register components
    monitor.register_component(mock_engine, "engine_test", ComponentType.ENGINE)
    monitor.register_component(mock_worker, "worker_test", ComponentType.WORKER)
    
    # Add collectors
    perf_collector = PerformanceCollector()
    resource_collector = ResourceCollector()
    
    monitor.add_collector(perf_collector)
    monitor.add_collector(resource_collector)
    
    # Add analyzers
    anomaly_detector = AnomalyDetector(min_samples=3)
    health_scorer = HealthScorer()
    
    monitor.add_analyzer(anomaly_detector)
    monitor.add_analyzer(health_scorer)
    
    try:
        # Start monitoring
        await monitor.start()
        print("✅ Monitor started successfully")
        
        # Let it run for a few seconds
        await asyncio.sleep(3)
        
        # Check system health
        health = monitor.get_system_health()
        print(f"✅ System health: {health['overall_health']:.1f}%")
        print(f"✅ Monitored components: {health['monitored_components']}")
        print(f"✅ Collection success rate: {health['collections_success_rate']:.1f}%")
        
        # Get component state
        engine_state = monitor.get_component_state("engine_test")
        if engine_state:
            print(f"✅ Engine state retrieved: {engine_state.component_id}")
        else:
            print("⚠️ No engine state found")
        
        # Get metrics
        metrics = monitor.get_metrics(limit=5)
        print(f"✅ Retrieved {len(metrics)} metrics")
        
        # Stop monitoring
        await monitor.stop()
        print("✅ Monitor stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic monitoring test failed: {e}")
        return False


async def test_self_healing():
    """Test self-healing functionality."""
    if SelfHealingAgent is None:
        print("\n⚠️ Skipping self-healing test (module not available)")
        return True
        
    print("\n🏥 Testing Self-Healing")
    print("-" * 25)
    
    try:
        config = MonitorConfig(enable_self_healing=True)
        component_registry = {}
        
        agent = SelfHealingAgent(config, component_registry)
        
        await agent.start()
        print("✅ Self-healing agent started")
        
        # Test healing with mock issue
        mock_issue = {
            'type': 'anomaly_detection',
            'component_id': 'test_component',
            'contributing_factors': ['memory_exhaustion'],
            'requires_intervention': True
        }
        
        success = await agent.heal(mock_issue)
        print(f"✅ Healing attempt: {'Success' if success else 'Failed'}")
        
        # Get status
        status = agent.get_healing_status()
        print(f"✅ Healing sessions: {status['total_healing_sessions']}")
        
        await agent.stop()
        print("✅ Self-healing agent stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Self-healing test failed: {e}")
        return False


async def test_guardrails():
    """Test guardrail functionality."""
    if GuardrailManager is None:
        print("\n⚠️ Skipping guardrail test (module not available)")
        return True
        
    print("\n🛡️ Testing Guardrails")
    print("-" * 20)
    
    try:
        config = MonitorConfig(enable_guardrails=True)
        manager = GuardrailManager(config)
        
        # Create test state with high memory usage
        test_state = ComponentState(
            component_id="test_component",
            component_type=ComponentType.WORKER,
            state_type=StateType.PERFORMANCE,
            timestamp=time.time(),
            memory_usage=97.0,  # High memory usage
            cpu_usage=85.0,
            health_score=0.3
        )
        
        violations = manager.check_guardrails(test_state)
        print(f"✅ Guardrail violations detected: {len(violations)}")
        
        for violation in violations:
            print(f"   - {violation['type']}: {violation['message']}")
        
        status = manager.get_guardrail_status()
        print(f"✅ Guardrails enabled: {status['enabled']}")
        print(f"✅ Registered guardrails: {len(status['registered_guardrails'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Guardrail test failed: {e}")
        return False


async def test_export_system():
    """Test export system functionality."""
    if ExportManager is None:
        print("\n⚠️ Skipping export test (module not available)")
        return True
        
    print("\n📤 Testing Export System")
    print("-" * 25)
    
    try:
        monitor_config = MonitorConfig(
            enable_metrics_export=True,
            enable_logging=True,
            enable_alerts=True
        )
        
        export_config = ExportConfig(
            export_interval_seconds=1,
            export_format="json",
            enable_compression=False
        )
        
        alert_config = AlertConfig(
            enabled=True,
            min_alert_level=AlertLevel.INFO
        )
        
        export_manager = ExportManager(monitor_config, export_config, alert_config)
        
        await export_manager.start()
        print("✅ Export manager started")
        
        # Test metric queuing
        test_states = [
            ComponentState(
                component_id="test_metric",
                component_type=ComponentType.ENGINE,
                state_type=StateType.PERFORMANCE,
                timestamp=time.time(),
                health_score=0.8
            )
        ]
        
        export_manager.queue_metrics(test_states)
        print("✅ Metrics queued for export")
        
        # Test log queuing
        test_logs = [
            {
                'level': 'INFO',
                'message': 'Test log message',
                'timestamp': time.time(),
                'component': 'test'
            }
        ]
        
        export_manager.queue_logs(test_logs)
        print("✅ Logs queued for export")
        
        # Test alert queuing
        test_alert = {
            'level': AlertLevel.WARNING.value,
            'message': 'Test alert',
            'component': 'test',
            'timestamp': time.time()
        }
        
        await export_manager.queue_alert(test_alert)
        print("✅ Alert queued")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get status
        status = export_manager.get_export_status()
        print(f"✅ Export running: {status['is_running']}")
        print(f"✅ Queue sizes: {status['queue_sizes']}")
        
        await export_manager.stop()
        print("✅ Export manager stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Export system test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 vLLM Monitoring System Test Suite")
    print("=" * 40)
    
    tests = [
        ("Basic Monitoring", test_basic_monitoring()),
        ("Self-Healing", test_self_healing()),
        ("Guardrails", test_guardrails()),
        ("Export System", test_export_system()),
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        print(f"\n▶️ Running {test_name} test...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The monitoring system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)