#!/usr/bin/env python3
"""
Simple test script to verify the vLLM monitoring system core functionality.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the monitoring system to the path
sys.path.append(str(Path(__file__).parent / "vllm_monitor"))

def test_imports():
    """Test that core modules can be imported."""
    print("🧪 Testing Module Imports")
    print("-" * 25)
    
    try:
        from core import (
            VLLMMonitor, MonitorConfig, ComponentType, ComponentState, 
            StateType, AlertLevel, CircularBuffer, PerformanceTimer
        )
        print("✅ Core module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core module: {e}")
        return False
    
    try:
        from collectors import BaseCollector, StateCollector
        print("✅ Collectors module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import collectors module: {e}")
        return False
    
    try:
        from analyzers import (
            AnomalyDetector, PerformanceAnalyzer, 
            FailurePredictor, HealthScorer
        )
        print("✅ Analyzers module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import analyzers module: {e}")
        return False
    
    try:
        from interventions import (
            InterventionEngine, SelfHealingAgent, GuardrailManager
        )
        print("✅ Interventions module imported successfully")
    except ImportError as e:
        print(f"⚠️ Interventions module import failed: {e}")
    
    try:
        from exporters import (
            ExportManager, ExportConfig, AlertConfig
        )
        print("✅ Exporters module imported successfully")
    except ImportError as e:
        print(f"⚠️ Exporters module import failed: {e}")
    
    return True


def test_core_functionality():
    """Test core data structures and utilities."""
    print("\n🔧 Testing Core Functionality")
    print("-" * 28)
    
    try:
        from core import (
            MonitorConfig, ComponentState, ComponentType, StateType,
            CircularBuffer, PerformanceTimer
        )
        
        # Test MonitorConfig
        config = MonitorConfig(
            enabled=True,
            collection_interval_ms=1000,
            enable_async_collection=True
        )
        print(f"✅ MonitorConfig created: enabled={config.enabled}")
        
        # Test ComponentState
        state = ComponentState(
            component_id="test_component",
            component_type=ComponentType.ENGINE,
            state_type=StateType.OPERATIONAL,
            timestamp=time.time(),
            health_score=0.9,
            is_healthy=True
        )
        print(f"✅ ComponentState created: {state.component_id}")
        
        # Test CircularBuffer
        buffer = CircularBuffer(maxsize=10)
        for i in range(15):
            buffer.append(f"item_{i}")
        
        latest = buffer.get_latest(5)
        print(f"✅ CircularBuffer working: {len(latest)} latest items")
        
        # Test PerformanceTimer
        with PerformanceTimer() as timer:
            time.sleep(0.001)  # 1ms
        
        print(f"✅ PerformanceTimer working: {timer.elapsed_us:.1f}μs")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False


async def test_monitor_basic():
    """Test basic monitor functionality without external dependencies."""
    print("\n🎯 Testing Basic Monitor")
    print("-" * 23)
    
    try:
        from core import VLLMMonitor, MonitorConfig, ComponentType
        
        # Create config
        config = MonitorConfig(
            enabled=True,
            collection_interval_ms=2000,
            enable_async_collection=False,  # Disable async for simplicity
            max_collection_time_us=1000.0
        )
        
        # Create monitor
        monitor = VLLMMonitor(config)
        print("✅ Monitor created successfully")
        
        # Create mock component
        class MockComponent:
            def __init__(self, name):
                self.name = name
                self.is_healthy = True
        
        mock_component = MockComponent("test_engine")
        
        # Register component
        monitor.register_component(
            mock_component, 
            "test_engine", 
            ComponentType.ENGINE
        )
        print("✅ Component registered successfully")
        
        # Check registration
        if "test_engine" in monitor._components:
            print("✅ Component found in registry")
        else:
            print("❌ Component not found in registry")
            return False
        
        # Test system health without starting
        health = monitor.get_system_health()
        print(f"✅ System health retrieved: {health['monitored_components']} components")
        
        # Test unregistration
        monitor.unregister_component("test_engine")
        if "test_engine" not in monitor._components:
            print("✅ Component unregistered successfully")
        else:
            print("❌ Component still in registry")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic monitor test failed: {e}")
        return False


def test_analyzers():
    """Test analyzer functionality."""
    print("\n🔍 Testing Analyzers")
    print("-" * 18)
    
    try:
        from analyzers import AnomalyDetector, HealthScorer
        from core import ComponentState, ComponentType, StateType
        
        # Test AnomalyDetector
        detector = AnomalyDetector(min_samples=3, sensitivity=2.0)
        print("✅ AnomalyDetector created")
        
        # Create test states
        states = []
        for i in range(10):
            state = ComponentState(
                component_id="test_component",
                component_type=ComponentType.WORKER,
                state_type=StateType.PERFORMANCE,
                timestamp=time.time() + i,
                health_score=0.9 if i < 8 else 0.3,  # Anomaly in last 2
                cpu_usage=50.0 + i * 2,
                memory_usage=60.0 + i * 3
            )
            states.append(state)
        
        # Analyze states
        result = detector.analyze(states)
        print(f"✅ Anomaly analysis completed: {result['anomaly_count']} anomalies")
        
        # Test HealthScorer
        scorer = HealthScorer()
        health_result = scorer.analyze(states)
        print(f"✅ Health analysis completed: {health_result['overall_health']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 vLLM Monitoring System Simple Test")
    print("=" * 40)
    
    tests = [
        ("Module Imports", test_imports),
        ("Core Functionality", test_core_functionality),
        ("Basic Monitor", lambda: asyncio.run(test_monitor_basic())),
        ("Analyzers", test_analyzers),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️ Running {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
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
        print("🎉 All tests passed! The monitoring system core is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)