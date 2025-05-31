#!/usr/bin/env python3
"""
Complete example showing vLLM integration with comprehensive monitoring.

This demonstrates:
1. Pre-startup validation
2. Engine creation with monitoring
3. Request processing with real-time monitoring
4. Predictive failure detection
5. Automatic mitigation
6. Continuous learning
"""

import os
import time
import torch
from typing import List, Optional

# Set environment variables for monitoring
os.environ['VLLM_ENABLE_MONITORING'] = 'true'
os.environ['VLLM_MONITOR_PREDICTIVE'] = 'true'
os.environ['VLLM_MONITOR_LEARNING'] = 'true'
os.environ['VLLM_MONITOR_AUTO_MITIGATE'] = 'true'

# Import vLLM components
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# Import monitoring components
from vllm_monitor.vllm_engine_integration import (
    VLLMEngineMonitor, 
    create_monitored_engine
)
from vllm_monitor.prestartup_check import PreStartupCheck


def run_prestartup_validation():
    """Run pre-startup validation and configuration."""
    print("=" * 80)
    print("Step 1: Pre-startup Validation")
    print("=" * 80)
    
    checker = PreStartupCheck(auto_fix=True)
    success = checker.run_checks()
    
    if not success:
        print("\n⚠️  Pre-startup validation failed. Please fix issues before continuing.")
        return False
    
    print("\n✅ Pre-startup validation passed!")
    return True


def create_monitored_llm(model_name: str = "gpt2"):
    """Create LLM with integrated monitoring."""
    print("\n" + "=" * 80)
    print("Step 2: Creating Monitored LLM")
    print("=" * 80)
    
    # Create engine args
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        enable_monitoring=True,  # Enable monitoring
        enable_predictive_detection=True,
        enable_continuous_learning=True,
        enable_auto_mitigation=True
    )
    
    # Create monitored engine
    engine, monitor = create_monitored_engine(
        engine_args,
        enable_predictive=True,
        enable_learning=True,
        enable_auto_mitigation=True
    )
    
    print(f"✅ Created monitored engine for {model_name}")
    print(f"   - Predictive detection: {monitor.predictive_detector is not None}")
    print(f"   - Continuous learning: {monitor.continuous_learner is not None}")
    print(f"   - Auto-mitigation: {monitor.enable_auto_mitigation}")
    
    return engine, monitor


def simulate_normal_operation(engine, monitor: VLLMEngineMonitor):
    """Simulate normal request processing."""
    print("\n" + "=" * 80)
    print("Step 3: Normal Operation")
    print("=" * 80)
    
    # Create some test prompts
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "The meaning of life is",
        "Python programming language was created by",
        "The largest planet in our solar system is"
    ]
    
    # Process requests
    print("\nProcessing requests...")
    for i, prompt in enumerate(prompts):
        print(f"\n[Request {i+1}] {prompt}")
        
        # Add request
        request_id = f"req_{i}"
        engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=SamplingParams(
                temperature=0.8,
                max_tokens=20
            )
        )
        
        # Step engine
        outputs = engine.step()
        
        # Process outputs
        while not outputs:
            outputs = engine.step()
            time.sleep(0.1)
        
        # Print results
        for output in outputs:
            if output.finished:
                generated_text = output.outputs[0].text
                print(f"   Generated: {generated_text}")
        
        # Check monitoring status
        status = monitor.get_engine_status()
        print(f"   Active requests: {status['active_requests']}")
        print(f"   Tokens generated: {status['tokens_generated']}")
        
        time.sleep(0.5)  # Small delay between requests


def simulate_high_load(engine, monitor: VLLMEngineMonitor):
    """Simulate high load scenario to trigger monitoring."""
    print("\n" + "=" * 80)
    print("Step 4: High Load Scenario")
    print("=" * 80)
    
    # Create many concurrent requests
    print("\nGenerating high load...")
    prompts = [f"Tell me about topic number {i}" for i in range(50)]
    
    # Add all requests quickly
    for i, prompt in enumerate(prompts):
        engine.add_request(
            request_id=f"load_req_{i}",
            prompt=prompt,
            params=SamplingParams(
                temperature=0.5,
                max_tokens=50
            )
        )
    
    print(f"Added {len(prompts)} requests to queue")
    
    # Process with monitoring
    processed = 0
    start_time = time.time()
    
    while engine.has_unfinished_requests():
        outputs = engine.step()
        
        # Count finished requests
        for output in outputs:
            if output.finished:
                processed += 1
        
        # Periodic status check
        if processed % 10 == 0 and processed > 0:
            status = monitor.get_engine_status()
            elapsed = time.time() - start_time
            
            print(f"\n[Progress] Processed: {processed}/{len(prompts)}")
            print(f"   Time elapsed: {elapsed:.1f}s")
            print(f"   Active requests: {status['active_requests']}")
            print(f"   Avg step duration: {status['avg_step_duration_ms']:.1f}ms")
            
            # Check if mitigations were triggered
            if monitor.mitigation_in_progress:
                print("   ⚠️  Mitigation in progress!")
    
    total_time = time.time() - start_time
    print(f"\n✅ Processed all {len(prompts)} requests in {total_time:.1f}s")
    print(f"   Average: {total_time/len(prompts):.2f}s per request")


def demonstrate_failure_prediction(engine, monitor: VLLMEngineMonitor):
    """Demonstrate predictive failure detection."""
    print("\n" + "=" * 80)
    print("Step 5: Predictive Failure Detection")
    print("=" * 80)
    
    if not monitor.predictive_detector:
        print("Predictive detection not enabled")
        return
    
    # Create a scenario that might trigger predictions
    print("\nCreating memory-intensive requests...")
    
    # Large prompts that might stress memory
    large_prompts = [
        "Write a very detailed essay about " * 20,  # Repetitive prompt
        "List all the numbers from 1 to 1000: " + " ".join(str(i) for i in range(100)),
    ]
    
    for i, prompt in enumerate(large_prompts):
        print(f"\n[Memory Test {i+1}]")
        
        # Check predictions before adding request
        current_state = monitor._determine_engine_state(engine)
        metrics = monitor._collect_engine_metrics(engine)
        
        # Create checkpoint for analysis
        from vllm_monitor.lifecycle_tracker import StateCheckpoint
        checkpoint = StateCheckpoint(
            timestamp=time.time(),
            state=current_state,
            previous_state=current_state,
            transition_type=None,
            arguments={},
            environment={},
            hardware_state={},
            metrics=metrics
        )
        
        # Get predictions
        predictions = monitor.predictive_detector.analyze_checkpoint(checkpoint)
        
        if predictions:
            print("   ⚠️  Failure predictions detected:")
            for pred in predictions[:3]:  # Show top 3
                print(f"      - {pred.failure_type}: {pred.probability:.1%} chance")
                print(f"        Time to failure: {pred.time_to_failure:.0f}s")
                if pred.preventive_actions:
                    print(f"        Suggested actions: {', '.join(pred.preventive_actions[:2])}")
        else:
            print("   ✅ No immediate failure risks detected")
        
        # Try to add the request
        try:
            engine.add_request(
                request_id=f"mem_test_{i}",
                prompt=prompt[:512],  # Limit length
                params=SamplingParams(max_tokens=10)
            )
            
            # Process
            while engine.has_unfinished_requests():
                engine.step()
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            print("   Monitor should have predicted this!")


def show_learning_insights(monitor: VLLMEngineMonitor):
    """Display what the system has learned."""
    print("\n" + "=" * 80)
    print("Step 6: Continuous Learning Insights")
    print("=" * 80)
    
    if not monitor.continuous_learner:
        print("Continuous learning not enabled")
        return
    
    learner = monitor.continuous_learner
    
    # Get learning stats
    print("\nLearning Statistics:")
    print(f"   Total mitigation attempts: {len(learner.attempt_history)}")
    
    # Show learned patterns
    if learner.attempt_history:
        # Count outcomes
        outcomes = {}
        strategies = {}
        
        for attempt in learner.attempt_history:
            outcome_name = attempt.outcome.name
            outcomes[outcome_name] = outcomes.get(outcome_name, 0) + 1
            
            strategy = attempt.strategy_name
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        print("\n   Mitigation Outcomes:")
        for outcome, count in outcomes.items():
            print(f"      - {outcome}: {count}")
        
        print("\n   Most Used Strategies:")
        for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      - {strategy}: {count} times")
    
    # Show current recommendations
    print("\n   Current Best Practices:")
    from vllm_monitor.lifecycle_tracker import LifecycleState
    
    test_states = [
        LifecycleState.ERROR,
        LifecycleState.SERVING,
        LifecycleState.DEGRADED
    ]
    
    for state in test_states:
        recs = learner.get_learning_recommendations(
            state,
            ["test_error"],
            {"gpu_memory_percent": 85}
        )
        
        if recs:
            print(f"\n   For {state.name} state:")
            for rec in recs[:2]:
                print(f"      - {rec['strategy']} (confidence: {rec['confidence']:.2f})")


def main():
    """Run the complete monitoring demonstration."""
    print("=" * 80)
    print("vLLM Complete Monitoring System Demonstration")
    print("=" * 80)
    
    # Step 1: Pre-startup validation
    if not run_prestartup_validation():
        return
    
    # Step 2: Create monitored LLM
    try:
        engine, monitor = create_monitored_llm()
    except Exception as e:
        print(f"\n❌ Failed to create engine: {e}")
        print("Using mock engine for demonstration")
        # In real usage, you would exit here
        return
    
    try:
        # Step 3: Normal operation
        simulate_normal_operation(engine, monitor)
        
        # Step 4: High load scenario
        simulate_high_load(engine, monitor)
        
        # Step 5: Predictive failure detection
        demonstrate_failure_prediction(engine, monitor)
        
        # Step 6: Learning insights
        show_learning_insights(monitor)
        
    finally:
        # Cleanup
        print("\n" + "=" * 80)
        print("Cleanup")
        print("=" * 80)
        
        # Get final status
        final_status = monitor.get_engine_status()
        print("\nFinal Statistics:")
        print(f"   Total tokens generated: {final_status['tokens_generated']}")
        print(f"   Monitoring active: {final_status['monitor_active']}")
        
        # Shutdown
        if hasattr(engine, 'model_executor'):
            engine.model_executor.shutdown()
        
        print("\n✅ Demonstration complete!")


if __name__ == "__main__":
    main()