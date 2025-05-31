"""
Complete example demonstrating predictive failure detection, continuous learning,
and automatic mitigation for vLLM deployments.

This example shows:
1. Pre-startup validation and configuration
2. Predictive failure detection with state projection
3. Continuous learning from mitigation outcomes
4. Automatic intervention execution
5. Comprehensive state tracking and guardrails
"""

import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List

# Import all monitoring components
from vllm_monitor import (
    VLLMMonitor,
    create_monitor_with_plugins,
    LifecycleState,
    StateTransition,
    StateCheckpoint,
    GuardrailPolicy
)

from vllm_monitor.predictive_failure_detection import (
    PredictiveFailureDetector,
    MitigationStrategy,
    FailurePattern,
    FailurePrediction
)

from vllm_monitor.continuous_learning import (
    ContinuousLearningSystem,
    MitigationAttempt,
    MitigationOutcome,
    LearningMethod
)

from vllm_monitor.vllm_integration_plugins import (
    PreStartupValidator,
    register_all_vllm_plugins
)

from vllm_monitor.vllm_mitigation_strategies import (
    register_all_mitigation_strategies,
    PreStartupConfiguration
)


def simulate_vllm_metrics() -> Dict[str, float]:
    """Simulate vLLM metrics for demonstration"""
    return {
        "memory_percent": random.uniform(60, 95),
        "gpu_0_memory_percent": random.uniform(70, 98),
        "gpu_0_utilization": random.uniform(40, 95),
        "gpu_0_temperature": random.uniform(60, 85),
        "error_rate": random.uniform(0, 0.15),
        "avg_latency_ms": random.uniform(50, 500),
        "queue_size": random.randint(0, 100),
        "tokens_per_second": random.uniform(100, 1000)
    }


def main():
    print("=== vLLM Predictive Monitoring System Demo ===\n")
    
    # 1. Initialize monitoring system with all components
    print("1. Initializing comprehensive monitoring system...")
    monitor = create_monitor_with_plugins()
    
    # Initialize predictive failure detector
    failure_detector = PredictiveFailureDetector(
        learning_rate=0.1,
        prediction_horizon=300.0  # 5 minutes
    )
    failure_detector.start_background_analysis()
    
    # Initialize continuous learning system
    learner = ContinuousLearningSystem(
        learning_method=LearningMethod.ENSEMBLE,
        learning_rate=0.1
    )
    learner.start_background_learning()
    
    # Register all vLLM plugins
    register_all_vllm_plugins(monitor.plugin_manager)
    
    # Register all mitigation strategies
    num_strategies = register_all_mitigation_strategies(learner)
    print(f"Registered {num_strategies} mitigation strategies")
    
    # 2. Pre-startup validation
    print("\n2. Running pre-startup validation...")
    
    # Execute pre-startup validator
    success, validation_result = monitor.plugin_manager.execute_plugin("pre_startup_validator")
    if success:
        print(f"Validation {'passed' if validation_result['passed'] else 'failed'}")
        if validation_result['errors']:
            print(f"Errors: {validation_result['errors']}")
        if validation_result['warnings']:
            print(f"Warnings: {validation_result['warnings'][:3]}...")  # First 3
        if validation_result['recommendations']:
            print(f"Recommendations: {validation_result['recommendations'][:3]}...")
    
    # Execute pre-startup configuration
    pre_config_strategy = PreStartupConfiguration()
    config_attempt = MitigationAttempt(
        attempt_id="pre_startup_config",
        timestamp=time.time(),
        initial_state=LifecycleState.NOT_STARTED,
        initial_metrics={},
        error_context=[],
        strategy_name=pre_config_strategy.name,
        interventions_executed=pre_config_strategy.interventions,
        execution_time=0,
        outcome=MitigationOutcome.NOT_APPLICABLE,
        final_state=LifecycleState.NOT_STARTED,
        final_metrics={}
    )
    
    print("\n3. Executing pre-startup configuration...")
    start_time = time.time()
    outcome = pre_config_strategy.execute({"logger": print})
    config_attempt.execution_time = time.time() - start_time
    config_attempt.outcome = outcome
    
    # Record attempt for learning
    learner.record_attempt(config_attempt)
    
    # 3. Simulate vLLM lifecycle with predictive monitoring
    print("\n4. Starting vLLM lifecycle simulation with predictive monitoring...")
    
    # Track startup
    startup_checkpoint = monitor.track_lifecycle_state(
        LifecycleState.INITIALIZING,
        StateTransition.STARTUP,
        {
            "model": "meta-llama/Llama-2-7b-hf",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.9
        }
    )
    
    # Simulate lifecycle progression
    lifecycle_sequence = [
        (LifecycleState.LOADING_MODEL, StateTransition.STARTUP),
        (LifecycleState.ALLOCATING_MEMORY, StateTransition.STARTUP),
        (LifecycleState.COMPILING_KERNELS, StateTransition.STARTUP),
        (LifecycleState.READY, StateTransition.STARTUP),
        (LifecycleState.SERVING, StateTransition.STARTUP)
    ]
    
    for state, transition in lifecycle_sequence:
        time.sleep(0.5)
        
        # Create checkpoint with simulated metrics
        checkpoint = StateCheckpoint(
            timestamp=time.time(),
            state=state,
            previous_state=monitor.lifecycle_tracker.current_state,
            transition_type=transition,
            arguments={},
            environment={},
            hardware_state={},
            metrics=simulate_vllm_metrics()
        )
        
        # Analyze for failure predictions
        predictions = failure_detector.analyze_checkpoint(checkpoint)
        
        if predictions:
            print(f"\n[{state.name}] Failure predictions detected:")
            for pred in predictions:
                print(f"  - {pred.failure_type}: {pred.probability:.2%} probability")
                print(f"    Time to failure: {pred.time_to_failure:.0f}s")
                print(f"    Confidence: {pred.confidence.name}")
                
                # Get learning-based recommendations
                recommendations = learner.get_learning_recommendations(
                    state,
                    [],  # No errors yet
                    checkpoint.metrics
                )
                
                if recommendations:
                    print("    Recommended mitigations:")
                    for rec in recommendations[:3]:
                        print(f"      {rec['rank']}. {rec['strategy']} (score: {rec['score']:.2f})")
                        print(f"         Rationale: {rec['rationale']}")
        
        # Update lifecycle state
        monitor.track_lifecycle_state(state, transition, checkpoint.metrics)
    
    # 4. Simulate failure scenario
    print("\n5. Simulating failure scenario...")
    
    # Create a high memory usage scenario
    critical_metrics = {
        "memory_percent": 92,
        "gpu_0_memory_percent": 96,
        "gpu_0_utilization": 95,
        "error_rate": 0.08,
        "avg_latency_ms": 450
    }
    
    failure_checkpoint = StateCheckpoint(
        timestamp=time.time(),
        state=LifecycleState.SERVING,
        previous_state=LifecycleState.SERVING,
        transition_type=None,
        arguments={},
        environment={},
        hardware_state={},
        metrics=critical_metrics,
        error_context={"errors": ["CUDA out of memory"]}
    )
    
    # Analyze critical situation
    predictions = failure_detector.analyze_checkpoint(failure_checkpoint)
    
    print("\nCritical situation detected!")
    print(f"Metrics: {json.dumps(critical_metrics, indent=2)}")
    
    # 5. Execute automatic mitigation
    print("\n6. Executing automatic mitigation...")
    
    if predictions:
        # Take the most urgent prediction
        most_urgent = max(predictions, key=lambda p: p.probability / p.time_to_failure)
        
        print(f"\nMost urgent threat: {most_urgent.failure_type}")
        print(f"Recommended mitigations: {len(most_urgent.recommended_mitigations)}")
        
        # Execute top mitigation
        if most_urgent.recommended_mitigations:
            top_mitigation = most_urgent.recommended_mitigations[0]
            
            print(f"\nExecuting mitigation path: {top_mitigation.path_id}")
            print(f"Interventions: {top_mitigation.interventions}")
            print(f"Expected success probability: {top_mitigation.success_probability:.2%}")
            
            # Simulate mitigation execution
            for intervention in top_mitigation.interventions:
                print(f"  - Executing: {intervention}")
                time.sleep(0.5)
            
            # Record mitigation outcome
            mitigation_attempt = MitigationAttempt(
                attempt_id=f"attempt_{int(time.time())}",
                timestamp=time.time(),
                initial_state=failure_checkpoint.state,
                initial_metrics=failure_checkpoint.metrics,
                error_context=["CUDA out of memory"],
                strategy_name=top_mitigation.interventions[0],
                interventions_executed=top_mitigation.interventions,
                execution_time=len(top_mitigation.interventions) * 0.5,
                outcome=MitigationOutcome.SUCCESS if random.random() < 0.8 else MitigationOutcome.PARTIAL_SUCCESS,
                final_state=LifecycleState.SERVING,
                final_metrics=simulate_vllm_metrics()  # New metrics after mitigation
            )
            
            # Record for learning
            learner.record_attempt(mitigation_attempt)
            
            print(f"\nMitigation outcome: {mitigation_attempt.outcome.name}")
            print(f"Effectiveness score: {mitigation_attempt.calculate_effectiveness():.2f}")
    
    # 6. Learn from failure pattern
    print("\n7. Learning from failure pattern...")
    
    # Teach the system about this failure pattern
    failure_detector.learn_pattern(
        state_sequence=[LifecycleState.SERVING] * 5,
        error_sequence=["CUDA out of memory"],
        metrics_sequence=[critical_metrics],
        resulted_in_failure=True
    )
    
    print("Pattern learned and added to knowledge base")
    
    # 7. Test guardrails
    print("\n8. Testing guardrail system...")
    
    # The guardrails should trigger based on the metrics
    health = monitor.get_system_health()
    print(f"System health score: {health['overall_score']:.2f}")
    print(f"Health status: {health['status']}")
    
    if health['triggered_guardrails']:
        print("Triggered guardrails:")
        for guardrail in health['triggered_guardrails']:
            print(f"  - {guardrail}")
    
    # 8. Generate comprehensive reports
    print("\n9. Generating comprehensive reports...")
    
    # Predictive failure detection report
    pred_report = failure_detector.get_prediction_report()
    print("\n--- Predictive Failure Detection Report ---")
    print(f"Active predictions: {pred_report['active_predictions']}")
    print(f"Learned patterns: {pred_report['learned_patterns']}")
    print(f"Prediction accuracy: {pred_report['prediction_metrics']['accuracy']:.2%}")
    print(f"High-risk states: {pred_report['state_graph']['high_risk_states']}")
    
    # Continuous learning report
    learning_report = learner.get_learning_report()
    print("\n--- Continuous Learning Report ---")
    print(f"Total mitigation attempts: {learning_report['learning_state']['total_attempts']}")
    print(f"Success rate: {learning_report['learning_state']['success_rate']:.2%}")
    print(f"Known strategies: {learning_report['learning_state']['known_strategies']}")
    print(f"Learned rules: {learning_report['learning_state']['learned_rules']}")
    
    if learning_report['top_strategies']:
        print("\nTop performing strategies:")
        for strategy in learning_report['top_strategies'][:3]:
            print(f"  - {strategy['name']}: {strategy['success_rate']:.2%} success rate")
    
    # 9. Demonstrate state recovery path finding
    print("\n10. Finding optimal recovery paths...")
    
    # Find path from ERROR to SERVING
    recovery_paths = failure_detector.state_graph.find_recovery_paths(
        LifecycleState.ERROR,
        LifecycleState.SERVING
    )
    
    if recovery_paths:
        print(f"Found {len(recovery_paths)} recovery paths")
        shortest_path = min(recovery_paths, key=len)
        print(f"Shortest path: {' -> '.join(s.name for s in shortest_path)}")
    
    # 10. Cleanup
    print("\n11. Shutting down monitoring system...")
    
    # Stop background threads
    failure_detector.stop_background_analysis()
    learner.stop_background_learning()
    
    # Track shutdown
    shutdown_checkpoint = monitor.lifecycle_tracker.track_shutdown(
        reason="demo_complete",
        cleanup_status={
            "predictive_detector": True,
            "continuous_learner": True,
            "plugins": True
        }
    )
    
    print("\nDemo complete!")
    print(f"Total runtime: {shutdown_checkpoint.metadata['uptime_seconds']:.1f} seconds")
    print(f"Total checkpoints: {shutdown_checkpoint.metadata['total_checkpoints']}")
    
    # Final summary
    print("\n=== System Capabilities Demonstrated ===")
    print("✓ Pre-startup validation and configuration")
    print("✓ Predictive failure detection with confidence scores")
    print("✓ Continuous learning from mitigation outcomes")
    print("✓ Automatic intervention execution")
    print("✓ State graph analysis and recovery path finding")
    print("✓ Comprehensive guardrail system")
    print("✓ Learning-based strategy recommendations")
    print("✓ Pattern recognition and knowledge accumulation")


if __name__ == "__main__":
    main()