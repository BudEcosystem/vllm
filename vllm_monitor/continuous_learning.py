"""
Continuous Learning System for Mitigation Strategy Optimization.

This module implements a self-improving system that learns from the success
and failure of mitigation strategies to optimize future interventions.
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum, auto
from collections import defaultdict, deque
import threading
import pickle
from pathlib import Path
import logging

from .predictive_failure_detection import (
    MitigationStrategy, FailurePattern, MitigationPath,
    HealthStatus, PredictionConfidence
)
from .lifecycle_tracker import LifecycleState, StateCheckpoint
from .core import CircularBuffer, get_logger


class MitigationOutcome(Enum):
    """Outcome of a mitigation attempt"""
    SUCCESS = auto()
    PARTIAL_SUCCESS = auto()
    FAILURE = auto()
    MADE_WORSE = auto()
    NOT_APPLICABLE = auto()


class LearningMethod(Enum):
    """Learning methods for strategy optimization"""
    REINFORCEMENT = auto()
    BAYESIAN = auto()
    NEURAL = auto()
    RULE_BASED = auto()
    ENSEMBLE = auto()


@dataclass
class MitigationAttempt:
    """Record of a mitigation attempt"""
    attempt_id: str
    timestamp: float
    initial_state: LifecycleState
    initial_metrics: Dict[str, float]
    error_context: List[str]
    
    # Mitigation details
    strategy_name: str
    interventions_executed: List[str]
    execution_time: float
    
    # Outcome
    outcome: MitigationOutcome
    final_state: LifecycleState
    final_metrics: Dict[str, float]
    side_effects_observed: List[str]
    
    # Context
    failure_pattern_id: Optional[str] = None
    prediction_id: Optional[str] = None
    confidence_score: float = 0.0
    
    def calculate_effectiveness(self) -> float:
        """Calculate effectiveness score (0-1)"""
        if self.outcome == MitigationOutcome.SUCCESS:
            base_score = 1.0
        elif self.outcome == MitigationOutcome.PARTIAL_SUCCESS:
            base_score = 0.7
        elif self.outcome == MitigationOutcome.FAILURE:
            base_score = 0.3
        elif self.outcome == MitigationOutcome.MADE_WORSE:
            base_score = 0.0
        else:
            base_score = 0.5
        
        # Adjust for execution time (faster is better)
        time_penalty = min(0.2, self.execution_time / 100)  # Max 0.2 penalty
        
        # Adjust for side effects
        side_effect_penalty = len(self.side_effects_observed) * 0.05
        
        return max(0, base_score - time_penalty - side_effect_penalty)


@dataclass
class StrategyPerformance:
    """Performance tracking for a mitigation strategy"""
    strategy_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    partial_success_attempts: int = 0
    failed_attempts: int = 0
    made_worse_attempts: int = 0
    
    # Performance metrics
    avg_execution_time: float = 0.0
    avg_effectiveness: float = 0.0
    success_rate: float = 0.0
    
    # Context-specific performance
    performance_by_state: Dict[LifecycleState, float] = field(default_factory=dict)
    performance_by_error: Dict[str, float] = field(default_factory=dict)
    performance_by_pattern: Dict[str, float] = field(default_factory=dict)
    
    # Learning parameters
    confidence: float = 0.5
    exploration_bonus: float = 0.1
    
    def update(self, attempt: MitigationAttempt):
        """Update performance metrics with new attempt"""
        self.total_attempts += 1
        
        # Update outcome counts
        if attempt.outcome == MitigationOutcome.SUCCESS:
            self.successful_attempts += 1
        elif attempt.outcome == MitigationOutcome.PARTIAL_SUCCESS:
            self.partial_success_attempts += 1
        elif attempt.outcome == MitigationOutcome.FAILURE:
            self.failed_attempts += 1
        elif attempt.outcome == MitigationOutcome.MADE_WORSE:
            self.made_worse_attempts += 1
        
        # Update averages
        effectiveness = attempt.calculate_effectiveness()
        self.avg_effectiveness = (
            (self.avg_effectiveness * (self.total_attempts - 1) + effectiveness) 
            / self.total_attempts
        )
        
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_attempts - 1) + attempt.execution_time)
            / self.total_attempts
        )
        
        self.success_rate = (
            (self.successful_attempts + 0.5 * self.partial_success_attempts) 
            / self.total_attempts
        )
        
        # Update context-specific performance
        self._update_context_performance(attempt, effectiveness)
        
        # Update confidence
        self.confidence = min(0.95, self.confidence + 0.05)
        self.exploration_bonus = max(0.01, self.exploration_bonus * 0.95)
    
    def _update_context_performance(self, attempt: MitigationAttempt, effectiveness: float):
        """Update context-specific performance metrics"""
        # By state
        state = attempt.initial_state
        current = self.performance_by_state.get(state, 0.5)
        self.performance_by_state[state] = current * 0.8 + effectiveness * 0.2
        
        # By error type
        for error in attempt.error_context[:3]:  # Top 3 errors
            current = self.performance_by_error.get(error, 0.5)
            self.performance_by_error[error] = current * 0.8 + effectiveness * 0.2
        
        # By pattern
        if attempt.failure_pattern_id:
            current = self.performance_by_pattern.get(attempt.failure_pattern_id, 0.5)
            self.performance_by_pattern[attempt.failure_pattern_id] = current * 0.8 + effectiveness * 0.2
    
    def get_expected_effectiveness(self, 
                                  state: LifecycleState,
                                  errors: List[str],
                                  pattern_id: Optional[str] = None) -> float:
        """Get expected effectiveness for given context"""
        # Start with overall average
        base_effectiveness = self.avg_effectiveness
        
        # Adjust based on context
        context_scores = []
        
        if state in self.performance_by_state:
            context_scores.append(self.performance_by_state[state])
        
        for error in errors:
            if error in self.performance_by_error:
                context_scores.append(self.performance_by_error[error])
        
        if pattern_id and pattern_id in self.performance_by_pattern:
            context_scores.append(self.performance_by_pattern[pattern_id])
        
        if context_scores:
            # Weighted average: 60% base, 40% context
            context_avg = sum(context_scores) / len(context_scores)
            return base_effectiveness * 0.6 + context_avg * 0.4
        
        return base_effectiveness


@dataclass
class LearningState:
    """State of the learning system"""
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    
    # Learning metrics
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    discount_factor: float = 0.95
    
    # Model performance
    prediction_accuracy: float = 0.0
    strategy_selection_accuracy: float = 0.0
    
    # Knowledge base size
    known_patterns: int = 0
    known_strategies: int = 0
    learned_rules: int = 0


class ContinuousLearningSystem:
    """
    Self-improving system that learns from mitigation outcomes.
    
    Features:
    - Multi-armed bandit for strategy selection
    - Reinforcement learning for strategy optimization
    - Pattern recognition for context matching
    - Rule extraction from successful mitigations
    - Ensemble learning combining multiple methods
    """
    
    def __init__(self,
                 learning_method: LearningMethod = LearningMethod.ENSEMBLE,
                 learning_rate: float = 0.1):
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # Configuration
        self.learning_method = learning_method
        self.learning_rate = learning_rate
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.strategy_registry: Dict[str, MitigationStrategy] = {}
        
        # Learning history
        self.attempt_history = CircularBuffer(10000)
        self.learning_events = CircularBuffer(1000)
        
        # Pattern learning
        self.success_patterns: Dict[str, Dict[str, Any]] = {}
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Rule base
        self.learned_rules: List[MitigationRule] = []
        self.rule_confidence: Dict[str, float] = {}
        
        # Q-learning table for reinforcement learning
        self.q_table: Dict[Tuple[str, str], float] = {}  # (state, action) -> value
        
        # Bayesian belief network
        self.belief_network = BayesianBeliefNetwork()
        
        # Neural network model (if using neural learning)
        self.neural_model = None
        
        # Learning state
        self.learning_state = LearningState()
        
        # Persistence
        self.model_path = Path("./continuous_learning_model.pkl")
        self._load_model()
        
        # Background learning
        self._learning_thread = None
        self._learning_active = False
        
        self.logger.info(f"Continuous learning system initialized with {learning_method.name} method")
    
    def record_attempt(self, attempt: MitigationAttempt) -> None:
        """Record a mitigation attempt and learn from it"""
        with self._lock:
            # Store in history
            self.attempt_history.append(attempt)
            self.learning_state.total_attempts += 1
            
            # Update performance tracking
            if attempt.strategy_name not in self.strategy_performance:
                self.strategy_performance[attempt.strategy_name] = StrategyPerformance(
                    strategy_name=attempt.strategy_name
                )
            
            self.strategy_performance[attempt.strategy_name].update(attempt)
            
            # Update success/failure counts
            if attempt.outcome in [MitigationOutcome.SUCCESS, MitigationOutcome.PARTIAL_SUCCESS]:
                self.learning_state.total_successes += 1
            else:
                self.learning_state.total_failures += 1
            
            # Learn from the attempt
            self._learn_from_attempt(attempt)
            
            # Log learning event
            self.learning_events.append({
                "timestamp": time.time(),
                "event": "attempt_recorded",
                "strategy": attempt.strategy_name,
                "outcome": attempt.outcome.name,
                "effectiveness": attempt.calculate_effectiveness()
            })
    
    def select_best_strategy(self,
                           state: LifecycleState,
                           errors: List[str],
                           metrics: Dict[str, float],
                           available_strategies: List[str],
                           pattern_id: Optional[str] = None) -> Tuple[str, float]:
        """
        Select the best mitigation strategy for current context.
        
        Returns:
            Tuple of (strategy_name, confidence_score)
        """
        with self._lock:
            if not available_strategies:
                return None, 0.0
            
            # Score each strategy
            strategy_scores = {}
            
            for strategy_name in available_strategies:
                if strategy_name in self.strategy_performance:
                    perf = self.strategy_performance[strategy_name]
                    
                    # Get base score from performance
                    base_score = perf.get_expected_effectiveness(state, errors, pattern_id)
                    
                    # Apply learning method specific adjustments
                    if self.learning_method == LearningMethod.REINFORCEMENT:
                        score = self._apply_reinforcement_learning(
                            strategy_name, state, base_score
                        )
                    elif self.learning_method == LearningMethod.BAYESIAN:
                        score = self._apply_bayesian_inference(
                            strategy_name, state, errors, metrics, base_score
                        )
                    elif self.learning_method == LearningMethod.RULE_BASED:
                        score = self._apply_rule_based_selection(
                            strategy_name, state, errors, metrics, base_score
                        )
                    else:  # ENSEMBLE
                        score = self._apply_ensemble_method(
                            strategy_name, state, errors, metrics, base_score
                        )
                    
                    # Add exploration bonus
                    score += perf.exploration_bonus
                    
                    strategy_scores[strategy_name] = score
                else:
                    # New strategy - give it a chance
                    strategy_scores[strategy_name] = 0.5 + self.learning_state.exploration_rate
            
            # Select best strategy
            if strategy_scores:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                return best_strategy[0], best_strategy[1]
            
            # Fallback to random selection
            import random
            return random.choice(available_strategies), 0.5
    
    def _learn_from_attempt(self, attempt: MitigationAttempt):
        """Learn from a mitigation attempt"""
        # Update Q-learning table
        if self.learning_method in [LearningMethod.REINFORCEMENT, LearningMethod.ENSEMBLE]:
            self._update_q_learning(attempt)
        
        # Update Bayesian network
        if self.learning_method in [LearningMethod.BAYESIAN, LearningMethod.ENSEMBLE]:
            self._update_bayesian_network(attempt)
        
        # Extract rules
        if self.learning_method in [LearningMethod.RULE_BASED, LearningMethod.ENSEMBLE]:
            self._extract_rules(attempt)
        
        # Update patterns
        self._update_patterns(attempt)
        
        # Decay exploration rate
        self.learning_state.exploration_rate *= 0.995
        self.learning_state.exploration_rate = max(0.05, self.learning_state.exploration_rate)
    
    def _update_q_learning(self, attempt: MitigationAttempt):
        """Update Q-learning table"""
        state_key = f"{attempt.initial_state.name}_{self._discretize_metrics(attempt.initial_metrics)}"
        action = attempt.strategy_name
        
        # Calculate reward
        reward = self._calculate_reward(attempt)
        
        # Get current Q-value
        current_q = self.q_table.get((state_key, action), 0.0)
        
        # Find max Q-value for next state
        next_state_key = f"{attempt.final_state.name}_{self._discretize_metrics(attempt.final_metrics)}"
        max_next_q = max(
            [self.q_table.get((next_state_key, a), 0.0) 
             for a in self.strategy_registry.keys()],
            default=0.0
        )
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.learning_state.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state_key, action)] = new_q
    
    def _update_bayesian_network(self, attempt: MitigationAttempt):
        """Update Bayesian belief network"""
        # Create evidence from attempt
        evidence = {
            "state": attempt.initial_state.name,
            "has_errors": len(attempt.error_context) > 0,
            "memory_high": attempt.initial_metrics.get("memory_percent", 0) > 80,
            "strategy": attempt.strategy_name,
            "outcome": attempt.outcome.name
        }
        
        self.belief_network.update_beliefs(evidence)
    
    def _extract_rules(self, attempt: MitigationAttempt):
        """Extract rules from successful attempts"""
        if attempt.outcome != MitigationOutcome.SUCCESS:
            return
        
        # Create rule from successful attempt
        rule = MitigationRule(
            rule_id=f"rule_{len(self.learned_rules)}",
            conditions={
                "state": attempt.initial_state,
                "error_patterns": attempt.error_context[:3],
                "metric_conditions": self._extract_metric_conditions(
                    attempt.initial_metrics
                )
            },
            recommended_strategy=attempt.strategy_name,
            confidence=0.7,
            success_count=1,
            failure_count=0
        )
        
        # Check if similar rule exists
        for existing_rule in self.learned_rules:
            if existing_rule.is_similar(rule):
                existing_rule.success_count += 1
                existing_rule.update_confidence()
                return
        
        # Add new rule
        self.learned_rules.append(rule)
        self.learning_state.learned_rules += 1
    
    def _update_patterns(self, attempt: MitigationAttempt):
        """Update success/failure patterns"""
        pattern_key = f"{attempt.initial_state.name}_{attempt.strategy_name}"
        
        pattern_data = {
            "state": attempt.initial_state,
            "strategy": attempt.strategy_name,
            "errors": attempt.error_context,
            "metrics": attempt.initial_metrics,
            "outcome": attempt.outcome,
            "effectiveness": attempt.calculate_effectiveness()
        }
        
        if attempt.outcome in [MitigationOutcome.SUCCESS, MitigationOutcome.PARTIAL_SUCCESS]:
            self.success_patterns[pattern_key] = pattern_data
        else:
            self.failure_patterns[pattern_key] = pattern_data
    
    def _calculate_reward(self, attempt: MitigationAttempt) -> float:
        """Calculate reward for reinforcement learning"""
        if attempt.outcome == MitigationOutcome.SUCCESS:
            reward = 1.0
        elif attempt.outcome == MitigationOutcome.PARTIAL_SUCCESS:
            reward = 0.5
        elif attempt.outcome == MitigationOutcome.FAILURE:
            reward = -0.2
        elif attempt.outcome == MitigationOutcome.MADE_WORSE:
            reward = -1.0
        else:
            reward = 0.0
        
        # Adjust for execution time
        time_penalty = min(0.3, attempt.execution_time / 300)  # Max 0.3 penalty
        
        return reward - time_penalty
    
    def _discretize_metrics(self, metrics: Dict[str, float]) -> str:
        """Discretize continuous metrics for state representation"""
        discretized = []
        
        # Memory usage
        mem = metrics.get("memory_percent", 0)
        if mem > 90:
            discretized.append("mem_critical")
        elif mem > 70:
            discretized.append("mem_high")
        else:
            discretized.append("mem_normal")
        
        # Error rate
        err = metrics.get("error_rate", 0)
        if err > 0.1:
            discretized.append("err_high")
        elif err > 0.01:
            discretized.append("err_medium")
        else:
            discretized.append("err_low")
        
        return "_".join(discretized)
    
    def _extract_metric_conditions(self, metrics: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Extract metric conditions for rule creation"""
        conditions = {}
        
        for key, value in metrics.items():
            # Create range around current value
            if key == "memory_percent":
                if value > 80:
                    conditions[key] = (80, 100)
                elif value > 60:
                    conditions[key] = (60, 80)
                else:
                    conditions[key] = (0, 60)
            elif key == "error_rate":
                if value > 0.1:
                    conditions[key] = (0.1, 1.0)
                elif value > 0.01:
                    conditions[key] = (0.01, 0.1)
                else:
                    conditions[key] = (0, 0.01)
        
        return conditions
    
    def _apply_reinforcement_learning(self,
                                    strategy: str,
                                    state: LifecycleState,
                                    base_score: float) -> float:
        """Apply Q-learning adjustments"""
        state_key = f"{state.name}_normal"  # Simplified for now
        q_value = self.q_table.get((state_key, strategy), 0.5)
        
        # Combine base score with Q-value
        return base_score * 0.6 + q_value * 0.4
    
    def _apply_bayesian_inference(self,
                                strategy: str,
                                state: LifecycleState,
                                errors: List[str],
                                metrics: Dict[str, float],
                                base_score: float) -> float:
        """Apply Bayesian inference"""
        evidence = {
            "state": state.name,
            "has_errors": len(errors) > 0,
            "memory_high": metrics.get("memory_percent", 0) > 80,
            "strategy": strategy
        }
        
        success_prob = self.belief_network.predict_outcome_probability(
            evidence, 
            "SUCCESS"
        )
        
        return base_score * 0.5 + success_prob * 0.5
    
    def _apply_rule_based_selection(self,
                                  strategy: str,
                                  state: LifecycleState,
                                  errors: List[str],
                                  metrics: Dict[str, float],
                                  base_score: float) -> float:
        """Apply learned rules"""
        matching_rules = []
        
        for rule in self.learned_rules:
            if rule.matches(state, errors, metrics) and rule.recommended_strategy == strategy:
                matching_rules.append(rule)
        
        if matching_rules:
            # Use highest confidence rule
            best_rule = max(matching_rules, key=lambda r: r.confidence)
            return base_score * 0.4 + best_rule.confidence * 0.6
        
        return base_score
    
    def _apply_ensemble_method(self,
                             strategy: str,
                             state: LifecycleState,
                             errors: List[str],
                             metrics: Dict[str, float],
                             base_score: float) -> float:
        """Combine multiple learning methods"""
        scores = [base_score]
        
        # Add Q-learning score
        rl_score = self._apply_reinforcement_learning(strategy, state, base_score)
        scores.append(rl_score)
        
        # Add Bayesian score
        bayes_score = self._apply_bayesian_inference(strategy, state, errors, metrics, base_score)
        scores.append(bayes_score)
        
        # Add rule-based score
        rule_score = self._apply_rule_based_selection(strategy, state, errors, metrics, base_score)
        scores.append(rule_score)
        
        # Weighted average
        weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for now
        return sum(s * w for s, w in zip(scores, weights))
    
    def predict_strategy_outcome(self,
                               strategy: str,
                               state: LifecycleState,
                               errors: List[str],
                               metrics: Dict[str, float]) -> Tuple[MitigationOutcome, float]:
        """
        Predict the likely outcome of a strategy.
        
        Returns:
            Tuple of (predicted_outcome, confidence)
        """
        with self._lock:
            if strategy not in self.strategy_performance:
                return MitigationOutcome.NOT_APPLICABLE, 0.0
            
            perf = self.strategy_performance[strategy]
            effectiveness = perf.get_expected_effectiveness(state, errors)
            
            # Predict outcome based on effectiveness
            if effectiveness > 0.8:
                outcome = MitigationOutcome.SUCCESS
                confidence = effectiveness
            elif effectiveness > 0.6:
                outcome = MitigationOutcome.PARTIAL_SUCCESS
                confidence = effectiveness * 0.9
            elif effectiveness > 0.3:
                outcome = MitigationOutcome.FAILURE
                confidence = 1.0 - effectiveness
            else:
                outcome = MitigationOutcome.MADE_WORSE
                confidence = 0.8
            
            return outcome, confidence
    
    def get_learning_recommendations(self,
                                   state: LifecycleState,
                                   errors: List[str],
                                   metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get learning-based recommendations"""
        recommendations = []
        
        with self._lock:
            # Get all applicable strategies
            applicable_strategies = []
            for name, strategy in self.strategy_registry.items():
                if strategy.is_applicable(state, errors):
                    applicable_strategies.append(name)
            
            # Rank strategies
            strategy_rankings = []
            for strategy_name in applicable_strategies:
                score, confidence = self.select_best_strategy(
                    state, errors, metrics, [strategy_name]
                )
                
                predicted_outcome, outcome_confidence = self.predict_strategy_outcome(
                    strategy_name, state, errors, metrics
                )
                
                strategy_rankings.append({
                    "strategy": strategy_name,
                    "score": score,
                    "confidence": confidence,
                    "predicted_outcome": predicted_outcome.name,
                    "outcome_confidence": outcome_confidence,
                    "performance_history": self._get_strategy_summary(strategy_name)
                })
            
            # Sort by score
            strategy_rankings.sort(key=lambda x: x["score"], reverse=True)
            
            # Create recommendations
            for rank, ranking in enumerate(strategy_rankings[:5]):  # Top 5
                recommendations.append({
                    "rank": rank + 1,
                    "strategy": ranking["strategy"],
                    "score": ranking["score"],
                    "confidence": ranking["confidence"],
                    "predicted_outcome": ranking["predicted_outcome"],
                    "rationale": self._generate_recommendation_rationale(
                        ranking["strategy"], state, errors, metrics
                    ),
                    "expected_duration": self.strategy_registry[ranking["strategy"]].expected_duration,
                    "side_effects": self.strategy_registry[ranking["strategy"]].side_effects
                })
        
        return recommendations
    
    def _get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance summary for a strategy"""
        if strategy_name not in self.strategy_performance:
            return {"total_attempts": 0, "success_rate": 0.0}
        
        perf = self.strategy_performance[strategy_name]
        return {
            "total_attempts": perf.total_attempts,
            "success_rate": perf.success_rate,
            "avg_execution_time": perf.avg_execution_time,
            "avg_effectiveness": perf.avg_effectiveness
        }
    
    def _generate_recommendation_rationale(self,
                                         strategy: str,
                                         state: LifecycleState,
                                         errors: List[str],
                                         metrics: Dict[str, float]) -> str:
        """Generate human-readable rationale for recommendation"""
        rationales = []
        
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            
            # Overall performance
            if perf.success_rate > 0.8:
                rationales.append(f"High success rate ({perf.success_rate:.1%})")
            
            # Context-specific performance
            if state in perf.performance_by_state:
                state_perf = perf.performance_by_state[state]
                if state_perf > 0.7:
                    rationales.append(f"Effective for {state.name} state")
            
            # Pattern matching
            matching_rules = sum(1 for rule in self.learned_rules 
                               if rule.matches(state, errors, metrics) 
                               and rule.recommended_strategy == strategy)
            if matching_rules > 0:
                rationales.append(f"Matches {matching_rules} learned rules")
        
        if not rationales:
            rationales.append("New strategy worth exploring")
        
        return "; ".join(rationales)
    
    def register_strategy(self, strategy: MitigationStrategy):
        """Register a new mitigation strategy"""
        with self._lock:
            self.strategy_registry[strategy.name] = strategy
            self.learning_state.known_strategies += 1
            self.logger.info(f"Registered strategy: {strategy.name}")
    
    def start_background_learning(self):
        """Start background learning thread"""
        if not self._learning_active:
            self._learning_active = True
            self._learning_thread = threading.Thread(
                target=self._background_learning_loop,
                daemon=True
            )
            self._learning_thread.start()
    
    def stop_background_learning(self):
        """Stop background learning"""
        self._learning_active = False
        if self._learning_thread:
            self._learning_thread.join(timeout=5)
    
    def _background_learning_loop(self):
        """Background learning and optimization"""
        while self._learning_active:
            try:
                # Periodic model optimization
                self._optimize_models()
                
                # Save model
                self._save_model()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Background learning error: {e}")
    
    def _optimize_models(self):
        """Optimize learning models"""
        with self._lock:
            # Prune low-confidence rules
            self.learned_rules = [
                rule for rule in self.learned_rules
                if rule.confidence > 0.3 or rule.success_count > 5
            ]
            
            # Update strategy exploration bonuses
            for perf in self.strategy_performance.values():
                if perf.total_attempts > 50:
                    perf.exploration_bonus *= 0.9
    
    def _cleanup_old_data(self):
        """Clean up old learning data"""
        # Keep only recent patterns
        cutoff_time = time.time() - 86400 * 7  # 7 days
        
        # This is simplified - in production would be more sophisticated
        pass
    
    def _save_model(self):
        """Save learning model"""
        try:
            model_data = {
                "strategy_performance": self.strategy_performance,
                "q_table": self.q_table,
                "learned_rules": self.learned_rules,
                "learning_state": self.learning_state,
                "success_patterns": self.success_patterns,
                "failure_patterns": self.failure_patterns
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save learning model: {e}")
    
    def _load_model(self):
        """Load learning model"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.strategy_performance = model_data.get("strategy_performance", {})
                self.q_table = model_data.get("q_table", {})
                self.learned_rules = model_data.get("learned_rules", [])
                self.learning_state = model_data.get("learning_state", LearningState())
                self.success_patterns = model_data.get("success_patterns", {})
                self.failure_patterns = model_data.get("failure_patterns", {})
                
                self.logger.info(f"Loaded learning model with {len(self.learned_rules)} rules")
                
            except Exception as e:
                self.logger.error(f"Failed to load learning model: {e}")
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning system report"""
        with self._lock:
            return {
                "learning_state": {
                    "total_attempts": self.learning_state.total_attempts,
                    "success_rate": (self.learning_state.total_successes / 
                                   self.learning_state.total_attempts 
                                   if self.learning_state.total_attempts > 0 else 0),
                    "exploration_rate": self.learning_state.exploration_rate,
                    "known_strategies": self.learning_state.known_strategies,
                    "learned_rules": self.learning_state.learned_rules
                },
                "top_strategies": [
                    {
                        "name": perf.strategy_name,
                        "success_rate": perf.success_rate,
                        "avg_effectiveness": perf.avg_effectiveness,
                        "total_attempts": perf.total_attempts
                    }
                    for perf in sorted(
                        self.strategy_performance.values(),
                        key=lambda p: p.avg_effectiveness,
                        reverse=True
                    )[:5]
                ],
                "learning_method": self.learning_method.name,
                "model_size": {
                    "q_table_entries": len(self.q_table),
                    "learned_rules": len(self.learned_rules),
                    "success_patterns": len(self.success_patterns),
                    "failure_patterns": len(self.failure_patterns)
                }
            }


@dataclass
class MitigationRule:
    """Learned rule for mitigation selection"""
    rule_id: str
    conditions: Dict[str, Any]
    recommended_strategy: str
    confidence: float
    success_count: int
    failure_count: int
    
    def matches(self, state: LifecycleState, errors: List[str], metrics: Dict[str, float]) -> bool:
        """Check if rule conditions match current context"""
        # Check state
        if "state" in self.conditions and self.conditions["state"] != state:
            return False
        
        # Check errors
        if "error_patterns" in self.conditions:
            required_errors = self.conditions["error_patterns"]
            if not any(err in errors for err in required_errors):
                return False
        
        # Check metrics
        if "metric_conditions" in self.conditions:
            for metric, (min_val, max_val) in self.conditions["metric_conditions"].items():
                if metric in metrics:
                    if not (min_val <= metrics[metric] <= max_val):
                        return False
        
        return True
    
    def update_confidence(self):
        """Update rule confidence based on success/failure counts"""
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = self.success_count / total
        
        # Boost confidence for rules with many successes
        if self.success_count > 10:
            self.confidence = min(0.95, self.confidence * 1.1)
    
    def is_similar(self, other: 'MitigationRule') -> bool:
        """Check if two rules are similar"""
        # Same strategy and state
        if (self.recommended_strategy == other.recommended_strategy and
            self.conditions.get("state") == other.conditions.get("state")):
            
            # Check if error patterns overlap
            self_errors = set(self.conditions.get("error_patterns", []))
            other_errors = set(other.conditions.get("error_patterns", []))
            
            if self_errors.intersection(other_errors):
                return True
        
        return False


class BayesianBeliefNetwork:
    """Simple Bayesian belief network for outcome prediction"""
    
    def __init__(self):
        self.probabilities = defaultdict(lambda: defaultdict(float))
        self.evidence_counts = defaultdict(int)
    
    def update_beliefs(self, evidence: Dict[str, Any]):
        """Update beliefs based on evidence"""
        evidence_key = self._make_key(evidence)
        self.evidence_counts[evidence_key] += 1
        
        outcome = evidence.get("outcome", "UNKNOWN")
        self.probabilities[evidence_key][outcome] += 1
    
    def predict_outcome_probability(self, 
                                  evidence: Dict[str, Any],
                                  outcome: str) -> float:
        """Predict probability of an outcome given evidence"""
        evidence_key = self._make_key(evidence)
        
        if evidence_key not in self.probabilities:
            return 0.5  # No prior knowledge
        
        total_count = self.evidence_counts[evidence_key]
        outcome_count = self.probabilities[evidence_key].get(outcome, 0)
        
        if total_count == 0:
            return 0.5
        
        # Apply Laplace smoothing
        return (outcome_count + 1) / (total_count + 5)
    
    def _make_key(self, evidence: Dict[str, Any]) -> str:
        """Create a key from evidence"""
        # Only use key features
        key_features = ["state", "has_errors", "memory_high", "strategy"]
        key_parts = []
        
        for feature in key_features:
            if feature in evidence:
                key_parts.append(f"{feature}:{evidence[feature]}")
        
        return "|".join(key_parts)