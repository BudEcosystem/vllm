"""
Predictive Failure Detection and State Projection System.

This module implements advanced failure prediction using state graph analysis,
pattern recognition, and probabilistic modeling to identify potential failure
paths and recommend preventive actions.
"""

import time
import json
import math
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from .lifecycle_tracker import LifecycleState, StateCheckpoint, StateTransition
from .core import CircularBuffer, get_logger


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = auto()
    DEGRADED = auto()
    AT_RISK = auto()
    CRITICAL = auto()
    FAILING = auto()
    FAILED = auto()


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    UNCERTAIN = auto()


@dataclass
class StateNode:
    """Node in the state graph representing a system state"""
    state: LifecycleState
    health_status: HealthStatus
    metrics: Dict[str, float]
    error_patterns: List[str] = field(default_factory=list)
    
    # Graph properties
    transitions_to: Dict[LifecycleState, float] = field(default_factory=dict)  # state -> probability
    transitions_from: Dict[LifecycleState, float] = field(default_factory=dict)
    
    # Historical data
    visit_count: int = 0
    failure_count: int = 0
    recovery_count: int = 0
    average_duration: float = 0.0
    
    # Computed properties
    failure_probability: float = 0.0
    health_score: float = 1.0
    
    def update_failure_probability(self):
        """Update failure probability based on historical data"""
        if self.visit_count > 0:
            self.failure_probability = self.failure_count / self.visit_count
        
        # Adjust based on error patterns
        if self.error_patterns:
            self.failure_probability = min(1.0, self.failure_probability + 0.1 * len(self.error_patterns))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "state": self.state.name,
            "health_status": self.health_status.name,
            "metrics": self.metrics,
            "error_patterns": self.error_patterns,
            "visit_count": self.visit_count,
            "failure_count": self.failure_count,
            "recovery_count": self.recovery_count,
            "failure_probability": self.failure_probability,
            "health_score": self.health_score
        }


@dataclass
class FailurePattern:
    """Pattern that leads to failures"""
    pattern_id: str
    state_sequence: List[LifecycleState]
    error_sequence: List[str]
    metrics_pattern: Dict[str, Tuple[float, float]]  # metric -> (min, max)
    
    # Pattern properties
    occurrence_count: int = 0
    failure_rate: float = 0.0
    avg_time_to_failure: float = 0.0
    confidence: PredictionConfidence = PredictionConfidence.LOW
    
    # Mitigation
    successful_mitigations: List[str] = field(default_factory=list)
    failed_mitigations: List[str] = field(default_factory=list)
    
    def matches(self, 
                state_history: List[LifecycleState],
                error_history: List[str],
                current_metrics: Dict[str, float]) -> bool:
        """Check if current state matches this failure pattern"""
        # Check state sequence
        if len(state_history) >= len(self.state_sequence):
            recent_states = state_history[-len(self.state_sequence):]
            if recent_states != self.state_sequence:
                return False
        else:
            return False
        
        # Check error sequence
        if self.error_sequence:
            recent_errors = error_history[-len(self.error_sequence):]
            if recent_errors != self.error_sequence:
                return False
        
        # Check metrics pattern
        for metric, (min_val, max_val) in self.metrics_pattern.items():
            if metric in current_metrics:
                if not (min_val <= current_metrics[metric] <= max_val):
                    return False
        
        return True


@dataclass
class MitigationPath:
    """Path to mitigate a predicted failure"""
    path_id: str
    start_state: LifecycleState
    target_state: LifecycleState
    interventions: List[str]
    expected_duration: float
    success_probability: float
    side_effects: List[str] = field(default_factory=list)
    
    def calculate_score(self) -> float:
        """Calculate path score (higher is better)"""
        # Consider success probability, duration, and side effects
        score = self.success_probability * 100
        score -= self.expected_duration * 0.1  # Penalize longer paths
        score -= len(self.side_effects) * 5   # Penalize side effects
        return max(0, score)


@dataclass
class FailurePrediction:
    """Prediction of potential failure"""
    prediction_id: str
    timestamp: float
    current_state: LifecycleState
    predicted_failure_state: Optional[LifecycleState]
    failure_type: str
    time_to_failure: float  # Estimated seconds until failure
    probability: float
    confidence: PredictionConfidence
    contributing_factors: List[str]
    recommended_mitigations: List[MitigationPath]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "current_state": self.current_state.name,
            "predicted_failure_state": self.predicted_failure_state.name if self.predicted_failure_state else None,
            "failure_type": self.failure_type,
            "time_to_failure": self.time_to_failure,
            "probability": self.probability,
            "confidence": self.confidence.name,
            "contributing_factors": self.contributing_factors,
            "recommended_mitigations": [
                {
                    "path_id": path.path_id,
                    "interventions": path.interventions,
                    "success_probability": path.success_probability,
                    "score": path.calculate_score()
                }
                for path in self.recommended_mitigations
            ]
        }


class StateGraph:
    """Graph representation of system states and transitions"""
    
    def __init__(self):
        self.nodes: Dict[LifecycleState, StateNode] = {}
        self.edges: Dict[Tuple[LifecycleState, LifecycleState], float] = {}
        self._lock = threading.RLock()
        
        # Initialize with known states
        self._initialize_states()
    
    def _initialize_states(self):
        """Initialize graph with known lifecycle states"""
        # Define healthy state progression
        healthy_progression = [
            (LifecycleState.NOT_STARTED, LifecycleState.VALIDATING_ENVIRONMENT),
            (LifecycleState.VALIDATING_ENVIRONMENT, LifecycleState.CHECKING_DEPENDENCIES),
            (LifecycleState.CHECKING_DEPENDENCIES, LifecycleState.LOADING_CONFIGURATIONS),
            (LifecycleState.LOADING_CONFIGURATIONS, LifecycleState.INITIALIZING),
            (LifecycleState.INITIALIZING, LifecycleState.LOADING_MODEL),
            (LifecycleState.LOADING_MODEL, LifecycleState.ALLOCATING_MEMORY),
            (LifecycleState.ALLOCATING_MEMORY, LifecycleState.COMPILING_KERNELS),
            (LifecycleState.COMPILING_KERNELS, LifecycleState.SETTING_UP_WORKERS),
            (LifecycleState.SETTING_UP_WORKERS, LifecycleState.READY),
            (LifecycleState.READY, LifecycleState.SERVING),
            (LifecycleState.SERVING, LifecycleState.PROCESSING_REQUESTS),
        ]
        
        # Create nodes for healthy progression
        for state, next_state in healthy_progression:
            if state not in self.nodes:
                self.nodes[state] = StateNode(
                    state=state,
                    health_status=HealthStatus.HEALTHY,
                    metrics={}
                )
            if next_state not in self.nodes:
                self.nodes[next_state] = StateNode(
                    state=next_state,
                    health_status=HealthStatus.HEALTHY,
                    metrics={}
                )
            
            # Add edge
            self.add_transition(state, next_state, 0.95)  # High probability for healthy transitions
        
        # Add error states and transitions
        error_states = [LifecycleState.ERROR, LifecycleState.CRITICAL_ERROR, LifecycleState.UNRECOVERABLE]
        for error_state in error_states:
            self.nodes[error_state] = StateNode(
                state=error_state,
                health_status=HealthStatus.FAILED,
                metrics={}
            )
        
        # Any state can transition to error states
        for state in LifecycleState:
            if state not in error_states:
                self.add_transition(state, LifecycleState.ERROR, 0.05)
    
    def add_transition(self, from_state: LifecycleState, to_state: LifecycleState, probability: float):
        """Add or update a state transition"""
        with self._lock:
            # Ensure nodes exist
            if from_state not in self.nodes:
                self.nodes[from_state] = StateNode(state=from_state, health_status=HealthStatus.HEALTHY, metrics={})
            if to_state not in self.nodes:
                self.nodes[to_state] = StateNode(state=to_state, health_status=HealthStatus.HEALTHY, metrics={})
            
            # Update edges
            self.edges[(from_state, to_state)] = probability
            self.nodes[from_state].transitions_to[to_state] = probability
            self.nodes[to_state].transitions_from[from_state] = probability
    
    def update_node(self, state: LifecycleState, checkpoint: StateCheckpoint):
        """Update node with checkpoint data"""
        with self._lock:
            if state not in self.nodes:
                self.nodes[state] = StateNode(state=state, health_status=HealthStatus.HEALTHY, metrics={})
            
            node = self.nodes[state]
            node.visit_count += 1
            node.metrics = checkpoint.metrics
            
            # Update health status based on metrics
            node.health_status = self._compute_health_status(checkpoint)
            
            # Update failure probability
            if checkpoint.error_context:
                node.failure_count += 1
                node.error_patterns.extend(checkpoint.error_context.get("errors", []))
            
            node.update_failure_probability()
    
    def _compute_health_status(self, checkpoint: StateCheckpoint) -> HealthStatus:
        """Compute health status from checkpoint"""
        metrics = checkpoint.metrics
        
        # Check critical metrics
        if metrics.get("memory_percent", 0) > 95:
            return HealthStatus.CRITICAL
        elif metrics.get("memory_percent", 0) > 90:
            return HealthStatus.AT_RISK
        elif metrics.get("error_rate", 0) > 0.1:
            return HealthStatus.DEGRADED
        elif checkpoint.error_context:
            return HealthStatus.AT_RISK
        
        return HealthStatus.HEALTHY
    
    def find_paths_to_failure(self, 
                             current_state: LifecycleState,
                             max_depth: int = 5) -> List[List[LifecycleState]]:
        """Find potential paths from current state to failure states"""
        failure_states = {LifecycleState.ERROR, LifecycleState.CRITICAL_ERROR, 
                         LifecycleState.UNRECOVERABLE}
        paths = []
        
        def dfs(state: LifecycleState, path: List[LifecycleState], depth: int):
            if depth > max_depth:
                return
            
            if state in failure_states:
                paths.append(path.copy())
                return
            
            if state in self.nodes:
                for next_state, prob in self.nodes[state].transitions_to.items():
                    if prob > 0.01:  # Only consider transitions with >1% probability
                        path.append(next_state)
                        dfs(next_state, path, depth + 1)
                        path.pop()
        
        with self._lock:
            dfs(current_state, [current_state], 0)
        
        return paths
    
    def find_recovery_paths(self,
                           current_state: LifecycleState,
                           target_state: LifecycleState) -> List[List[LifecycleState]]:
        """Find paths from current state to target healthy state"""
        paths = []
        visited = set()
        
        def dfs(state: LifecycleState, path: List[LifecycleState]):
            if state == target_state:
                paths.append(path.copy())
                return
            
            if state in visited or len(path) > 10:
                return
            
            visited.add(state)
            
            if state in self.nodes:
                # Sort by probability (highest first)
                transitions = sorted(
                    self.nodes[state].transitions_to.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for next_state, prob in transitions:
                    if prob > 0.1:  # Consider transitions with >10% probability
                        path.append(next_state)
                        dfs(next_state, path)
                        path.pop()
            
            visited.remove(state)
        
        with self._lock:
            dfs(current_state, [current_state])
        
        return paths


class PredictiveFailureDetector:
    """
    Advanced failure detection system with predictive capabilities.
    
    Features:
    - State graph analysis
    - Pattern recognition
    - Probabilistic failure prediction
    - Mitigation path finding
    - Continuous learning
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 prediction_horizon: float = 300.0,  # 5 minutes
                 persistence_manager: Optional['PersistenceManager'] = None):
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # Configuration
        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        
        # Persistence manager
        self.persistence_manager = persistence_manager
        
        # State tracking
        self.state_graph = StateGraph()
        self.state_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        self.checkpoint_history = CircularBuffer(1000)
        
        # Pattern detection
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.pattern_matcher = PatternMatcher()
        
        # Predictions
        self.active_predictions: Dict[str, FailurePrediction] = {}
        self.prediction_history = CircularBuffer(1000)
        
        # Mitigation strategies
        self.mitigation_strategies: Dict[str, MitigationStrategy] = {}
        self.mitigation_history = CircularBuffer(500)
        
        # Learning
        self.learning_enabled = True
        self.model_path = Path("./failure_detector_model.pkl")
        
        # Metrics
        self.prediction_accuracy = 0.0
        self.false_positive_rate = 0.0
        self.true_positive_rate = 0.0
        
        # Background analysis
        self._analyzer_thread = None
        self._analyzing = False
        
        # Load learned patterns
        if self.persistence_manager:
            self._load_from_persistence()
        else:
            self._load_model()
        
        self.logger.info("Predictive failure detector initialized")
    
    def _load_from_persistence(self):
        """Load failure detection data from persistence"""
        try:
            # Load failure patterns from model
            try:
                patterns_model, metadata = self.persistence_manager.load_model("failure_patterns")
                self.failure_patterns = patterns_model.get('patterns', {})
                self.logger.info(f"Loaded {len(self.failure_patterns)} failure patterns")
            except:
                pass
            
            # Load state graph from model
            try:
                graph_model, metadata = self.persistence_manager.load_model("state_graph")
                if graph_model:
                    # Reconstruct state graph
                    self._restore_state_graph(graph_model)
                    self.logger.info("Loaded state graph from persistence")
            except:
                pass
            
            # Load historical predictions
            predictions = self._get_historical_predictions()
            
            for pred_data in predictions:
                # Reconstruct prediction object
                prediction = self._reconstruct_prediction(pred_data)
                if prediction:
                    self.prediction_history.append(prediction.to_dict())
            
            self.logger.info(f"Loaded {len(predictions)} historical predictions")
            
            # Load metrics from model
            try:
                metrics_model, metadata = self.persistence_manager.load_model("failure_detector_metrics")
                if metrics_model:
                    self.prediction_accuracy = metrics_model.get('prediction_accuracy', 0.0)
                    self.false_positive_rate = metrics_model.get('false_positive_rate', 0.0)
                    self.true_positive_rate = metrics_model.get('true_positive_rate', 0.0)
                    self.logger.info("Loaded failure detector metrics")
            except:
                pass
            
            # Load recent checkpoints to rebuild state
            checkpoints = self.persistence_manager.get_checkpoints(
                start_time=time.time() - 3600,  # Last hour
                limit=100
            )
            
            for checkpoint in checkpoints:
                self.state_history.append(checkpoint.state)
                self.checkpoint_history.append(checkpoint)
                # Update state graph
                self.state_graph.update_node(checkpoint.state, checkpoint)
            
            self.logger.info(f"Loaded {len(checkpoints)} recent checkpoints")
            
        except Exception as e:
            self.logger.error(f"Error loading from persistence: {e}")
            # Fall back to file-based loading
            self._load_model()
    
    def analyze_checkpoint(self, checkpoint: StateCheckpoint) -> List[FailurePrediction]:
        """Analyze a checkpoint for failure predictions"""
        with self._lock:
            # Update state graph
            self.state_graph.update_node(checkpoint.state, checkpoint)
            
            # Update history
            self.state_history.append(checkpoint.state)
            if checkpoint.error_context:
                self.error_history.extend(checkpoint.error_context.get("errors", []))
            self.checkpoint_history.append(checkpoint)
            
            # Detect patterns
            patterns = self._detect_failure_patterns(checkpoint)
            
            # Generate predictions
            predictions = []
            
            # Pattern-based predictions
            for pattern in patterns:
                prediction = self._generate_prediction_from_pattern(checkpoint, pattern)
                if prediction:
                    predictions.append(prediction)
                    self.active_predictions[prediction.prediction_id] = prediction
            
            # Graph-based predictions
            graph_predictions = self._analyze_state_graph(checkpoint)
            predictions.extend(graph_predictions)
            
            # Anomaly-based predictions
            anomaly_predictions = self._detect_anomalies(checkpoint)
            predictions.extend(anomaly_predictions)
            
            # Store predictions
            for pred in predictions:
                self.prediction_history.append(pred.to_dict())
                
                # Persist prediction if enabled
                if self.persistence_manager:
                    self.persistence_manager.save_prediction(pred)
            
            # Learn from feedback
            if self.learning_enabled:
                self._update_learning(checkpoint, predictions)
            
            return predictions
    
    def _detect_failure_patterns(self, checkpoint: StateCheckpoint) -> List[FailurePattern]:
        """Detect matching failure patterns"""
        matching_patterns = []
        
        current_metrics = checkpoint.metrics
        state_list = list(self.state_history)
        error_list = list(self.error_history)
        
        for pattern in self.failure_patterns.values():
            if pattern.matches(state_list, error_list, current_metrics):
                matching_patterns.append(pattern)
        
        return matching_patterns
    
    def _generate_prediction_from_pattern(self, 
                                        checkpoint: StateCheckpoint,
                                        pattern: FailurePattern) -> Optional[FailurePrediction]:
        """Generate prediction from a matched pattern"""
        # Calculate time to failure based on pattern
        time_to_failure = pattern.avg_time_to_failure * (1 - pattern.failure_rate)
        
        # Find mitigation paths
        mitigations = self._find_mitigation_paths(
            checkpoint.state,
            pattern,
            checkpoint.metrics
        )
        
        # Create prediction
        prediction = FailurePrediction(
            prediction_id=f"pred_{int(time.time() * 1000)}_{pattern.pattern_id}",
            timestamp=time.time(),
            current_state=checkpoint.state,
            predicted_failure_state=None,  # Pattern doesn't specify exact failure state
            failure_type=f"Pattern: {pattern.pattern_id}",
            time_to_failure=time_to_failure,
            probability=pattern.failure_rate,
            confidence=pattern.confidence,
            contributing_factors=[
                f"State sequence: {' -> '.join(s.name for s in pattern.state_sequence[-3:])}",
                f"Error patterns: {len(pattern.error_sequence)} errors detected",
                f"Occurrence count: {pattern.occurrence_count}"
            ],
            recommended_mitigations=mitigations
        )
        
        return prediction
    
    def _analyze_state_graph(self, checkpoint: StateCheckpoint) -> List[FailurePrediction]:
        """Analyze state graph for failure paths"""
        predictions = []
        
        # Find paths to failure states
        failure_paths = self.state_graph.find_paths_to_failure(checkpoint.state)
        
        for path in failure_paths:
            if len(path) < 2:
                continue
            
            # Calculate path probability
            path_probability = 1.0
            for i in range(len(path) - 1):
                edge_prob = self.state_graph.edges.get((path[i], path[i + 1]), 0.0)
                path_probability *= edge_prob
            
            if path_probability > 0.01:  # Only consider paths with >1% probability
                # Estimate time to failure
                time_to_failure = len(path) * 30  # Assume 30 seconds per transition
                
                # Find mitigations
                mitigations = self._find_graph_based_mitigations(
                    checkpoint.state,
                    path[-1]  # Target failure state
                )
                
                prediction = FailurePrediction(
                    prediction_id=f"pred_graph_{int(time.time() * 1000)}",
                    timestamp=time.time(),
                    current_state=checkpoint.state,
                    predicted_failure_state=path[-1],
                    failure_type="State progression to failure",
                    time_to_failure=time_to_failure,
                    probability=path_probability,
                    confidence=self._calculate_confidence(path_probability),
                    contributing_factors=[
                        f"Path: {' -> '.join(s.name for s in path[:3])}...",
                        f"Path length: {len(path)} transitions",
                        f"Current health: {self.state_graph.nodes[checkpoint.state].health_status.name}"
                    ],
                    recommended_mitigations=mitigations
                )
                
                predictions.append(prediction)
        
        return predictions
    
    def _detect_anomalies(self, checkpoint: StateCheckpoint) -> List[FailurePrediction]:
        """Detect anomalies that might lead to failures"""
        predictions = []
        
        # Check for metric anomalies
        anomalies = []
        
        # Memory anomaly
        if checkpoint.metrics.get("memory_percent", 0) > 85:
            anomalies.append({
                "type": "high_memory_usage",
                "severity": "critical" if checkpoint.metrics["memory_percent"] > 95 else "warning",
                "value": checkpoint.metrics["memory_percent"]
            })
        
        # Error rate anomaly
        if checkpoint.metrics.get("error_rate", 0) > 0.05:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "warning",
                "value": checkpoint.metrics["error_rate"]
            })
        
        # GPU temperature anomaly
        if checkpoint.metrics.get("gpu_temperature", 0) > 80:
            anomalies.append({
                "type": "gpu_overheating",
                "severity": "critical" if checkpoint.metrics["gpu_temperature"] > 85 else "warning",
                "value": checkpoint.metrics["gpu_temperature"]
            })
        
        # Create predictions for anomalies
        for anomaly in anomalies:
            if anomaly["severity"] == "critical":
                time_to_failure = 60  # 1 minute
                probability = 0.8
            else:
                time_to_failure = 300  # 5 minutes
                probability = 0.3
            
            prediction = FailurePrediction(
                prediction_id=f"pred_anomaly_{anomaly['type']}_{int(time.time() * 1000)}",
                timestamp=time.time(),
                current_state=checkpoint.state,
                predicted_failure_state=LifecycleState.ERROR,
                failure_type=f"Anomaly: {anomaly['type']}",
                time_to_failure=time_to_failure,
                probability=probability,
                confidence=PredictionConfidence.HIGH if anomaly["severity"] == "critical" else PredictionConfidence.MEDIUM,
                contributing_factors=[
                    f"{anomaly['type']}: {anomaly['value']:.2f}",
                    f"Severity: {anomaly['severity']}"
                ],
                recommended_mitigations=self._get_anomaly_mitigations(anomaly)
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _find_mitigation_paths(self,
                              current_state: LifecycleState,
                              pattern: FailurePattern,
                              metrics: Dict[str, float]) -> List[MitigationPath]:
        """Find mitigation paths for a failure pattern"""
        paths = []
        
        # Get successful mitigations from pattern history
        for mitigation_name in pattern.successful_mitigations:
            if mitigation_name in self.mitigation_strategies:
                strategy = self.mitigation_strategies[mitigation_name]
                
                # Calculate success probability based on history
                total_attempts = len(pattern.successful_mitigations) + len(pattern.failed_mitigations)
                success_count = pattern.successful_mitigations.count(mitigation_name)
                success_prob = success_count / total_attempts if total_attempts > 0 else 0.5
                
                path = MitigationPath(
                    path_id=f"path_{mitigation_name}_{pattern.pattern_id}",
                    start_state=current_state,
                    target_state=LifecycleState.READY,  # Assume we want to get to READY state
                    interventions=[mitigation_name],
                    expected_duration=strategy.expected_duration,
                    success_probability=success_prob,
                    side_effects=strategy.side_effects
                )
                
                paths.append(path)
        
        # Sort by score
        paths.sort(key=lambda p: p.calculate_score(), reverse=True)
        
        return paths[:3]  # Return top 3 paths
    
    def _find_graph_based_mitigations(self,
                                    current_state: LifecycleState,
                                    failure_state: LifecycleState) -> List[MitigationPath]:
        """Find mitigations based on state graph analysis"""
        paths = []
        
        # Find recovery paths to healthy states
        healthy_states = [LifecycleState.READY, LifecycleState.SERVING]
        
        for target_state in healthy_states:
            recovery_paths = self.state_graph.find_recovery_paths(current_state, target_state)
            
            for path in recovery_paths[:3]:  # Consider top 3 paths
                # Map states to interventions
                interventions = self._map_path_to_interventions(path)
                
                if interventions:
                    # Calculate path probability
                    path_probability = 1.0
                    for i in range(len(path) - 1):
                        edge_prob = self.state_graph.edges.get((path[i], path[i + 1]), 0.5)
                        path_probability *= edge_prob
                    
                    mitigation_path = MitigationPath(
                        path_id=f"graph_path_{current_state.name}_to_{target_state.name}",
                        start_state=current_state,
                        target_state=target_state,
                        interventions=interventions,
                        expected_duration=len(interventions) * 10,  # 10 seconds per intervention
                        success_probability=path_probability,
                        side_effects=[]
                    )
                    
                    paths.append(mitigation_path)
        
        return paths
    
    def _map_path_to_interventions(self, path: List[LifecycleState]) -> List[str]:
        """Map a state path to required interventions"""
        interventions = []
        
        # Define state transition interventions
        transition_interventions = {
            (LifecycleState.ERROR, LifecycleState.RECOVERING): ["clear_errors", "reset_components"],
            (LifecycleState.RECOVERING, LifecycleState.INITIALIZING): ["reinitialize_system"],
            (LifecycleState.PAUSED, LifecycleState.READY): ["resume_operations"],
            (LifecycleState.SHUTTING_DOWN, LifecycleState.READY): ["abort_shutdown", "stabilize_system"],
        }
        
        for i in range(len(path) - 1):
            transition = (path[i], path[i + 1])
            if transition in transition_interventions:
                interventions.extend(transition_interventions[transition])
        
        return interventions
    
    def _get_anomaly_mitigations(self, anomaly: Dict[str, Any]) -> List[MitigationPath]:
        """Get mitigations for specific anomalies"""
        mitigations = []
        
        anomaly_interventions = {
            "high_memory_usage": ["cleanup_memory", "reduce_batch_size", "offload_to_cpu"],
            "high_error_rate": ["reduce_load", "restart_workers", "enable_error_recovery"],
            "gpu_overheating": ["throttle_gpu", "reduce_compute_load", "improve_cooling"]
        }
        
        if anomaly["type"] in anomaly_interventions:
            interventions = anomaly_interventions[anomaly["type"]]
            
            for i, intervention in enumerate(interventions):
                path = MitigationPath(
                    path_id=f"anomaly_{anomaly['type']}_{intervention}",
                    start_state=LifecycleState.SERVING,  # Assume we're serving
                    target_state=LifecycleState.SERVING,  # Stay in serving
                    interventions=[intervention],
                    expected_duration=5.0 + i * 5,  # Stagger interventions
                    success_probability=0.9 - i * 0.2,  # First intervention most likely
                    side_effects=["temporary_performance_degradation"] if i > 0 else []
                )
                mitigations.append(path)
        
        return mitigations
    
    def _calculate_confidence(self, probability: float) -> PredictionConfidence:
        """Calculate confidence level from probability"""
        if probability > 0.8:
            return PredictionConfidence.HIGH
        elif probability > 0.5:
            return PredictionConfidence.MEDIUM
        elif probability > 0.2:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.UNCERTAIN
    
    def _update_learning(self, checkpoint: StateCheckpoint, predictions: List[FailurePrediction]):
        """Update learning based on checkpoint and predictions"""
        # This would implement online learning algorithms
        # For now, we just update pattern statistics
        
        # Check if any previous predictions came true
        for pred_id, prediction in list(self.active_predictions.items()):
            if time.time() - prediction.timestamp > prediction.time_to_failure:
                # Check if failure occurred
                if checkpoint.state in [LifecycleState.ERROR, LifecycleState.CRITICAL_ERROR]:
                    # True positive
                    self.true_positive_rate = (self.true_positive_rate * 0.9 + 0.1)
                else:
                    # False positive
                    self.false_positive_rate = (self.false_positive_rate * 0.9 + 0.1)
                
                # Remove from active predictions
                del self.active_predictions[pred_id]
    
    def register_mitigation_strategy(self, strategy: 'MitigationStrategy'):
        """Register a new mitigation strategy"""
        with self._lock:
            self.mitigation_strategies[strategy.name] = strategy
            self.logger.info(f"Registered mitigation strategy: {strategy.name}")
    
    def record_mitigation_outcome(self,
                                 pattern_id: str,
                                 mitigation_name: str,
                                 success: bool):
        """Record the outcome of a mitigation attempt"""
        with self._lock:
            if pattern_id in self.failure_patterns:
                pattern = self.failure_patterns[pattern_id]
                if success:
                    pattern.successful_mitigations.append(mitigation_name)
                else:
                    pattern.failed_mitigations.append(mitigation_name)
                
                # Update pattern confidence based on feedback
                total = len(pattern.successful_mitigations) + len(pattern.failed_mitigations)
                if total > 10:
                    pattern.confidence = PredictionConfidence.HIGH
                elif total > 5:
                    pattern.confidence = PredictionConfidence.MEDIUM
    
    def learn_pattern(self, 
                     state_sequence: List[LifecycleState],
                     error_sequence: List[str],
                     metrics_sequence: List[Dict[str, float]],
                     resulted_in_failure: bool):
        """Learn a new failure pattern"""
        if not resulted_in_failure:
            return
        
        # Create pattern ID
        pattern_hash = hashlib.md5(
            f"{state_sequence}{error_sequence}".encode()
        ).hexdigest()[:8]
        pattern_id = f"learned_{pattern_hash}"
        
        # Calculate metrics pattern (min/max for each metric)
        metrics_pattern = {}
        for metrics in metrics_sequence:
            for key, value in metrics.items():
                if key not in metrics_pattern:
                    metrics_pattern[key] = (value, value)
                else:
                    min_val, max_val = metrics_pattern[key]
                    metrics_pattern[key] = (min(min_val, value), max(max_val, value))
        
        # Create or update pattern
        if pattern_id in self.failure_patterns:
            pattern = self.failure_patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.failure_rate = pattern.occurrence_count / (pattern.occurrence_count + 1)
        else:
            pattern = FailurePattern(
                pattern_id=pattern_id,
                state_sequence=state_sequence[-5:],  # Last 5 states
                error_sequence=error_sequence[-3:],   # Last 3 errors
                metrics_pattern=metrics_pattern,
                occurrence_count=1,
                failure_rate=1.0,
                avg_time_to_failure=len(state_sequence) * 30  # Rough estimate
            )
            self.failure_patterns[pattern_id] = pattern
        
        self.logger.info(f"Learned failure pattern: {pattern_id}")
    
    def start_background_analysis(self):
        """Start background analysis thread"""
        if not self._analyzing:
            self._analyzing = True
            self._analyzer_thread = threading.Thread(
                target=self._background_analysis_loop,
                daemon=True
            )
            self._analyzer_thread.start()
    
    def stop_background_analysis(self):
        """Stop background analysis"""
        self._analyzing = False
        if self._analyzer_thread:
            self._analyzer_thread.join(timeout=5)
    
    def _background_analysis_loop(self):
        """Background analysis loop"""
        while self._analyzing:
            try:
                # Periodic model saving
                if len(self.failure_patterns) > 0:
                    self._save_model()
                
                # Clean up old predictions
                current_time = time.time()
                for pred_id in list(self.active_predictions.keys()):
                    if current_time - self.active_predictions[pred_id].timestamp > 3600:  # 1 hour
                        del self.active_predictions[pred_id]
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Background analysis error: {e}")
    
    def _save_model(self):
        """Save learned patterns and model"""
        try:
            # Save to persistence manager if available
            if self.persistence_manager:
                # Save failure patterns
                self.persistence_manager.save_model(
                    "failure_patterns",
                    {"patterns": self.failure_patterns},
                    metadata={
                        "pattern_count": len(self.failure_patterns),
                        "last_updated": time.time()
                    }
                )
                
                # Save state graph
                graph_data = {
                    "nodes": {state.name: node.to_dict() 
                             for state, node in self.state_graph.nodes.items()},
                    "edges": {f"{s1.name}->{s2.name}": prob 
                             for (s1, s2), prob in self.state_graph.edges.items()}
                }
                self.persistence_manager.save_model(
                    "state_graph",
                    graph_data,
                    metadata={
                        "node_count": len(self.state_graph.nodes),
                        "edge_count": len(self.state_graph.edges)
                    }
                )
                
                # Save metrics
                self.persistence_manager.save_model(
                    "failure_detector_metrics",
                    {
                        "prediction_accuracy": self.prediction_accuracy,
                        "false_positive_rate": self.false_positive_rate,
                        "true_positive_rate": self.true_positive_rate
                    },
                    metadata={
                        "last_updated": time.time(),
                        "total_predictions": len(self.prediction_history._buffer)
                    }
                )
            else:
                # Fall back to file-based storage
                model_data = {
                    "failure_patterns": self.failure_patterns,
                    "state_graph": {
                        "nodes": {state.name: node.to_dict() 
                                 for state, node in self.state_graph.nodes.items()},
                        "edges": {f"{s1.name}->{s2.name}": prob 
                                 for (s1, s2), prob in self.state_graph.edges.items()}
                    },
                    "metrics": {
                        "prediction_accuracy": self.prediction_accuracy,
                        "false_positive_rate": self.false_positive_rate,
                        "true_positive_rate": self.true_positive_rate
                    }
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load learned patterns and model"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.failure_patterns = model_data.get("failure_patterns", {})
                
                # Restore metrics
                metrics = model_data.get("metrics", {})
                self.prediction_accuracy = metrics.get("prediction_accuracy", 0.0)
                self.false_positive_rate = metrics.get("false_positive_rate", 0.0)
                self.true_positive_rate = metrics.get("true_positive_rate", 0.0)
                
                self.logger.info(f"Loaded {len(self.failure_patterns)} failure patterns")
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
    
    def get_prediction_report(self) -> Dict[str, Any]:
        """Get comprehensive prediction report"""
        with self._lock:
            return {
                "active_predictions": len(self.active_predictions),
                "learned_patterns": len(self.failure_patterns),
                "prediction_metrics": {
                    "accuracy": self.prediction_accuracy,
                    "false_positive_rate": self.false_positive_rate,
                    "true_positive_rate": self.true_positive_rate
                },
                "state_graph": {
                    "total_states": len(self.state_graph.nodes),
                    "total_transitions": len(self.state_graph.edges),
                    "high_risk_states": [
                        state.name for state, node in self.state_graph.nodes.items()
                        if node.failure_probability > 0.3
                    ]
                },
                "recent_predictions": [
                    pred.to_dict() for pred in list(self.active_predictions.values())[:5]
                ]
            }
    
    def _restore_state_graph(self, graph_data: Dict[str, Any]):
        """Restore state graph from persisted data"""
        # Clear existing graph
        self.state_graph = StateGraph()
        
        # Restore nodes
        nodes = graph_data.get("nodes", {})
        for state_name, node_data in nodes.items():
            state = LifecycleState[state_name]
            node = StateNode(
                state=state,
                health_status=HealthStatus[node_data["health_status"]],
                metrics=node_data["metrics"],
                error_patterns=node_data.get("error_patterns", []),
                visit_count=node_data.get("visit_count", 0),
                failure_count=node_data.get("failure_count", 0),
                recovery_count=node_data.get("recovery_count", 0),
                failure_probability=node_data.get("failure_probability", 0.0),
                health_score=node_data.get("health_score", 1.0)
            )
            self.state_graph.nodes[state] = node
        
        # Restore edges
        edges = graph_data.get("edges", {})
        for edge_key, probability in edges.items():
            states = edge_key.split("->")
            if len(states) == 2:
                from_state = LifecycleState[states[0]]
                to_state = LifecycleState[states[1]]
                self.state_graph.add_transition(from_state, to_state, probability)
    
    def _get_historical_predictions(self) -> List[Dict[str, Any]]:
        """Get historical predictions from persistence"""
        # This would be implemented based on the persistence layer's API
        # For now, return empty list
        return []
    
    def _reconstruct_prediction(self, pred_data: Dict[str, Any]) -> Optional[FailurePrediction]:
        """Reconstruct a FailurePrediction from persisted data"""
        try:
            prediction = FailurePrediction(
                prediction_id=pred_data.get('prediction_id', ''),
                timestamp=pred_data.get('timestamp', 0),
                current_state=LifecycleState[pred_data.get('current_state', 'NOT_STARTED')],
                predicted_failure_state=LifecycleState[pred_data['predicted_failure_state']] 
                    if pred_data.get('predicted_failure_state') else None,
                failure_type=pred_data.get('failure_type', ''),
                time_to_failure=pred_data.get('time_to_failure', 0),
                probability=pred_data.get('probability', 0),
                confidence=PredictionConfidence[pred_data.get('confidence', 'LOW')],
                contributing_factors=pred_data.get('contributing_factors', []),
                recommended_mitigations=[]  # Would need to reconstruct these as well
            )
            return prediction
        except Exception as e:
            self.logger.error(f"Failed to reconstruct prediction: {e}")
            return None
    
    def persist_patterns(self):
        """Persist current failure patterns"""
        if self.persistence_manager:
            # Save patterns
            self.persistence_manager.save_model(
                "failure_patterns",
                {"patterns": self.failure_patterns},
                metadata={
                    "pattern_count": len(self.failure_patterns),
                    "last_updated": time.time()
                }
            )


class PatternMatcher:
    """Pattern matching engine for failure detection"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def match_patterns(self,
                      state_history: List[LifecycleState],
                      error_history: List[str],
                      metrics: Dict[str, float],
                      patterns: Dict[str, FailurePattern]) -> List[FailurePattern]:
        """Match current state against known patterns"""
        matches = []
        
        for pattern in patterns.values():
            if pattern.matches(state_history, error_history, metrics):
                matches.append(pattern)
        
        return matches
    
    def extract_pattern(self,
                       checkpoints: List[StateCheckpoint],
                       failure_occurred: bool) -> Optional[FailurePattern]:
        """Extract a pattern from checkpoint sequence"""
        if len(checkpoints) < 2 or not failure_occurred:
            return None
        
        # Extract sequences
        state_sequence = [cp.state for cp in checkpoints]
        error_sequence = []
        metrics_pattern = {}
        
        for cp in checkpoints:
            if cp.error_context:
                error_sequence.extend(cp.error_context.get("errors", []))
            
            # Update metrics pattern
            for key, value in cp.metrics.items():
                if key not in metrics_pattern:
                    metrics_pattern[key] = (value, value)
                else:
                    min_val, max_val = metrics_pattern[key]
                    metrics_pattern[key] = (min(min_val, value), max(max_val, value))
        
        # Create pattern
        pattern_id = f"extracted_{int(time.time() * 1000)}"
        
        return FailurePattern(
            pattern_id=pattern_id,
            state_sequence=state_sequence,
            error_sequence=error_sequence,
            metrics_pattern=metrics_pattern,
            occurrence_count=1,
            failure_rate=1.0 if failure_occurred else 0.0,
            avg_time_to_failure=(checkpoints[-1].timestamp - checkpoints[0].timestamp)
        )


@dataclass
class MitigationStrategy:
    """Mitigation strategy definition"""
    name: str
    description: str
    applicable_states: List[LifecycleState]
    applicable_errors: List[str]
    interventions: List[str]
    expected_duration: float
    success_rate: float
    side_effects: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def is_applicable(self, state: LifecycleState, errors: List[str]) -> bool:
        """Check if strategy is applicable to current situation"""
        state_match = not self.applicable_states or state in self.applicable_states
        error_match = not self.applicable_errors or any(
            error in errors for error in self.applicable_errors
        )
        return state_match and error_match