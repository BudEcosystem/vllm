"""
Advanced analyzers for vLLM monitoring data.

These analyzers provide anomaly detection, performance analysis, failure prediction,
and health scoring capabilities with statistical and ML-based approaches.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from scipy import stats
from dataclasses import dataclass
import pickle
import os

from .core import (
    ComponentState, ComponentType, StateType, Analyzer, AlertLevel,
    PerformanceTimer
)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    component_id: str
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    description: str
    severity: AlertLevel
    contributing_factors: List[str]


@dataclass
class PerformanceInsight:
    """Performance analysis insight."""
    metric_name: str
    current_value: float
    baseline_value: float
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0.0 to 1.0
    recommendation: str
    impact_level: AlertLevel


@dataclass
class FailurePrediction:
    """Failure prediction result."""
    component_id: str
    failure_probability: float
    time_to_failure_hours: Optional[float]
    failure_type: str
    confidence: float
    warning_signs: List[str]
    recommended_actions: List[str]


class StatisticalAnalyzer:
    """Statistical analysis utilities for monitoring data."""
    
    def __init__(self, window_size: int = 100, confidence_level: float = 0.95):
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.z_threshold = stats.norm.ppf((1 + confidence_level) / 2)
    
    def detect_outliers_zscore(self, values: List[float], threshold: float = None) -> List[bool]:
        """Detect outliers using Z-score method."""
        if len(values) < 3:
            return [False] * len(values)
        
        threshold = threshold or self.z_threshold
        values_array = np.array(values)
        z_scores = np.abs(stats.zscore(values_array, nan_policy='omit'))
        return (z_scores > threshold).tolist()
    
    def detect_outliers_iqr(self, values: List[float], multiplier: float = 1.5) -> List[bool]:
        """Detect outliers using Interquartile Range method."""
        if len(values) < 4:
            return [False] * len(values)
        
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return [v < lower_bound or v > upper_bound for v in values]
    
    def detect_trend(self, values: List[float], timestamps: List[float]) -> Tuple[str, float]:
        """Detect trend in time series data."""
        if len(values) < 3:
            return 'stable', 0.0
        
        # Use linear regression to detect trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
        
        # Determine trend direction and strength
        if abs(r_value) < 0.1:
            return 'stable', abs(r_value)
        elif slope > 0:
            return 'improving' if r_value > 0 else 'degrading', abs(r_value)
        else:
            return 'degrading' if r_value > 0 else 'improving', abs(r_value)
    
    def calculate_control_limits(self, values: List[float]) -> Tuple[float, float, float]:
        """Calculate statistical control limits."""
        if len(values) < 10:
            return 0.0, 0.0, 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ucl = mean_val + 3 * std_val  # Upper Control Limit
        lcl = mean_val - 3 * std_val  # Lower Control Limit
        
        return mean_val, ucl, lcl


class AnomalyDetector(Analyzer):
    """Advanced anomaly detection for vLLM components."""
    
    def __init__(self, sensitivity: float = 2.0, min_samples: int = 10):
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.component_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.statistical_analyzer = StatisticalAnalyzer()
        self._lock = threading.RLock()
        
        # Initialize ML models if available
        self.isolation_forests: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
    def analyze(self, states: List[ComponentState]) -> Dict[str, Any]:
        """Analyze states for anomalies."""
        with PerformanceTimer() as timer:
            anomalies = []
            
            # Group states by component
            component_states = defaultdict(list)
            for state in states:
                component_states[state.component_id].append(state)
            
            # Analyze each component
            for component_id, comp_states in component_states.items():
                component_anomalies = self._analyze_component(component_id, comp_states)
                anomalies.extend(component_anomalies)
            
            # Update baseline statistics
            self._update_baselines(states)
            
            # Determine overall alert level
            alert_level = AlertLevel.INFO
            if anomalies:
                max_severity = max(a.severity for a in anomalies)
                alert_level = max_severity
            
            return {
                'type': 'anomaly_detection',
                'timestamp': time.time(),
                'anomalies': [self._anomaly_to_dict(a) for a in anomalies],
                'anomaly_count': len(anomalies),
                'analysis_time_us': timer.elapsed_us,
                'alert_level': alert_level.value,
                'requires_intervention': any(a.severity.value >= AlertLevel.ERROR.value for a in anomalies),
                'message': f"Detected {len(anomalies)} anomalies" if anomalies else "No anomalies detected"
            }
    
    def _analyze_component(self, component_id: str, states: List[ComponentState]) -> List[AnomalyResult]:
        """Analyze a single component for anomalies."""
        anomalies = []
        
        with self._lock:
            # Add states to history
            for state in states:
                self.component_histories[component_id].append(state)
            
            history = list(self.component_histories[component_id])
            
        if len(history) < self.min_samples:
            return anomalies
        
        # Extract numerical features for analysis
        features = self._extract_features(history)
        if not features:
            return anomalies
        
        # Statistical anomaly detection
        stat_anomalies = self._detect_statistical_anomalies(component_id, history, features)
        anomalies.extend(stat_anomalies)
        
        # ML-based anomaly detection if available
        if HAS_SKLEARN:
            ml_anomalies = self._detect_ml_anomalies(component_id, history, features)
            anomalies.extend(ml_anomalies)
        
        # Health score anomalies
        health_anomalies = self._detect_health_anomalies(component_id, history)
        anomalies.extend(health_anomalies)
        
        return anomalies
    
    def _extract_features(self, states: List[ComponentState]) -> Dict[str, List[float]]:
        """Extract numerical features from component states."""
        features = defaultdict(list)
        
        for state in states:
            # Core metrics
            if state.cpu_usage is not None:
                features['cpu_usage'].append(state.cpu_usage)
            if state.memory_usage is not None:
                features['memory_usage'].append(state.memory_usage)
            if state.gpu_usage is not None:
                features['gpu_usage'].append(state.gpu_usage)
            
            features['health_score'].append(state.health_score)
            features['error_count'].append(state.error_count)
            features['requests_processed'].append(state.requests_processed)
            features['average_latency_ms'].append(state.average_latency_ms)
            features['throughput_rps'].append(state.throughput_rps)
            
            # Extract data metrics
            for key, value in state.data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    features[f"data_{key}"].append(float(value))
        
        # Remove features with insufficient data
        return {k: v for k, v in features.items() if len(v) >= self.min_samples}
    
    def _detect_statistical_anomalies(self, 
                                    component_id: str, 
                                    states: List[ComponentState], 
                                    features: Dict[str, List[float]]) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        current_time = time.time()
        
        for feature_name, values in features.items():
            if len(values) < self.min_samples:
                continue
            
            # Z-score based detection
            outliers_z = self.statistical_analyzer.detect_outliers_zscore(values, self.sensitivity)
            
            # IQR based detection
            outliers_iqr = self.statistical_analyzer.detect_outliers_iqr(values)
            
            # Check recent values for anomalies
            recent_count = min(5, len(values))
            recent_outliers_z = outliers_z[-recent_count:]
            recent_outliers_iqr = outliers_iqr[-recent_count:]
            
            if any(recent_outliers_z) or any(recent_outliers_iqr):
                current_value = values[-1]
                mean_value = np.mean(values[:-recent_count]) if len(values) > recent_count else np.mean(values)
                
                # Calculate anomaly score
                z_score = abs((current_value - mean_value) / max(np.std(values), 1e-6))
                anomaly_score = min(z_score / self.sensitivity, 1.0)
                
                # Determine severity
                if anomaly_score > 0.8:
                    severity = AlertLevel.CRITICAL
                elif anomaly_score > 0.6:
                    severity = AlertLevel.ERROR
                elif anomaly_score > 0.4:
                    severity = AlertLevel.WARNING
                else:
                    severity = AlertLevel.INFO
                
                anomaly = AnomalyResult(
                    component_id=component_id,
                    timestamp=current_time,
                    anomaly_score=anomaly_score,
                    is_anomaly=True,
                    confidence=min(anomaly_score * 0.8, 0.95),
                    description=f"Statistical anomaly in {feature_name}: {current_value:.2f} (baseline: {mean_value:.2f})",
                    severity=severity,
                    contributing_factors=[f"{feature_name}_outlier"]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_ml_anomalies(self, 
                           component_id: str, 
                           states: List[ComponentState], 
                           features: Dict[str, List[float]]) -> List[AnomalyResult]:
        """Detect anomalies using ML methods."""
        if not HAS_SKLEARN:
            return []
        
        anomalies = []
        current_time = time.time()
        
        # Prepare feature matrix
        feature_names = list(features.keys())
        if len(feature_names) < 2:
            return anomalies
        
        # Create feature matrix (samples x features)
        min_length = min(len(values) for values in features.values())
        feature_matrix = np.array([
            features[name][-min_length:] for name in feature_names
        ]).T
        
        if feature_matrix.shape[0] < self.min_samples:
            return anomalies
        
        # Get or create isolation forest for this component
        model_key = f"{component_id}_isolation"
        if model_key not in self.isolation_forests:
            self.isolation_forests[model_key] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=50
            )
            self.scalers[model_key] = StandardScaler()
        
        # Scale features
        scaler = self.scalers[model_key]
        if hasattr(scaler, 'mean_'):
            # Transform using existing scaler
            scaled_features = scaler.transform(feature_matrix)
        else:
            # Fit and transform
            scaled_features = scaler.fit_transform(feature_matrix)
        
        # Train/update model
        model = self.isolation_forests[model_key]
        model.fit(scaled_features)
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(scaled_features)
        anomaly_labels = model.predict(scaled_features)
        
        # Check recent samples for anomalies
        recent_count = min(3, len(anomaly_labels))
        recent_labels = anomaly_labels[-recent_count:]
        recent_scores = anomaly_scores[-recent_count:]
        
        if any(label == -1 for label in recent_labels):
            # Found anomaly in recent data
            anomaly_score = abs(min(recent_scores))  # More negative = more anomalous
            
            # Determine severity based on score
            if anomaly_score > 0.7:
                severity = AlertLevel.CRITICAL
            elif anomaly_score > 0.5:
                severity = AlertLevel.ERROR
            elif anomaly_score > 0.3:
                severity = AlertLevel.WARNING
            else:
                severity = AlertLevel.INFO
            
            anomaly = AnomalyResult(
                component_id=component_id,
                timestamp=current_time,
                anomaly_score=anomaly_score,
                is_anomaly=True,
                confidence=anomaly_score * 0.9,
                description=f"ML-detected anomaly in component behavior (score: {anomaly_score:.3f})",
                severity=severity,
                contributing_factors=["multivariate_anomaly"]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_health_anomalies(self, component_id: str, states: List[ComponentState]) -> List[AnomalyResult]:
        """Detect anomalies in health scores."""
        anomalies = []
        current_time = time.time()
        
        health_scores = [state.health_score for state in states]
        if len(health_scores) < self.min_samples:
            return anomalies
        
        # Check for sudden health degradation
        recent_scores = health_scores[-5:]
        baseline_score = np.mean(health_scores[:-5]) if len(health_scores) > 5 else np.mean(health_scores)
        current_score = recent_scores[-1]
        
        health_drop = baseline_score - current_score
        if health_drop > 0.3:  # Significant health degradation
            severity = AlertLevel.CRITICAL if health_drop > 0.7 else AlertLevel.ERROR
            
            anomaly = AnomalyResult(
                component_id=component_id,
                timestamp=current_time,
                anomaly_score=health_drop,
                is_anomaly=True,
                confidence=0.9,
                description=f"Health score degradation: {current_score:.2f} (baseline: {baseline_score:.2f})",
                severity=severity,
                contributing_factors=["health_degradation"]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _update_baselines(self, states: List[ComponentState]) -> None:
        """Update baseline statistics for components."""
        with self._lock:
            for state in states:
                if state.component_id not in self.baseline_stats:
                    self.baseline_stats[state.component_id] = {}
                
                baseline = self.baseline_stats[state.component_id]
                
                # Update moving averages
                alpha = 0.1  # Smoothing factor
                if 'health_score' in baseline:
                    baseline['health_score'] = alpha * state.health_score + (1 - alpha) * baseline['health_score']
                else:
                    baseline['health_score'] = state.health_score
                
                if state.cpu_usage is not None:
                    if 'cpu_usage' in baseline:
                        baseline['cpu_usage'] = alpha * state.cpu_usage + (1 - alpha) * baseline['cpu_usage']
                    else:
                        baseline['cpu_usage'] = state.cpu_usage
                
                if state.memory_usage is not None:
                    if 'memory_usage' in baseline:
                        baseline['memory_usage'] = alpha * state.memory_usage + (1 - alpha) * baseline['memory_usage']
                    else:
                        baseline['memory_usage'] = state.memory_usage
    
    def _anomaly_to_dict(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Convert anomaly result to dictionary."""
        return {
            'component_id': anomaly.component_id,
            'timestamp': anomaly.timestamp,
            'anomaly_score': anomaly.anomaly_score,
            'is_anomaly': anomaly.is_anomaly,
            'confidence': anomaly.confidence,
            'description': anomaly.description,
            'severity': anomaly.severity.name,
            'contributing_factors': anomaly.contributing_factors
        }


class PerformanceAnalyzer(Analyzer):
    """Analyze performance trends and provide optimization insights."""
    
    def __init__(self, lookback_hours: float = 1.0):
        self.lookback_hours = lookback_hours
        self.lookback_seconds = lookback_hours * 3600
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.statistical_analyzer = StatisticalAnalyzer()
        self._lock = threading.RLock()
    
    def analyze(self, states: List[ComponentState]) -> Dict[str, Any]:
        """Analyze performance trends and generate insights."""
        with PerformanceTimer() as timer:
            insights = []
            current_time = time.time()
            cutoff_time = current_time - self.lookback_seconds
            
            # Update history
            with self._lock:
                for state in states:
                    if state.timestamp >= cutoff_time:
                        self.performance_history[state.component_id].append(state)
            
            # Analyze each component's performance
            for component_id, history in self.performance_history.items():
                component_insights = self._analyze_component_performance(component_id, list(history))
                insights.extend(component_insights)
            
            # Determine overall alert level
            alert_level = AlertLevel.INFO
            if insights:
                max_impact = max(i.impact_level for i in insights)
                alert_level = max_impact
            
            return {
                'type': 'performance_analysis',
                'timestamp': current_time,
                'insights': [self._insight_to_dict(i) for i in insights],
                'insight_count': len(insights),
                'analysis_time_us': timer.elapsed_us,
                'alert_level': alert_level.value,
                'requires_intervention': any(i.impact_level.value >= AlertLevel.WARNING.value for i in insights),
                'message': f"Generated {len(insights)} performance insights" if insights else "No performance issues detected"
            }
    
    def _analyze_component_performance(self, component_id: str, states: List[ComponentState]) -> List[PerformanceInsight]:
        """Analyze performance for a single component."""
        insights = []
        
        if len(states) < 10:
            return insights
        
        # Extract performance metrics
        timestamps = [s.timestamp for s in states]
        
        metrics = {
            'cpu_usage': [s.cpu_usage for s in states if s.cpu_usage is not None],
            'memory_usage': [s.memory_usage for s in states if s.memory_usage is not None],
            'gpu_usage': [s.gpu_usage for s in states if s.gpu_usage is not None],
            'health_score': [s.health_score for s in states],
            'average_latency_ms': [s.average_latency_ms for s in states if s.average_latency_ms > 0],
            'throughput_rps': [s.throughput_rps for s in states if s.throughput_rps > 0],
        }
        
        # Analyze each metric
        for metric_name, values in metrics.items():
            if len(values) < 10:
                continue
            
            # Get timestamps corresponding to these values
            metric_timestamps = timestamps[-len(values):]
            
            insight = self._analyze_metric_trend(component_id, metric_name, values, metric_timestamps)
            if insight:
                insights.append(insight)
        
        return insights
    
    def _analyze_metric_trend(self, 
                            component_id: str, 
                            metric_name: str, 
                            values: List[float], 
                            timestamps: List[float]) -> Optional[PerformanceInsight]:
        """Analyze trend for a specific metric."""
        if len(values) < 10:
            return None
        
        # Detect trend
        trend_direction, trend_strength = self.statistical_analyzer.detect_trend(values, timestamps)
        
        # Calculate baseline and current values
        baseline_count = max(len(values) // 3, 5)
        baseline_value = np.mean(values[:baseline_count])
        current_value = np.mean(values[-5:])  # Last 5 values
        
        # Skip if no significant change
        relative_change = abs(current_value - baseline_value) / max(abs(baseline_value), 1e-6)
        if relative_change < 0.1 and trend_strength < 0.3:
            return None
        
        # Generate insights based on metric type and trend
        recommendation, impact_level = self._generate_recommendation(
            metric_name, trend_direction, current_value, baseline_value, trend_strength
        )
        
        if impact_level == AlertLevel.DEBUG:
            return None  # Skip low-impact insights
        
        return PerformanceInsight(
            metric_name=f"{component_id}:{metric_name}",
            current_value=current_value,
            baseline_value=baseline_value,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            recommendation=recommendation,
            impact_level=impact_level
        )
    
    def _generate_recommendation(self, 
                               metric_name: str, 
                               trend_direction: str, 
                               current_value: float, 
                               baseline_value: float,
                               trend_strength: float) -> Tuple[str, AlertLevel]:
        """Generate recommendation based on metric analysis."""
        
        # Define thresholds and recommendations for different metrics
        recommendations = {
            'cpu_usage': {
                'high_threshold': 80.0,
                'degrading': ("CPU usage is increasing. Consider optimizing CPU-intensive operations or scaling horizontally.", AlertLevel.WARNING),
                'improving': ("CPU usage is decreasing, indicating good performance optimization.", AlertLevel.INFO),
                'stable_high': ("CPU usage is consistently high. Monitor for potential bottlenecks.", AlertLevel.WARNING),
            },
            'memory_usage': {
                'high_threshold': 85.0,
                'degrading': ("Memory usage is increasing. Check for memory leaks or consider increasing memory allocation.", AlertLevel.WARNING),
                'improving': ("Memory usage is decreasing, indicating efficient memory management.", AlertLevel.INFO),
                'stable_high': ("Memory usage is consistently high. Monitor for potential memory pressure.", AlertLevel.ERROR),
            },
            'gpu_usage': {
                'high_threshold': 90.0,
                'degrading': ("GPU usage is increasing. Monitor GPU memory and consider optimizing GPU operations.", AlertLevel.WARNING),
                'improving': ("GPU usage is decreasing, which may indicate underutilization or optimization.", AlertLevel.INFO),
                'stable_high': ("GPU usage is consistently high. Ensure adequate GPU cooling and memory.", AlertLevel.WARNING),
            },
            'health_score': {
                'low_threshold': 0.7,
                'degrading': ("Health score is declining. Investigate component issues and consider intervention.", AlertLevel.ERROR),
                'improving': ("Health score is improving, indicating successful optimizations.", AlertLevel.INFO),
                'stable_low': ("Health score is consistently low. Immediate attention required.", AlertLevel.CRITICAL),
            },
            'average_latency_ms': {
                'high_threshold': 1000.0,
                'degrading': ("Response latency is increasing. Optimize processing pipeline or scale resources.", AlertLevel.WARNING),
                'improving': ("Response latency is decreasing, indicating performance improvements.", AlertLevel.INFO),
                'stable_high': ("Response latency is consistently high. Optimize critical path operations.", AlertLevel.ERROR),
            },
            'throughput_rps': {
                'low_threshold': 1.0,
                'degrading': ("Throughput is decreasing. Check for bottlenecks in the processing pipeline.", AlertLevel.WARNING),
                'improving': ("Throughput is increasing, indicating good performance scaling.", AlertLevel.INFO),
                'stable_low': ("Throughput is consistently low. Investigate processing bottlenecks.", AlertLevel.WARNING),
            },
        }
        
        if metric_name not in recommendations:
            return "No specific recommendation available.", AlertLevel.DEBUG
        
        config = recommendations[metric_name]
        
        # Determine recommendation based on trend and thresholds
        if trend_direction == 'degrading':
            if trend_strength > 0.5:
                return config['degrading']
            else:
                return config['degrading'][0], AlertLevel.INFO
        elif trend_direction == 'improving':
            return config['improving']
        else:  # stable
            if metric_name in ['cpu_usage', 'memory_usage', 'gpu_usage', 'average_latency_ms']:
                threshold = config.get('high_threshold', 100.0)
                if current_value > threshold:
                    return config['stable_high']
            elif metric_name in ['health_score', 'throughput_rps']:
                threshold = config.get('low_threshold', 0.0)
                if current_value < threshold:
                    return config.get('stable_low', config['degrading'])
        
        return "Metric is stable within normal ranges.", AlertLevel.DEBUG
    
    def _insight_to_dict(self, insight: PerformanceInsight) -> Dict[str, Any]:
        """Convert performance insight to dictionary."""
        return {
            'metric_name': insight.metric_name,
            'current_value': insight.current_value,
            'baseline_value': insight.baseline_value,
            'trend_direction': insight.trend_direction,
            'trend_strength': insight.trend_strength,
            'recommendation': insight.recommendation,
            'impact_level': insight.impact_level.name
        }


class FailurePredictor(Analyzer):
    """Predict potential failures based on historical patterns."""
    
    def __init__(self, prediction_horizon_hours: float = 4.0):
        self.prediction_horizon_hours = prediction_horizon_hours
        self.prediction_horizon_seconds = prediction_horizon_hours * 3600
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.component_trends: Dict[str, Dict] = defaultdict(dict)
        self._lock = threading.RLock()
    
    def analyze(self, states: List[ComponentState]) -> Dict[str, Any]:
        """Analyze states to predict potential failures."""
        with PerformanceTimer() as timer:
            predictions = []
            current_time = time.time()
            
            # Group states by component
            component_states = defaultdict(list)
            for state in states:
                component_states[state.component_id].append(state)
            
            # Analyze each component for failure risk
            for component_id, comp_states in component_states.items():
                prediction = self._predict_component_failure(component_id, comp_states, current_time)
                if prediction:
                    predictions.append(prediction)
            
            # Update failure patterns
            self._update_failure_patterns(states)
            
            # Determine alert level
            alert_level = AlertLevel.INFO
            if predictions:
                max_prob = max(p.failure_probability for p in predictions)
                if max_prob > 0.8:
                    alert_level = AlertLevel.CRITICAL
                elif max_prob > 0.6:
                    alert_level = AlertLevel.ERROR
                elif max_prob > 0.4:
                    alert_level = AlertLevel.WARNING
            
            return {
                'type': 'failure_prediction',
                'timestamp': current_time,
                'predictions': [self._prediction_to_dict(p) for p in predictions],
                'prediction_count': len(predictions),
                'analysis_time_us': timer.elapsed_us,
                'alert_level': alert_level.value,
                'requires_intervention': any(p.failure_probability > 0.6 for p in predictions),
                'message': f"Predicted {len(predictions)} potential failures" if predictions else "No failures predicted"
            }
    
    def _predict_component_failure(self, 
                                 component_id: str, 
                                 states: List[ComponentState], 
                                 current_time: float) -> Optional[FailurePrediction]:
        """Predict failure for a single component."""
        if len(states) < 20:  # Need sufficient data for prediction
            return None
        
        # Analyze recent trends
        recent_states = states[-50:]  # Last 50 states
        warning_signs = []
        failure_indicators = []
        
        # Check health score trend
        health_scores = [s.health_score for s in recent_states]
        if len(health_scores) >= 10:
            recent_health = np.mean(health_scores[-5:])
            baseline_health = np.mean(health_scores[:-5])
            
            if recent_health < 0.3:
                failure_indicators.append("critically_low_health")
            elif recent_health < 0.5:
                warning_signs.append("low_health_score")
            
            if baseline_health - recent_health > 0.3:
                warning_signs.append("rapid_health_degradation")
        
        # Check error rate trend
        error_counts = [s.error_count for s in recent_states]
        if len(error_counts) >= 10:
            recent_errors = sum(error_counts[-5:])
            baseline_errors = sum(error_counts[:-5]) / len(error_counts[:-5]) * 5
            
            if recent_errors > baseline_errors * 3:
                failure_indicators.append("increasing_error_rate")
            elif recent_errors > baseline_errors * 1.5:
                warning_signs.append("elevated_error_rate")
        
        # Check resource utilization trends
        cpu_usage = [s.cpu_usage for s in recent_states if s.cpu_usage is not None]
        memory_usage = [s.memory_usage for s in recent_states if s.memory_usage is not None]
        
        if len(cpu_usage) >= 10:
            recent_cpu = np.mean(cpu_usage[-5:])
            if recent_cpu > 95:
                failure_indicators.append("cpu_exhaustion")
            elif recent_cpu > 85:
                warning_signs.append("high_cpu_usage")
        
        if len(memory_usage) >= 10:
            recent_memory = np.mean(memory_usage[-5:])
            if recent_memory > 95:
                failure_indicators.append("memory_exhaustion")
            elif recent_memory > 85:
                warning_signs.append("high_memory_usage")
        
        # Check for component-specific indicators
        component_warnings = self._check_component_specific_indicators(component_id, recent_states)
        warning_signs.extend(component_warnings)
        
        # Calculate failure probability
        failure_probability = self._calculate_failure_probability(
            failure_indicators, warning_signs, health_scores, error_counts
        )
        
        if failure_probability < 0.2:
            return None  # Low risk, no prediction needed
        
        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(
            failure_probability, health_scores, error_counts
        )
        
        # Determine failure type
        failure_type = self._determine_failure_type(failure_indicators, warning_signs)
        
        # Generate recommended actions
        recommended_actions = self._generate_failure_recommendations(
            failure_type, failure_indicators, warning_signs
        )
        
        return FailurePrediction(
            component_id=component_id,
            failure_probability=failure_probability,
            time_to_failure_hours=time_to_failure,
            failure_type=failure_type,
            confidence=min(failure_probability * 0.8, 0.9),
            warning_signs=warning_signs + failure_indicators,
            recommended_actions=recommended_actions
        )
    
    def _check_component_specific_indicators(self, 
                                           component_id: str, 
                                           states: List[ComponentState]) -> List[str]:
        """Check for component-specific failure indicators."""
        warnings = []
        
        # Extract component type from the latest state
        if not states:
            return warnings
        
        latest_state = states[-1]
        component_type = latest_state.component_type
        
        if component_type == ComponentType.SCHEDULER:
            # Check queue buildup
            waiting_requests = [
                s.data.get('waiting_requests', 0) for s in states 
                if 'waiting_requests' in s.data
            ]
            if waiting_requests and np.mean(waiting_requests[-5:]) > 100:
                warnings.append("request_queue_buildup")
        
        elif component_type == ComponentType.CACHE_ENGINE:
            # Check cache block availability
            free_blocks = [
                s.data.get('free_gpu_blocks', 0) for s in states 
                if 'free_gpu_blocks' in s.data
            ]
            if free_blocks and np.mean(free_blocks[-5:]) < 5:
                warnings.append("cache_exhaustion")
        
        elif component_type == ComponentType.WORKER:
            # Check GPU memory usage
            gpu_utilization = [
                s.data.get('gpu_utilization', 0) for s in states 
                if 'gpu_utilization' in s.data
            ]
            if gpu_utilization and np.mean(gpu_utilization[-5:]) > 0.95:
                warnings.append("gpu_memory_pressure")
        
        return warnings
    
    def _calculate_failure_probability(self, 
                                     failure_indicators: List[str], 
                                     warning_signs: List[str],
                                     health_scores: List[float],
                                     error_counts: List[int]) -> float:
        """Calculate probability of failure based on indicators."""
        base_probability = 0.0
        
        # Failure indicators contribute heavily
        base_probability += len(failure_indicators) * 0.3
        
        # Warning signs contribute moderately
        base_probability += len(warning_signs) * 0.1
        
        # Health score contribution
        if health_scores:
            recent_health = np.mean(health_scores[-5:])
            health_penalty = max(0, (0.5 - recent_health) * 0.5)
            base_probability += health_penalty
        
        # Error rate contribution
        if error_counts:
            recent_errors = sum(error_counts[-5:])
            if recent_errors > 0:
                error_penalty = min(recent_errors * 0.05, 0.3)
                base_probability += error_penalty
        
        return min(base_probability, 1.0)
    
    def _estimate_time_to_failure(self, 
                                failure_probability: float,
                                health_scores: List[float],
                                error_counts: List[int]) -> Optional[float]:
        """Estimate time until failure occurs."""
        if failure_probability < 0.3:
            return None
        
        # Base time estimate based on probability
        if failure_probability > 0.8:
            base_hours = 0.5  # 30 minutes
        elif failure_probability > 0.6:
            base_hours = 2.0  # 2 hours
        elif failure_probability > 0.4:
            base_hours = 8.0  # 8 hours
        else:
            base_hours = 24.0  # 24 hours
        
        # Adjust based on trend velocity
        if len(health_scores) >= 10:
            health_trend_slope = np.polyfit(range(len(health_scores)), health_scores, 1)[0]
            if health_trend_slope < -0.1:  # Rapid degradation
                base_hours *= 0.5
            elif health_trend_slope < -0.05:  # Moderate degradation
                base_hours *= 0.8
        
        return base_hours
    
    def _determine_failure_type(self, 
                              failure_indicators: List[str], 
                              warning_signs: List[str]) -> str:
        """Determine the most likely type of failure."""
        all_indicators = failure_indicators + warning_signs
        
        if any('memory' in indicator for indicator in all_indicators):
            return "memory_exhaustion"
        elif any('cpu' in indicator for indicator in all_indicators):
            return "cpu_exhaustion" 
        elif any('error' in indicator for indicator in all_indicators):
            return "error_cascade"
        elif any('health' in indicator for indicator in all_indicators):
            return "component_degradation"
        elif any('cache' in indicator for indicator in all_indicators):
            return "cache_overflow"
        elif any('queue' in indicator for indicator in all_indicators):
            return "request_backlog"
        else:
            return "general_instability"
    
    def _generate_failure_recommendations(self, 
                                        failure_type: str,
                                        failure_indicators: List[str],
                                        warning_signs: List[str]) -> List[str]:
        """Generate recommended actions to prevent failure."""
        recommendations = []
        
        if failure_type == "memory_exhaustion":
            recommendations.extend([
                "Increase memory allocation or scale horizontally",
                "Check for memory leaks in the application",
                "Reduce batch sizes or model parameters if possible",
                "Enable memory monitoring and alerts"
            ])
        elif failure_type == "cpu_exhaustion":
            recommendations.extend([
                "Scale CPU resources or add more workers",
                "Optimize CPU-intensive operations",
                "Reduce concurrent request processing",
                "Consider CPU affinity optimization"
            ])
        elif failure_type == "error_cascade":
            recommendations.extend([
                "Investigate root cause of increasing errors",
                "Implement circuit breaker patterns",
                "Increase error handling robustness",
                "Consider temporary load reduction"
            ])
        elif failure_type == "cache_overflow":
            recommendations.extend([
                "Increase cache memory allocation",
                "Optimize cache block management",
                "Reduce sequence lengths or batch sizes",
                "Monitor cache hit rates"
            ])
        elif failure_type == "request_backlog":
            recommendations.extend([
                "Scale processing capacity",
                "Implement request throttling",
                "Optimize request processing pipeline",
                "Consider load balancing improvements"
            ])
        else:
            recommendations.extend([
                "Monitor component health closely",
                "Prepare for potential restart",
                "Review recent configuration changes",
                "Implement additional monitoring"
            ])
        
        return recommendations
    
    def _update_failure_patterns(self, states: List[ComponentState]) -> None:
        """Update failure patterns for future predictions."""
        # This would implement machine learning pattern recognition
        # For now, we'll just track basic statistics
        pass
    
    def _prediction_to_dict(self, prediction: FailurePrediction) -> Dict[str, Any]:
        """Convert failure prediction to dictionary."""
        return {
            'component_id': prediction.component_id,
            'failure_probability': prediction.failure_probability,
            'time_to_failure_hours': prediction.time_to_failure_hours,
            'failure_type': prediction.failure_type,
            'confidence': prediction.confidence,
            'warning_signs': prediction.warning_signs,
            'recommended_actions': prediction.recommended_actions
        }


class HealthScorer(Analyzer):
    """Calculate overall system health scores."""
    
    def __init__(self):
        self.component_weights = {
            ComponentType.ENGINE: 0.25,
            ComponentType.SCHEDULER: 0.20,
            ComponentType.WORKER: 0.20,
            ComponentType.CACHE_ENGINE: 0.15,
            ComponentType.MODEL_RUNNER: 0.10,
            ComponentType.BLOCK_MANAGER: 0.10,
        }
    
    def analyze(self, states: List[ComponentState]) -> Dict[str, Any]:
        """Calculate comprehensive health scores."""
        with PerformanceTimer() as timer:
            current_time = time.time()
            
            # Group states by component type
            component_health = defaultdict(list)
            for state in states:
                if state.component_type in self.component_weights:
                    component_health[state.component_type].append(state.health_score)
            
            # Calculate component-level health scores
            component_scores = {}
            for comp_type, scores in component_health.items():
                if scores:
                    component_scores[comp_type] = {
                        'average_health': np.mean(scores),
                        'min_health': np.min(scores),
                        'component_count': len(scores),
                        'healthy_count': sum(1 for s in scores if s > 0.7),
                    }
            
            # Calculate weighted overall health
            overall_health = self._calculate_overall_health(component_scores)
            
            # Generate health insights
            insights = self._generate_health_insights(component_scores, overall_health)
            
            # Determine alert level
            if overall_health < 0.3:
                alert_level = AlertLevel.CRITICAL
            elif overall_health < 0.5:
                alert_level = AlertLevel.ERROR
            elif overall_health < 0.7:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.INFO
            
            return {
                'type': 'health_scoring',
                'timestamp': current_time,
                'overall_health': overall_health,
                'component_scores': {k.value: v for k, v in component_scores.items()},
                'insights': insights,
                'analysis_time_us': timer.elapsed_us,
                'alert_level': alert_level.value,
                'requires_intervention': overall_health < 0.5,
                'message': f"System health: {overall_health:.1%}"
            }
    
    def _calculate_overall_health(self, component_scores: Dict[ComponentType, Dict]) -> float:
        """Calculate weighted overall health score."""
        if not component_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for comp_type, weight in self.component_weights.items():
            if comp_type in component_scores:
                score_data = component_scores[comp_type]
                # Use minimum health to be conservative
                health = score_data['min_health']
                weighted_sum += health * weight
                total_weight += weight
        
        return weighted_sum / max(total_weight, 1e-6)
    
    def _generate_health_insights(self, 
                                component_scores: Dict[ComponentType, Dict],
                                overall_health: float) -> List[str]:
        """Generate health insights and recommendations."""
        insights = []
        
        # Overall health insights
        if overall_health < 0.3:
            insights.append("CRITICAL: System health is severely degraded. Immediate intervention required.")
        elif overall_health < 0.5:
            insights.append("WARNING: System health is poor. Investigation recommended.")
        elif overall_health < 0.7:
            insights.append("CAUTION: System health is below optimal. Monitor closely.")
        elif overall_health > 0.9:
            insights.append("EXCELLENT: System health is optimal.")
        
        # Component-specific insights
        for comp_type, scores in component_scores.items():
            avg_health = scores['average_health']
            min_health = scores['min_health']
            healthy_count = scores['healthy_count']
            total_count = scores['component_count']
            
            if min_health < 0.3:
                insights.append(f"CRITICAL: {comp_type.value} has components in critical state.")
            elif avg_health < 0.5:
                insights.append(f"WARNING: {comp_type.value} average health is poor.")
            elif healthy_count / total_count < 0.8:
                insights.append(f"CAUTION: Only {healthy_count}/{total_count} {comp_type.value} components are healthy.")
        
        return insights