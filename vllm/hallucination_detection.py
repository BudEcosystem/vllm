"""
Hallucination detection module for vLLM.

This module provides real-time hallucination detection capabilities for LLM outputs,
integrating the WhiteBox scorer with vLLM's generation pipeline.
"""

import numpy as np
from numba import jit, njit, prange
from typing import Union, Tuple, Optional, List, Dict, Any
import math
from dataclasses import dataclass
from enum import Enum


@njit(fastmath=True, cache=True)
def exp_safe(x: float) -> float:
    """Fast, safe exponential function"""
    if x < -700:  # Prevent underflow
        return 0.0
    elif x > 700:  # Prevent overflow
        return 1e308
    return math.exp(x)


@njit(fastmath=True, cache=True)
def min_probability_score_numba(probabilities: np.ndarray) -> float:
    """JIT-compiled minimum probability calculation"""
    return np.min(probabilities)


@njit(fastmath=True, cache=True)
def normalized_probability_score_numba(probabilities: np.ndarray) -> float:
    """JIT-compiled geometric mean calculation"""
    n = len(probabilities)
    if n == 0:
        return 0.0
    
    # Use log-sum for numerical stability
    log_sum = 0.0
    for i in range(n):
        if probabilities[i] > 0:
            log_sum += math.log(probabilities[i])
        else:
            log_sum += math.log(1e-10)  # Small epsilon
    
    return exp_safe(log_sum / n)


@njit(fastmath=True, cache=True)
def batch_min_probability_numba(batch_probs: np.ndarray) -> np.ndarray:
    """Vectorized batch minimum probability calculation"""
    n_sequences = batch_probs.shape[0]
    results = np.empty(n_sequences, dtype=np.float32)
    
    for i in prange(n_sequences):
        results[i] = np.min(batch_probs[i])
    
    return results


@njit(fastmath=True, cache=True)
def batch_normalized_probability_numba(batch_probs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Vectorized batch geometric mean calculation"""
    n_sequences = batch_probs.shape[0]
    results = np.empty(n_sequences, dtype=np.float32)
    
    for i in prange(n_sequences):
        length = lengths[i]
        if length == 0:
            results[i] = 0.0
            continue
            
        log_sum = 0.0
        for j in range(length):
            if batch_probs[i, j] > 0:
                log_sum += math.log(batch_probs[i, j])
            else:
                log_sum += math.log(1e-10)
        
        results[i] = exp_safe(log_sum / length)
    
    return results


@njit(fastmath=True, cache=True)
def logprobs_to_probs_vectorized(logprobs: np.ndarray) -> np.ndarray:
    """Convert log probabilities to probabilities efficiently"""
    n = len(logprobs)
    probs = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        probs[i] = exp_safe(logprobs[i])
    
    return probs


@njit(fastmath=True, cache=True)
def get_risk_level_fast(score: float) -> int:
    """
    Fast risk level calculation returning integer codes:
    0: CRITICAL, 1: HIGH, 2: MEDIUM, 3: LOW, 4: MINIMAL
    """
    if score < 0.5:
        return 0  # CRITICAL
    elif score < 0.7:
        return 1  # HIGH
    elif score < 0.85:
        return 2  # MEDIUM
    elif score < 0.95:
        return 3  # LOW
    else:
        return 4  # MINIMAL


class HallucinationRiskLevel(Enum):
    """Hallucination risk levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


@dataclass
class HallucinationInfo:
    """Information about hallucination detection for a token or sequence"""
    confidence_score: float
    risk_level: HallucinationRiskLevel
    token_logprob: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "confidence_score": self.confidence_score,
            "hallucination_probability": self.risk_level.value,
            "token_log_probability": self.token_logprob
        }


class OptimizedWhiteBoxScorer:
    """
    High-performance WhiteBox scorer optimized for production use.
    
    Features:
    - Minimal memory allocations
    - JIT-compiled critical paths
    - Vectorized operations
    - No string operations in hot paths
    - Stateless design for thread safety
    """
    
    # Class-level constants to avoid repeated allocations
    RISK_LEVELS = [
        HallucinationRiskLevel.CRITICAL,
        HallucinationRiskLevel.HIGH,
        HallucinationRiskLevel.MEDIUM,
        HallucinationRiskLevel.LOW,
        HallucinationRiskLevel.MINIMAL
    ]
    METHOD_MIN = 0
    METHOD_NORM = 1
    
    def __init__(self, preallocate_size: int = 512):
        """
        Initialize scorer with optional preallocation.
        
        Args:
            preallocate_size: Size for preallocated buffers
        """
        # Preallocate buffers for common sizes to avoid repeated allocations
        self._buffer = np.empty(preallocate_size, dtype=np.float32)
        self._compiled = False
        self._warmup()
    
    def _warmup(self):
        """Warm up JIT compilation with dummy data"""
        if not self._compiled:
            dummy = np.array([0.5, 0.6, 0.7], dtype=np.float32)
            _ = min_probability_score_numba(dummy)
            _ = normalized_probability_score_numba(dummy)
            self._compiled = True
    
    def score_single(self, 
                    probabilities: Union[np.ndarray, list], 
                    method: int = METHOD_NORM) -> float:
        """
        Score a single sequence with minimal overhead.
        
        Args:
            probabilities: Token probabilities (numpy array preferred)
            method: 0 for min_probability, 1 for normalized_probability
            
        Returns:
            Confidence score between 0 and 1
        """
        # Convert to numpy if needed (avoid if possible)
        if not isinstance(probabilities, np.ndarray):
            probs = np.asarray(probabilities, dtype=np.float32)
        else:
            probs = probabilities
        
        if method == self.METHOD_MIN:
            return float(min_probability_score_numba(probs))
        else:
            return float(normalized_probability_score_numba(probs))
    
    def score_batch(self,
                   batch_probabilities: np.ndarray,
                   lengths: Optional[np.ndarray] = None,
                   method: int = METHOD_NORM) -> np.ndarray:
        """
        Score multiple sequences efficiently.
        
        Args:
            batch_probabilities: 2D array of shape (batch_size, max_length)
            lengths: Array of actual sequence lengths (if None, uses full length)
            method: 0 for min_probability, 1 for normalized_probability
            
        Returns:
            Array of confidence scores
        """
        if lengths is None:
            lengths = np.full(batch_probabilities.shape[0], 
                            batch_probabilities.shape[1], 
                            dtype=np.int32)
        
        if method == self.METHOD_MIN:
            return batch_min_probability_numba(batch_probabilities)
        else:
            return batch_normalized_probability_numba(batch_probabilities, lengths)
    
    @staticmethod
    def logprobs_to_scores(logprobs: Union[np.ndarray, list],
                          method: int = 1) -> float:
        """
        Direct conversion from log probabilities to confidence score.
        Most efficient method for single sequence scoring.
        
        Args:
            logprobs: Log probabilities from model
            method: 0 for min_probability, 1 for normalized_probability
            
        Returns:
            Confidence score
        """
        if not isinstance(logprobs, np.ndarray):
            logprobs = np.asarray(logprobs, dtype=np.float32)
        
        # Direct computation without intermediate probability conversion
        if method == 0:  # min_probability
            return float(exp_safe(np.max(logprobs)))
        else:  # normalized_probability
            return float(exp_safe(np.mean(logprobs)))
    
    def get_risk_level(self, score: float) -> HallucinationRiskLevel:
        """Get risk level from confidence score"""
        level_idx = get_risk_level_fast(score)
        return self.RISK_LEVELS[level_idx]
    
    @staticmethod
    def get_risk_levels(scores: np.ndarray) -> np.ndarray:
        """
        Vectorized risk level assignment.
        
        Args:
            scores: Array of confidence scores
            
        Returns:
            Array of risk level codes (0-4)
        """
        # Vectorized comparison
        risk_codes = np.zeros(len(scores), dtype=np.int8)
        risk_codes[scores >= 0.95] = 4  # MINIMAL
        risk_codes[(scores >= 0.85) & (scores < 0.95)] = 3  # LOW
        risk_codes[(scores >= 0.7) & (scores < 0.85)] = 2  # MEDIUM
        risk_codes[(scores >= 0.5) & (scores < 0.7)] = 1  # HIGH
        # CRITICAL (0) is default
        
        return risk_codes
    
    def stream_score(self, 
                    window_size: int = 50,
                    method: int = METHOD_NORM):
        """
        Create a streaming scorer for real-time token processing.
        
        Args:
            window_size: Size of probability window to maintain
            method: Scoring method to use
            
        Returns:
            StreamingScorer instance
        """
        return StreamingScorer(window_size, method, self)


class StreamingScorer:
    """
    Efficient streaming scorer for real-time token-by-token scoring.
    Maintains a sliding window with O(1) updates.
    """
    
    def __init__(self, window_size: int = 50, method: int = 1, scorer: Optional[OptimizedWhiteBoxScorer] = None):
        self.window_size = window_size
        self.method = method
        self.buffer = np.zeros(window_size, dtype=np.float32)
        self.position = 0
        self.count = 0
        self.log_sum = 0.0
        self.min_val = 1.0
        self.scorer = scorer or OptimizedWhiteBoxScorer()
        # Store per-token hallucination info
        self.token_hallucination_info: List[HallucinationInfo] = []
    
    def update(self, prob: float, logprob: Optional[float] = None) -> HallucinationInfo:
        """
        Update with new probability and return hallucination info.
        O(1) complexity for both methods.
        """
        # Update circular buffer
        old_prob = self.buffer[self.position]
        self.buffer[self.position] = prob
        
        # Update running statistics
        if self.count < self.window_size:
            self.count += 1
            self.log_sum += math.log(max(prob, 1e-10))
            self.min_val = min(self.min_val, prob)
        else:
            # Remove old value from running sum
            self.log_sum -= math.log(max(old_prob, 1e-10))
            self.log_sum += math.log(max(prob, 1e-10))
            
            # Update min if necessary
            if old_prob == self.min_val or prob < self.min_val:
                self.min_val = np.min(self.buffer[:self.count])
        
        # Move to next position
        self.position = (self.position + 1) % self.window_size
        
        # Calculate score based on method
        if self.method == 0:  # min_probability
            score = self.min_val
        else:  # normalized_probability
            score = exp_safe(self.log_sum / self.count)
        
        # Create hallucination info
        risk_level = self.scorer.get_risk_level(score)
        info = HallucinationInfo(
            confidence_score=score,
            risk_level=risk_level,
            token_logprob=logprob
        )
        
        self.token_hallucination_info.append(info)
        return info
    
    def get_sequence_score(self) -> float:
        """Get overall sequence score"""
        if self.count == 0:
            return 0.0
        
        if self.method == 0:  # min_probability
            return self.min_val
        else:  # normalized_probability
            return exp_safe(self.log_sum / self.count)
    
    def get_sequence_hallucination_info(self) -> HallucinationInfo:
        """Get overall sequence hallucination info"""
        score = self.get_sequence_score()
        risk_level = self.scorer.get_risk_level(score)
        return HallucinationInfo(
            confidence_score=score,
            risk_level=risk_level
        )
    
    def reset(self):
        """Reset scorer state"""
        self.buffer.fill(0)
        self.position = 0
        self.count = 0
        self.log_sum = 0.0
        self.min_val = 1.0
        self.token_hallucination_info.clear()


# Utility functions for common operations
@njit(fastmath=True, cache=True)
def find_low_confidence_positions(probabilities: np.ndarray, 
                                 threshold: float = 0.5) -> np.ndarray:
    """Find positions where confidence drops below threshold"""
    return np.where(probabilities < threshold)[0]


@njit(fastmath=True, cache=True)
def find_confidence_drops(probabilities: np.ndarray,
                         drop_threshold: float = 0.3) -> np.ndarray:
    """Find positions with sudden confidence drops"""
    n = len(probabilities)
    drops = []
    
    for i in range(1, n):
        drop = probabilities[i-1] - probabilities[i]
        if drop > drop_threshold:
            drops.append(i)
    
    return np.array(drops, dtype=np.int32)


# Global scorer instance for reuse
_global_scorer = None


def get_global_scorer() -> OptimizedWhiteBoxScorer:
    """Get or create global scorer instance"""
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = OptimizedWhiteBoxScorer()
    return _global_scorer


def calculate_hallucination_info_from_logprobs(
    logprobs: List[float],
    method: int = OptimizedWhiteBoxScorer.METHOD_NORM
) -> HallucinationInfo:
    """
    Calculate hallucination info from a list of logprobs.
    
    Args:
        logprobs: List of log probabilities
        method: Scoring method (0=min, 1=normalized)
    
    Returns:
        HallucinationInfo object
    """
    scorer = get_global_scorer()
    score = scorer.logprobs_to_scores(logprobs, method)
    risk_level = scorer.get_risk_level(score)
    
    return HallucinationInfo(
        confidence_score=score,
        risk_level=risk_level,
        token_logprob=logprobs[-1] if logprobs else None
    )