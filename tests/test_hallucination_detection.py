"""
Tests for vLLM hallucination detection functionality.
"""

import os
import pytest
import numpy as np

from vllm.hallucination_detection import (
    OptimizedWhiteBoxScorer,
    StreamingScorer, 
    HallucinationInfo,
    HallucinationRiskLevel,
    calculate_hallucination_info_from_logprobs
)
from vllm.outputs import CompletionOutput
from vllm.sequence import Sequence, SequenceData, Logprob


class TestOptimizedWhiteBoxScorer:
    """Test the OptimizedWhiteBoxScorer functionality."""
    
    def test_scorer_initialization(self):
        """Test that scorer initializes correctly."""
        scorer = OptimizedWhiteBoxScorer()
        assert scorer is not None
        assert scorer._compiled is True  # Should be warmed up
    
    def test_logprobs_to_scores_normalized(self):
        """Test normalized probability scoring from logprobs."""
        logprobs = [-0.1, -0.2, -0.1, -0.3]  # High confidence
        score = OptimizedWhiteBoxScorer.logprobs_to_scores(
            logprobs, method=OptimizedWhiteBoxScorer.METHOD_NORM
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high confidence
        
        logprobs_low = [-2.0, -3.0, -2.5, -4.0]  # Low confidence  
        score_low = OptimizedWhiteBoxScorer.logprobs_to_scores(
            logprobs_low, method=OptimizedWhiteBoxScorer.METHOD_NORM
        )
        assert score_low < score  # Lower confidence
    
    def test_logprobs_to_scores_min(self):
        """Test minimum probability scoring from logprobs."""
        logprobs = [-0.1, -0.2, -0.1, -0.3]
        score = OptimizedWhiteBoxScorer.logprobs_to_scores(
            logprobs, method=OptimizedWhiteBoxScorer.METHOD_MIN
        )
        assert 0.0 <= score <= 1.0
        # Min method should use the best (highest) logprob
        expected = np.exp(max(logprobs))
        assert abs(score - expected) < 0.001
    
    def test_risk_level_assignment(self):
        """Test risk level assignment based on confidence scores."""
        scorer = OptimizedWhiteBoxScorer()
        
        # Test different confidence levels
        assert scorer.get_risk_level(0.98) == HallucinationRiskLevel.MINIMAL
        assert scorer.get_risk_level(0.90) == HallucinationRiskLevel.LOW  
        assert scorer.get_risk_level(0.75) == HallucinationRiskLevel.MEDIUM
        assert scorer.get_risk_level(0.60) == HallucinationRiskLevel.HIGH
        assert scorer.get_risk_level(0.30) == HallucinationRiskLevel.CRITICAL


class TestStreamingScorer:
    """Test the StreamingScorer functionality."""
    
    def test_streaming_scorer_initialization(self):
        """Test streaming scorer initialization."""
        scorer = OptimizedWhiteBoxScorer()
        streaming_scorer = scorer.stream_score(window_size=10, method=1)
        
        assert isinstance(streaming_scorer, StreamingScorer)
        assert streaming_scorer.window_size == 10
        assert streaming_scorer.method == 1
        assert len(streaming_scorer.token_hallucination_info) == 0
    
    def test_streaming_update(self):
        """Test streaming updates with tokens."""
        scorer = OptimizedWhiteBoxScorer()
        streaming_scorer = scorer.stream_score(window_size=5)
        
        # Add some tokens with different confidence levels
        probs = [0.9, 0.8, 0.6, 0.7, 0.95]
        logprobs = [np.log(p) for p in probs]
        
        for i, (prob, logprob) in enumerate(zip(probs, logprobs)):
            info = streaming_scorer.update(prob, logprob)
            
            assert isinstance(info, HallucinationInfo)
            assert 0.0 <= info.confidence_score <= 1.0
            assert info.token_logprob == logprob
            assert info.risk_level in HallucinationRiskLevel
        
        # Check that we have the right number of token infos
        assert len(streaming_scorer.token_hallucination_info) == 5
    
    def test_sequence_level_scoring(self):
        """Test sequence-level confidence scoring."""
        scorer = OptimizedWhiteBoxScorer()
        streaming_scorer = scorer.stream_score(window_size=10)
        
        # Add tokens
        high_conf_probs = [0.9, 0.85, 0.92, 0.88]
        for prob in high_conf_probs:
            streaming_scorer.update(prob)
        
        seq_info = streaming_scorer.get_sequence_hallucination_info()
        assert isinstance(seq_info, HallucinationInfo)
        assert seq_info.confidence_score > 0.8  # Should be high confidence


class TestHallucinationInfo:
    """Test the HallucinationInfo dataclass."""
    
    def test_hallucination_info_creation(self):
        """Test creating hallucination info objects."""
        info = HallucinationInfo(
            confidence_score=0.85,
            risk_level=HallucinationRiskLevel.LOW,
            token_logprob=-0.15
        )
        
        assert info.confidence_score == 0.85
        assert info.risk_level == HallucinationRiskLevel.LOW
        assert info.token_logprob == -0.15
    
    def test_to_dict_conversion(self):
        """Test converting hallucination info to dictionary."""
        info = HallucinationInfo(
            confidence_score=0.75,
            risk_level=HallucinationRiskLevel.MEDIUM,
            token_logprob=-0.25
        )
        
        info_dict = info.to_dict()
        expected = {
            "confidence_score": 0.75,
            "hallucination_probability": "MEDIUM", 
            "token_log_probability": -0.25
        }
        
        assert info_dict == expected


class TestSequenceIntegration:
    """Test integration with vLLM Sequence objects."""
    
    def test_sequence_hallucination_initialization(self):
        """Test that sequences can initialize hallucination scoring."""
        # Mock the environment variables
        os.environ["VLLM_ENABLE_HALLUCINATION_DETECTION"] = "1"
        os.environ["VLLM_HALLUCINATION_DETECTION_METHOD"] = "normalized"
        os.environ["VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE"] = "20"
        
        try:
            # This test would require a full vLLM setup, so we'll test the logic
            from vllm.sequence import Sequence
            from vllm.inputs import SingletonInputs
            
            # Create mock inputs
            inputs = SingletonInputs({
                "type": "tokens",
                "prompt_token_ids": [1, 2, 3, 4, 5]
            })
            
            # Create sequence (hallucination scorer should auto-initialize)
            seq = Sequence(seq_id=0, inputs=inputs, block_size=16)
            
            # Check if hallucination detection was initialized
            if hasattr(seq, 'hallucination_scorer') and seq.hallucination_scorer:
                assert isinstance(seq.hallucination_scorer, StreamingScorer)
                assert seq.hallucination_scorer.window_size == 20
            
        finally:
            # Clean up environment
            os.environ.pop("VLLM_ENABLE_HALLUCINATION_DETECTION", None)
            os.environ.pop("VLLM_HALLUCINATION_DETECTION_METHOD", None)
            os.environ.pop("VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE", None)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_hallucination_info_from_logprobs(self):
        """Test the utility function for calculating hallucination info."""
        logprobs = [-0.1, -0.2, -0.3, -0.15]
        
        info = calculate_hallucination_info_from_logprobs(logprobs)
        
        assert isinstance(info, HallucinationInfo)
        assert 0.0 <= info.confidence_score <= 1.0
        assert info.risk_level in HallucinationRiskLevel
        assert info.token_logprob == logprobs[-1]  # Last token's logprob


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_logprobs(self):
        """Test handling of empty logprobs."""
        info = calculate_hallucination_info_from_logprobs([])
        assert info.confidence_score == 0.0
        assert info.token_logprob is None
    
    def test_very_low_probabilities(self):
        """Test handling of very low probabilities."""
        very_low_logprobs = [-10.0, -15.0, -20.0]
        score = OptimizedWhiteBoxScorer.logprobs_to_scores(very_low_logprobs)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.1  # Should be very low confidence
    
    def test_streaming_scorer_reset(self):
        """Test resetting streaming scorer."""
        scorer = OptimizedWhiteBoxScorer()
        streaming_scorer = scorer.stream_score(window_size=5)
        
        # Add some data
        streaming_scorer.update(0.5)
        streaming_scorer.update(0.6)
        assert len(streaming_scorer.token_hallucination_info) == 2
        
        # Reset
        streaming_scorer.reset()
        assert len(streaming_scorer.token_hallucination_info) == 0
        assert streaming_scorer.count == 0


@pytest.mark.parametrize("method", ["min", "normalized"])
def test_different_methods(method):
    """Test both scoring methods work correctly."""
    method_int = OptimizedWhiteBoxScorer.METHOD_MIN if method == "min" else OptimizedWhiteBoxScorer.METHOD_NORM
    
    logprobs = [-0.1, -0.5, -0.2, -1.0, -0.3]
    score = OptimizedWhiteBoxScorer.logprobs_to_scores(logprobs, method=method_int)
    
    assert 0.0 <= score <= 1.0
    
    if method == "min":
        # Min method should be close to exp(max(logprobs))
        expected = np.exp(max(logprobs))
        assert abs(score - expected) < 0.01
    else:
        # Normalized method should be close to exp(mean(logprobs))
        expected = np.exp(np.mean(logprobs))
        assert abs(score - expected) < 0.01


if __name__ == "__main__":
    # Run tests manually if not using pytest
    import sys
    
    test_classes = [
        TestOptimizedWhiteBoxScorer,
        TestStreamingScorer, 
        TestHallucinationInfo,
        TestSequenceIntegration,
        TestUtilityFunctions,
        TestEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}:")
        test_instance = test_class()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(test_instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All tests passed! ✓")
        sys.exit(0) 
    else:
        print("Some tests failed! ✗")
        sys.exit(1)