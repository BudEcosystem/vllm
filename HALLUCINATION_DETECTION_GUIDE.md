# vLLM Hallucination Detection Guide

This guide explains how to use vLLM's integrated hallucination detection functionality to monitor model confidence and detect potential hallucinations in real-time.

## Overview

vLLM's hallucination detection system provides:

1. **Real-time token-level confidence scoring** during text generation
2. **Sequence-level confidence assessment** for entire responses  
3. **Hallucination risk classification** (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)
4. **WhiteBox scoring** based on token probabilities
5. **Streaming and batch processing support**
6. **OpenAI-compatible API integration**

## Configuration

### Environment Variables

Enable and configure hallucination detection using environment variables:

```bash
# Enable hallucination detection (required)
export VLLM_ENABLE_HALLUCINATION_DETECTION=1

# Choose scoring method: "normalized" (default) or "min" 
export VLLM_HALLUCINATION_DETECTION_METHOD=normalized

# Set sliding window size for streaming detection (default: 50)
export VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE=50
```

### Scoring Methods

- **Normalized (default)**: Uses geometric mean of token probabilities
  - Better for overall sequence confidence
  - Balances all tokens in the sequence
  - Recommended for most use cases

- **Min**: Uses minimum token probability in the sequence
  - More sensitive to individual low-confidence tokens
  - Better for detecting specific uncertain tokens
  - Useful for critical applications

## API Usage

### 1. Offline Inference

```python
import os
os.environ["VLLM_ENABLE_HALLUCINATION_DETECTION"] = "1"

from vllm import LLM, SamplingParams

# Initialize LLM
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Sampling parameters with logprobs (required for hallucination detection)
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=100,
    logprobs=10  # Required: must be > 0
)

# Generate with hallucination detection
outputs = llm.generate(["What is the capital of France?"], sampling_params)

for output in outputs:
    completion = output.outputs[0]
    
    # Sequence-level confidence
    if completion.sequence_hallucination_info:
        seq_info = completion.sequence_hallucination_info
        print(f"Overall confidence: {seq_info.confidence_score:.3f}")
        print(f"Risk level: {seq_info.risk_level.value}")
    
    # Token-level analysis
    if completion.hallucination_info:
        for i, token_info in enumerate(completion.hallucination_info):
            print(f"Token {i}: confidence={token_info.confidence_score:.3f}, "
                  f"risk={token_info.risk_level.value}")
```

### 2. OpenAI-Compatible Completion API

```python
import requests

# Start vLLM server with: vllm serve model_name --api-key your-key

response = requests.post("http://localhost:8000/v1/completions", 
    headers={"Authorization": "Bearer your-key"},
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "Explain quantum physics:",
        "max_tokens": 150,
        "logprobs": 5,  # Required for hallucination detection
        "temperature": 0.7
    }
)

result = response.json()
choice = result["choices"][0]

# Check for hallucination info in response
if "hallucination_info" in choice:
    hal_info = choice["hallucination_info"]
    print(f"Confidence: {hal_info['confidence_score']}")
    print(f"Risk: {hal_info['hallucination_probability']}")

# Token-level info in logprobs
logprobs = choice.get("logprobs", {})
if logprobs.get("confidence_scores"):
    for token, conf, risk in zip(
        logprobs["tokens"],
        logprobs["confidence_scores"], 
        logprobs["hallucination_probabilities"]
    ):
        print(f"'{token}': {conf:.3f} ({risk})")
```

### 3. Streaming Chat Completion

```python
import requests
import json

response = requests.post("http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer your-key"},
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf", 
        "messages": [{"role": "user", "content": "Write a story about AI"}],
        "max_tokens": 200,
        "logprobs": True,
        "top_logprobs": 3,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        data = json.loads(line[6:])
        if data.get("choices"):
            choice = data["choices"][0]
            
            # Real-time hallucination info
            if "hallucination_info" in choice:
                hal_info = choice["hallucination_info"]
                conf = hal_info.get("confidence_score")
                if conf is not None:
                    print(f" [conf: {conf:.2f}]", end="")
            
            # Token content
            delta = choice.get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)
```

## Response Format

### Completion API Response

The completion API includes hallucination information in two places:

1. **Choice-level** (sequence confidence):
```json
{
  "choices": [
    {
      "text": "Generated text...",
      "hallucination_info": {
        "confidence_score": 0.85,
        "hallucination_probability": "LOW",
        "token_log_probability": -0.15
      }
    }
  ]
}
```

2. **Token-level** (in logprobs):
```json
{
  "logprobs": {
    "tokens": ["The", "capital", "is", "Paris"],
    "token_logprobs": [-0.1, -0.2, -0.05, -0.3],
    "confidence_scores": [0.90, 0.82, 0.95, 0.74],
    "hallucination_probabilities": ["MINIMAL", "LOW", "MINIMAL", "MEDIUM"],
    "token_log_probabilities": [-0.1, -0.2, -0.05, -0.3]
  }
}
```

### Chat API Response

Chat completions include similar information:

```json
{
  "choices": [
    {
      "message": {"role": "assistant", "content": "Generated text..."},
      "hallucination_info": {
        "confidence_score": 0.78,
        "hallucination_probability": "MEDIUM"
      },
      "logprobs": {
        "content": [
          {
            "token": "The",
            "logprob": -0.1,
            "confidence_score": 0.90,
            "hallucination_probability": "MINIMAL",
            "token_log_probability": -0.1
          }
        ]
      }
    }
  ]
}
```

## Risk Level Classification

Confidence scores are mapped to risk levels:

| Confidence Score | Risk Level | Description |
|-----------------|------------|-------------|
| ≥ 0.95          | MINIMAL    | Very high confidence, minimal hallucination risk |
| 0.85 - 0.95     | LOW        | High confidence, low hallucination risk |
| 0.70 - 0.85     | MEDIUM     | Moderate confidence, some caution advised |
| 0.50 - 0.70     | HIGH       | Low confidence, high hallucination risk |
| < 0.50          | CRITICAL   | Very low confidence, critical hallucination risk |

## Best Practices

### 1. Model Selection
- Works with any text generation model in vLLM
- Larger models typically show better calibrated confidence scores
- Chat-tuned models may have different confidence characteristics

### 2. Sampling Parameters
- **Always enable logprobs** (`logprobs > 0` or `logprobs: true`)
- Lower temperature often correlates with higher confidence
- Consider using `top_p` for more diverse outputs with confidence tracking

### 3. Threshold Setting
- Set confidence thresholds based on your use case:
  - **Critical applications**: Only accept MINIMAL/LOW risk (≥0.85)
  - **General use**: Accept up to MEDIUM risk (≥0.70)
  - **Creative tasks**: May accept HIGH risk (≥0.50)

### 4. Token-Level Analysis
- Monitor sustained periods of low confidence
- Look for confidence drops mid-sequence (may indicate hallucination onset)
- Consider regenerating if multiple consecutive tokens show HIGH/CRITICAL risk

### 5. Sequence-Level Monitoring
- Use sequence confidence for overall response quality assessment
- Combine with token-level analysis for fine-grained control
- Consider confidence trends over time

## Integration Examples

### 1. Confidence-Based Response Filtering

```python
def filter_by_confidence(outputs, min_confidence=0.70):
    """Filter outputs by minimum confidence score."""
    filtered = []
    for output in outputs:
        completion = output.outputs[0]
        if (completion.sequence_hallucination_info and 
            completion.sequence_hallucination_info.confidence_score >= min_confidence):
            filtered.append(output)
    return filtered
```

### 2. Real-time Confidence Monitoring

```python
class ConfidenceMonitor:
    def __init__(self, warning_threshold=0.70, stop_threshold=0.50):
        self.warning_threshold = warning_threshold
        self.stop_threshold = stop_threshold
        self.low_confidence_streak = 0
    
    def check_token(self, confidence_score):
        if confidence_score < self.stop_threshold:
            return "STOP"  # Stop generation
        elif confidence_score < self.warning_threshold:
            self.low_confidence_streak += 1
            if self.low_confidence_streak >= 3:
                return "WARNING"  # Warn about sustained low confidence
        else:
            self.low_confidence_streak = 0
        return "OK"
```

### 3. Batch Quality Assessment

```python
def assess_batch_quality(outputs):
    """Assess quality metrics for a batch of outputs."""
    metrics = {
        "total_responses": len(outputs),
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "avg_confidence": 0.0
    }
    
    confidences = []
    for output in outputs:
        completion = output.outputs[0]
        if completion.sequence_hallucination_info:
            conf = completion.sequence_hallucination_info.confidence_score
            confidences.append(conf)
            
            if conf >= 0.85:
                metrics["high_confidence"] += 1
            elif conf >= 0.70:
                metrics["medium_confidence"] += 1
            else:
                metrics["low_confidence"] += 1
    
    if confidences:
        metrics["avg_confidence"] = sum(confidences) / len(confidences)
    
    return metrics
```

## Performance Considerations

### 1. Computational Overhead
- Minimal overhead: ~1-3% additional latency
- JIT compilation optimizes repeated operations
- Preallocated buffers reduce memory allocations

### 2. Memory Usage
- Streaming scorer maintains small sliding window (default 50 tokens)
- Token-level info stored only when needed
- Automatic cleanup for finished sequences

### 3. Scaling
- Hallucination detection scales linearly with sequence length
- Batch processing optimized with vectorized operations
- No impact on model inference speed

## Troubleshooting

### Common Issues

1. **No hallucination info in responses**
   - Ensure `VLLM_ENABLE_HALLUCINATION_DETECTION=1`
   - Verify `logprobs > 0` in sampling parameters
   - Check that numba is installed for optimizations

2. **All confidence scores are similar**
   - May indicate poorly calibrated model
   - Try adjusting temperature or sampling parameters
   - Consider different scoring method (min vs normalized)

3. **Performance degradation**
   - Reduce window size for streaming detection
   - Disable token-level tracking if only sequence-level needed
   - Ensure JIT compilation is working (first few requests may be slower)

### Debug Mode

Enable debug logging to troubleshoot:

```python
import logging
logging.getLogger("vllm.hallucination_detection").setLevel(logging.DEBUG)
```

## Advanced Configuration

### Custom Scoring Windows

For different use cases, adjust the sliding window size:

```bash
# Short sequences or real-time applications
export VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE=20

# Long sequences or high accuracy needs  
export VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE=100
```

### Method Selection Guidelines

Choose scoring method based on your needs:

- **Normalized method** for:
  - General purpose applications
  - Balanced confidence assessment
  - Long sequence generation

- **Min method** for:
  - Critical applications requiring high certainty
  - Detecting individual problematic tokens
  - Conservative confidence estimates

## Limitations

1. **Model Dependency**: Confidence calibration varies by model
2. **Logprobs Required**: Must enable logprobs (increases memory usage)
3. **Probability-Based**: Only detects uncertainty, not factual correctness
4. **Token-Level Only**: Doesn't detect higher-level semantic issues

## Support

For issues or questions:
- Check the troubleshooting section above
- Review example code in `/examples/hallucination_detection_example.py`
- Run tests with `python tests/test_hallucination_detection.py`
- File issues on the vLLM GitHub repository