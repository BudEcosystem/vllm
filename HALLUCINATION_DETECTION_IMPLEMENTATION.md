# Hallucination Detection Integration in vLLM

## Summary

This implementation integrates real-time hallucination detection into vLLM's generation pipeline. The hallucination scorer is automatically initialized when a Sequence object is created if the `VLLM_ENABLE_HALLUCINATION_DETECTION` environment variable is set to `1`.

## Key Changes

### 1. Sequence Initialization (`vllm/sequence.py`)

- Added `_initialize_hallucination_scorer()` method to the `Sequence` class
- The method is called during `__init__` to check if hallucination detection is enabled
- If enabled, it creates an `OptimizedWhiteBoxScorer` and initializes a `StreamingScorer`
- Configuration is controlled by environment variables:
  - `VLLM_ENABLE_HALLUCINATION_DETECTION`: Enable/disable detection (default: 0)
  - `VLLM_HALLUCINATION_DETECTION_METHOD`: Scoring method - "min" or "normalized" (default: "normalized")
  - `VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE`: Window size for streaming detection (default: 50)

### 2. Token Scoring (`vllm/sequence.py`)

- The existing `append_token_id()` method already contains logic to update the hallucination scorer
- For each generated token, it:
  - Extracts the chosen token's log probability
  - Converts it to probability using `math.exp()`
  - Updates the streaming scorer and gets hallucination info
  - Appends the info to the sequence's `hallucination_info` list

### 3. Environment Variables (`vllm/envs.py`)

- Already defined in the environment variables system:
  - `VLLM_ENABLE_HALLUCINATION_DETECTION`
  - `VLLM_HALLUCINATION_DETECTION_METHOD`
  - `VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE`

### 4. Output Integration (`vllm/outputs.py`)

- `CompletionOutput` class already has fields for hallucination info:
  - `hallucination_info`: List of per-token hallucination information
  - `sequence_hallucination_info`: Overall sequence hallucination assessment
- The `RequestOutput.from_seq_group()` method populates these fields from the sequence data

### 5. API Response Integration

- OpenAI-compatible API endpoints (`serving_chat.py` and `serving_completion.py`) already include hallucination info in responses
- Per-token info is included in logprobs when available
- Sequence-level info is included in the choice data

## Usage

To enable hallucination detection:

```bash
export VLLM_ENABLE_HALLUCINATION_DETECTION=1
export VLLM_HALLUCINATION_DETECTION_METHOD=normalized  # or "min"
export VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE=50
```

Then start vLLM normally. The hallucination detection will be automatically initialized for all new sequences.

## Implementation Details

### Initialization Flow

1. When a `Sequence` object is created in `llm_engine.py` (line 585), the constructor is called
2. The constructor calls `_initialize_hallucination_scorer()`
3. This method:
   - Imports `envs` to check if detection is enabled
   - If enabled, imports the hallucination detection module
   - Creates an `OptimizedWhiteBoxScorer` instance
   - Initializes a `StreamingScorer` with the configured method and window size
   - Assigns it to `self.hallucination_scorer`

### Scoring Flow

1. During generation, `append_token_id()` is called for each new token
2. If `hallucination_scorer` is not None:
   - Extract the chosen token's probability
   - Call `scorer.update()` to get hallucination info
   - Append to the sequence's hallucination info list

### Output Flow

1. When creating output responses, the hallucination info is extracted from sequences
2. Both per-token and sequence-level info are included in API responses
3. The info includes confidence scores, risk levels, and token log probabilities

## Notes

- The implementation uses lazy imports to avoid circular dependencies
- The hallucination detection module (`hallucination_detection.py`) is already optimized with JIT compilation
- The integration is minimal and non-invasive - it only activates when explicitly enabled
- All necessary hooks and data structures were already in place in the codebase