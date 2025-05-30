# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It uses PagedAttention for efficient memory management and supports various optimization techniques including continuous batching, CUDA/HIP graph execution, and multiple quantization methods.

## Key Architecture Components

### Core Engine Architecture
- **LLMEngine/AsyncLLMEngine** (`vllm/engine/`): Central orchestrator managing request lifecycle, scheduling, and model execution
- **Scheduler** (`vllm/core/scheduler.py`): Implements continuous batching and request scheduling with preemption support
- **BlockManager** (`vllm/core/block_manager.py`): Manages KV cache memory using PagedAttention algorithm
- **Worker/ModelRunner** (`vllm/worker/`): Handles actual model execution on GPUs/CPUs with support for various backends

### Attention Mechanism
- **PagedAttention**: Core innovation that manages attention KV cache as virtual memory with paging
- Custom CUDA kernels in `csrc/attention/` for optimized attention computation
- Support for various attention backends including FlashAttention and FlashInfer

### Model Execution Flow
1. Requests enter through API server or LLM entrypoint
2. Scheduler batches requests based on available memory
3. BlockManager allocates KV cache blocks
4. Worker executes model with optimized kernels
5. Results are post-processed and returned

## Development Commands

### Setup and Installation
```bash
# Install from source
pip install -e .

# Install development dependencies
pip install -r requirements/dev.txt

# Set up pre-commit hooks (required for development)
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/core/              # Core functionality tests
pytest tests/models/            # Model-specific tests
pytest tests/kernels/           # CUDA kernel tests
pytest tests/distributed/       # Multi-GPU tests
pytest tests/engine/           # Engine tests

# Run a single test file
pytest tests/core/test_scheduler.py

# Run with specific markers or patterns
pytest -k "test_scheduler" 
pytest -v tests/                # Verbose output
```

### Linting and Formatting
```bash
# Run all pre-commit hooks manually
pre-commit run --all-files

# The linting system includes:
# - yapf (Python formatter)
# - ruff (Python linter)
# - isort (import sorter)
# - clang-format (C++/CUDA formatter)
# - codespell (spell checker)
# - mypy (type checker)
```

### Building Documentation
```bash
# Install docs dependencies
pip install -r requirements/docs.txt

# Serve documentation locally
mkdocs serve

# Documentation will be available at http://localhost:8000
```

### Running vLLM
```bash
# Serve a model with OpenAI-compatible API
vllm serve <model_name> [options]

# Example with common options
vllm serve meta-llama/Llama-2-7b-hf \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# Run offline inference
python examples/offline_inference/basic/basic.py
```

### Benchmarking
```bash
# Throughput benchmark
python benchmarks/benchmark_throughput.py \
    --model <model_name> \
    --input-len 256 \
    --output-len 128

# Latency benchmark
python benchmarks/benchmark_latency.py \
    --model <model_name>

# Serving benchmark
python benchmarks/benchmark_serving.py \
    --model <model_name> \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json
```

## Code Organization

### Python Package Structure
- `vllm/entrypoints/`: API server and CLI entry points
- `vllm/engine/`: Core engine implementation (scheduling, async handling)
- `vllm/core/`: Core algorithms (scheduler, block manager)
- `vllm/model_executor/`: Model loading and execution logic
- `vllm/worker/`: GPU/CPU worker implementations
- `vllm/attention/`: Attention layer implementations
- `vllm/multimodal/`: Multi-modal input handling
- `vllm/distributed/`: Distributed execution utilities
- `vllm/quantization/`: Quantization method implementations

### C++/CUDA Code
- `csrc/attention/`: PagedAttention CUDA kernels
- `csrc/quantization/`: Quantized operation kernels
- `csrc/moe/`: Mixture of Experts kernels
- `csrc/cache_kernels.cu`: KV cache manipulation kernels
- `csrc/cuda_utils.h`: CUDA utility functions

### Key Abstractions
- **SequenceGroup**: Groups sequences sharing the same prompt for efficient batching
- **Block**: Unit of KV cache memory (typically 16 tokens)
- **LogicalBlock/PhysicalBlock**: Logical vs physical memory mapping for PagedAttention
- **SamplingParams**: Controls text generation behavior
- **RequestOutput**: Encapsulates generated tokens and metadata

## Testing Guidelines

- Tests use pytest framework with custom fixtures in `conftest.py` files
- Model tests compare outputs against HuggingFace implementations
- Kernel tests verify correctness of CUDA operations
- Use `@pytest.mark.parametrize` for testing multiple configurations
- Distributed tests require multiple GPUs and use special decorators

## Build System Notes

- CMake handles C++/CUDA compilation with custom extensions in `setup.py`
- Supports multiple CUDA architectures and compute capabilities
- Environment variables control build options (e.g., `VLLM_TARGET_DEVICE`)
- Uses ninja for faster builds when available

## Common Development Patterns

### Adding New Models
1. Implement model in `vllm/model_executor/models/`
2. Register in `vllm/model_executor/models/registry.py`
3. Add tests in `tests/models/`
4. Update supported models documentation

### Adding New Quantization Methods
1. Implement in `vllm/quantization/`
2. Add corresponding CUDA kernels if needed
3. Register in quantization config
4. Add tests for accuracy and performance

### Optimizing Kernels
1. CUDA kernels go in `csrc/`
2. Python bindings in `vllm/_custom_ops.py`
3. Benchmark against existing implementations
4. Ensure compatibility across GPU architectures

## Environment Variables

Key environment variables that affect vLLM behavior:
- `VLLM_ATTENTION_BACKEND`: Force specific attention implementation
- `VLLM_USE_TRITON_FLASH_ATTN`: Enable Triton-based FlashAttention
- `VLLM_CPU_ONLY`: Run in CPU-only mode
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility
- `VLLM_WORKER_MULTIPROC_METHOD`: Control multiprocessing method