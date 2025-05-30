#!/usr/bin/env python3
"""
vLLM Hallucination Detection Example

This example demonstrates how to use vLLM's integrated hallucination detection
functionality to monitor model confidence and detect potential hallucinations
in real-time during text generation.

Features demonstrated:
1. Real-time token-level hallucination detection
2. Sequence-level confidence scoring  
3. Streaming and non-streaming API usage
4. Batch processing with hallucination metrics
5. Configuration options for different detection methods

Requirements:
- vLLM with hallucination detection enabled
- numba for optimized computations
- A compatible model (any text generation model)
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Any

# Set environment variables to enable hallucination detection
os.environ["VLLM_ENABLE_HALLUCINATION_DETECTION"] = "1"
os.environ["VLLM_HALLUCINATION_DETECTION_METHOD"] = "normalized"  # or "min" 
os.environ["VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE"] = "50"

import requests
from vllm import LLM, SamplingParams


def example_offline_inference():
    """Example using vLLM offline inference with hallucination detection."""
    print("=== Offline Inference with Hallucination Detection ===")
    
    # Initialize the LLM (hallucination detection is auto-enabled via env vars)
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",  # Use any available model
        max_model_len=512,
        gpu_memory_utilization=0.8
    )
    
    # Prepare sampling parameters with logprobs enabled
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
        logprobs=10,  # Required for hallucination detection
    )
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "Write a short story about a dragon who",
        "Explain quantum physics in simple terms:",
        "What is 2+2? Let me think step by step:",
    ]
    
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Display results with hallucination info
    for i, output in enumerate(outputs):
        print(f"\n--- Prompt {i+1}: {prompts[i]} ---")
        
        for j, completion in enumerate(output.outputs):
            print(f"\nCompletion {j+1}:")
            print(f"Text: {completion.text}")
            
            # Display sequence-level hallucination info
            if completion.sequence_hallucination_info:
                info = completion.sequence_hallucination_info
                print(f"Overall Confidence: {info.confidence_score:.3f}")
                print(f"Risk Level: {info.risk_level.value}")
            
            # Display token-level hallucination info
            if completion.hallucination_info:
                print(f"\nToken-level Analysis ({len(completion.hallucination_info)} tokens):")
                for k, token_info in enumerate(completion.hallucination_info[:5]):  # Show first 5
                    print(f"  Token {k+1}: conf={token_info.confidence_score:.3f}, "
                          f"risk={token_info.risk_level.value}, "
                          f"logprob={token_info.token_logprob:.3f}")
                if len(completion.hallucination_info) > 5:
                    print(f"  ... and {len(completion.hallucination_info) - 5} more tokens")


def example_openai_api_completion():
    """Example using OpenAI-compatible completion API with hallucination detection."""
    print("\n=== OpenAI Completion API with Hallucination Detection ===")
    
    # Note: This assumes vLLM server is running with hallucination detection enabled
    # Start server with: vllm serve meta-llama/Llama-2-7b-chat-hf --api-key token-123
    
    url = "http://localhost:8000/v1/completions"
    headers = {
        "Authorization": "Bearer token-123",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "The three laws of robotics are:",
        "max_tokens": 150,
        "temperature": 0.7,
        "logprobs": 5,  # Enable logprobs for hallucination detection
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        choice = result["choices"][0]
        
        print(f"Generated text: {choice['text']}")
        
        # Display hallucination info if available
        if "hallucination_info" in choice:
            hal_info = choice["hallucination_info"]
            print(f"Sequence confidence: {hal_info.get('confidence_score', 'N/A')}")
            print(f"Hallucination probability: {hal_info.get('hallucination_probability', 'N/A')}")
        
        # Display token-level hallucination info from logprobs
        if choice.get("logprobs") and hasattr(choice["logprobs"], "confidence_scores"):
            logprobs = choice["logprobs"]
            if logprobs.get("confidence_scores"):
                print(f"\nToken-level confidence scores:")
                for i, (token, conf, risk) in enumerate(zip(
                    logprobs["tokens"][:5],
                    logprobs["confidence_scores"][:5], 
                    logprobs["hallucination_probabilities"][:5]
                )):
                    print(f"  '{token}': conf={conf:.3f}, risk={risk}")
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print("Make sure vLLM server is running with hallucination detection enabled")


def example_streaming_chat():
    """Example using streaming chat completion with hallucination detection."""
    print("\n=== Streaming Chat with Hallucination Detection ===")
    
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer token-123", 
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [
            {"role": "user", "content": "Explain the process of photosynthesis"}
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "logprobs": True,
        "top_logprobs": 3,
        "stream": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        print("Streaming response with real-time hallucination detection:")
        print("=" * 60)
        
        full_text = ""
        confidence_scores = []
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk["choices"][0]
                        delta = choice["delta"]
                        
                        # Display token content
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            full_text += content
                            print(content, end="", flush=True)
                        
                        # Display hallucination info if available
                        if "hallucination_info" in choice and choice["hallucination_info"]:
                            hal_info = choice["hallucination_info"]
                            conf = hal_info.get("confidence_score")
                            if conf is not None:
                                confidence_scores.append(conf)
                        
                        # Display logprobs hallucination info
                        if choice.get("logprobs") and choice["logprobs"].get("content"):
                            for token_info in choice["logprobs"]["content"]:
                                if "confidence_score" in token_info:
                                    print(f" [{token_info['confidence_score']:.2f}]", end="")
                    
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n\nFinal Statistics:")
        print(f"Total text length: {len(full_text)} characters")
        if confidence_scores:
            avg_conf = sum(confidence_scores) / len(confidence_scores)
            min_conf = min(confidence_scores)
            print(f"Average confidence: {avg_conf:.3f}")
            print(f"Minimum confidence: {min_conf:.3f}")
        
    except requests.exceptions.RequestException as e:
        print(f"Streaming request failed: {e}")


def example_batch_analysis():
    """Example of batch processing with hallucination analysis."""
    print("\n=== Batch Analysis with Hallucination Detection ===")
    
    # Test scenarios with different expected confidence levels
    test_cases = [
        ("High confidence", "What is 2 + 2?"),
        ("Medium confidence", "Write a creative story about a time-traveling cat"),
        ("Low confidence", "Predict the exact stock price of AAPL in 2030"),
        ("Mixed confidence", "Explain both the scientific facts and cultural myths about dreams")
    ]
    
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", max_model_len=256)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        logprobs=5
    )
    
    prompts = [case[1] for case in test_cases]
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        completion = output.outputs[0]
        
        result = {
            "category": test_cases[i][0],
            "prompt": test_cases[i][1],
            "text": completion.text,
            "sequence_confidence": None,
            "sequence_risk": None,
            "avg_token_confidence": None,
            "min_token_confidence": None,
            "high_risk_tokens": 0
        }
        
        # Analyze sequence-level metrics
        if completion.sequence_hallucination_info:
            info = completion.sequence_hallucination_info
            result["sequence_confidence"] = info.confidence_score
            result["sequence_risk"] = info.risk_level.value
        
        # Analyze token-level metrics
        if completion.hallucination_info:
            token_confidences = [info.confidence_score for info in completion.hallucination_info]
            result["avg_token_confidence"] = sum(token_confidences) / len(token_confidences)
            result["min_token_confidence"] = min(token_confidences)
            result["high_risk_tokens"] = sum(1 for info in completion.hallucination_info 
                                           if info.risk_level.value in ["HIGH", "CRITICAL"])
        
        results.append(result)
    
    # Display analysis
    print(f"{'Category':<20} {'Seq Conf':<10} {'Risk':<10} {'Avg Conf':<10} {'Min Conf':<10} {'High Risk':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['category']:<20} "
              f"{result['sequence_confidence']:<10.3f} " 
              f"{result['sequence_risk']:<10} "
              f"{result['avg_token_confidence']:<10.3f} "
              f"{result['min_token_confidence']:<10.3f} "
              f"{result['high_risk_tokens']:<10}")
    
    return results


def main():
    """Run all examples to demonstrate hallucination detection functionality."""
    print("vLLM Hallucination Detection Examples")
    print("=" * 50)
    
    print("Environment Configuration:")
    print(f"  VLLM_ENABLE_HALLUCINATION_DETECTION = {os.getenv('VLLM_ENABLE_HALLUCINATION_DETECTION')}")
    print(f"  VLLM_HALLUCINATION_DETECTION_METHOD = {os.getenv('VLLM_HALLUCINATION_DETECTION_METHOD')}")
    print(f"  VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE = {os.getenv('VLLM_HALLUCINATION_DETECTION_WINDOW_SIZE')}")
    print()
    
    try:
        # Run offline inference example
        example_offline_inference()
        
        # Run API examples (requires running server)
        print("\nAPI Examples (requires vLLM server running):")
        print("Start server with: vllm serve <model> --api-key token-123")
        
        # Uncomment these if you have a vLLM server running:
        # example_openai_api_completion()
        # example_streaming_chat()
        
        # Run batch analysis
        example_batch_analysis()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. vLLM installed with hallucination detection support")
        print("2. Required dependencies (numba)")
        print("3. A compatible model available")
        print("4. Sufficient GPU memory")


if __name__ == "__main__":
    main()