# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMX-accelerated MoE implementation for CPU backend."""

import torch
from typing import Optional, Any
import logging

from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import AMX operations
try:
    from .amx_ops import (
        is_amx_available,
        amx_expert_compute,
        amx_mxfp4_linear,
        amx_matmul_bf16,
        AMXConfig
    )
    _AMX_AVAILABLE = is_amx_available()
except ImportError:
    _AMX_AVAILABLE = False
    logger.debug("AMX operations not available for MoE")


def amx_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    w1_precision: Optional[Any] = None,
    w2_precision: Optional[Any] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    AMX-accelerated MoE forward pass.
    
    This implementation uses Intel AMX for efficient BF16 matrix operations
    and properly routes tokens to experts.
    
    Args:
        hidden_states: Input tensor [num_tokens, hidden_dim]
        w1: First expert weights [num_experts, intermediate_dim, hidden_dim]
        w2: Second expert weights [num_experts, hidden_dim, intermediate_dim]
        gating_output: Router logits [num_tokens, num_experts]
        topk: Number of experts to use per token
        renormalize: Whether to renormalize router weights
        activation: Activation function name
        w1_bias: Optional bias for w1
        w2_bias: Optional bias for w2
        w1_precision: Precision config for w1 (for quantized weights)
        w2_precision: Precision config for w2 (for quantized weights)
        
    Returns:
        Output tensor [num_tokens, hidden_dim]
    """
    if not _AMX_AVAILABLE:
        logger.info_once(
            "AMX not available, falling back to standard MoE implementation"
        )
        # Import fallback implementation
        from .moe_cpu import cpu_moe_forward
        return cpu_moe_forward(
            hidden_states, w1, w2, gating_output, topk, renormalize,
            activation, w1_bias, w2_bias, w1_precision, w2_precision,
            global_num_experts, expert_map, apply_router_weight_on_input,
            **kwargs
        )
    
    logger.debug("Using AMX-accelerated MoE forward")
    
    # Get dimensions
    if hidden_states.dim() == 2:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states
    elif hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len
        hidden_states_flat = hidden_states.view(num_tokens, hidden_dim)
    else:
        raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
    
    num_experts = gating_output.shape[-1]
    
    # Convert to BF16 for AMX operations
    hidden_states_bf16 = hidden_states_flat.to(torch.bfloat16)
    
    # Compute top-k experts per token
    topk_weights, topk_ids = torch.topk(gating_output, topk, dim=-1)
    
    # Renormalize weights if requested
    if renormalize:
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)
    else:
        # Use raw router scores
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    # Initialize output tensor
    output = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=hidden_states_bf16.dtype,
        device=hidden_states.device
    )
    
    # Process each expert
    for expert_idx in range(num_experts):
        # Find tokens assigned to this expert
        expert_mask = (topk_ids == expert_idx).any(dim=-1)
        
        if not expert_mask.any():
            continue
        
        # Get tokens for this expert
        expert_tokens = hidden_states_bf16[expert_mask]
        
        # Get routing weights for this expert
        # Shape: [num_tokens, topk]
        expert_weights_mask = (topk_ids == expert_idx)
        # Get the weights where this expert was selected
        expert_routing_weights = torch.where(
            expert_weights_mask,
            topk_weights,
            torch.zeros_like(topk_weights)
        )
        # Sum across topk dimension and filter to relevant tokens
        expert_routing_weights = expert_routing_weights.sum(dim=-1)[expert_mask]
        
        # Get expert parameters
        if w1.dim() >= 3:
            expert_w1 = w1[expert_idx]
            expert_w2 = w2[expert_idx]
        else:
            # Weights might be packed differently
            expert_w1 = w1
            expert_w2 = w2
        
        expert_w1_bias = w1_bias[expert_idx] if w1_bias is not None else None
        expert_w2_bias = w2_bias[expert_idx] if w2_bias is not None else None
        
        # First expert layer (gate/up projection)
        if expert_w1.dtype == torch.uint8 and w1_precision is not None:
            # Quantized weights - use AMX mxfp4 linear
            if hasattr(w1_precision, 'weight_scale'):
                intermediate = amx_mxfp4_linear(
                    expert_tokens,
                    expert_w1,
                    w1_precision.weight_scale[expert_idx],
                    expert_w1_bias
                )
            else:
                # Fallback for other quantization formats
                intermediate = torch.matmul(expert_tokens, expert_w1.t())
        else:
            # Non-quantized weights
            expert_w1_bf16 = expert_w1.to(torch.bfloat16)
            intermediate = amx_matmul_bf16(
                expert_tokens,
                expert_w1_bf16,
                expert_w1_bias
            )
        
        # Apply activation
        if activation == "silu":
            intermediate = torch.nn.functional.silu(intermediate)
        elif activation == "gelu":
            intermediate = torch.nn.functional.gelu(intermediate)
        elif activation == "relu":
            intermediate = torch.nn.functional.relu(intermediate)
        else:
            # Default to SiLU
            intermediate = torch.nn.functional.silu(intermediate)
        
        # Second expert layer (down projection)
        if expert_w2.dtype == torch.uint8 and w2_precision is not None:
            # Quantized weights
            if hasattr(w2_precision, 'weight_scale'):
                expert_output = amx_mxfp4_linear(
                    intermediate,
                    expert_w2,
                    w2_precision.weight_scale[expert_idx],
                    expert_w2_bias
                )
            else:
                expert_output = torch.matmul(intermediate, expert_w2.t())
        else:
            # Non-quantized weights
            expert_w2_bf16 = expert_w2.to(torch.bfloat16)
            expert_output = amx_matmul_bf16(
                intermediate,
                expert_w2_bf16,
                expert_w2_bias
            )
        
        # Apply routing weights and accumulate
        expert_output = expert_output * expert_routing_weights.unsqueeze(-1)
        output[expert_mask] += expert_output
    
    # Convert back to original dtype and shape
    output = output.to(hidden_states.dtype)
    
    if hidden_states.dim() == 3:
        output = output.view(batch_size, seq_len, hidden_dim)
    
    # Add residual connection for stability
    output = hidden_states + output
    
    logger.debug(f"AMX MoE forward completed: input shape {hidden_states.shape}, output shape {output.shape}")
    
    return output


def amx_moe_forward_optimized(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    **kwargs
) -> torch.Tensor:
    """
    Optimized AMX MoE forward with better batching.
    
    This version batches tokens by expert for better AMX utilization.
    """
    if not _AMX_AVAILABLE:
        return amx_moe_forward(
            hidden_states, w1, w2, gating_output, topk, renormalize,
            activation, **kwargs
        )
    
    # Flatten input if needed
    original_shape = hidden_states.shape
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len
        hidden_states = hidden_states.view(num_tokens, hidden_dim)
    else:
        num_tokens, hidden_dim = hidden_states.shape
    
    num_experts = gating_output.shape[-1]
    
    # Convert to BF16
    hidden_states = hidden_states.to(torch.bfloat16)
    
    # Get top-k routing
    topk_weights, topk_ids = torch.topk(gating_output, topk, dim=-1)
    
    if renormalize:
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)
    
    # Create permutation for token sorting by expert
    # This improves memory locality for AMX operations
    sorted_indices = []
    expert_boundaries = [0]
    
    for expert_idx in range(num_experts):
        expert_mask = (topk_ids == expert_idx).any(dim=-1)
        expert_token_indices = torch.where(expert_mask)[0]
        sorted_indices.append(expert_token_indices)
        expert_boundaries.append(expert_boundaries[-1] + len(expert_token_indices))
    
    if sorted_indices:
        sorted_indices = torch.cat(sorted_indices)
        sorted_hidden = hidden_states[sorted_indices]
        
        # Process all experts in batch
        output_sorted = torch.zeros_like(sorted_hidden)
        
        for expert_idx in range(num_experts):
            start_idx = expert_boundaries[expert_idx]
            end_idx = expert_boundaries[expert_idx + 1]
            
            if start_idx == end_idx:
                continue
            
            expert_hidden = sorted_hidden[start_idx:end_idx]
            
            # Get expert weights
            expert_w1 = w1[expert_idx] if w1.dim() >= 3 else w1
            expert_w2 = w2[expert_idx] if w2.dim() >= 3 else w2
            
            # First layer with AMX
            intermediate = amx_matmul_bf16(expert_hidden, expert_w1.to(torch.bfloat16))
            
            # Activation
            if activation == "silu":
                intermediate = torch.nn.functional.silu(intermediate)
            elif activation == "gelu":
                intermediate = torch.nn.functional.gelu(intermediate)
            else:
                intermediate = torch.nn.functional.relu(intermediate)
            
            # Second layer with AMX
            expert_output = amx_matmul_bf16(intermediate, expert_w2.to(torch.bfloat16))
            
            output_sorted[start_idx:end_idx] = expert_output
        
        # Unsort the output
        output = torch.zeros((num_tokens, hidden_dim), dtype=torch.bfloat16, device=hidden_states.device)
        output[sorted_indices] = output_sorted
    else:
        output = torch.zeros_like(hidden_states)
    
    # Apply routing weights
    for i in range(topk):
        expert_ids = topk_ids[:, i]
        weights = topk_weights[:, i].unsqueeze(-1)
        
        for expert_idx in range(num_experts):
            mask = (expert_ids == expert_idx)
            if mask.any():
                output[mask] *= weights[mask]
    
    # Reshape and convert back
    if len(original_shape) == 3:
        output = output.view(original_shape)
    
    output = output.to(hidden_states.dtype)
    
    return output