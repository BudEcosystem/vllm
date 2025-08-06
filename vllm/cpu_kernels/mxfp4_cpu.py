# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-compatible implementations for mxfp4 quantization.

This module provides CPU equivalents for triton_kernels components used in mxfp4
quantization, enabling mxfp4 models to run on CPU backend.
"""

import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import AMX operations
try:
    from .amx_ops import (
        is_amx_available,
        amx_dequant_mxfp4_to_bf16,
        amx_mxfp4_linear,
        AMXConfig
    )
    _AMX_AVAILABLE = is_amx_available()
except ImportError:
    _AMX_AVAILABLE = False
    logger.debug("AMX operations not available, using standard CPU implementation")


class InFlexData:
    """CPU version of InFlexData for flexible data context."""
    def __init__(self):
        pass


class FlexCtx:
    """CPU version of FlexCtx for flexible data context."""
    def __init__(self, rhs_data: Optional[InFlexData] = None):
        self.rhs_data = rhs_data if rhs_data is not None else InFlexData()


class PrecisionConfig:
    """CPU version of PrecisionConfig for precision configuration."""
    def __init__(self, 
                 weight_scale: Optional[torch.Tensor] = None, 
                 flex_ctx: Optional[FlexCtx] = None,
                 limit: float = 1.0):
        self.weight_scale = weight_scale
        self.flex_ctx = flex_ctx if flex_ctx is not None else FlexCtx()
        self.limit = limit


def cpu_swizzle_mxfp4(quant_tensor: torch.Tensor, 
                      scale: torch.Tensor, 
                      num_warps: int) -> Tuple[torch.Tensor, InFlexData, torch.Tensor]:
    """CPU-optimized memory layout for mxfp4 tensors.
    
    For CPU, we don't need GPU-specific swizzling patterns.
    We transpose for better cache locality on CPU.
    
    Args:
        quant_tensor: Quantized tensor in mxfp4 format
        scale: Per-block scale factors
        num_warps: Number of warps (ignored on CPU)
        
    Returns:
        Tuple of (transposed tensor, flex data, transposed scale)
    """
    # Transpose for better CPU memory access patterns
    quant_tensor_t = quant_tensor.transpose(-2, -1).contiguous()
    scale_t = scale.transpose(-2, -1).contiguous()
    
    return quant_tensor_t, InFlexData(), scale_t


def unpack_4bit_to_8bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack 4-bit values (2 per byte) to 8-bit tensor.
    
    Args:
        packed: Packed tensor with 2 4-bit values per byte
        
    Returns:
        Unpacked tensor with one value per byte
    """
    # Extract high and low nibbles
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    
    # Create output tensor with doubled last dimension
    shape = list(packed.shape)
    shape[-1] *= 2
    unpacked = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    
    # Interleave to restore original order
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    
    return unpacked


def mxfp4_to_float(value: int, exponent: int) -> float:
    """Convert a 4-bit mxfp4 value to float using shared exponent.
    
    mxfp4 format: 1 sign bit, 3 mantissa bits
    The exponent is shared across a block of values.
    
    Args:
        value: 4-bit integer value
        exponent: Shared exponent for the block
        
    Returns:
        Float representation
    """
    # Extract sign and mantissa from 4-bit value
    sign = (value >> 3) & 0x1
    mantissa = value & 0x7
    
    # Convert to float
    # mxfp4 uses a biased representation
    if mantissa == 0 and sign == 0:
        return 0.0
    
    # Construct the float value
    float_val = (1.0 + mantissa / 8.0) * (2.0 ** exponent)
    
    return -float_val if sign else float_val


def apply_block_scale(data: torch.Tensor, 
                     scale: torch.Tensor, 
                     block_size: int = 32) -> torch.Tensor:
    """Apply per-block scaling to dequantize mxfp4 data.
    
    Args:
        data: Unpacked 8-bit data from mxfp4
        scale: Per-block scale factors
        block_size: Size of each scaling block (default: 32)
        
    Returns:
        Dequantized tensor
    """
    *batch_dims, seq_len = data.shape
    
    # Handle case where sequence length is not divisible by block_size
    if seq_len % block_size != 0:
        # Pad to nearest block_size
        pad_len = block_size - (seq_len % block_size)
        data = torch.nn.functional.pad(data, (0, pad_len))
        seq_len_padded = seq_len + pad_len
    else:
        seq_len_padded = seq_len
    
    num_blocks = seq_len_padded // block_size
    
    # Reshape to expose blocks
    data_blocks = data[..., :seq_len_padded].reshape(*batch_dims, num_blocks, block_size)
    
    # Ensure scale has the right shape
    if scale.shape[-1] != num_blocks:
        # Adjust scale if needed
        scale = scale[..., :num_blocks]
    
    # Expand scale to match data blocks
    scale_expanded = scale.unsqueeze(-1).expand_as(data_blocks)
    
    # Apply scaling
    dequantized = data_blocks.float() * scale_expanded
    
    # Reshape back and trim padding if needed
    dequantized = dequantized.reshape(*batch_dims, seq_len_padded)
    if seq_len != seq_len_padded:
        dequantized = dequantized[..., :seq_len]
    
    return dequantized


def dequant_mxfp4_cpu(x: torch.Tensor, 
                      scale: torch.Tensor, 
                      float_dtype: torch.dtype) -> torch.Tensor:
    """CPU implementation of mxfp4 dequantization.
    
    Based on the mxfp4 paper:
    - Each 4-bit value is dequantized using its block's shared scale
    - Block size is typically 32 elements
    
    This function will use AMX acceleration when available and the
    target dtype is BF16, falling back to standard implementation otherwise.
    
    Args:
        x: Packed mxfp4 tensor (4-bit values, 2 per byte)
        scale: Per-block scale factors
        float_dtype: Target floating point dtype
        
    Returns:
        Dequantized tensor in float_dtype
    """
    # Use AMX-accelerated path if available and dtype is BF16
    if _AMX_AVAILABLE and float_dtype == torch.bfloat16:
        logger.debug("Using AMX-accelerated mxfp4 dequantization")
        return amx_dequant_mxfp4_to_bf16(x, scale, block_size=32)
    # Unpack 4-bit values to 8-bit
    x_unpacked = unpack_4bit_to_8bit(x)
    
    # Convert to signed values (mxfp4 uses signed representation)
    # 4-bit signed range: -8 to 7
    x_signed = x_unpacked.to(torch.int8)
    x_signed = torch.where(x_signed > 7, x_signed - 16, x_signed)
    
    # Apply per-block scaling
    output = apply_block_scale(x_signed, scale, block_size=32)
    
    return output.to(float_dtype)


def mxfp4_linear_cpu(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """CPU implementation of mxfp4 quantized linear layer.
    
    Performs fused dequantization and matrix multiplication,
    using AMX acceleration when available.
    
    Args:
        input: Input tensor
        packed_weight: Packed mxfp4 weights
        weight_scales: Per-block scales for weights
        bias: Optional bias tensor
        output_dtype: Output dtype (defaults to input dtype)
        
    Returns:
        Output tensor
    """
    output_dtype = output_dtype or input.dtype
    
    # Use AMX-accelerated path if available
    if _AMX_AVAILABLE and output_dtype in [torch.bfloat16, torch.float32]:
        logger.debug("Using AMX-accelerated mxfp4 linear")
        return amx_mxfp4_linear(input, packed_weight, weight_scales, bias)
    
    # Fallback to standard implementation
    # Dequantize weights
    weight = dequant_mxfp4_cpu(packed_weight, weight_scales, input.dtype)
    
    # Perform matrix multiplication
    output = torch.matmul(input, weight.t() if weight.dim() == 2 else weight)
    
    if bias is not None:
        output = output + bias
    
    return output.to(output_dtype)


def quant_dequant_mxfp4_cpu(x: torch.Tensor,
                            scale_calculation_mode: str = "even") -> torch.Tensor:
    """CPU implementation of mxfp4 quantization and dequantization.
    
    This is used for fake quantization during training.
    
    Args:
        x: Input tensor to quantize and dequantize
        scale_calculation_mode: Mode for calculating scales ("even" or other)
        
    Returns:
        Quantized and dequantized tensor (same shape as input)
    """
    # For CPU, we implement a simple fake quantization
    # that approximates the mxfp4 behavior
    
    block_size = 32
    *batch_dims, seq_len = x.shape
    
    # Pad if necessary
    if seq_len % block_size != 0:
        pad_len = block_size - (seq_len % block_size)
        x_padded = torch.nn.functional.pad(x, (0, pad_len))
        seq_len_padded = seq_len + pad_len
    else:
        x_padded = x
        seq_len_padded = seq_len
    
    num_blocks = seq_len_padded // block_size
    
    # Reshape to blocks
    x_blocks = x_padded.reshape(*batch_dims, num_blocks, block_size)
    
    # Calculate per-block scales (max absolute value per block)
    scales = x_blocks.abs().max(dim=-1, keepdim=True)[0]
    
    # Avoid division by zero
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    
    # Normalize by scale
    x_normalized = x_blocks / scales
    
    # Quantize to 4-bit range (-7 to 7, avoiding -8 for symmetry)
    x_quantized = torch.clamp(torch.round(x_normalized * 7), -7, 7)
    
    # Dequantize
    x_dequantized = (x_quantized / 7) * scales
    
    # Reshape back
    x_dequantized = x_dequantized.reshape(*batch_dims, seq_len_padded)
    
    # Remove padding if added
    if seq_len != seq_len_padded:
        x_dequantized = x_dequantized[..., :seq_len]
    
    return x_dequantized