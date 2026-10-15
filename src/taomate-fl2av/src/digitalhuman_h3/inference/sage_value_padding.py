"""Exact V padding/transpose fusion for H3's BF16/NHD Sage SM90 path.

Permutation adapted from SageAttention TransposePadPermuteKernel,
copyright (c) 2024 SageAttention team, Apache License 2.0.
"""

import torch
import triton
import triton.language as tl
from sageattention import core
from sageattention.quant import _fused


@triton.jit
def _transpose_pad_value(V, OUT, N, P, vb, vn, vh,
                         HEADS: tl.constexpr, DIM: tl.constexpr,
                         BN: tl.constexpr, BD: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    bh = tl.program_id(2)
    batch, head = bh // HEADS, bh % HEADS
    m = n % 16
    source_n = n // 16 * 16 + m // 4 * 2 + m % 4 // 2 * 8 + m % 2
    value = tl.load(
        V + batch * vb + source_n[:, None] * vn + head * vh + d[None, :],
        mask=(source_n[:, None] < N) & (d[None, :] < DIM), other=0,
    )
    tl.store(
        OUT + batch * DIM * HEADS * P + d[None, :] * HEADS * P + head * P + n[:, None],
        value, mask=(n[:, None] < P) & (d[None, :] < DIM),
    )


def transpose_pad_permute_value(v):
    """Return BF16 [B,D,H,ceil(N/128)*128] without materializing padded V."""
    if v.ndim != 4 or v.dtype != torch.bfloat16 or v.shape[-1] != 128 or v.stride(-1) != 1:
        raise ValueError("H3 Sage V fusion supports only BF16 NHD with D=128")
    batch, rows, heads, dim = v.shape
    padded = triton.cdiv(rows, 128) * 128
    transposed = torch.empty((batch, dim, heads, padded), dtype=v.dtype, device=v.device)
    _transpose_pad_value[(triton.cdiv(padded, 64), 1, batch * heads)](
        v, transposed, rows, padded, v.stride(0), v.stride(1), v.stride(2),
        HEADS=heads, DIM=dim, BN=64, BD=128,
    )
    return transposed


def quantize_value_with_padding(v):
    transposed = transpose_pad_permute_value(v)
    batch, dim, heads, padded = transposed.shape
    quantized = torch.empty(transposed.shape, dtype=torch.float8_e4m3fn, device=v.device)
    scale = torch.empty((batch, heads, dim), dtype=torch.float32, device=v.device)
    # Match Sage's original padded length for scale calculation.
    _fused.scale_fuse_quant_cuda(transposed, quantized, scale, padded, 448.0, 0)
    return quantized, scale


def sage_with_fused_value_padding(q, k, v, *, sm_scale=None):
    """Noncausal per-thread QK quantization; unchanged Sage attention kernel."""
    if any(x.ndim != 4 or x.dtype != torch.bfloat16 or x.shape[-1] != 128 or x.stride(-1) != 1
           for x in (q, k, v)):
        raise ValueError("H3 Sage V fusion supports only BF16 NHD with D=128")
    torch.cuda.set_device(v.device)
    mean = k.mean(dim=1, keepdim=True)
    qi, qs, ki, ks = core.per_thread_int8_triton(
        q, k, mean, tensor_layout="NHD", BLKQ=64, WARPQ=16, BLKK=128, WARPK=128,
    )
    output = torch.empty(q.size(), dtype=q.dtype, device=q.device)
    vf, vs = quantize_value_with_padding(v)
    core.sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        qi, ki, vf, output, qs, ks, vs, 0, 0, 3,
        128**-0.5 if sm_scale is None else sm_scale, 0,
    )
    return output
