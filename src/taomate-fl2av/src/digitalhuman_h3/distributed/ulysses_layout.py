"""Inference-only layout copies with zero-padded Ulysses head shards."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_send(X, Y, TOKENS, HEADS: tl.constexpr, DIM: tl.constexpr,
              WORLD: tl.constexpr, LOCAL: tl.constexpr,
              STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % DIM
    h = (i // DIM) % LOCAL
    t = (i // (DIM * LOCAL)) % TOKENS
    qkv = (i // (DIM * LOCAL * TOKENS)) % 3
    dest = i // (DIM * LOCAL * TOKENS * 3)
    head = dest * LOCAL + h
    valid = i < WORLD * 3 * TOKENS * LOCAL * DIM
    value = tl.load(
        X + t * STRIDE + (qkv * HEADS + head) * DIM + d,
        valid & (head < HEADS), other=0,
    )
    tl.store(Y + i, value, valid)


def qkv_send_layout(value, *, num_heads, head_dim, world_size):
    tokens = value.shape[0]
    local = triton.cdiv(num_heads, world_size)
    result = torch.empty(
        (world_size, 3, tokens, local, head_dim),
        dtype=value.dtype, device=value.device,
    )
    _qkv_send[(triton.cdiv(result.numel(), 1024),)](
        value, result, tokens, num_heads, head_dim, world_size, local,
        value.stride(0), 1024,
    )
    return result


@triton.jit
def _attention_receive(X, Y, TOKENS, HEADS: tl.constexpr,
                       DIM: tl.constexpr, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % DIM
    head = (i // DIM) % HEADS
    t = i // (DIM * HEADS)
    source = head // LOCAL
    source_offset = ((source * TOKENS + t) * LOCAL + head % LOCAL) * DIM + d
    valid = i < TOKENS * HEADS * DIM
    tl.store(Y + i, tl.load(X + source_offset, valid, other=0), valid)


def attention_receive_layout(value, *, num_heads):
    _, tokens, local, dim = value.shape
    result = torch.empty((tokens, num_heads, dim), dtype=value.dtype, device=value.device)
    _attention_receive[(triton.cdiv(result.numel(), 1024),)](
        value, result, tokens, num_heads, dim, local, 1024,
    )
    return result
