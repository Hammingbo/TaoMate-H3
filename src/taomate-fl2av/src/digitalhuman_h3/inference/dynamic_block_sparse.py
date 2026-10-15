from __future__ import annotations

import math
from typing import Any


_COMPILED_FLEX_ATTENTION: Any | None = None


def _block_means(torch: Any, tensor: Any, *, block_size: int) -> Any:
    length = int(tensor.shape[-2])
    blocks = math.ceil(length / block_size)
    padded_length = blocks * block_size
    if padded_length != length:
        tensor = torch.nn.functional.pad(
            tensor,
            (0, 0, 0, padded_length - length),
        )
    grouped = tensor.reshape(
        *tensor.shape[:-2],
        blocks,
        block_size,
        tensor.shape[-1],
    )
    counts = torch.full(
        (blocks,),
        block_size,
        dtype=torch.float32,
        device=tensor.device,
    )
    if padded_length != length:
        counts[-1] = length - (blocks - 1) * block_size
    return grouped.float().sum(dim=-2) / counts.view(
        *((1,) * (grouped.ndim - 3)), blocks, 1
    )


def _compiled_flex_attention(torch: Any) -> Any:
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention,
            dynamic=False,
        )
    return _COMPILED_FLEX_ATTENTION


def _dynamic_history_mask(
    torch: Any,
    *,
    query: Any,
    history_key: Any,
    total_key_tokens: int,
    density: float,
    block_size: int,
) -> tuple[Any, dict[str, int | float]]:
    from torch.nn.attention.flex_attention import BlockMask

    query_tokens = int(query.shape[-2])
    history_tokens = int(history_key.shape[-2])
    full_history_blocks = history_tokens // block_size
    total_history_blocks = math.ceil(history_tokens / block_size)
    total_key_blocks = math.ceil(total_key_tokens / block_size)
    if full_history_blocks < 1:
        raise ValueError("dynamic sparse history has no complete K/V block")
    selected_count = max(
        1,
        min(full_history_blocks, math.ceil(full_history_blocks * density)),
    )
    query_summary = _block_means(torch, query, block_size=block_size)
    key_summary = _block_means(
        torch,
        history_key[..., : full_history_blocks * block_size, :],
        block_size=block_size,
    )
    scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        query_summary,
        key_summary,
    ).mul_(query.shape[-1] ** -0.5)
    selected = scores.topk(selected_count, dim=-1, sorted=False).indices
    # The partial historical tail and every current-document block stay dense.
    # Keeping them in the same BlockMask lets FlexAttention execute the exact
    # "sparse history + dense current" semantics in one kernel call.
    always_dense = torch.arange(
        full_history_blocks,
        total_key_blocks,
        device=selected.device,
        dtype=selected.dtype,
    ).view(1, 1, 1, -1)
    always_dense = always_dense.expand(*selected.shape[:-1], -1)
    indices = torch.cat((selected, always_dense), dim=-1)
    indices = indices.sort(dim=-1).values.to(torch.int32)
    counts = torch.full(
        indices.shape[:-1],
        indices.shape[-1],
        device=indices.device,
        dtype=torch.int32,
    )
    padded_indices = torch.zeros(
        (*indices.shape[:-1], total_key_blocks),
        device=indices.device,
        dtype=torch.int32,
    )
    padded_indices[..., : indices.shape[-1]].copy_(indices)
    mask = BlockMask.from_kv_blocks(
        counts,
        padded_indices,
        BLOCK_SIZE=block_size,
        seq_lengths=(query_tokens, total_key_tokens),
    )
    return mask, {
        "history_tokens": history_tokens,
        "history_blocks": total_history_blocks,
        "selected_history_blocks": selected_count
        + int(total_history_blocks > full_history_blocks),
        "effective_history_density": float(
            selected_count + int(total_history_blocks > full_history_blocks)
        )
        / total_history_blocks,
        "always_dense_key_blocks": int(always_dense.shape[-1]),
    }


def dynamic_history_block_sparse_attention(
    torch: Any,
    *,
    query: Any,
    history_key: Any,
    history_value: Any,
    current_key: Any,
    current_value: Any,
    softmax_scale: float,
    history_density: float,
    block_size: int = 128,
) -> tuple[Any, dict[str, int | float]]:
    """Attend to dynamically selected history blocks and dense current K/V.

    All tensors use H3's post-Ulysses ``[tokens, heads, head_dim]`` layout.
    Historical selection is refreshed from the current post-RoPE Q/K for every
    call.  The current document remains fully dense inside the same forward-only
    BlockMask, so no second attention call or log-sum-exp merge is required.
    """

    if not 0.0 < history_density <= 1.0:
        raise ValueError("dynamic history density must be in (0, 1]")
    if block_size != 128:
        raise ValueError("SparSTAR-compatible H3 block size must be 128")
    q_bhsd = query.transpose(0, 1).unsqueeze(0)
    history_k_bhsd = history_key.transpose(0, 1).unsqueeze(0)
    combined_key = torch.cat((history_key, current_key), dim=0)
    combined_value = torch.cat((history_value, current_value), dim=0)
    combined_k_bhsd = combined_key.transpose(0, 1).unsqueeze(0)
    combined_v_bhsd = combined_value.transpose(0, 1).unsqueeze(0)
    block_mask, receipt = _dynamic_history_mask(
        torch,
        query=q_bhsd,
        history_key=history_k_bhsd,
        total_key_tokens=int(combined_key.shape[0]),
        density=history_density,
        block_size=block_size,
    )
    output = _compiled_flex_attention(torch)(
        q_bhsd,
        combined_k_bhsd,
        combined_v_bhsd,
        block_mask=block_mask,
        scale=softmax_scale,
    )
    return output.transpose(1, 2).squeeze(0).to(query.dtype), receipt
