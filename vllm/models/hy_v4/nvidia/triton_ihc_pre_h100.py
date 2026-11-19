# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""H100 iHC pre schedule for HC=4 and hidden=3072.

The split reductions and token tiling are adapted from FlagGems-vllm's
Apache-2.0 iHC implementation. All arithmetic and reductions remain FP32.
"""

from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_DECODE_LAUNCHES = {}
_PREFILL_LAUNCHES = {}


@cache
@torch.compiler.assume_constant_result
def is_h100(device_index: int) -> bool:
    return "H100" in current_platform.get_device_name(device_index)


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    sumsq_slot_ptr,
    mix_slot_ptr,
    N,
    X_STRIDE: tl.constexpr,
    K_CHUNK: tl.constexpr,
    TWO_HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NSPLIT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    token_block_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    m_off = token_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < N
    k_start = split_idx * K_CHUNK
    sum_squares = tl.zeros((BLOCK_M,), dtype=tl.float32)
    mixes = ()
    for _ in tl.static_range(TWO_HC):
        mixes = mixes + (tl.zeros((BLOCK_M,), dtype=tl.float32),)
    x_base = x_ptr + m_off[:, None] * X_STRIDE
    for k_offset in tl.range(0, K_CHUNK, BLOCK_K, num_stages=NUM_STAGES):
        k_off = k_start + k_offset + tl.arange(0, BLOCK_K)
        k_mask = k_off < X_STRIDE
        x = tl.load(
            x_base + k_off[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)
        sum_squares += tl.sum(x * x, axis=1)
        next_mixes = ()
        for j in tl.static_range(TWO_HC):
            weight_row = tl.load(
                weight_ptr + j * X_STRIDE + k_off, mask=k_mask, other=0.0
            )
            contribution = tl.sum(x * weight_row[None, :], axis=1)
            next_mixes = next_mixes + (mixes[j] + contribution,)
        mixes = next_mixes
    tl.store(sumsq_slot_ptr + split_idx * N + m_off, sum_squares, mask=m_mask)
    mix_slot = mix_slot_ptr + split_idx * N * TWO_HC + m_off * TWO_HC
    for j in tl.static_range(TWO_HC):
        tl.store(mix_slot + j, mixes[j], mask=m_mask)


@triton.jit
def _ihc_pre_stage1_decode(
    x_ptr,
    weight_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    TWO_HC: tl.constexpr,
    TWO_HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    k_off = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K_TOTAL
    x = tl.load(x_ptr + token_idx * K_TOTAL + k_off, mask=k_mask, other=0.0).to(
        tl.float32
    )
    sum_squares = tl.sum(x * x, axis=0)
    mix_idx = tl.arange(0, TWO_HC_POW2)
    mix_mask = mix_idx < TWO_HC
    weight = tl.load(
        weight_ptr + mix_idx[:, None] * K_TOTAL + k_off[None, :],
        mask=mix_mask[:, None] & k_mask[None, :],
        other=0.0,
    )
    mixes = tl.sum(weight * x[None, :], axis=1)
    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + mix_idx, mixes, mask=mix_mask)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    mix_slot_ptr,
    sumsq_slot_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    N,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    TWO_HC: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    split_idx = tl.arange(0, BLOCK_S)
    split_mask = split_idx < NSPLIT
    sum_squares = tl.sum(
        tl.load(sumsq_slot_ptr + split_idx * N + token_idx, mask=split_mask, other=0.0),
        axis=0,
    )
    mix_idx = tl.arange(0, 2 * HC_POW2)
    mix_parts = tl.load(
        mix_slot_ptr
        + split_idx[:, None] * N * TWO_HC
        + token_idx * TWO_HC
        + mix_idx[None, :],
        mask=split_mask[:, None] & (mix_idx[None, :] < TWO_HC),
        other=0.0,
    )
    mixes = tl.sum(mix_parts, axis=0)
    is_pre = mix_idx[None, :] == hc_idx[:, None]
    is_post = mix_idx[None, :] == hc_idx[:, None] + HC_MULT
    pre_mix = tl.sum(tl.where(is_pre, mixes[None, :], 0.0), axis=1)
    post_mix = tl.sum(tl.where(is_post, mixes[None, :], 0.0), axis=1)
    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS
    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)
    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel
    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _ihc_pre_stage2_decode(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    TWO_HC: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    split_idx = tl.arange(0, BLOCK_S)
    split_mask = split_idx < NUM_SPLITS
    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    slot_ptrs = partial_row + split_idx * PARTIAL_STRIDE
    sqrsum_parts = tl.load(slot_ptrs, mask=split_mask, other=0.0)
    sum_squares = tl.sum(sqrsum_parts, axis=0)
    mix_idx = tl.arange(0, 2 * HC_POW2)
    mix_ptrs = slot_ptrs[:, None] + 1 + mix_idx[None, :]
    mix_parts = tl.load(
        mix_ptrs, mask=split_mask[:, None] & (mix_idx[None, :] < TWO_HC), other=0.0
    )
    mixes = tl.sum(mix_parts, axis=0)
    is_pre = mix_idx[None, :] == hc_idx[:, None]
    is_post = mix_idx[None, :] == hc_idx[:, None] + HC_MULT
    pre_mix = tl.sum(tl.where(is_pre, mixes[None, :], 0.0), axis=1)
    post_mix = tl.sum(tl.where(is_post, mixes[None, :], 0.0), axis=1)
    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS
    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)
    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel
    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def _launch_decode(
    x,
    weight,
    partial,
    scale,
    base,
    output,
    post,
    n,
    hc,
    d,
    hc_pad,
    block_k,
    splits,
    partial_stride,
    block_d,
    magnitude,
    norm_eps,
    hc_eps,
    layout_tag,
    stage2_num_warps=4,
):
    k = hc * d
    first_constants = (k, 2 * hc, 2 * hc_pad, splits, block_k, partial_stride)
    second_constants = (
        d,
        k,
        hc,
        hc_pad,
        2 * hc,
        splits,
        partial_stride,
        triton.next_power_of_2(splits),
        block_d,
        magnitude,
        norm_eps,
        hc_eps,
    )
    first_grid = (n, splits, 1)
    second_grid = (n, triton.cdiv(d, block_d), 1)
    if torch.compiler.is_compiling() or not triton.__version__.startswith("3.6."):
        _ihc_pre_stage1_decode[first_grid](
            x, weight, partial, *first_constants, num_warps=1, enable_fp_fusion=False
        )
        _ihc_pre_stage2_decode[second_grid](
            x,
            partial,
            scale,
            base,
            output,
            post,
            *second_constants,
            num_warps=stage2_num_warps,
            enable_fp_fusion=False,
        )
        return
    key = (
        x.device.index,
        x.dtype,
        n,
        hc,
        d,
        layout_tag,
        first_constants,
        second_constants[:-3],
        magnitude.hex(),
        norm_eps.hex(),
        hc_eps.hex(),
        first_grid,
        second_grid,
        1,
        stage2_num_warps,
        False,
        x.data_ptr() % 16 == 0,
        weight.data_ptr() % 16 == 0,
        partial.data_ptr() % 16 == 0,
        scale.data_ptr() % 16 == 0,
        base.data_ptr() % 16 == 0,
        output.data_ptr() % 16 == 0,
        post.data_ptr() % 16 == 0,
    )
    cacheable = magnitude == magnitude and norm_eps == norm_eps and (hc_eps == hc_eps)
    cached = _DECODE_LAUNCHES.get(key) if cacheable else None
    if cached is not None:
        first, second, first_constants, second_constants = cached
        first(x, weight, partial, *first_constants)
        second(x, partial, scale, base, output, post, *second_constants)
        return
    first = _ihc_pre_stage1_decode[first_grid](
        x, weight, partial, *first_constants, num_warps=1, enable_fp_fusion=False
    )
    second = _ihc_pre_stage2_decode[second_grid](
        x,
        partial,
        scale,
        base,
        output,
        post,
        *second_constants,
        num_warps=stage2_num_warps,
        enable_fp_fusion=False,
    )
    if cacheable:
        _DECODE_LAUNCHES[key] = (
            first[first_grid],
            second[second_grid],
            first_constants,
            second_constants,
        )


def _launch_prefill(
    x,
    weight,
    sumsq,
    mix,
    scale,
    base,
    output,
    post,
    n,
    hc,
    d,
    hc_pad,
    grid_m,
    splits,
    block_m,
    block_k,
    block_d,
    num_warps,
    pipeline_stages,
    magnitude,
    norm_eps,
    hc_eps,
    layout_tag,
):
    k = hc * d
    first_constants = (
        k,
        k // splits,
        2 * hc,
        block_m,
        block_k,
        splits,
        pipeline_stages,
    )
    second_constants = (
        d,
        k,
        hc,
        hc_pad,
        2 * hc,
        splits,
        triton.next_power_of_2(splits),
        block_d,
        magnitude,
        norm_eps,
        hc_eps,
    )
    first_grid = (grid_m, splits, 1)
    second_grid = (n, triton.cdiv(d, block_d), 1)
    if torch.compiler.is_compiling() or not triton.__version__.startswith("3.6."):
        _ihc_pre_stage1[first_grid](
            x,
            weight,
            sumsq,
            mix,
            n,
            *first_constants,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        _ihc_pre_stage2[second_grid](
            x,
            mix,
            sumsq,
            scale,
            base,
            output,
            post,
            n,
            *second_constants,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return
    key = (
        x.device.index,
        x.dtype,
        n,
        hc,
        d,
        layout_tag,
        first_constants,
        second_constants[:-3],
        magnitude.hex(),
        norm_eps.hex(),
        hc_eps.hex(),
        first_grid,
        second_grid,
        num_warps,
        4,
        False,
        x.data_ptr() % 16 == 0,
        weight.data_ptr() % 16 == 0,
        sumsq.data_ptr() % 16 == 0,
        mix.data_ptr() % 16 == 0,
        scale.data_ptr() % 16 == 0,
        base.data_ptr() % 16 == 0,
        output.data_ptr() % 16 == 0,
        post.data_ptr() % 16 == 0,
    )
    cacheable = magnitude == magnitude and norm_eps == norm_eps and (hc_eps == hc_eps)
    cached = _PREFILL_LAUNCHES.get(key) if cacheable else None
    if cached is not None:
        first, second, first_constants, second_constants = cached
        first(x, weight, sumsq, mix, n, *first_constants)
        second(x, mix, sumsq, scale, base, output, post, n, *second_constants)
        return
    first = _ihc_pre_stage1[first_grid](
        x,
        weight,
        sumsq,
        mix,
        n,
        *first_constants,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    second = _ihc_pre_stage2[second_grid](
        x,
        mix,
        sumsq,
        scale,
        base,
        output,
        post,
        n,
        *second_constants,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if cacheable:
        _PREFILL_LAUNCHES[key] = (
            first[first_grid],
            second[second_grid],
            first_constants,
            second_constants,
        )


def h100_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the measured H100 schedule after public input normalization."""
    n, hc, d = x.shape
    k = hc * d
    magnitude, hc_eps, norm_eps = float(magnitude), float(hc_eps), float(norm_eps)
    output = torch.empty((n, d), dtype=x.dtype, device=x.device)
    post = torch.empty((n, hc), dtype=torch.float32, device=x.device)
    hc_pad = triton.next_power_of_2(hc)
    if n < 256:
        block_k = 512
        block_d, stage2_warps = (256, 1) if n < 128 else (1024, 4)
        splits = triton.cdiv(k, block_k)
        partial_stride = 1 + 2 * hc
        partial = torch.empty(
            (n, splits, partial_stride), dtype=torch.float32, device=x.device
        )
        _launch_decode(
            x,
            weight,
            partial,
            scale,
            base,
            output,
            post,
            n,
            hc,
            d,
            hc_pad,
            block_k,
            splits,
            partial_stride,
            block_d,
            magnitude,
            norm_eps,
            hc_eps,
            0,
            stage2_num_warps=stage2_warps,
        )
    else:
        block_m, block_k, block_d = 8, 128, 1024
        grid_m = triton.cdiv(n, block_m)
        splits = 1
        while grid_m * splits * 2 <= 512 and k % (splits * 2 * block_k) == 0:
            splits *= 2
        if 512 <= n < 2048:
            splits *= 2
        elif 2048 <= n <= 4096:
            splits *= 3
            block_d = 2048
        sumsq = torch.empty(splits * n, dtype=torch.float32, device=x.device)
        mix = torch.empty(splits * n * 2 * hc, dtype=torch.float32, device=x.device)
        _launch_prefill(
            x,
            weight,
            sumsq,
            mix,
            scale,
            base,
            output,
            post,
            n,
            hc,
            d,
            hc_pad,
            grid_m,
            splits,
            block_m,
            block_k,
            block_d,
            1,
            3,
            magnitude,
            norm_eps,
            hc_eps,
            0,
        )
    return output, post
