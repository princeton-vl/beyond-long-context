# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
import os
import warnings
from typing import Any

import torch
import triton
import triton.language as tl

from fla.ops.deltaformer import invcum

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

from fla.layers.utils import pad_input, unpad_input

BLOCK_SIZE_C = 512
DELTAFORMER_EPS = float(os.environ.get('DELTAFORMER_EPS', '1e-9'))
DELTAFORMER_DEBUG = bool(int(os.environ.get('DELTAFORMER_DEBUG', '0')))


def _ensure_finite(name: str, tensor: torch.Tensor, **context: Any) -> None:
    if not DELTAFORMER_DEBUG:
        return
    with torch.no_grad():
        tensor = tensor.detach()
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        if nan_count == 0 and inf_count == 0:
            return
        finite_mask = torch.isfinite(tensor)
        finite_vals = tensor.masked_select(finite_mask)
        stats = 'no finite values'
        if finite_vals.numel() > 0:
            finite_vals = finite_vals.float()
            stats = (
                f"min={finite_vals.min().item():.4e} "
                f"max={finite_vals.max().item():.4e} "
                f"mean={finite_vals.mean().item():.4e} "
                f"std={finite_vals.std(unbiased=False).item():.4e}"
            )
        ctx = dict(context)
        ctx_str = ', '.join(f"{k}={v}" for k, v in ctx.items()) if ctx else 'no-context'
        raise RuntimeError(
            f"[DeltaFormerDebug] Non-finite values detected in {name} "
            f"(shape={tuple(tensor.shape)}, {ctx_str}) "
            f"nan={nan_count} inf={inf_count} {stats}"
        )


def parallel_deltaformer_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    qk_scale: float,
    beta: torch.Tensor,
):
    C, H, D = q.size()
    T, _H, _D = k.size()
    __C, __H = beta.size()
    assert H == _H and D == _D and H == __H and __C == C
    w = torch.empty(C, H, C, device=q.device, dtype=q.dtype)
    lse = torch.empty(C, H, device=q.device, dtype=torch.float)
    parallel_deltaformer_kernel(q, k, v, u, w, lse, qk_scale, beta)
    return w, lse


def parallel_deltaformer_bwd_u_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    fa_scale: float,
    beta: torch.Tensor,
    debug_info: dict | None = None,
):
    C, H, D = q.size()
    T, _H, _D = k.size()
    grad_u = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(C, META['BLOCK_C']), H)

    parallel_deltaformer_bwd_kernel_u[grid](
        grad_u, q, k, grad_v, lse, beta,
        H, T, C, D, fa_scale,
    )
    ctx = dict(debug_info or {})
    _ensure_finite('parallel_deltaformer_bwd_u_chunk', grad_u, H=H, T=T, C=C, **ctx)
    return grad_u


def parallel_deltaformer_bwd_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    qk_scale: float,
    fa_scale: float,
    beta: torch.Tensor,
    debug_info: dict | None = None,
):
    T, H, D = k.size()
    row_dot_sum = torch.empty_like(lse)

    def grid_bp(META):
        return (triton.cdiv(T, META['BLOCK_C']), H)

    parallel_deltaformer_bwd_kernel_row_sum[grid_bp](
        row_dot_sum, q, k, grad_v, u, lse,
        H, T, D,
        fa_scale,
    )
    ctx = dict(debug_info or {})
    ctx['stage'] = 'row_sum'
    if DELTAFORMER_DEBUG:
        finite = torch.isfinite(row_dot_sum)
        if not finite.all():
            bad_idx = torch.nonzero(~finite, as_tuple=False)
            row_id = int(bad_idx[0, 0].item())
            head_id = int(bad_idx[0, 1].item())
            ctx['bad_row'] = row_id
            ctx['bad_head'] = head_id
            # Log representative stats around the offending row/head.
            row_slice = {
                'q_norm': float(q[row_id, head_id].float().norm().item()),
                'k_norm': float(k[row_id, head_id].float().norm().item()),
                'u_norm': float(u[row_id, head_id].float().norm().item()),
                'grad_v_norm': float(grad_v[row_id, head_id].float().norm().item()),
                'lse': float(lse[row_id, head_id].item()),
                'beta': float(beta[row_id, head_id].item()),
            }
            print(f"[DeltaFormerDebug] row_dot_sum non-finite at row={row_id} head={head_id}: {row_slice}")
    _ensure_finite('parallel_deltaformer_row_dot', row_dot_sum, **ctx)
    grad_k = torch.empty_like(k)
    grad_q = torch.empty_like(q)

    parallel_deltaformer_bwd_kernel_qk[grid_bp](
        grad_q, grad_k, q, k, grad_v, u, lse, beta, row_dot_sum,
        H, T, D,
        fa_scale, qk_scale,
    )
    ctx['stage'] = 'grad_q'
    _ensure_finite('parallel_deltaformer_grad_q', grad_q, **ctx)
    ctx['stage'] = 'grad_k'
    _ensure_finite('parallel_deltaformer_grad_k', grad_k, **ctx)
    return grad_q, grad_k, row_dot_sum


def parallel_deltaformer_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    lse: torch.Tensor,
    qk_scale: float,
    beta: torch.Tensor,
    eps: float = DELTAFORMER_EPS,
) -> None:
    C, H, D = q.size()
    T, _H, _D = k.size()

    def grid(META):
        return (triton.cdiv(C, META['BLOCK_C']), H)

    parallel_deltaformer_fwd_kernel[grid](
        q, k, v, u, w, lse, beta,
        H, T, C, D, qk_scale,
        eps,
    )


def _config_deltaformer():
    return [
        triton.Config({'BLOCK_C': BC, 'BLOCK_T': BT}, num_stages=ns, num_warps=nw)
        for BC in [128, 64]
        for BT in [64, 32]
        for ns in [3, 2]
        for nw in [8, 4]
    ]


@triton.autotune(configs=_config_deltaformer(), key=['C', 'D'])
@triton.jit
def parallel_deltaformer_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    u_ptr,
    w_ptr,
    lse_ptr,
    beta_ptr,
    H,
    T,
    C,
    D: tl.constexpr,
    qk_scale: float,
    eps: float,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    rowmax = tl.zeros([BLOCK_C], dtype=tl.float32) - float('inf')
    rowsum = tl.zeros([BLOCK_C], dtype=tl.float32) + 1
    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    q_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    q = tl.load(q_blk_ptr, boundary_check=(0,))

    for kv_i in range(0, T - C, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        rowmax_i = tl.maximum(rowmax, tl.max(qk, axis=1))
        qk -= rowmax_i[:, None]
        p = tl.math.exp2(qk)

        rowsum_i = tl.sum(p, axis=1)
        alpha = tl.math.exp2(rowmax - rowmax_i)
        rowsum = rowsum * alpha + rowsum_i
        acc = acc * alpha[:, None]
        rowmax = rowmax_i

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_h * D,
            shape=(T, D),
            strides=(H * D, 1),
            offsets=(kv_i, 0),
            block_shape=(BLOCK_T, D),
            order=(1, 0),
        )
        u = tl.load(u_blk_ptr, boundary_check=(0,))
        acc = tl.dot(p.to(u_ptr.dtype.element_ty), u, acc)

    for kv_i in range(T - C, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        mask = (T - C - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
        qk = tl.where(mask, -1e6, qk)

        rowmax_i = tl.maximum(rowmax, tl.max(qk, axis=1))
        qk -= rowmax_i[:, None]
        p = tl.math.exp2(qk)

        rowsum_i = tl.sum(p, axis=1)
        alpha = tl.math.exp2(rowmax - rowmax_i)
        rowsum = rowsum * alpha + rowsum_i
        acc = acc * alpha[:, None]
        rowmax = rowmax_i

    lse = rowmax + tl.math.log2(rowsum + eps)
    lse_block_ptr = lse_ptr + pid_h + rowid_block * H
    lse_mask = rowid_block < C
    tl.store(lse_block_ptr, lse, mask=lse_mask)

    v_ptr = tl.make_block_ptr(
        base=v_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc / (rowsum[:, None] + eps)

    beta_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_h,
        shape=(C,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    beta = tl.load(beta_ptr, boundary_check=(0,))
    acc = acc * beta[:, None]

    v = tl.load(v_ptr, boundary_check=(0,))
    u = v - acc.to(v_ptr.dtype.element_ty)
    u_block_ptr = tl.make_block_ptr(
        base=u_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(T - C + pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(u_block_ptr, u, boundary_check=(0, 1))

    for kv_i in range(T - C, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        mask = (T - C - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
        qk -= rowmax[:, None]
        p = tl.math.exp2(qk) / (rowsum[:, None] + eps)
        p = tl.where(mask, 0, p)
        w_blk_ptr = tl.make_block_ptr(
            base=w_ptr + pid_h * C,
            shape=(C, C),
            strides=(H * C, 1),
            offsets=(pid_c * BLOCK_C, kv_i - (T - C)),
            block_shape=(BLOCK_C, BLOCK_T),
            order=(1, 0),
        )
        tl.store(w_blk_ptr, p.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_config_deltaformer(), key=['C', 'D'])
@triton.jit
def parallel_deltaformer_bwd_kernel_u(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    lse_ptr,
    beta_ptr,
    H,
    T,
    C,
    D: tl.constexpr,
    fa_scale,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    q_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    q = tl.load(q_blk_ptr, boundary_check=(0,))

    for kv_i in range(0, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * fa_scale

        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr, boundary_check=(0,))
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr, boundary_check=(0,))

        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        v_blk_ptr = tl.make_block_ptr(
            base=v_ptr + pid_h * D,
            shape=(T, D),
            strides=(H * D, 1),
            offsets=(kv_i, 0),
            block_shape=(BLOCK_T, D),
            order=(1, 0),
        )
        v = tl.load(v_blk_ptr, boundary_check=(0,))
        acc = tl.dot(p.to(v_ptr.dtype.element_ty), v, acc)

    o_blk_ptr = tl.make_block_ptr(
        base=o_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(o_blk_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0,))


@triton.autotune(configs=_config_deltaformer(), key=['T', 'D'])
@triton.jit
def parallel_deltaformer_bwd_kernel_row_sum(
    row_dot_ptr,
    q_ptr,
    k_ptr,
    grad_v_ptr,
    u_ptr,
    lse_ptr,
    H,
    T,
    D: tl.constexpr,
    fa_scale,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr, boundary_check=(0,))
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr, boundary_check=(0,))
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr, boundary_check=(0,))

    for kv_i in range(0, (pid_c + 1) * BLOCK_C, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(k_row, k) * fa_scale
        p = tl.math.exp2(qk - lse[:, None])

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        ut = tl.load(u_blk_ptr, boundary_check=(1,))
        dp = tl.dot(grad_v_row, ut)
        if kv_i + BLOCK_T >= pid_c * BLOCK_C:
            mask = (rowid_block[:, None] <= colid_block[None, :] + kv_i)
            p = tl.where(mask, 0., p)
            dp = tl.where(mask, 0., dp)
        acc += tl.sum(p * dp, axis=1)
    row_dot_block_ptr = tl.make_block_ptr(
        base=row_dot_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    tl.store(row_dot_block_ptr, acc, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({'BLOCK_C': BC}, num_stages=ns, num_warps=nw)
                          for BC in [64, 32]
                          for ns in [4, 3]
                          for nw in [4]], key=['T', 'D'])
@triton.jit
def parallel_deltaformer_bwd_kernel_qk(
    grad_q_ptr,
    grad_k_ptr,
    q_ptr,
    k_ptr,
    grad_v_ptr,
    u_ptr,
    lse_ptr,
    beta_ptr,
    row_dot_ptr,
    H,
    T,
    D: tl.constexpr,
    fa_scale: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    block_i = tl.arange(0, BLOCK_C)

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr, boundary_check=(0,))
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr, boundary_check=(0,))
    beta_blk_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    beta = tl.load(beta_blk_ptr, boundary_check=(0,))
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr, boundary_check=(0,))
    row_dot_blk_ptr = tl.make_block_ptr(
        base=row_dot_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    row_dot_row = tl.load(row_dot_blk_ptr, boundary_check=(0,)).to(k_ptr.dtype.element_ty)

    for kv_i in range(0, pid_c * BLOCK_C, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        ut = tl.load(u_blk_ptr)
        dp = tl.dot(grad_v_row, ut)
        da = p * (dp - row_dot_row[:, None])
        k = tl.trans(kt, 1, 0)
        acc = tl.dot(da.to(k.dtype), k, acc)

    k_row_blk_ptr = tl.make_block_ptr(
        base=k_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row_true = tl.load(k_row_blk_ptr, boundary_check=(0,))
    qk = tl.dot(k_row, tl.trans(k_row_true, 1, 0)) * fa_scale
    p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]
    u_blk_ptr = tl.make_block_ptr(
        base=u_ptr + pid_h * D,
        shape=(D, T),
        strides=(1, H * D),
        offsets=(0, pid_c * BLOCK_C),
        block_shape=(D, BLOCK_C),
        order=(0, 1),
    )
    ut = tl.load(u_blk_ptr)
    dp = tl.dot(grad_v_row, ut)
    dpm = dp - row_dot_row[:, None]
    mask = block_i[None, :] < block_i[:, None]
    p = tl.where(mask, p, 0.)
    dpm = tl.where(mask, dpm, 0.)
    da = p * dpm
    daat = da
    acc = tl.dot(daat.to(k_row.dtype), k_row_true, acc)

    grad_q_blk_ptr = tl.make_block_ptr(
        base=grad_q_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(BLOCK_C * pid_c, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc * qk_scale
    tl.store(grad_q_blk_ptr, acc.to(grad_q_ptr.dtype.element_ty), boundary_check=(0,))

    daat = tl.trans(da, 1, 0)
    acc = tl.dot(daat.to(k_row.dtype), k_row)
    k_row = k_row_true
    nu = -tl.trans(ut, 1, 0)
    for kv_i in range((pid_c + 1) * BLOCK_C, T, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=q_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr, boundary_check=(1,))
        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr, boundary_check=(0,))
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr, boundary_check=(0,))
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        grad_vt_blk_ptr = tl.make_block_ptr(
            base=grad_v_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        grad_vt = tl.load(grad_vt_blk_ptr, boundary_check=(1,))
        row_dot_blk_ptr = tl.make_block_ptr(
            base=row_dot_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        row_dot = tl.load(row_dot_blk_ptr, boundary_check=(0,)).to(k_ptr.dtype.element_ty)
        dp = tl.dot(nu, grad_vt)
        da = p * (dp - row_dot[None, :])
        k = tl.trans(kt, 1, 0)
        acc = tl.dot(da.to(k.dtype), k, acc)

    grad_k_blk_ptr = tl.make_block_ptr(
        base=grad_k_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(BLOCK_C * pid_c, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc * qk_scale
    tl.store(grad_k_blk_ptr, acc.to(grad_k_ptr.dtype.element_ty), boundary_check=(0,))


class ParallelDeltaformerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qo: torch.Tensor,
        ko: torch.Tensor,
        vo: torch.Tensor,
        betao: torch.Tensor | None = None,
        C: int = BLOCK_SIZE_C,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        B, T, H, D = ko.size()
        C = min(C, T)
        ctx.C = C
        ctx.cu_seqlens = cu_seqlens

        if cu_seqlens is not None:
            need_aux = qo.requires_grad or ko.requires_grad or vo.requires_grad or (betao is not None and betao.requires_grad)
            u, ws, lses = ParallelDeltaformerFunction._forward_impl(
                qo, ko, vo, betao, C, need_aux=need_aux, cu_seqlens=cu_seqlens)
            saved_beta = betao if betao is not None else torch.ones(B, T, H, device=ko.device, dtype=ko.dtype)
            ctx.beta_is_none = betao is None
            if need_aux:
                ctx.save_for_backward(qo, ko, vo, u, ws, lses, saved_beta)
            else:
                ctx.save_for_backward()
            return u

        u, ws, lses = ParallelDeltaformerFunction._forward_impl(qo, ko, vo, betao, C, need_aux=True)
        saved_beta = betao if betao is not None else torch.ones(B, T, H, device=ko.device, dtype=ko.dtype)
        ctx.save_for_backward(qo, ko, vo, u, ws, lses, saved_beta)
        ctx.beta_is_none = betao is None
        return u

    @staticmethod
    def backward(
        ctx,
        grad_u: torch.Tensor,
    ):
        if getattr(ctx, 'cu_seqlens', None) is not None:
            cu = ctx.cu_seqlens
            qo, ko, vo, u_full, ws, lses, betao = ctx.saved_tensors
            B, T_max, H, D = ko.size()
            qk_scale = 1.0 / math.sqrt(D)
            fa_scale = qk_scale / math.log(2)

            dq = torch.zeros_like(qo)
            dk = torch.zeros_like(ko)
            dv = torch.zeros_like(vo)
            dbeta = None if ctx.beta_is_none else torch.zeros_like(betao)

            C = ctx.C
            N = len(cu) - 1
            chunk_bases = []
            total = 0
            lengths = []
            for b in range(N):
                L = int(cu[b + 1].item() - cu[b].item())
                lengths.append(L)
                chunk_bases.append(total)
                if L > 0:
                    total += (L + C - 1) // C

            for b in range(N):
                L = lengths[b]
                if L == 0:
                    continue
                base = chunk_bases[b]
                seq_start = int(cu[b].item())

                seq_end = seq_start + L
                q_seq = qo[0, seq_start:seq_end, :, :]
                k_seq = ko[0, seq_start:seq_end, :, :]
                u_seq = u_full[0, seq_start:seq_end, :, :]
                beta_seq = betao[0, seq_start:seq_end, :]
                lse_seq = lses[0, seq_start:seq_end, :]
                go_seq = grad_u[0, seq_start:seq_end, :, :]

                gv_seq = torch.zeros_like(u_seq)
                start = ((L - 1) // C) * C
                for i_local in range(start, -1, -C):
                    Ci = min(C, L - i_local)
                    i0 = i_local
                    i1 = i_local + Ci
                    do = go_seq[i0:i1, :, :]
                    if i_local < L - C:
                        qi = k_seq[i0:i1, :, :]
                        ki = q_seq[i1:L, :, :]
                        lse_tail = lse_seq[i1:L, :]
                        beta_tail = beta_seq[i1:L, :]
                        du_tail = parallel_deltaformer_bwd_u_chunk(
                            qi,
                            ki,
                            lse_tail,
                            gv_seq[i1:L, :, :],
                            fa_scale,
                            beta_tail,
                            debug_info={'batch': b, 'chunk_start': i_local, 'phase': 'varlen_tail'},
                        )
                        do = do - du_tail
                    Wpad = ws[base + (i_local // C)]
                    W = Wpad[:Ci, :, :Ci]
                    W_t = W.transpose(0, 1).contiguous()
                    du_chunk = invcum.backward_x(do.transpose(0, 1).contiguous(), W_t).transpose(0, 1).contiguous()
                    _ensure_finite(
                        'parallel_deltaformer_invcum_chunk',
                        du_chunk,
                        batch=b,
                        chunk_start=i_local,
                        phase='varlen_chunk',
                    )
                    gv_seq[i0:i1, :, :].copy_(du_chunk)

                gq, gk, gbeta = parallel_deltaformer_bwd_qk(
                    q_seq,
                    k_seq,
                    u_seq,
                    lse_seq,
                    gv_seq,
                    qk_scale,
                    fa_scale,
                    beta_seq,
                    debug_info={'batch': b, 'phase': 'varlen', 'seq_start': seq_start, 'seq_end': seq_end},
                )
                dq[0, seq_start:seq_end, :, :].copy_(gq)
                dk[0, seq_start:seq_end, :, :].copy_(gk)
                dv[0, seq_start:seq_end, :, :].copy_(gv_seq)
                if dbeta is not None:
                    dbeta[0, seq_start:seq_end, :].copy_(gbeta)

            return dq, dk, dv, dbeta, None, None
        qo, ko, vo, u, ws, lses, betao = ctx.saved_tensors
        C = ctx.C
        B, T, H, D = ko.size()

        grad_q = torch.zeros_like(qo)
        grad_k = torch.zeros_like(ko)
        grad_v = torch.zeros_like(vo)
        grad_beta_out = None if ctx.beta_is_none else torch.zeros_like(betao)

        qk_scale = 1.0 / math.sqrt(D)
        fa_scale = qk_scale / math.log(2)

        chunk_base = 0
        for b in range(B):
            grad_v_seq = torch.empty(T, H, D, device=ko.device, dtype=ko.dtype)
            num_chunks = (T + C - 1) // C
            for chunk_idx in range(num_chunks - 1, -1, -1):
                i = chunk_idx * C
                Ci = min(C, T - i)
                do = grad_u[b, i:i + Ci, :, :]

                if chunk_idx < num_chunks - 1:
                    tail_start = i + Ci
                    qi = ko[b, i:i + Ci, :, :]
                    ki = qo[b, tail_start:, :, :]
                    lse = lses[b, tail_start:, :]
                    if not ctx.beta_is_none:
                        beta_single = betao[b, tail_start:, :]
                    else:
                        beta_single = torch.ones(T - tail_start, H, device=ko.device, dtype=ko.dtype)
                    du = parallel_deltaformer_bwd_u_chunk(
                        qi,
                        ki,
                        lse,
                        grad_v_seq[tail_start:, :, :],
                        fa_scale,
                        beta_single,
                        debug_info={'batch': b, 'chunk_start': i, 'phase': 'dense_tail'},
                    )
                    do = grad_u[b, i:i + Ci, :, :] - du

                W = ws[chunk_base + chunk_idx][:Ci, :, :Ci]
                W_t = W.transpose(0, 1).contiguous()
                du = invcum.backward_x(do.transpose(0, 1).contiguous(), W_t).transpose(0, 1).contiguous()
                _ensure_finite(
                    'parallel_deltaformer_invcum_chunk',
                    du,
                    batch=b,
                    chunk_start=i,
                    phase='dense_chunk',
                )
                grad_v_seq[i:i + Ci, :, :].copy_(du)

            q_seq = qo[b]
            k_seq = ko[b]
            u_seq = u[b]
            lse_seq = lses[b]
            beta_seq = betao[b] if not ctx.beta_is_none else torch.ones(T, H, device=ko.device, dtype=ko.dtype)

            gq, gk, gbeta = parallel_deltaformer_bwd_qk(
                q_seq,
                k_seq,
                u_seq,
                lse_seq,
                grad_v_seq,
                qk_scale,
                fa_scale,
                beta_seq,
                debug_info={'batch': b, 'phase': 'dense'},
            )

            grad_q[b].copy_(gq)
            grad_k[b].copy_(gk)
            grad_v[b].copy_(grad_v_seq)
            if not ctx.beta_is_none:
                grad_beta_out[b].copy_(gbeta)

            chunk_base += num_chunks

        return grad_q, grad_k, grad_v, grad_beta_out, None, None

    @staticmethod
    def _forward_impl(
        qo: torch.Tensor,
        ko: torch.Tensor,
        vo: torch.Tensor,
        betao: torch.Tensor | None,
        C: int,
        need_aux: bool,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, T_max, H, D = ko.size()
        C = min(C, T_max)
        qk_scale = 1.0 / math.sqrt(D)
        fa_scale = qk_scale / math.log(2)

        if cu_seqlens is None:
            if betao is None:
                beta_full = torch.ones(B, T_max, H, device=ko.device, dtype=ko.dtype)
            else:
                beta_full = betao

            u_full = torch.empty_like(vo)
            if need_aux:
                total_chunks = B * ((T_max + C - 1) // C)
                ws = torch.empty(total_chunks, C, H, C, device=ko.device, dtype=ko.dtype)
                lses = torch.empty(B, T_max, H, device=ko.device, dtype=torch.float)
                chunk_base = 0
            else:
                ws = None
                lses = None
                chunk_base = 0

            for b in range(B):
                for i in range(0, T_max, C):
                    Ci = min(C, T_max - i)

                    qi = qo[b, i:i + Ci, :, :]
                    ki = ko[b, :i + Ci, :, :]
                    vi = vo[b, i:i + Ci, :, :]
                    ui_prev = u_full[b, :i + Ci, :, :]
                    betai = beta_full[b, i:i + Ci, :]

                    w, lse_chunk = parallel_deltaformer_chunk_fwd(qi, ki, vi, ui_prev, fa_scale, betai)
                    w = w * betai.unsqueeze(-1).to(torch.float32)
                    if need_aux:
                        wpad = torch.zeros(C, H, C, device=ko.device, dtype=ko.dtype)
                        wpad[:Ci, :, :Ci].copy_(w)
                        ws[chunk_base + (i // C)].copy_(wpad)
                        lses[b, i:i + Ci, :].copy_(lse_chunk)

                    u_chunk_view = u_full[b, i:i + Ci, :, :]
                    w_t = w.transpose(0, 1).contiguous()
                    u_chunk_view_t = u_chunk_view.transpose(0, 1).contiguous()
                    invcum.forward_inplace(u_chunk_view_t, w_t)
                    u_chunk_view.copy_(u_chunk_view_t.transpose(0, 1))

                chunk_base += (T_max + C - 1) // C

            return u_full, ws, lses

        N = len(cu_seqlens) - 1
        assert cu_seqlens.dim() == 1 and cu_seqlens.size(0) == N + 1, "cu_seqlens must be [N+1]"
        device = ko.device
        dtype_k = ko.dtype
        if betao is None:
            beta_full = torch.ones(B, T_max, H, device=device, dtype=dtype_k)
        else:
            beta_full = betao

        u_full = torch.empty_like(vo)
        if need_aux:
            total_chunks = sum((max(0, int(cu_seqlens[b + 1].item() - cu_seqlens[b].item())) + C - 1) // C
                               for b in range(N))
            ws = torch.empty(total_chunks, C, H, C, device=device, dtype=dtype_k)
            lses = torch.empty(B, T_max, H, device=device, dtype=torch.float)
            chunk_base = 0
        else:
            ws = None
            lses = None
            chunk_base = 0

        for b in range(N):
            seq_start = int(cu_seqlens[b].item())
            seq_end = int(cu_seqlens[b + 1].item())
            L = max(0, seq_end - seq_start)
            if L == 0:
                continue

            for i_local in range(0, L, C):
                Ci = min(C, L - i_local)
                li0 = i_local
                li1 = i_local + Ci

                abs_start = seq_start + li0
                abs_end = seq_start + li1
                abs_context_end = seq_start + li1

                qi = qo[0, abs_start:abs_end, :, :]
                ki = ko[0, seq_start:abs_context_end, :, :]
                vi = vo[0, abs_start:abs_end, :, :]
                ui_prev = u_full[0, seq_start:abs_context_end, :, :]
                betai = beta_full[0, abs_start:abs_end, :]

                w, lse_chunk = parallel_deltaformer_chunk_fwd(qi, ki, vi, ui_prev, fa_scale, betai)
                w = w * betai.unsqueeze(-1).to(torch.float32)
                if need_aux:
                    wpad = torch.zeros(C, H, C, device=device, dtype=dtype_k)
                    wpad[:Ci, :, :Ci].copy_(w)
                    ws[chunk_base + (i_local // C)].copy_(wpad)
                    lses[0, abs_start:abs_end, :].copy_(lse_chunk)

                u_chunk_view = u_full[0, abs_start:abs_end, :, :]
                w_t = w.transpose(0, 1).contiguous()
                u_chunk_view_t = u_chunk_view.transpose(0, 1).contiguous()
                invcum.forward_inplace(u_chunk_view_t, w_t)
                u_chunk_view.copy_(u_chunk_view_t.transpose(0, 1))

            chunk_base += (L + C - 1) // C

        return u_full, ws, lses


def deltaformer_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    C: int = BLOCK_SIZE_C,
) -> torch.Tensor:
    if flash_attn_func is None:
        raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

    B, T, H, D = k.shape
    C = min(C, T)

    u = ParallelDeltaformerFunction.apply(q, k, v, beta, C, cu_seqlens)

    if attention_mask is not None:
        q_padded, (k_padded, u_padded), indices_q, cu_seqlens_lens, max_seq_lens = unpad_input(q, (k, u), attention_mask, T)
        cu_seqlens_q, cu_seqlens_k = cu_seqlens_lens
        max_seqlen_q, max_seqlen_k = max_seq_lens
        o = flash_attn_varlen_func(
            q_padded, k_padded, u_padded,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            window_size=(-1, -1),
        )
        o = pad_input(o, indices_q, B, T)
    elif cu_seqlens is not None:
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        o = flash_attn_varlen_func(
            q.squeeze(0), k.squeeze(0), u.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=(-1, -1),
        ).unsqueeze(0)
    else:
        o = flash_attn_func(q, k, u, causal=True, window_size=(-1, -1))

    return o


__all__ = [
    'deltaformer_attn',
]
