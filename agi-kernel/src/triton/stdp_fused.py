"""Optional fused STDP kernel in Triton.

For small columns where the CSR fits in SMEM, we can fuse the STDP observe
+ apply passes and skip one round of global-memory traffic. For large,
cross-area CSRs the CUDA kernel in src/cuda/stdp_update.cu is preferred
because it exploits warp-level `__shfl_xor_sync`.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def stdp_fused_kernel(
    row_ptr_ptr, col_idx_ptr, w_q8_ptr, e_trace_ptr,
    pre_bits_ptr, post_bits_ptr,
    last_pre_ptr, last_post_ptr,
    t_ms, tau_plus, tau_minus, a_plus, a_minus,
    tau_elig, dt, da_scalar, lr,
    N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BLOCK_N
    row_range = n0 + tl.arange(0, BLOCK_N)
    mask = row_range < N

    row_begin = tl.load(row_ptr_ptr + row_range,     mask=mask, other=0)
    row_end   = tl.load(row_ptr_ptr + row_range + 1, mask=mask, other=0)

    post_bits_word = tl.load(post_bits_ptr + row_range // 32, mask=mask, other=0)
    post_fired = ((post_bits_word >> (row_range % 32)) & 1).to(tl.int1)
    post_last = tl.load(last_post_ptr + row_range, mask=mask, other=-1.0)

    decay = tl.exp(-dt / tau_elig)

    for s in tl.static_range(0, 4096, BLOCK_S):
        p = row_begin[:, None] + s + tl.arange(0, BLOCK_S)[None, :]
        active = p < row_end[:, None]

        col = tl.load(col_idx_ptr + p, mask=active, other=0)
        wq  = tl.load(w_q8_ptr    + p, mask=active, other=0).to(tl.int32)
        e   = tl.load(e_trace_ptr + p, mask=active, other=0.0)

        pre_bits_word = tl.load(pre_bits_ptr + (col // 32), mask=active, other=0)
        pre_fired = ((pre_bits_word >> (col % 32)) & 1).to(tl.int1)
        pre_last = tl.load(last_pre_ptr + col, mask=active, other=-1.0)

        e = e * decay

        # LTP branch (post fires after pre)
        dt_pair = t_ms - pre_last
        ltp = tl.where(post_fired[:, None] & (pre_last > 0.0),
                       a_plus * tl.exp(-dt_pair / tau_plus), 0.0)

        # LTD branch (pre fires after post)
        dt_pair2 = t_ms - post_last[:, None]
        ltd = tl.where(pre_fired & (post_last[:, None] > 0.0),
                       a_minus * tl.exp(-dt_pair2 / tau_minus), 0.0)

        e = e + ltp - ltd

        # Apply immediately (three-factor).
        dw = lr * da_scalar * e
        wq_new = tl.minimum(tl.maximum(wq + (dw * 32.0).to(tl.int32), 0), 255)

        tl.store(w_q8_ptr    + p, wq_new.to(tl.uint8), mask=active)
        tl.store(e_trace_ptr + p, e,                    mask=active)
