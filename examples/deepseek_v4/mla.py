import tilelang 
import tilelang.language as T
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mqa_fwd(
    heads,
    dim,
    topk,
    sm_scale=None,
    block_I=16,
    num_stages=2,
    threads=512,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be power of 2, got {dim}"
    assert topk % block_I == 0, f"topk ({topk}) must be divisible by block_I ({block_I})"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    attn_sink_shape = [heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = heads
    padded_H = max(tilelang.math.next_power_of_2(heads), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    HEAD_TILE = 64
    if heads > HEAD_TILE:
        assert heads % HEAD_TILE == 0, "heads should be a multiple of 64"
        REPLICATE_H = heads // HEAD_TILE
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else HEAD_TILE
    WARP_SIZE = 64
    QK_GROUPS = 2
    WAVES_PER_QK_GROUP = 4
    THREADS_PER_QK_GROUP = WARP_SIZE * WAVES_PER_QK_GROUP
    QK_K_PER_GROUP = D // QK_GROUPS

    assert threads == QK_GROUPS * THREADS_PER_QK_GROUP, "split-K requires one 512-thread CTA (2 groups x 4 waves)"
    assert D % (QK_GROUPS * 16) == 0, "split-K requires D to be divisible by 32"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, threads=threads) as (bx, by):
            # Keep each half of Q in the registers of exactly one 4-wave
            # group. This avoids a 64 KiB Q shared-memory allocation.
            Q_chunk0 = T.alloc_fragment([H_per_block, QK_K_PER_GROUP], dtype)
            Q_chunk1 = T.alloc_fragment([H_per_block, QK_K_PER_GROUP], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            mask = T.alloc_fragment([BI], "bool")
            Indices_share = T.alloc_shared([topk], indices_dtype)
            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            # Each guarded high-level GEMM has its own accumulator. The two
            # partial score tiles are exchanged through shared memory.
            P_chunk0 = T.alloc_fragment([H_per_block, BI], accum_dtype)
            P_chunk1 = T.alloc_fragment([H_per_block, BI], accum_dtype)
            P_partial_shared = T.alloc_shared([QK_GROUPS * H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_shared([H_per_block], accum_dtype)
            sumexp_i = T.alloc_shared([H_per_block], accum_dtype)
            alpha = T.alloc_shared([H_per_block], accum_dtype)
            m_i = T.alloc_shared([H_per_block], accum_dtype)
            m_i_prev = T.alloc_shared([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * HEAD_TILE
            H1 = H0 + H_per_block
            tx, _, _ = T.get_thread_bindings()

            T.copy(Q[b_i, s_i, H0:H1, :QK_K_PER_GROUP], Q_chunk0)
            T.copy(Q[b_i, s_i, H0:H1, QK_K_PER_GROUP:D], Q_chunk1)
            T.copy(Indices[b_i, s_i, :], Indices_share)
            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices_share[i_i * BI + bi_i] != -1

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices_share[i_i * BI + bi_i], d_i]

                # Two independent 4-wave groups each run a complete
                # high-level GEMM. They consume matching Q/K halves and
                # publish their 64x16 partial score tiles for pair-reduce.
                T.sync_threads()
                if tx < THREADS_PER_QK_GROUP:
                    T.clear(P_chunk0)
                    T.gemm(
                        Q_chunk0,
                        KV_shared[:, :QK_K_PER_GROUP],
                        P_chunk0,
                        transpose_B=True,
                        # policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(P_chunk0, P_partial_shared[:H_per_block, :])
                elif tx < QK_GROUPS * THREADS_PER_QK_GROUP:
                    T.clear(P_chunk1)
                    T.gemm(
                        Q_chunk1,
                        KV_shared[:, QK_K_PER_GROUP:D],
                        P_chunk1,
                        transpose_B=True,
                        # policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(P_chunk1, P_partial_shared[H_per_block : QK_GROUPS * H_per_block, :])

                # Make both 64x16 partial tiles visible before the pair
                # reduction. Invalid sparse positions are masked here, after
                # the raw dot-product reduction.
                T.sync_threads()
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i],
                        P_partial_shared[h_i, bi_i] + P_partial_shared[H_per_block + h_i, bi_i],
                        -T.infinity(acc_s.dtype),
                    )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                # PV's reduction dimension is BI=16. MACA k_pack=2 makes
                # the effective K step 32, so it emits zero MMA iterations
                # for this tile and silently breaks correctness.
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # attn_sink: add exp(attn_sink[h] - max_scaled) to softmax denominator
            # attn_sink is a pre-scaled logit (same space as scores*sm_scale), so only convert to log2 base
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] += T.exp2(AttnSink[H0 + h_i] * 1.44269504 - m_i[h_i] * sm_scale)

            # Rescale output
            for h_i in T.Parallel(H_per_block):
                alpha[h_i] = 1.0 / sumexp[h_i]
            T.sync_threads()
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] *= alpha[h_i]
            # LSE = log2(sumexp) + m_i * sm_scale (in log2 space)
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main