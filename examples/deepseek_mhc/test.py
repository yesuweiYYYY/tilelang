import tilelang
import tilelang.language as T
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(n, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

        if ENABLE_PDL:
            T.pdl_trigger()

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=128) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            if T.get_thread_binding() < 64:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)
            if T.get_thread_binding() < 64:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2] + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)
                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)
                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)
                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )
                ###################################################################
                # _mhc_pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)
                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)
                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]
                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)



