import MLX

/// Custom Metal kernels for the transformer attention prologue.
enum FusedKernels {
    /// Per-head RMSNorm followed by interleaved-pair RoPE, for Q and K in a single launch.
    ///
    /// The composed path reads and writes each `[1, L, H, D]` tensor ~6 times (cast, norm, cast, split pairs,
    /// rotate, stack, cast). Here every thread owns one `(2i, 2i+1)` pair and works in registers, but rounds to
    /// `T` at exactly the points the reference does: diffusers `RMSNorm` normalizes in fp32 and rounds, multiplies
    /// by the bf16 weight and rounds, then RoPE rotates in fp32 and rounds on store.
    /// One threadgroup of `D / 2` threads handles one `(token, head)` row.
    static let qkNormRopeSource = """
        uint row = threadgroup_position_in_grid.x;
        uint i = thread_position_in_threadgroup.x;
        int L = q_shape[1];
        int H = q_shape[2];
        int D = q_shape[3];
        int half_d = D / 2;
        int l = (row / H) % L;
        uint off = row * D + 2 * i;

        float aq = static_cast<float>(q[off]);
        float bq = static_cast<float>(q[off + 1]);
        float ak = static_cast<float>(k[off]);
        float bk = static_cast<float>(k[off + 1]);

        float sq = simd_sum(aq * aq + bq * bq);
        float sk = simd_sum(ak * ak + bk * bk);
        threadgroup float part_q[8];
        threadgroup float part_k[8];
        if (thread_index_in_simdgroup == 0) {
            part_q[simdgroup_index_in_threadgroup] = sq;
            part_k[simdgroup_index_in_threadgroup] = sk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tq = 0.0f;
        float tk = 0.0f;
        for (uint g = 0; g < simdgroups_per_threadgroup; g++) {
            tq += part_q[g];
            tk += part_k[g];
        }
        float rq = metal::precise::rsqrt(tq / D + eps[0]);
        float rk = metal::precise::rsqrt(tk / D + eps[0]);

        float nqa = static_cast<float>(T(static_cast<float>(T(aq * rq)) * static_cast<float>(wq[2 * i])));
        float nqb = static_cast<float>(T(static_cast<float>(T(bq * rq)) * static_cast<float>(wq[2 * i + 1])));
        float nka = static_cast<float>(T(static_cast<float>(T(ak * rk)) * static_cast<float>(wk[2 * i])));
        float nkb = static_cast<float>(T(static_cast<float>(T(bk * rk)) * static_cast<float>(wk[2 * i + 1])));

        float c = cos_t[l * half_d + i];
        float s = sin_t[l * half_d + i];
        out_q[off] = T(nqa * c - nqb * s);
        out_q[off + 1] = T(nqa * s + nqb * c);
        out_k[off] = T(nka * c - nkb * s);
        out_k[off + 1] = T(nka * s + nkb * c);
        """

    static let qkNormRope = MLXFast.metalKernel(
        name: "lingxi_qk_norm_rope",
        inputNames: ["q", "k", "wq", "wk", "cos_t", "sin_t", "eps"],
        outputNames: ["out_q", "out_k"],
        source: qkNormRopeSource)

    /// `q`, `k`: `[1, L, H, D]` (D a multiple of 64, at most 512); `cos`/`sin`: fp32 `[L, D / 2]`.
    static func qkNormRope(
        q: MLXArray, k: MLXArray, weightQ: MLXArray, weightK: MLXArray, cos: MLXArray, sin: MLXArray, eps: Float
    ) -> (MLXArray, MLXArray) {
        let shape = q.shape
        let (l, h, d) = (shape[1], shape[2], shape[3])
        precondition(d % 64 == 0 && d <= 512 && cos.shape == [l, d / 2])
        let out = qkNormRope(
            [q, k, weightQ, weightK, cos, sin, MLXArray([eps])],
            template: [("T", q.dtype)],
            grid: (shape[0] * l * h * d / 2, 1, 1),
            threadGroup: (d / 2, 1, 1),
            outputShapes: [shape, shape],
            outputDTypes: [q.dtype, q.dtype])
        return (out[0], out[1])
    }
}
