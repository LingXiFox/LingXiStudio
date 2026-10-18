import Foundation

/// Process-wide MLX settings that must be in place before the first MLX GPU call.
public enum MLXRuntime {
    /// MLX commits a Metal command buffer once it holds `MLX_MAX_OPS_PER_BUFFER` kernels or has touched
    /// `MLX_MAX_MB_PER_BUFFER` MB of inputs (weights included). The defaults for this GPU class (40 / 40) commit after
    /// almost every DiT matmul. Each commit pays a fixed kernel-side submit cost, and with the many short LoRA
    /// matmuls the submits fall behind and leave the GPU idle. Larger buffers remove those gaps at the cost of holding a few
    /// more intermediates in flight (see docs/phase-1.5-report.md for the measurements behind these values).
    public static let commandBufferOps = 100
    public static let commandBufferMB = 200

    /// MLX runs float32 matmuls, convolutions and attention as TF32 on Apple GPUs by default (`MLX_ENABLE_TF32=1`),
    /// about 1e-3 relative error per product. The stages this runtime keeps in fp32 on purpose (vision tower, VAE)
    /// get true fp32 instead: measured against the reference, the vision tower goes from 1.4e-2 to 4.6e-5 rel-L2,
    /// for about 0.5 s more per 1024² VAE decode. bf16 work (the DiT and text encoder matmuls) is unaffected.
    public static let tf32 = false

    /// Sets the defaults above unless the environment already chooses a value. MLX reads them once when it creates
    /// the Metal device, so call this before any array is evaluated; later calls have no effect.
    public static func configure() {
        setenv("MLX_MAX_OPS_PER_BUFFER", String(commandBufferOps), 0)
        setenv("MLX_MAX_MB_PER_BUFFER", String(commandBufferMB), 0)
        setenv("MLX_ENABLE_TF32", tf32 ? "1" : "0", 0)
    }
}
