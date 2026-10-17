import MLX

func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}

/// Tanh-approximated GELU, evaluated in fp32 like PyTorch's opmath for bf16 inputs.
func geluTanh(_ x: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let inner = Float((2.0 / Double.pi).squareRoot()) * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1 + tanh(inner))).asType(x.dtype)
}
