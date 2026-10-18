// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LingXiStudio",
    platforms: [.macOS("26.2")],
    products: [
        .library(name: "QwenImageCore", targets: ["QwenImageCore"]),
        .executable(name: "qwen-image", targets: ["qwen-image"]),
        .executable(name: "LingXiStudio", targets: ["LingXiStudio"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
    ],
    targets: [
        .target(
            name: "QwenImageCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "qwen-image",
            dependencies: ["QwenImageCore"]
        ),
        .executableTarget(
            name: "LingXiStudio",
            dependencies: ["QwenImageCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "WorkbenchTests",
            dependencies: ["LingXiStudio", "QwenImageCore"],
            path: "Tests/Workbench"
        ),
    ]
)
