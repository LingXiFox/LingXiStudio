# macOS 测试与交付准备

## 自动化分层

- 常规 XCTest 不下载模型。`.github/workflows/check.yml` 使用标准 Apple Silicon `xcode-27` runner、Release 构建与本仓库锁定的 MLX Swift，2 个编译任务。
- CI 的 MLX 小张量测试显式在 CPU 上运行；不改变生产 Runtime 的 GPU 路径。Metal 编译图生命周期用例在标准虚拟机中**未覆盖**，必须在本机 GPU 运行。
- SwiftPM 并行 xUnit 曾将 skipped 用例输出成没有 skipped 节点的 testcase。因此 CI 直接排除 Metal 专用用例，并在覆盖清单明确输出 notCovered，不把它当作通过。
- `Scripts/check-tests.py` 将 `swift test list` 与实际 xUnit 的完整用例集合比较，拒绝空报告、遗漏、重复、失败、错误与非预期跳过。硬件测试不属于普通 CI。
- 四项真实模型测试仍通过显式环境开关运行，输入、LoRA 与数十 GB 模型由真机本地提供；详见 [Runtime 回归](runtime-v1-candidate.md)。

```sh
swift test list -c release --jobs 2 > .tmp/tests.txt
swift test -c release --jobs 2 --parallel --num-workers 1 --skip Hardware --xunit-output .tmp/tests.xml
python3 Scripts/check-tests.py .tmp/tests.txt .tmp/tests.xml

LINGXI_TEST_DEVICE=cpu swift test -c release --jobs 2 --parallel --num-workers 1 \
  --skip 'Hardware|TransformerLifetimeTests' --xunit-output .tmp/tests-cpu.xml
python3 Scripts/check-tests.py .tmp/tests.txt .tmp/tests-cpu.xml --cpu
```

GitHub runner 与标签依据 [官方 Xcode 27 公告](https://github.blog/changelog/2026-10-15-xcode-27-runner-image-generally-available/) 和 [runner 规格](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。CPU scope 依据 [MLX Swift 官方文档](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Documentation.docc/Articles/using-streams.md)，并已在本机实跑。

## 最终 App 检查

```sh
Scripts/make-app.sh
python3 Scripts/verify-app.py dist/LingXiStudio.app
```

检查最终可执行文件 arm64 与 `LC_BUILD_VERSION minos 26.2`，不能只看 Package.swift。要求动态依赖都来自系统目录；存在 Icon Composer 编译产物 `Assets.car` / `icon.icns`、MLX Bundle 内的 `Contents/Resources/default.metallib` 和法律声明；验证签名完整性。发布包不得包含验收用 `LSEnvironment`。

本机 Release 包已通过上述资源、架构、依赖与 ad-hoc 签名检查。当前实际运行系统为 macOS 27.2，不能等同于已在最低 macOS 26.2 上验证。

首次启动测试使用新的绝对路径 `LINGXI_PROFILE_DIRECTORY`，模型缺失时显示“未选择模型”，生成禁用，仍可打开模型库/设置。该 profile 的输出与设置均隔离；普通用户输出为 `~/Pictures/LingXiStudio`，Application Support 存输入副本、任务、预设与日志，UserDefaults 存偏好。模型权重独立存放，不打入 App。profile 无效路径会明确拒绝，不隐式混用用户数据。

## 公开测试之前仍须完成

当前包仅 ad-hoc 签名，没有 Developer ID、hardened runtime、公证或装订票据，不宣称可直接通过下载来源的 Gatekeeper 检查。保持系统安全设置，不指导用户绕过保护。

取得正式签名身份后，用 Developer ID Application 对最终包按内到外签名，启用 hardened runtime，基于实际权限决定最小 entitlements；在归档副本上验证签名。使用已配置的 keychain profile 提交 `notarytool`，成功后 staple，再用 `stapler validate` / Gatekeeper 验证。凭据、证书与 keychain profile 不提交仓库。本阶段不创建签名身份、不读取凭据、不正式发布。

发布前应在另一台干净 macOS 26.2+ Apple Silicon Mac 验证首次启动、无模型引导、模型选择、完整生成、资源定位、用户数据升级与退出恢复；并做更长时间交互稳定性试用。未完成这些项目之前不标记正式稳定版。
