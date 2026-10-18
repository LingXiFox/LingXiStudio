#!/bin/zsh
# Builds dist/LingXiStudio.app: the release SwiftUI executable plus MLX's Metal library bundle, ad-hoc signed.
# MLX finds `mlx-swift_Cmlx.bundle/default.metallib` through the main bundle's Resources folder.
set -euo pipefail
root=${0:A:h:h}
version=${VERSION:-0.2.0}
cd "$root"
swift build -c release --jobs "${BUILD_JOBS:-2}" --product LingXiStudio
bin=$(swift build -c release --show-bin-path)
app=$root/dist/LingXiStudio.app
# Updated in place: the executable and plist are overwritten, the Metal bundle is merged with ditto.
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp -f "$bin/LingXiStudio" "$app/Contents/MacOS/LingXiStudio"
ditto "$bin/mlx-swift_Cmlx.bundle" "$app/Contents/Resources/mlx-swift_Cmlx.bundle"
mkdir -p "$app/Contents/Resources/Legal"
cp "$root/LICENSE" "$root/NOTICE" "$app/Contents/Resources/Legal/"
ditto "$root/LICENSES" "$app/Contents/Resources/Legal/LICENSES"
# Compile the layered Icon Composer source for native macOS appearances.
mkdir -p "$root/.tmp"
xcrun actool "$root/ICONs/icon.icon" --compile "$app/Contents/Resources" \
  --platform macosx --target-device mac --minimum-deployment-target 26.2 \
  --app-icon icon --output-partial-info-plist "$root/.tmp/app-icon-info.plist" \
  --output-format human-readable-text --warnings --notices
test -s "$app/Contents/Resources/Assets.car"
test -s "$app/Contents/Resources/icon.icns"
cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>LingXiStudio</string>
  <key>CFBundleIdentifier</key><string>com.lingxifox.LingXiStudio</string>
  <key>CFBundleName</key><string>LingXiStudio</string>
  <key>CFBundleDisplayName</key><string>LingXiStudio</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundleIconName</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>26.2</string>
  <key>NSHumanReadableCopyright</key><string>Copyright © 2026 LingXiFox. MIT License; third-party terms apply.</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
codesign --force --sign - "$app"
echo "built $app"
