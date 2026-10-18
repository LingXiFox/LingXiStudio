#!/usr/bin/env python3
"""Check the deliverable itself, without accepting build settings as binary evidence."""
import argparse
import json
import plistlib
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('app', type=Path)
args = parser.parse_args()
app = args.app.resolve()
contents = app / 'Contents'
info = plistlib.loads((contents / 'Info.plist').read_bytes())
assert info['CFBundleIdentifier'] == 'com.lingxifox.LingXiStudio'
assert info['LSMinimumSystemVersion'] == '26.2'
assert 'LSEnvironment' not in info, 'Acceptance profile must not ship'
exe = contents / 'MacOS' / info['CFBundleExecutable']
assert subprocess.check_output(['lipo', '-archs', str(exe)], text=True).strip() == 'arm64'
load = subprocess.check_output(['xcrun', 'vtool', '-show-build', str(exe)], text=True)
assert 'minos 26.2' in load, load
libs = subprocess.check_output(['otool', '-L', str(exe)], text=True)
paths = [line.strip().split(' (')[0] for line in libs.splitlines()[1:]]
assert paths and all(p.startswith(('/System/Library/', '/usr/lib/')) for p in paths), libs
for resource in ['Assets.car', 'icon.icns', 'mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib', 'Legal/LICENSE', 'Legal/NOTICE']:
    p = contents / 'Resources' / resource
    assert p.is_file() and p.stat().st_size > 0, resource
subprocess.run(['plutil', '-lint', str(contents / 'Info.plist')], check=True, stdout=sys.stderr)
subprocess.run(['codesign', '--verify', '--deep', '--strict', '--verbose=2', str(app)], check=True, stdout=sys.stderr)
print(json.dumps({'app': str(app), 'architecture': 'arm64', 'minimumOS': '26.2',
                  'dynamicDependencies': paths, 'signed': 'verified (identity checked separately)',
                  'resources': 'icon, Metal library, legal notices present'}, indent=2))
