#!/usr/bin/env python3
"""Prepare an isolated ms-swift runtime without touching installed packages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '4.2.0'
WHEEL_SHA256 = '2c0883817ba4175801db610bc0b522d7243e6e10601eecf9e10a7b7d74955729'


def install(wheel):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise SystemExit('Wheel checksum differs from the verified ms-swift 4.2.0 wheel.')
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                parts = Path(name).parts
                if not name.startswith('swift/') or name.endswith('/'):
                    continue
                if '..' in parts:
                    raise SystemExit('Unsafe archive path.')
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        for file in (ROOT / 'compat/swift').rglob('*'):
            if file.is_file():
                target = stage / 'swift' / file.relative_to(ROOT / 'compat/swift')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
        target = runtime / 'swift'
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(stage / 'swift'), target)
    (runtime / 'runtime.json').write_text(json.dumps({'ms-swift': VERSION, 'wheel_sha256': WHEEL_SHA256}, indent=2)+'\n')
    print('Prepared .runtime/swift with the local inference compatibility file.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, help='Use an already downloaded official 4.2.0 wheel.')
    args = parser.parse_args()
    if args.wheel:
        install(args.wheel.resolve())
    else:
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                            f'ms-swift=={VERSION}', '--dest', temporary], check=True)
            install(next(Path(temporary).glob('ms_swift-4.2.0-*.whl')))


if __name__ == '__main__':
    main()
