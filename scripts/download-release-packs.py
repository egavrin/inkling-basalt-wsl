#!/usr/bin/env python3
"""Download and reassemble the exact packs from the model-packs-v1 release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/egavrin/inkling-basalt-wsl/releases/download/model-packs-v1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(cache, part):
    path = cache / part['name']
    if not path.exists() or path.stat().st_size < part['bytes']:
        subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-all-errors',
                        '--continue-at', '-', '--output', str(path), f"{BASE}/{part['name']}"], check=True)
    if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
        path.unlink()
        raise ValueError(f"Corrupt part {part['name']}. Run the downloader again to fetch a clean copy.")
    return path


def assemble(destination, profile, pack):
    output = destination / pack['filename']
    if output.exists():
        if output.stat().st_size == pack['bytes'] and digest(output) == pack['sha256']:
            print(f'Verified {output}', flush=True)
            return
        raise FileExistsError(f'An existing pack has different bytes: {output}')

    cache = destination / f'.{profile}-release-parts'
    cache.mkdir(exist_ok=True)
    paths = [fetch(cache, part) for part in pack['parts']]
    partial = cache / f"{pack['filename']}.partial"
    with partial.open('wb') as joined:
        for path in paths:
            with path.open('rb') as chunk:
                shutil.copyfileobj(chunk, joined, 4 * 1024**2)
    if partial.stat().st_size != pack['bytes'] or digest(partial) != pack['sha256']:
        raise ValueError(f'Assembled pack checksum failed: {partial}')
    os.link(partial, output)
    partial.unlink()
    for path in paths:
        path.unlink()
    cache.rmdir()
    print(f'Verified {output}', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('profile', choices=('gsq', 'mx8', 'both'))
    cli.add_argument('destination', type=Path)
    args = cli.parse_args()
    manifest = json.loads((REPO / 'model-packs.json').read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    wanted = ('gsq', 'mx8') if args.profile == 'both' else (args.profile,)
    for profile in wanted:
        assemble(args.destination, profile, manifest[profile])


if __name__ == '__main__':
    main()
