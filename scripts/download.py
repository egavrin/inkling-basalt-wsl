#!/usr/bin/env python3
"""Pinned, resumable downloads. No Hugging Face login or remote code execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get('INKLING_STATE', ROOT / '.state')).resolve()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f.fileno(), 0, f.tell(), os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profile', nargs='?', choices=['pi', 'gsq', 'mx8', 'both'], default='both')
    a = ap.parse_args()
    for item in json.loads((ROOT / 'manifests/inputs.json').read_text())['files']:
        if item['local'].startswith('pi/') and a.profile != 'pi':
            continue
        if a.profile == 'mx8' and item['local'].startswith('gsq/'):
            continue
        out = STATE / 'inputs' / item['local']
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if digest(out) != item['sha256']:
                raise SystemExit(f'Checksum mismatch: {out}; remove this file explicitly before retrying')
            print(f'Already verified: {item["local"]}', flush=True)
            continue
        partial = out.with_name(out.name + '.partial')
        subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--continue-at', '-',
                        '--output', str(partial), item['url']], check=True)
        if digest(partial) != item['sha256']:
            raise SystemExit(f'Checksum mismatch: {partial}; remove partial explicitly before retrying')
        partial.replace(out)
        print(f'Verified: {item["local"]}', flush=True)


if __name__ == '__main__':
    main()
