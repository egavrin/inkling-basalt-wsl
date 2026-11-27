#!/usr/bin/env python3
"""Recreate DFlash with the pinned upstream converter; verify benchmark equivalence."""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import types

from download import digest, ROOT, STATE

sys.path[:0] = [str(STATE / 'basalt/tools/convert'), str(STATE / 'basalt/tools')]
from model_file import ModelFile, ModelWriter, Stream
from checkpoint_inkling2 import Dflash2

EXPECTED = json.loads((ROOT / 'manifests/equivalence.json').read_text())
NAMES = {'gsq': 'Inkling-2-Mini-GSQ-RCO-IQ3_S-DFlash2.basalt',
         'mx8': 'Inkling-2-Mini-DFlash2-MX8.basalt'}
ACTIVE_WRITER = None


def chunks(path, offset, size):
    with open(path, 'rb') as f:
        f.seek(offset)
        while size:
            b = f.read(min(size, 8 << 20))
            if not b:
                raise ValueError('Short tensor/section read')
            size -= len(b)
            yield b
            os.posix_fadvise(f.fileno(), 0, f.tell(), os.POSIX_FADV_DONTNEED)
            if ACTIVE_WRITER is not None and not ACTIVE_WRITER.f.closed:
                ACTIVE_WRITER.f.flush()
                os.fdatasync(ACTIVE_WRITER.f.fileno())
                os.posix_fadvise(ACTIVE_WRITER.f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def sha(parts):
    h = hashlib.sha256()
    for b in parts:
        h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def verify(path, profile):
    p = ModelFile(path)
    m = json.loads(p.read('meta'))
    if m['arch'] != 'inkling2':
        raise ValueError('Wrong model architecture')
    for name, (off, size, checksum) in p.sections.items():
        assert sha(chunks(path, off, size)) == checksum.hex(), name
    for name, expected in EXPECTED['sections'][profile].items():
        _, size, checksum = p.sections[name]
        assert {'bytes': size, 'sha256': checksum.hex()} == expected, name
    actual_kv = {k: canonical(v) for k, v in m['gguf_kv'].items()
                 if k.startswith('inkling2.') or k.startswith('tokenizer.')}
    assert actual_kv == EXPECTED['geometry'][profile], 'Model/drafter/tokenizer geometry differs'
    expected_tensors = EXPECTED[profile] | EXPECTED['dflash2']
    assert set(m['tensors']) == set(expected_tensors), 'Tensor set differs'
    section_off, section_size = p.section('gguf')
    for name, t in m['tensors'].items():
        assert {k: t[k] for k in ('shape', 'type', 'bytes')} == EXPECTED['tensor_geometry'][profile][name], name
        assert 0 <= t['offset'] and t['offset'] + t['bytes'] <= section_size, name
        assert sha(chunks(path, section_off + t['offset'], t['bytes'])) == expected_tensors[name], name
    return {'pass': True, 'profile': profile, 'file': Path(path).name,
            'sha256': digest(Path(path)), 'bytes': Path(path).stat().st_size,
            'target_tensors': len(EXPECTED[profile]), 'dflash2_tensors': len(EXPECTED['dflash2']),
            'geometry_and_tokenizer_match': True, 'non_tensor_sections_match': True}


def build(profile, drafter, draft_files):
    global ACTIVE_WRITER
    output = STATE / 'models' / NAMES[profile]
    if output.exists():
        return verify(output, profile)
    source = (STATE / 'models/gsq-base.basalt' if profile == 'gsq' else
              STATE / 'inputs/mx8/Inkling-2-Mini-NVFP4-MX8.basalt')
    base = ModelFile(source)
    meta = json.loads(base.read('meta'))
    for name, (off, size, checksum) in base.sections.items():
        assert sha(chunks(source, off, size)) == checksum.hex(), name
    selected = {n: t for n, t in meta['tensors'].items() if not n.startswith(('dspark.', 'dflash2.'))}
    assert set(selected) == set(EXPECTED[profile]), 'Base target tensor set differs'
    meta.pop('dspark', None)
    meta['gguf_kv'] = {k: v for k, v in meta['gguf_kv'].items() if not k.startswith(('inkling2.dspark.', 'inkling2.dflash2.'))}
    meta['gguf_kv'].update(drafter.kv())
    meta['dflash2'] = {'repo': 'incoai/Inkling-2-Mini-DFlash2',
                      'revision': '4f9088ea66c21f2b99ed657d1941c306b03e84d2',
                      'config': drafter.config, 'converter_tree': '0026804d2191e2c55ff7cb4d9eddfa122f4ac511'}
    meta['reproduction'] = {'format': 1, 'target_profile': profile,
                            'note': 'Tensor bytes and geometry verified against benchmark manifest'}
    partial = output.with_name(output.name + '.partial')
    writer = ModelWriter(str(partial))
    ACTIVE_WRITER = writer
    for name in base.names():
        if name not in ('meta', 'gguf'):
            off, size = base.section(name)
            writer.add_chunks(name, chunks(source, off, size))
    entries = list(selected.items())
    parts = [Stream(t['bytes'], chunks(source, base.section('gguf')[0] + t['offset'], t['bytes']))
             for n, t in entries]
    for n, t in drafter.tensors.items():
        entries.append((n, {'shape': t.ne, 'type': t.type_name, 'type_id': t.type_id, 'bytes': t.expected_bytes()}))
        parts.append(Stream(t.expected_bytes(), chunks(draft_files[n], 0, t.expected_bytes())))
    offsets = writer.add_parts('gguf', parts)
    meta['tensors'] = {n: dict(t, offset=off) for (n, t), off in zip(entries, offsets)}
    writer.add_bytes('meta', json.dumps(meta, separators=(',', ':')).encode())
    writer.close()
    ACTIVE_WRITER = None
    report = verify(partial, profile)
    partial.replace(output)
    report['file'] = output.name
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['build', 'verify'])
    ap.add_argument('profile', nargs='?', choices=['gsq', 'mx8', 'both'], default='both')
    a = ap.parse_args()
    profiles = list(NAMES) if a.profile == 'both' else [a.profile]
    reports = []
    if a.action == 'build':
        cfg = json.loads((ROOT / 'manifests/target-config.json').read_text())
        drafter = Dflash2(STATE / 'inputs/dflash2', types.SimpleNamespace(text=cfg['text_config']))
        tmp = STATE / 'draft-tensors'
        tmp.mkdir(exist_ok=True)
        files = {}
        with multiprocessing.Pool(1) as pool:
            for n, t in drafter.tensors.items():
                path = tmp / n
                with path.open('wb') as f:
                    parts = drafter.chunks(n, pool, depth=1) if t.type_name == 'Q8_0' else [drafter.float_bytes(n)]
                    for part in parts:
                        f.write(part)
                assert digest(path) == EXPECTED['dflash2'][n], n
                files[n] = path
        for profile in profiles:
            reports.append(build(profile, drafter, files))
    else:
        for profile in profiles:
            reports.append(verify(STATE / 'models' / NAMES[profile], profile))
    previous = STATE / 'pack-verification.json'
    if previous.exists():
        reports += [r for r in json.loads(previous.read_text()) if r.get('profile') not in profiles]
    (STATE / 'pack-verification.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
