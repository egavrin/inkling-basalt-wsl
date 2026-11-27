#!/usr/bin/env python3
"""Foreground WSL2 GPU sandbox. No network except an explicit inference socket."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess

from download import ROOT, STATE, digest

NAMES = {'gsq': 'Inkling-2-Mini-GSQ-RCO-IQ3_S-DFlash2.basalt',
         'mx8': 'Inkling-2-Mini-DFlash2-MX8.basalt'}
REFERENCE = {'gsq': '1fc2e87ab9fa12446319c1429d9edcda38c541ed0db064ec437b60247e6c680e',
             'mx8': '5c1f52101cd598c1578832f6c17c20a9c72a0a50ede0839b31d971369301907a'}


def command(profile, context, expert_cache=None, vram_reserve_mib=512):
    args = ['/basalt/bin/basalt', 'serve', '/model/' + NAMES[profile],
            '--alias', 'inkling-2-mini', '--host', '127.0.0.1', '--port', '8191',
            '--parallel', '1', '--max-context', str(context), '--kv', 'fp8', '--spec', '8',
            '--suffix-draft', '0', '--vision', 'off', '--pool-workers', '10',
            '--vram-reserve-mib', str(vram_reserve_mib), '--ram-reserve-mib', '4096', '--fit-max-tokens', '--request-lines']
    if expert_cache is not None:
        args += ['--expert-cache', str(expert_cache)]
    return args


def sandbox(cuda):
    args = ['bwrap', '--die-with-parent', '--new-session', '--unshare-all', '--clearenv',
            '--proc', '/proc', '--dev', '/dev', '--dev-bind', '/dev/dxg', '/dev/dxg', '--tmpfs', '/tmp']
    for path in ('/usr', '/bin', '/lib', '/lib64', '/sys'):
        args += ['--ro-bind', path, path]
    args += ['--ro-bind', str(cuda), '/cuda', '--setenv', 'HOME', '/tmp',
             '--setenv', 'PATH', '/venv/bin:/usr/bin:/bin:/usr/lib/wsl/lib',
             '--setenv', 'LD_LIBRARY_PATH', '/cuda/lib64:/cuda/lib:/usr/lib/wsl/lib',
             '--setenv', 'PYTHONUNBUFFERED', '1']
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=NAMES, default='gsq')
    ap.add_argument('--context', type=int, default=270336,
                    help='explicit total context cap; larger values require independent memory validation')
    ap.add_argument('--expert-cache', type=int,
                    help='explicit expert-cache count (1-3072); omitted uses Basalt automatic sizing')
    ap.add_argument('--vram-reserve-mib', type=int, choices=(512, 768, 1024), default=512,
                    help='VRAM reserve in MiB (default: 512)')
    ap.add_argument('--dry-run', action='store_true', help='print contract without loading files or GPU')
    ap.add_argument('--check-sandbox', action='store_true', help='check namespace visibility without loading the model')
    a = ap.parse_args()
    if not 1024 <= a.context <= 1048576:
        ap.error('context must be between 1024 and the model limit 1048576')
    if a.expert_cache is not None and not 1 <= a.expert_cache <= 3072:
        ap.error('expert-cache must be between 1 and 3072')
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0')).resolve()
    mounts = sandbox(cuda)
    cmd = command(a.profile, a.context, a.expert_cache, a.vram_reserve_mib)
    if a.check_sandbox:
        subprocess.run(mounts + ['/usr/bin/python3', '-c',
            "import os,socket; assert not os.path.exists('/home'); assert not os.path.exists('/mnt/c'); "
            "assert socket.if_nameindex()==[(1,'lo')]; assert socket.socket().connect_ex(('1.1.1.1',443))!=0; "
            "assert os.path.exists('/dev/dxg'); print('Private network, hidden homes, GPU device visible: PASS')"], check=True)
        return
    mounts += ['--ro-bind', str(STATE / 'basalt'), '/basalt', '--ro-bind', str(STATE / 'venv'), '/venv',
               '--ro-bind', str(ROOT / 'scripts'), '/scripts', '--ro-bind',
               str(STATE / 'models' / NAMES[a.profile]), '/model/' + NAMES[a.profile],
               '--bind', str(STATE / 'run'), '/bridge', '--chdir', '/basalt']
    launch = mounts + ['/usr/bin/python3', '/scripts/service.py', json.dumps(cmd)]
    if a.dry_run:
        print(json.dumps({'profile': a.profile, 'command': cmd, 'sandbox': launch}, indent=2))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binary_record = (STATE / 'binary.sha256').read_text().split()[0]
        if digest(STATE / 'basalt/bin/basalt') != binary_record:
            raise SystemExit('Binary changed since build; rebuild before serving')
        expected = REFERENCE[a.profile]
        verification = STATE / 'pack-verification.json'
        if verification.exists():
            matches = [r for r in json.loads(verification.read_text())
                       if r.get('pass') is True and r.get('profile') == a.profile and r.get('file') == NAMES[a.profile]]
            if matches:
                expected = matches[0]['sha256']
        if digest(STATE / 'models' / NAMES[a.profile]) != expected:
            raise SystemExit('Pack checksum mismatch; run pack.py verify or download verified release packs')
        run = STATE / 'run'
        run.mkdir(mode=0o700, exist_ok=True)
        run.chmod(0o700)
        (run / 'inference.sock').unlink(missing_ok=True)
        (run / 'health.json').unlink(missing_ok=True)
        (run / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        print(f'Private API: curl --unix-socket {run}/inference.sock http://localhost/health', flush=True)
        process = subprocess.Popen(launch)

        def stop(*args):
            if process.poll() is None:
                process.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            raise SystemExit(process.wait())
        finally:
            stop()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (run / 'inference.sock').unlink(missing_ok=True)
            (run / 'health.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
