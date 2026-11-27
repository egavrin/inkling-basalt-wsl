#!/usr/bin/env python3
"""Runs only inside the network namespace; exports a single inference socket."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from relay import server


def main():
    command = json.loads(sys.argv[1])
    expected_context = int(command[command.index('--max-context') + 1])
    alias = command[command.index('--alias') + 1]
    stop = threading.Event()
    process = subprocess.Popen(command)
    bridge = None

    def terminate(*args):
        stop.set()
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    try:
        deadline = time.monotonic() + 900
        while not stop.is_set() and process.poll() is None:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8191/health', timeout=5) as response:
                    health = json.load(response)
                if not (health.get('loaded') is True and health.get('loading') is False
                        and health.get('model') == alias and health.get('max_context') == expected_context):
                    raise ValueError('Weights/context not ready')
                bridge = server('/bridge/inference.sock', ('127.0.0.1', 8191))
                Path('/bridge/health.json').write_text(json.dumps(health, indent=2) + '\n')
                print('Ready: private inference.sock', flush=True)
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Model did not become ready within 900 seconds')
                stop.wait(1)
        if bridge is None:
            raise RuntimeError('Model stopped before readiness')
        returncode = process.wait()
        if returncode and not stop.is_set():
            raise RuntimeError(f'Basalt exited with status {returncode}')
    finally:
        if bridge is not None:
            bridge.shutdown()
            bridge.server_close()
        terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        Path('/bridge/inference.sock').unlink(missing_ok=True)
        Path('/bridge/health.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
