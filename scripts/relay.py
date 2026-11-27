#!/usr/bin/env python3
"""Explicit host-loopback to private Unix-socket bridge (no outbound relay)."""
import argparse
import select
import socket
import socketserver
import threading


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        family = socket.AF_UNIX if isinstance(self.server.target, str) else socket.AF_INET
        with socket.socket(family) as target:
            target.connect(self.server.target)
            while True:
                ready, _, _ = select.select([target, self.request], [], [], 7200)
                if not ready:
                    return
                for source in ready:
                    data = source.recv(65536)
                    destination = self.request if source is target else target
                    if not data:
                        return
                    destination.sendall(data)


def server(address, target):
    cls = socketserver.ThreadingUnixStreamServer if isinstance(address, str) else socketserver.ThreadingTCPServer
    cls.allow_reuse_address = True
    cls.daemon_threads = True
    instance = cls(address, Handler)
    instance.target = target
    threading.Thread(target=instance.serve_forever, daemon=True).start()
    return instance


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--socket', required=True)
    ap.add_argument('--port', type=int, default=8191)
    a = ap.parse_args()
    if not 1024 <= a.port <= 65535:
        ap.error('port must be between 1024 and 65535')
    bridge = server(('127.0.0.1', a.port), a.socket)
    print(f'Loopback bridge: http://127.0.0.1:{a.port} -> {a.socket}', flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        bridge.server_close()
