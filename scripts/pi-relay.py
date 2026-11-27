"""Byte relay: sandbox loopback can reach only the host's local inference port."""
import socket,socketserver,select,sys,threading,subprocess

class Handler(socketserver.BaseRequestHandler):
 def handle(self):
  target=socket.socket(socket.AF_UNIX if self.server.unix_target else socket.AF_INET)
  try:
   target.connect(self.server.target)
   while True:
    ready,_,_=select.select([target,self.request],[],[],7200)
    if not ready:return
    for src in ready:
     data=src.recv(65536)
     if not data:return
     (self.request if src is target else target).sendall(data)
  finally:target.close()

def server(address,target,unix_target=False):
 cls=socketserver.ThreadingTCPServer if isinstance(address,tuple) else socketserver.ThreadingUnixStreamServer
 cls.allow_reuse_address=True;cls.daemon_threads=True
 s=cls(address,Handler);s.target=target;s.unix_target=unix_target
 threading.Thread(target=s.serve_forever,daemon=True).start()
 return s

if __name__=='__main__':
 s=server(('127.0.0.1',8190),'/bridge/inference.sock',True)
 try:sys.exit(subprocess.call(sys.argv[1:]))
 finally:s.shutdown()
