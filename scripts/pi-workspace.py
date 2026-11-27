#!/usr/bin/env python3
"""Interactive Pi workspace in WSL; host-only, narrowly scoped Exa search."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from relay import server
from download import STATE

D = Path(__file__).resolve().parent
B = STATE
H = B / 'pi-runtime'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('workspace', type=Path, help='Existing directory exposed read/write as /workspace')
    endpoint = ap.add_mutually_exclusive_group()
    endpoint.add_argument('--service', type=Path, help='Managed service evidence directory containing endpoint-socket.txt')
    endpoint.add_argument('--endpoint-socket', type=Path, default=STATE/'run/inference.sock', help='Inference Unix socket (including a merged LoRA service)')
    ap.add_argument('--model', default='inkling-2-mini')
    ap.add_argument('--no-search', action='store_true', help='Disable web_search; no credential is read')
    ap.add_argument('--print', action='store_true', dest='print_mode')
    ap.add_argument('--json', action='store_true', help='One-shot native JSONL events on stdout')
    ap.add_argument('--prompt', help='Optional initial prompt; required for one-shot modes')
    ap.add_argument('--continue', action='store_true', dest='continue_session')
    ap.add_argument('--max-output-tokens', type=int, default=32768)
    ap.add_argument('--check', action='store_true', help='CPU-only isolation and Pi extension load check; no inference or search')
    a = ap.parse_args()
    bwrap = shutil.which('bwrap')
    if bwrap is None:
        ap.error('bubblewrap is required; install it or add its directory to PATH')
    bwrap = str(Path(bwrap).resolve(strict=True))
    if not 1 <= a.max_output_tokens <= 65536:
        ap.error('--max-output-tokens must be 1..65536')
    if (a.print_mode or a.json) and not a.prompt and not a.check:
        ap.error('--print/--json requires --prompt')
    workspace = a.workspace.resolve(strict=True)
    if not workspace.is_dir():
        ap.error('workspace must be a directory')
    # Resolve before comparing, including a user-selected symlink to an ancestor.
    # The complete private tree is excluded, protecting credentials and runtime.
    private = B.resolve()
    if workspace == private or workspace in private.parents or private in workspace.parents:
        ap.error('workspace must not contain or be inside INKLING_STATE (credentials and private runtime)')
    credential = B / '.env'
    if credential.is_symlink():
        ap.error('shared .env must not be a symlink')
    target = (Path(a.service.resolve().joinpath('endpoint-socket.txt').read_text().strip()) if a.service else a.endpoint_socket).resolve()
    if not a.check and not target.is_socket():
        ap.error('Inference endpoint is not a Unix socket; start the selected service first')

    context = 270336
    if not a.check:
        def api(path):
            connection = http.client.HTTPConnection('localhost', timeout=20)
            connection.sock = socket.socket(socket.AF_UNIX)
            connection.sock.settimeout(20)
            try:
                connection.sock.connect(str(target))
                connection.request('GET', path)
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError('Service readiness request failed')
                return json.load(response)
            finally:
                connection.close()
        health = api('/health')
        context = health.get('max_context')
        if type(context) is not int or not 1024 <= context <= 1048576:
            ap.error('Service health lacks a valid max_context')
        if a.model not in [model.get('id') for model in api('/v1/models').get('data', [])]:
            ap.error('Selected endpoint does not advertise --model')

    os.umask(0o077)
    runs = B / 'pi-workspaces'
    runs.mkdir(parents=True, exist_ok=True)
    state = runs / hashlib.sha256(str(workspace).encode()).hexdigest()[:24]
    state.mkdir(exist_ok=True)
    if state.is_symlink():
        ap.error('Private workspace state must not be a symlink')
    # Serialize launches so the shared per-workspace session/config cannot race.
    import fcntl
    lock = (state / 'launch.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('Another Pi launcher is using this workspace')
    for name in ['home', 'config', 'cache', 'data', 'state', 'pi', 'sessions']:
        if (state / name).is_symlink():
            ap.error('Private workspace state directories must not be symlinks')
        (state / name).mkdir(exist_ok=True)
    tools = ['read', 'edit', 'write', 'grep', 'find', 'ls', 'codemode', 'tool_search']
    if not a.no_search:
        tools.append('web_search')
    def save(path, value):
        # Replace instead of following an old state-file symlink on the host.
        fd, temporary = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, 'w', encoding='utf8') as output:
            json.dump(value, output, indent=2)
        os.replace(temporary, path)
    save(state / 'pi/models.json', {'providers': {'inkling': {'baseUrl':'http://127.0.0.1:8190/v1','api':'openai-completions','apiKey':'local','models':[{'id':a.model,'name':a.model,'reasoning':False,'input':['text'],'contextWindow':context,'maxTokens':a.max_output_tokens,'cost':{'input':0,'output':0,'cacheRead':0,'cacheWrite':0},'compat':{'supportsDeveloperRole':False,'supportsStore':False,'maxTokensField':'max_tokens'}}]}}})
    save(state / 'pi/settings.json', {'defaultProvider':'inkling','defaultModel':a.model,'defaultThinkingLevel':'off','enabledModels':['inkling/'+a.model],'defaultTools':tools,'codemode':{'mode':'on'},'compaction':{'enabled':False},'retry':{'enabled':False},'cacheWarming':'off','packages':[]})
    with tempfile.TemporaryDirectory(prefix='pi-', dir=runs) as temporary:
        run = Path(temporary)
        config = run / 'config'
        bridge = run / 'bridge'
        config.mkdir()
        bridge.mkdir()
        for name in ['pi-relay.py', 'pi-scope.ts', 'pi-web-search.ts']:
            shutil.copyfile(D / name, config / name)
        relay = server(str(bridge / 'inference.sock'), str(target))
        search = None
        try:
            if not a.no_search:
                search = subprocess.Popen([str(H / 'runtime/node'), str(D / 'exa-bridge.cjs'), str(bridge / 'search.sock'), str(credential)], env={'PATH':'/usr/bin:/bin'}, stdout=subprocess.DEVNULL)
                for _ in range(100):
                    if (bridge / 'search.sock').is_socket():
                        break
                    if search.poll() is not None:
                        return 1
                    time.sleep(.05)
                else:
                    raise RuntimeError('Search bridge did not become ready')
            env = {'HOME':'/state/home','PATH':'/opt/runtime:/usr/bin:/bin','LANG':'C.UTF-8','TERM':os.environ.get('TERM','xterm-256color'),'XDG_CONFIG_HOME':'/state/config','XDG_CACHE_HOME':'/state/cache','XDG_DATA_HOME':'/state/data','XDG_STATE_HOME':'/state/state','PI_CODING_AGENT_DIR':'/state/pi','PI_OFFLINE':'1','INKLING_HARNESS_MAX_OUTPUT_TOKENS':str(a.max_output_tokens)}
            base = [bwrap,'--die-with-parent','--new-session','--unshare-all','--clearenv']
            for source, dest in [('/usr','/usr'),('/bin','/bin'),('/lib','/lib'),('/lib64','/lib64'),(str(H / 'pi'),'/opt/pi'),(str(H / 'runtime'),'/opt/runtime'),(str(config),'/config')]:
                if Path(source).exists():
                    base += ['--ro-bind',source,dest]
            base += ['--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/etc','--ro-bind','/etc/ssl','/etc/ssl','--bind',str(workspace),'/workspace','--bind',str(state),'/state','--ro-bind',str(bridge),'/bridge','--chdir','/workspace']
            for key, value in env.items():
                base += ['--setenv',key,value]
            base += ['/usr/bin/python3','/config/pi-relay.py']
            outer = {'PATH':'/usr/bin:/bin'}
            extensions = ['--no-extensions','-e','builtin:codemode','-e','builtin:tool-search','-e','/opt/pi/node_modules/@aliou/pi-guardrails/extensions/path-access/index.ts','-e','/opt/pi/node_modules/@aliou/pi-guardrails/extensions/guardrails/index.ts','-e','/opt/pi/node_modules/pi-token-stats/extensions/token-stats.ts','-e','/config/pi-scope.ts']
            if not a.no_search:
                extensions += ['-e','/config/pi-web-search.ts']
            cmd = ['/opt/pi/node_modules/.bin/pi',*extensions,'--offline','--no-mcp','--no-skills','--no-prompt-templates','--no-themes','--no-approve','--provider','inkling','--model',a.model,'--thinking','off','--tools',','.join(tools),'--session-dir','/state/sessions']
            if a.check:
                check = "import os,pathlib,socket; assert 'EXA_API_KEY' not in os.environ; assert not pathlib.Path('/home').exists(); assert not pathlib.Path('/mnt/c').exists(); s=socket.socket(); s.settimeout(.3); assert s.connect_ex(('1.1.1.1',443))!=0; print('credential environment, host home, Windows mounts and outbound network isolation PASS')"
                rc = subprocess.call(base + ['/usr/bin/python3','-c',check], env=outer)
                if rc:
                    return rc
                checked = subprocess.run(base + cmd + ['--mode','rpc'], input='{"id":"check","type":"get_state"}\n', text=True, capture_output=True, env=outer, timeout=45)
                print(checked.stdout, end='')
                print(checked.stderr, end='', file=sys.stderr)
                if checked.returncode or '"id":"check"' not in checked.stdout or 'Failed to load extension' in checked.stderr:
                    return 1
                print('Pi RPC startup and explicit extension load PASS')
                return 0
            if a.continue_session:
                cmd.append('--continue')
            if a.json:
                cmd += ['--mode','json']
            elif a.print_mode:
                cmd.append('--print')
            if a.prompt is not None:
                cmd += ['--',a.prompt]
            return subprocess.call(base + cmd, env=outer)
        finally:
            if search is not None:
                search.terminate()
                search.wait(timeout=5)
            relay.shutdown()
            relay.server_close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        # Do not dump environment, credential contents, or HTTP response bodies.
        print(f'Pi workspace launch failed: {exc}', file=sys.stderr)
        sys.exit(1)
