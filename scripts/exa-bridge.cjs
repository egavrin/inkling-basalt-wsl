// Host-only service. No credential or arbitrary proxy interface enters Pi.
const fs = require('node:fs');
const http = require('node:http');
const https = require('node:https');

function readKey(path) {
  const fd = fs.openSync(path, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile() || stat.size > 65536 || (stat.mode & 0o077)) throw Error('permissions');
    const lines = fs.readFileSync(fd, 'utf8').split(/\r?\n/);
    const values = lines.filter(line => /^\s*(?:export\s+)?EXA_API_KEY\s*=/.test(line));
    if (values.length !== 1) throw Error('missing or duplicate key');
    let key = values[0].replace(/^\s*(?:export\s+)?EXA_API_KEY\s*=\s*/, '').trim();
    if ((key.startsWith('"') && key.endsWith('"')) || (key.startsWith("'") && key.endsWith("'"))) key = key.slice(1, -1);
    if (!key || /[\s\x00-\x1f]/.test(key)) throw Error('invalid key');
    return key; // Data only: no interpolation, sourcing, or executable dotenv.
  } finally { fs.closeSync(fd); }
}

function createBridge(key) {
  let active = 0;
  const server = http.createServer((req, res) => {
    const reply = (status, data) => {
      if (!res.destroyed && !res.writableEnded) {
        res.writeHead(status, { 'content-type': 'application/json' });
        res.end(JSON.stringify(data));
      }
    };
    if (req.method !== 'POST' || req.url !== '/search') { reply(404, {error:'Only POST /search is supported'}); req.resume(); return; }
    if (active >= 4) { reply(429, {error:'Search concurrency limit reached'}); req.resume(); return; }
    active++;
    let upstream, timer, finished = false;
    const finish = () => { if (finished) return; finished = true; active--; clearTimeout(timer); upstream?.destroy(); };
    res.on('close', finish);
    timer = setTimeout(() => { reply(504, {error:'Exa search timed out'}); finish(); }, 25000);
    let body = '';
    req.on('error', finish);
    req.on('data', chunk => {
      body += chunk;
      if (Buffer.byteLength(body) > 8192) { reply(413, {error:'Search request too large'}); finish(); req.destroy(); }
    });
    req.on('end', () => {
      if (finished) return;
      let input;
      try {
        input = JSON.parse(body);
        if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).some(k => !['query','numResults'].includes(k))) throw Error();
        if (typeof input.query !== 'string' || !input.query.trim() || input.query.length > 2000) throw Error();
        if (input.numResults !== undefined && (!Number.isInteger(input.numResults) || input.numResults < 1 || input.numResults > 8)) throw Error();
      } catch { reply(400, {error:'Expected query (1..2000 characters) and optional numResults (1..8)'}); finish(); return; }
      const count = input.numResults ?? 5;
      const payload = JSON.stringify({query:input.query.trim(), numResults:count, type:'auto', contents:{highlights:true}});
      upstream = https.request('https://api.exa.ai/search', {method:'POST', headers:{'x-api-key':key,'content-type':'application/json','content-length':Buffer.byteLength(payload)}}, response => {
        // Never follow redirects, expose provider errors, or forward provider headers.
        if (response.statusCode !== 200) { response.resume(); reply(502, {error:`Exa search failed (HTTP ${response.statusCode})`}); finish(); return; }
        let data = '';
        response.on('data', chunk => {
          data += chunk;
          if (Buffer.byteLength(data) > 1048576) { reply(502, {error:'Exa response too large'}); finish(); }
        });
        response.on('error', () => { reply(502, {error:'Exa response interrupted'}); finish(); });
        response.on('end', () => {
          if (finished) return;
          try {
            const parsed = JSON.parse(data);
            if (!Array.isArray(parsed.results)) throw Error();
            const clean = (value, limit) => typeof value === 'string' ? value.split(key).join('[redacted]').slice(0, limit) : '';
            const results = parsed.results.slice(0,count).map(item => ({title:clean(item.title,300), url:clean(item.url,2048), highlights:Array.isArray(item.highlights) ? item.highlights.slice(0,3).map(x => clean(x,1000)) : []}));
            reply(200, {results});
          } catch { reply(502, {error:'Invalid Exa response'}); }
          finish();
        });
      });
      upstream.on('error', () => { reply(502, {error:'Exa search connection failed'}); finish(); });
      upstream.end(payload);
    });
  });
  server.requestTimeout = 26000;
  server.headersTimeout = 10000;
  return server;
}

if (require.main === module) {
  try {
    const key = readKey(process.argv[3]);
    const server = createBridge(key);
    server.on('error', () => { console.error('Cannot start Exa bridge'); process.exit(1); });
    server.listen(process.argv[2], () => fs.chmodSync(process.argv[2], 0o600));
  } catch {
    console.error('Search requires one EXA_API_KEY in the shared .env (regular file, chmod 600). Use --no-search to launch without search.');
    process.exit(1);
  }
}
module.exports = {createBridge, readKey};
