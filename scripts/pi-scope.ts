import { resolve, relative, isAbsolute } from 'node:path';
import { realpathSync, existsSync } from 'node:fs';
// Defense in depth, in addition to mount and network namespaces. Code Mode's
// nested calls pass through the same tool_call hooks.
export default function(pi: any) {
 const maxOutputTokens = Number(process.env.INKLING_HARNESS_MAX_OUTPUT_TOKENS || 32768);
 pi.on('before_provider_request', async (event: any) => ({...event.payload, temperature:0.7, max_tokens:maxOutputTokens, reasoning_effort:'none'}));
 pi.on('tool_call', async (event: any) => {
  if (['bash','powershell'].includes(event.toolName)) return {block:true,reason:'Shell tools are disabled in this workspace'};
  if (!['read','write','edit','grep','find','ls'].includes(event.toolName)) return;
  const raw=event.input.path ?? event.input.filePath ?? '.';
  let p=resolve('/workspace',raw), anchor=p;
  while(!existsSync(anchor) && anchor!=='/') anchor=resolve(anchor,'..');
  const actual=resolve(realpathSync(anchor),relative(anchor,p));
  const rel=relative('/workspace',actual);
  if(rel==='..'||rel.startsWith('../')||isAbsolute(rel)) return {block:true,reason:'Only /workspace files are allowed'};
 });
}
