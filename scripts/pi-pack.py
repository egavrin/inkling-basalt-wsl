#!/usr/bin/env python3
"""CPU-only pinned preview pack; run build or verify under the existing Basalt venv."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, json, struct, hashlib, re
import numpy as np

from download import ROOT, STATE, digest
WORK = STATE / 'pi-merge'
SRC = STATE / 'basalt'
sys.path[:0] = [str(SRC/'tools'), str(SRC/'tools/convert')]
from model_file import ModelFile, ModelWriter
from model_sections import q8_0, q8_0_values, _bf16_to_f32
from pack import verify as verify_stock
BASE = STATE/'models/Inkling-2-Mini-GSQ-RCO-IQ3_S-DFlash2.basalt'
OUT = STATE/'models/Inkling-2-Mini-GSQ-RCO-IQ3_S-DFlash2-Pi-7788a53.basalt'
EXPECTED = json.loads((ROOT/'manifests/pi-equivalence.json').read_text())
ADAPTER_SHA = EXPECTED['adapter_sha256']
ACTIVE_WRITER = None

def chunks(path, offset=0, size=None):
    with open(path,'rb') as f:
        f.seek(offset)
        left = path.stat().st_size-offset if size is None else size
        while left:
            b=f.read(min(left,8<<20)); assert b
            left-=len(b); yield b
            os.posix_fadvise(f.fileno(),offset,f.tell()-offset,os.POSIX_FADV_DONTNEED)
            if ACTIVE_WRITER is not None and not ACTIVE_WRITER.f.closed:
                ACTIVE_WRITER.f.flush()
                os.fdatasync(ACTIVE_WRITER.f.fileno())
                os.posix_fadvise(ACTIVE_WRITER.f.fileno(),0,0,os.POSIX_FADV_DONTNEED)

def sha(parts):
    h=hashlib.sha256()
    for b in parts: h.update(b)
    return h.hexdigest()

def st_header(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return json.loads(f.read(n)),n+8

def array(path,header,offset,key):
    t=header[key]; a,b=t['data_offsets']
    raw=b''.join(chunks(path,offset+a,b-a))
    if t['dtype']=='BF16': x=_bf16_to_f32(np.frombuffer(raw,dtype='<u2'))
    else:
        assert t['dtype'] in ('F32','F16'), t
        x=np.frombuffer(raw,dtype='<f4' if t['dtype']=='F32' else '<f2').astype(np.float32)
    return x.reshape(t['shape'])

def verify(candidate=OUT):
    base,out=ModelFile(str(BASE)),ModelFile(str(candidate))
    verify_stock(BASE, 'gsq')
    report = {}
    targets = set(EXPECTED['targets'])
    assert base.names()==out.names()
    assert base.read('meta')==out.read('meta'), 'Metadata must remain byte-identical'
    meta=json.loads(base.read('meta'))
    sections={}
    for name, (offset,size,digest) in out.sections.items():
        actual=sha(chunks(candidate,offset,size)); assert actual==digest.hex(),name
        bo,bs,bd=base.sections[name]
        assert sha(chunks(BASE,bo,bs))==bd.hex(),name
        if name!='gguf': assert (size,digest)==(bs,bd),name
        sections[name]={'bytes':size,'sha256':actual,'unchanged':digest==bd}
    unchanged=0; dflash=0
    for name,t in meta['tensors'].items():
        h0=sha(chunks(BASE,base.section('gguf')[0]+t['offset'],t['bytes']))
        h1=sha(chunks(candidate,out.section('gguf')[0]+t['offset'],t['bytes']))
        if name in targets:
            record=EXPECTED['targets'][name]
            assert h0!=h1 and h0==record['base_sha256'] and h1==record['merged_sha256'],name
        else:
            assert h0==h1,name
            unchanged+=1; dflash+=name.startswith('dflash2.')
    report['verification']={'pass':True,'sections':sections,'unchanged_tensors':unchanged,'unchanged_dflash2_tensors':dflash,'changed_tensors':len(targets),'metadata_byte_identical':True,'output_sha256':sha(chunks(candidate)),'output_bytes':candidate.stat().st_size}
    assert unchanged == 641 and dflash == 81 and len(targets) == 96
    (WORK/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['verification'],indent=2),flush=True)
    return report['verification']

def build():
    global ACTIVE_WRITER
    if OUT.exists():
        return verify()
    adapter=STATE/'inputs/pi/adapter_model.safetensors'
    assert sha(chunks(adapter))==ADAPTER_SHA
    verify_stock(BASE, 'gsq')
    cfg=json.loads((ROOT/'manifests/pi-adapter-config.json').read_text())
    assert cfg['r']==32 and cfg['lora_alpha']==64
    assert not any(cfg[k] for k in ('fan_in_fan_out','use_rslora','use_dora','use_qalora','rank_pattern','alpha_pattern','modules_to_save','lora_bias'))
    ah,ao=st_header(adapter)
    base=ModelFile(str(BASE)); meta=json.loads(base.read('meta')); gg=base.section('gguf')[0]
    mapping={'q':('wq_du','q'),'k':('wk_dv','k'),'v':('wv_dv','v'),'o':('wo_ud','output')}
    report={'adapter_sha256':ADAPTER_SHA,'base_sha256':digest(BASE),'output':str(OUT),'approach':'Exact baseline Q8_0 dequantized to FP32 + (alpha/r=2)*(FP32 B @ FP32 A), then pinned converter Q8_0; no intermediate BF16 rounding. Original BF16 source rejected because pristine quantization differs from baseline.','targets':[]}
    replacements=[]; consumed=set()
    for key in sorted(ah):
        if key=='__metadata__' or '.lora_A.' not in key: continue
        match=re.fullmatch(r'base_model.model.model.language_model.layers.(\d+).self_attn.([qkvo])_proj.linear.lora_A.weight',key)
        assert match,key
        layer,proj=match.groups(); src,dst=mapping[proj]
        source=f'model.llm.layers.{layer}.attn.{src}.weight'; name=f'blk.{layer}.attn_{dst}.weight'
        bkey=key.replace('.lora_A.','.lora_B.'); consumed.update((key,bkey))
        a=array(adapter,ah,ao,key); b=array(adapter,ah,ao,bkey)
        t=meta['tensors'][name]; st={'dtype':'BF16','shape':EXPECTED['targets'][name]['shape']}
        assert st['dtype']=='BF16' and t['type']=='Q8_0' and t['type_id']==8
        assert st['shape']==list(reversed(t['shape'])) and a.shape==(32,st['shape'][1]) and b.shape==(st['shape'][0],32),(name,st,t,a.shape,b.shape)
        original=b''.join(chunks(BASE,gg+t['offset'],t['bytes']))
        x=q8_0_values(np.frombuffer(original,dtype=np.uint8)).reshape(st['shape'])
        assert q8_0(x)==original, f'Zero-delta Q8 roundtrip failed: {name}'
        delta=(b@a)*np.float32(2); merged=x+delta
        assert np.isfinite(merged).all() and np.count_nonzero(delta)>0
        result=q8_0(merged); assert len(result)==t['bytes'] and result!=original
        dest=WORK/(name+'.q8'); dest.write_bytes(result)
        replacements.append((t['offset'],t['bytes'],dest))
        report['targets'].append({'tensor':name,'source_geometry_reference':source,'a':key,'b':bkey,'shape':st['shape'],'adapter_dtype':ah[key]['dtype'],'zero_delta_q8_roundtrip':True,'base_sha256':sha([original]),'merged_sha256':sha([result]),'delta_rms':float(np.sqrt(np.mean(delta**2))),'delta_max':float(np.max(np.abs(delta)))})
        print('merged',name,flush=True)
    assert consumed==set(ah)-{'__metadata__'}, set(ah)-consumed
    assert len(replacements)==96,len(replacements)
    replacements.sort()
    def merged_chunks():
        pos=0
        for off,size,path in replacements:
            assert off>=pos
            yield from chunks(BASE,gg+pos,off-pos)
            yield from chunks(path)
            pos=off+size
        yield from chunks(BASE,gg+pos,base.section('gguf')[1]-pos)
    partial=OUT.with_suffix('.basalt.partial'); writer=ModelWriter(str(partial)); ACTIVE_WRITER=writer
    for name in base.names():
        off,size=base.section(name)
        writer.add_chunks(name,merged_chunks() if name=='gguf' else chunks(BASE,off,size))
    writer.close()
    ACTIVE_WRITER=None
    (WORK/'merge-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    checked = verify(partial)
    partial.replace(OUT)
    return checked

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['build', 'verify'])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    result = build() if a.action == 'build' else verify()
    stamp = STATE / 'pack-verification.json'
    reports = json.loads(stamp.read_text()) if stamp.exists() else []
    reports = [r for r in reports if r.get('profile') != 'pi']
    reports.append({'pass':True, 'profile':'pi', 'file':OUT.name,
                    'sha256':result['output_sha256'], 'bytes':result['output_bytes']})
    stamp.write_text(json.dumps(reports, indent=2)+'\n')

if __name__ == '__main__':
    main()
