import json
import os
import signal
import subprocess
import time
import httpx
from transformers import AutoTokenizer
from server.config import configuration
from server.supervisor import engine_command
from .common import Evidence, digest, write, entry

def main():
    evidence=Evidence('prefix', {'prefix_tokens':512,'requests':8,'repeats':3,
        'variants':['off','on'],'groups':['shared','unique'],'max_new_tokens':8,
        'temperature':0,'ttft_slo_ms':200,'e2e_slo_ms':1500})
    tok=AutoTokenizer.from_pretrained(os.environ['MODEL_ID'],revision=os.environ['MODEL_REVISION'])
    prefixes=[]
    for i in range(8):
        tokens=tok.encode('<|im_start|>system\n'+(f'Reference document {i}: topic {chr(65+i)}. '*600),add_special_tokens=False)
        prefixes.append(tokens[:512]); assert len(prefixes[-1])==512
    suffixes=[tok.encode(f'<|im_end|>\n<|im_start|>user\nRequest {i}: reply with exactly OK and nothing else.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False) for i in range(8)]
    assert len(set(map(len,suffixes)))==1
    groups={name:[(prefixes[0] if name=='shared' else prefixes[i])+suffixes[i] for i in range(8)] for name in ['shared','unique']}
    write(evidence.out/'inputs.json',groups)
    evidence.manifest['input_hash']=digest(groups)
    _,cfg=configuration();cfg['max_running_requests']=8
    summaries=[]
    for enabled in [False,True]:
        variant='on' if enabled else 'off'
        command=engine_command(cfg,cache=enabled)
        write(evidence.out/f'{variant}-command.json',command)
        log=(evidence.out/f'{variant}-engine.log').open('w')
        process=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
        try:
            with httpx.Client(base_url='http://127.0.0.1:30000',timeout=60,trust_env=False) as client:
                deadline=time.monotonic()+160
                while True:
                    if process.poll() is not None: raise RuntimeError('engine exited')
                    try:
                        if client.get('/health',timeout=2).status_code==200:break
                    except httpx.HTTPError:pass
                    if time.monotonic()>deadline:raise RuntimeError('engine readiness deadline')
                    time.sleep(1)
                # Shape/kernel warmup is excluded; flush AFTER warmup for every trial.
                warm=client.post('/generate',json={'input_ids':groups['shared'][0], 'sampling_params':{'temperature':0,'max_new_tokens':8}})
                warm.raise_for_status();write(evidence.out/f'{variant}-kernel-warmup.json',warm.json())
                for group,inputs in groups.items():
                    for repeat in range(3):
                        flush=client.post('/flush_cache');flush.raise_for_status()
                        write(evidence.out/f'{variant}-{group}-{repeat}-flush.json',{'status':flush.status_code,'time':time.time()})
                        group_rows=[];window=time.monotonic()
                        for index,ids in enumerate(inputs):
                            start=time.monotonic();first=None;last=None;obj={};done=False;error=None
                            try:
                                with client.stream('POST','/generate',json={'input_ids':ids,'stream':True,
                                    'sampling_params':{'temperature':0,'max_new_tokens':8}}) as response:
                                    response.raise_for_status()
                                    for line in response.iter_lines():
                                        if not line.startswith('data:'):continue
                                        data=line[5:].strip()
                                        if data=='[DONE]':done=True;break
                                        if not data:continue
                                        obj=json.loads(data)
                                        if obj.get('text'):
                                            first=first or time.monotonic();last=time.monotonic()
                                if not done:raise RuntimeError('truncated native stream')
                            except Exception as exc:error=type(exc).__name__
                            elapsed=(time.monotonic()-start)*1000
                            text=obj.get('text','');meta=obj.get('meta_info',{})
                            quality=text.strip().strip('.!')=='OK'
                            state='cold' if index==0 else ('hot' if enabled and group=='shared' else 'subsequent')
                            row={'variant':variant,'group':group,'repeat':repeat,'index':index,'cache_state':state,
                                'input_hash':digest(ids),'prefix_hash':digest(ids[:512]),'prefix_length':512,
                                'input_tokens':len(ids),'cached_tokens':meta.get('cached_tokens'),
                                'output':text,'quality_pass':quality,'ttft_ms':None if first is None else (first-start)*1000,
                                'e2e_ms':elapsed,'completion_tokens':meta.get('completion_tokens'),
                                'ok':error is None,'error':error}
                            evidence.row(**row);group_rows.append(row)
                        duration=time.monotonic()-window
                        summaries.append({'variant':variant,'group':group,'repeat':repeat,'seconds':duration,
                            'success_rate':sum(r['ok'] for r in group_rows)/8,
                            'quality_rate':sum(r['quality_pass'] for r in group_rows)/8,
                            'quality_goodput':sum(r['ok'] and r['quality_pass'] and r['ttft_ms']<=200 and r['e2e_ms']<=1500 for r in group_rows)/duration})
                (evidence.out/f'{variant}-metrics.txt').write_text(client.get('/metrics').text)
        finally:
            os.killpg(process.pid,signal.SIGTERM) if process.poll() is None else None
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            log.close();time.sleep(2)
    evidence.finish({'groups':summaries,'counter_source':'SGLang /generate meta_info.cached_tokens',
        'input_note':'Exactly 512 shared prefix IDs; all full token IDs saved. Unique group retains the short common chat header, which is legitimate reuse.'})
if __name__=='__main__': entry(main)
