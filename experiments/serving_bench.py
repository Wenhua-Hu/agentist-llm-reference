"""Local measurement client; platform scoring remains on its trusted controller.
measurement.py is reused from the Agentist Project 2 scaffold (see SOURCES.md).
Adds per-request cancellation/error accounting and reproducible SVG reporting.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import httpx
from .measurement import request_once, summarize
from .report import plot

async def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--requests',type=int,default=32);p.add_argument('--base',default='http://127.0.0.1:8000')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[];sem=asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=30,trust_env=False) as c:
        async def one(i):
            async with sem:
                fields={'id':10000+i,'city':'Boston','units':i%9+1}
                item={'id':str(i),'messages':[{'role':'system','content':'Extract exactly id, city, units. Return only JSON.'},
                    {'role':'user','content':json.dumps(fields)}],'max_tokens':96,'quality':{'json_fields':fields}}
                row=await request_once(c,args.base+'/v1/chat/completions',os.environ['MODEL_ID'],item,30,os.environ['API_KEY'],str(i),True)
                row.update(variant=os.environ.get('BENCH_PROFILE','unknown'),group='c'+str(args.concurrency),ttft_ms=row.get('ttft_proxy_ms'))
                rows.append(row)
        start=time.monotonic();await asyncio.gather(*(one(i) for i in range(args.requests)))
        summary=summarize(rows,time.monotonic()-start,200,1500)
        info=(await c.get(args.base+'/info',headers={'Authorization':'Bearer '+os.environ['API_KEY']})).json()
    (out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'manifest.json').write_text(json.dumps({'info':info,'concurrency':args.concurrency,'requests':args.requests,'warmup':0,
        'note':'Exploratory run; formal scoring independently uses warmup and 3 repeats.'},indent=2))
    plot(out)
    print(json.dumps(summary))
if __name__=='__main__':asyncio.run(main())
