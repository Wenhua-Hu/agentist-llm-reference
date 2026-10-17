"""Explicit serving cancellation diagnostic; never runs during startup."""
import asyncio
import json
import os
from pathlib import Path
import httpx

async def main():
    out=Path(os.environ.get('RUNTIME_DIR', '.runtime'))/'cancellation';out.mkdir(parents=True, exist_ok=True)
    headers={'Authorization':'Bearer '+os.environ['API_KEY']}
    base='http://127.0.0.1:'+os.environ['PORT']
    rows=[]
    async with httpx.AsyncClient(timeout=30,trust_env=False) as client:
        (out/'engine-before.txt').write_text((await client.get('http://127.0.0.1:30000/metrics')).text)
        for i in range(20):
            async with client.stream('POST',base+'/v1/chat/completions',headers=headers,json={
                'model':os.environ['MODEL_ID'],'messages':[{'role':'user','content':'Write a very long numbered list of different animals, at least 500 entries.'}],
                'max_tokens':2048,'stream':True}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith('data:') and 'content' in line:break
            await asyncio.sleep(.3)
            snapshot=(await client.get(base+'/metrics',headers=headers)).json()
            rows.append({'iteration':i,'snapshot':snapshot})
        await asyncio.sleep(2)
        final=(await client.get(base+'/metrics',headers=headers)).json()
        (out/'engine-after.txt').write_text((await client.get('http://127.0.0.1:30000/metrics')).text)
        (out/'results.json').write_text(json.dumps({'rows':rows,'final':final},indent=2))
        assert final['active']==0 and final['queued']==0
        assert final['counters'].get('abort_acknowledged',0)>=20
        print(json.dumps({'cancellation_cycles':20,'final':final}))
if __name__=='__main__':asyncio.run(main())
