"""Trusted token counting and lifecycle recovery. No client-provided billing fields."""
import asyncio
import fcntl
import os
from pathlib import Path
from .storage import LedgerError

class Counter:
    def __init__(self,backend,model,revision,contract_test=False):
        self.backend,self.model,self.revision,self.contract_test=backend,model,revision,contract_test
        self.tokenizer=None
        self.lock=asyncio.Lock()
    async def count(self,payload):
        # Explicit backend injection is used by CPU tests; production backend has no count hook.
        if hasattr(self.backend,'count_input'):
            return await self.backend.count_input(payload)
        if self.contract_test:
            response=await self.backend.client.post('/tokenize',json={'model':self.model,'messages':payload['messages'],'add_generation_prompt':True})
            response.raise_for_status()
            count=response.json().get('count')
        else:
            async with self.lock:
                if self.tokenizer is None:
                    from transformers import AutoTokenizer
                    self.tokenizer=await asyncio.to_thread(AutoTokenizer.from_pretrained,self.model,revision=self.revision,trust_remote_code=False)
                ids=self.tokenizer.apply_chat_template(payload['messages'],tokenize=True,add_generation_prompt=True)
                count=len(ids)
        if type(count) is not int or count<1 or count>4096:
            raise LedgerError(502,'invalid_engine_token_count')
        if count+payload['max_tokens']>4096:
            raise LedgerError(400,'context_limit_exceeded')
        return count

class Recovery:
    def __init__(self,store,backend):
        self.store,self.backend=store,backend
        self.live=set()
        self.task=None
        self.lockfile=None
    async def start(self):
        self.lockfile=open(self.store.path.parent/'gateway.lock','a')
        try:
            fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lockfile.close();self.lockfile=None
            raise RuntimeError('DATA_DIR already has a live gateway; one worker per ledger')
        (self.store.path.parent/'gateway.pid').write_text(str(os.getpid()))
        await self.sweep(startup=True)
        self.task=asyncio.create_task(self.run())
    async def sweep(self,startup=False):
        for row in self.store.unfinished():
            rid=row['request_id']
            if rid in self.live: continue
            if not startup and row['status'] not in ('pending_reconciliation','usage_recorded'): continue
            if row['engine_started'] and not row['usage_evidence']:
                try: await asyncio.wait_for(self.backend.abort(rid),3)
                except Exception: pass
            self.store.reconcile(rid,'restart_without_terminal_usage' if startup else 'interrupted_without_terminal_usage')
    async def run(self):
        while True:
            await asyncio.sleep(2)
            await self.sweep()
    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task,return_exceptions=True)
        await self.sweep(startup=True)
        if self.lockfile:
            fcntl.flock(self.lockfile,fcntl.LOCK_UN);self.lockfile.close()
