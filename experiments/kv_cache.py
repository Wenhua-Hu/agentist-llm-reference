import gc
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .common import Evidence, tensor_hash, entry

def main():
    dtype_name=os.environ.get('KV_DTYPE','bfloat16')
    if dtype_name not in ('bfloat16','float32'): raise ValueError('KV_DTYPE must be bfloat16 or float32')
    dtype=getattr(torch,dtype_name)
    atol,rtol=(.15,.02) if dtype_name=='bfloat16' else (.001,.0001)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    parameters={'lengths':[128,512],'decode_tokens':32,'warmup':1,'repeats':3,
                'dtype':dtype_name,'atol':atol,'rtol':rtol,'tf32':False,'seed':20261017}
    evidence=Evidence('kv',parameters)
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_ID'],revision=os.environ['MODEL_REVISION'])
    model=AutoModelForCausalLM.from_pretrained(os.environ['MODEL_ID'],revision=os.environ['MODEL_REVISION'],
        torch_dtype=dtype,attn_implementation='eager').cuda().eval()
    cfg=model.config; dh=cfg.hidden_size//cfg.num_attention_heads
    bytes_per_token=2*cfg.num_hidden_layers*cfg.num_key_value_heads*dh*model.get_input_embeddings().weight.element_size()
    text=('A model server processes requests while preserving correctness. '*200)
    ids=tokenizer(text,return_tensors='pt').input_ids.cuda()
    summaries=[]
    with torch.inference_mode():
        for length in parameters['lengths']:
            prompt=ids[:,:length]
            generated=model.generate(prompt,max_new_tokens=32,min_new_tokens=32,do_sample=False,
                use_cache=True,pad_token_id=tokenizer.eos_token_id)
            continuation=generated[:,length:length+32]
            assert continuation.shape[-1]==32
            def replay(use_cache, capture=False):
                cache=None; full=prompt; logits=[]
                if use_cache:
                    prefill=model(prompt,use_cache=True)
                    cache=prefill.past_key_values
                    del prefill
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                begin=time.perf_counter()
                for i in range(32):
                    token=continuation[:,i:i+1]
                    full=torch.cat([full,token],dim=1)
                    result=model(token if use_cache else full,past_key_values=cache,use_cache=use_cache)
                    if use_cache: cache=result.past_key_values
                    if capture: logits.append(result.logits[:,-1,:].float().clone())
                torch.cuda.synchronize()
                duration=(time.perf_counter()-begin)*1000
                allocated=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
                captured=torch.stack(logits).cpu() if capture else None
                return duration,allocated,reserved,captured
            # Validate every decode position outside the measured runs.
            off=replay(False,True)[3]; on=replay(True,True)[3]
            error=float((off-on).abs().max()); valid=bool(torch.allclose(off,on,atol=atol,rtol=rtol))
            top1=float((off.argmax(-1)==on.argmax(-1)).float().mean())
            for cache in (False,True): replay(cache)
            for repeat in range(3):
                for cache in ([False,True] if repeat%2==0 else [True,False]):
                    gc.collect();torch.cuda.empty_cache()
                    elapsed,allocated,reserved,_=replay(cache)
                    evidence.row(variant='cache_on' if cache else 'cache_off',length=length,repeat=repeat,
                        input_hash=tensor_hash(prompt),continuation_hash=tensor_hash(continuation),decode_tokens=32,
                        elapsed_ms=elapsed,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                        max_abs_error=error,atol=atol,rtol=rtol,top1_agreement=top1,dtype=dtype_name,ok=valid,
                        error=None if valid else 'logits_tolerance_exceeded',cache_state='prefilled' if cache else 'none')
            summaries.append({'length':length,'max_abs_error':error,'correct':valid,'top1_agreement':top1,
                'KV_bytes_one_request':bytes_per_token*(length+32),'KV_bytes_eight_requests':8*bytes_per_token*(length+32)})
    evidence.finish({'layers':cfg.num_hidden_layers,'KV_heads':cfg.num_key_value_heads,'head_dim':dh,
        'dtype':dtype_name,'dtype_bytes':model.get_input_embeddings().weight.element_size(),'KV_bytes_per_token':bytes_per_token,'measurements':summaries,
        'memory_note':'Logical active KV is distinct from allocator reserved bytes and the SGLang preallocated pool; weights also occupy process VRAM.'})
if __name__=='__main__': entry(main)
