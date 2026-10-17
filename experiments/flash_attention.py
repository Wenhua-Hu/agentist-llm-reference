import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa
from .common import Evidence, tensor_hash, write, entry

def main():
    parameters = {'B':1,'H':8,'dh':64,'lengths':[128,512,1024], 'dtype':'bfloat16',
                  'seed':20261017,'warmup':5,'groups':3,'iterations':10,'atol':.02,'rtol':.02,
                  'causal':True,'dropout':0}
    evidence = Evidence('attention', parameters)
    torch.manual_seed(parameters['seed'])
    summaries = []
    for length in parameters['lengths']:
        q,k,v = [torch.randn(1,8,length,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        hashes = [tensor_hash(t) for t in (q,k,v)]
        with sdpa_kernel(SDPBackend.MATH): reference = sdpa(q,k,v,is_causal=True,dropout_p=0)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION): optimized = sdpa(q,k,v,is_causal=True,dropout_p=0)
        error = float((reference.float()-optimized.float()).abs().max())
        valid = bool(torch.allclose(reference,optimized,atol=.02,rtol=.02))
        for name,backend in [('math',SDPBackend.MATH),('flash',SDPBackend.FLASH_ATTENTION)]:
            with sdpa_kernel(backend), torch.inference_mode():
                for _ in range(5): sdpa(q,k,v,is_causal=True,dropout_p=0)
                torch.cuda.synchronize()
                for repeat in range(3):
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); start = time.perf_counter()
                    for _ in range(10): result = sdpa(q,k,v,is_causal=True,dropout_p=0)
                    torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)*1000/10
                    evidence.row(variant=name,length=length,repeat=repeat,shape=list(q.shape),input_hashes=hashes,
                        elapsed_ms=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), max_abs_error=error,
                        atol=.02,rtol=.02,ok=valid, output_shape=list(result.shape),cache_state=None)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,profile_memory=True) as profiler:
                    sdpa(q,k,v,is_causal=True,dropout_p=0); torch.cuda.synchronize()
                trace=evidence.out/f'{name}-{length}.trace.json'; profiler.export_chrome_trace(str(trace))
                keys=[event.key for event in profiler.key_averages()]
                write(evidence.out/f'{name}-{length}-operators.json',keys)
                import hashlib
                write(evidence.out/f'{name}-{length}-trace-sha256.json', {'sha256':hashlib.sha256(trace.read_bytes()).hexdigest()})
        summaries.append({'length':length,'max_abs_error':error,'correct':valid,
                          'score_matrix_elements':8*length*length,'bf16_S_and_P_bytes':2*8*length*length*2})
    evidence.finish({'shapes':summaries,'note':'Microbenchmark only; optimized service uses FlashInfer, baseline uses Triton. Neither is claimed to be this PyTorch SDPA Flash implementation.'})
if __name__=='__main__': entry(main)
