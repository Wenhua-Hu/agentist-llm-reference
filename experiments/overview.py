"""Rebuild the report overview from published raw data; CPU only.
Install requirements-analysis.lock in a separate local venv, not during a GPU lease.
"""
import argparse
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results')
    args=parser.parse_args();root=Path(args.results)
    report=json.loads((root/'formal/report.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    colors={'baseline':'#718096','optimized':'#2458d3'}
    for axis,key,title,unit in zip(axes[0],['ttft_p95_ms','e2e_p95_ms','output_tokens_per_second'],
            ['Time to first content (p95)','End-to-end latency (p95)','Successful output throughput'],['ms','ms','tokens/s (usage not independently verified)']):
        for profile in ['baseline','optimized']:
            points=sorted([r for r in report['performance'] if r['profile']==profile],key=lambda r:r['concurrency'])
            axis.plot([r['concurrency'] for r in points],[r[key] for r in points],marker='o',label=profile,color=colors[profile])
        axis.set(title=title,xlabel='Client concurrency',ylabel=unit,xticks=[1,4,8]);axis.legend();axis.grid(alpha=.2)
    for axis,case,variants in [(axes[1,0],'kv',['cache_off','cache_on']),(axes[1,1],'attention',['math','flash'])]:
        data=rows(root/case/'rehearsal/raw.jsonl');lengths=sorted({r['length'] for r in data if 'length' in r})
        for index,variant in enumerate(variants):
            values=[mean(r['elapsed_ms'] for r in data if r.get('variant')==variant and r.get('length')==length) for length in lengths]
            axis.plot(lengths,values,marker='o',label=variant,color=['#718096','#2458d3'][index])
        axis.set(title='KV BF16: FAILED logits tolerance\nTiming only; not a correctness-qualified speedup' if case=='kv' else 'Attention: forced SDPA backends',xlabel='Input tokens',ylabel='Mean latency (ms)',xticks=lengths)
        axis.legend();axis.grid(alpha=.2)
    data=rows(root/'prefix/rehearsal/raw.jsonl');ax=axes[1,2];labels=[];values=[]
    for group in ['shared','unique']:
        for variant in ['off','on']:
            measured=[r['ttft_ms'] for r in data if r.get('group')==group and r.get('variant')==variant and r.get('index',0)>0 and r.get('ttft_ms') is not None]
            labels.append(group+'\n'+variant);values.append(mean(measured))
    ax.bar(labels,values,color=['#718096','#2458d3']*2);ax.set(title='Prefix reuse: subsequent requests',ylabel='Mean TTFT (ms)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Qwen2.5-1.5B / A10 / BF16 — measured reference implementation\n576 scored serving requests; independent optimization microbenchmarks',fontsize=16)
    out=root/'analysis';out.mkdir(exist_ok=True)
    fig.savefig(out/'overview.png',dpi=160);fig.savefig(out/'overview.svg')
    print(out/'overview.png')
if __name__=='__main__':main()
