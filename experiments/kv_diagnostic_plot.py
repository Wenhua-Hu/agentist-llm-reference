"""Plot independent precision controls; never merge them into one speedup."""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/kv-diagnostics');fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for ax,case,dtype in [(axes[0],'bf16','BF16'),(axes[1],'fp32','FP32 / TF32 disabled')]:
        directory=root/case
        rows=[json.loads(x) for x in (directory/'raw.jsonl').read_text().splitlines()]
        summary=json.loads((directory/'summary.json').read_text())
        lengths=sorted({r['length'] for r in rows})
        for variant,color in [('cache_off','#718096'),('cache_on','#2458d3')]:
            values=[mean(r['elapsed_ms'] for r in rows if r['variant']==variant and r['length']==n) for n in lengths]
            ax.plot(lengths,values,marker='o',label=variant,color=color)
        passed=all(m['correct'] for m in summary['measurements'])
        ax.set(title=f'{dtype}: '+('PASS' if passed else 'FAIL')+' predefined logits tolerance',xlabel='Input tokens',ylabel='32-step decode time (ms)',xticks=lengths)
        ax.legend();ax.grid(alpha=.2)
        points=summary['measurements']
        axes[2].plot([p['length'] for p in points],[p['max_abs_error'] for p in points],marker='o',label=dtype)
    axes[2].set(yscale='log',title='Max absolute logits error',xlabel='Input tokens',ylabel='Absolute error (log scale)',xticks=[128,512]);axes[2].legend();axes[2].grid(alpha=.2)
    fig.suptitle('Same-model cache on/off within each precision; cross-precision timings are not a speedup claim')
    out=Path('results/analysis');out.mkdir(exist_ok=True)
    fig.savefig(out/'kv-diagnostics.png',dpi=160);fig.savefig(out/'kv-diagnostics.svg')
if __name__=='__main__':main()
