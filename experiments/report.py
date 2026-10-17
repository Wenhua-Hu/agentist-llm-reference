"""Dependency-free SVG generated solely from raw measurements."""
import html
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

def plot(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory/'raw.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        value = row.get('elapsed_ms', row.get('ttft_ms'))
        if value is not None:
            groups[f"{row.get('variant')} / {row.get('length',row.get('group',''))} / {row.get('cache_state','')}"] .append(value)
    width = 1000; height = 90 + len(groups)*36
    maximum = max([mean(v) for v in groups.values()] or [1]) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fafafa"/>',
             '<text x="24" y="30" font-family="sans-serif" font-size="20">Measured mean latency (ms); all recorded runs</text>']
    for i,(label,values) in enumerate(groups.items()):
        y = 70+i*36; value=mean(values)
        parts += [f'<text x="20" y="{y}" font-family="monospace" font-size="13">{html.escape(label)}</text>',
                  f'<rect x="360" y="{y-16}" width="{value/maximum*490:.1f}" height="22" fill="#496ad5"/>',
                  f'<text x="{370+value/maximum*490:.1f}" y="{y}" font-family="monospace" font-size="13">{value:.3f}</text>']
    parts.append('</svg>'); (directory/'comparison.svg').write_text('\n'.join(parts))

if __name__ == '__main__':
    import sys
    plot(sys.argv[1])
