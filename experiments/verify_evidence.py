"""CPU-only integrity checks for published evidence, not a new GPU evaluation."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root, formal_name="formal"):
    formal=root/formal_name
    report=json.loads((formal/'report.json').read_text())
    workload=json.loads((formal/'workload.json').read_text())
    assert hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==report['workload_sha256']
    expected={item['id']:item['quality']['json_fields'] for item in workload}
    profiles={p:read_rows(formal/(p+'-raw.jsonl')) for p in ('baseline','optimized')}
    assert sum(map(len,profiles.values()))==576
    for rows in profiles.values():
        for row in rows:
            assert hashlib.sha256(row['text'].encode()).hexdigest()==row['output_sha256']
            try:
                output=json.loads(row['text'])
                correct=isinstance(output,dict) and all(output.get(k)==v for k,v in expected[row['workload_id']].items())
            except ValueError:
                correct=False
            assert correct==row['quality_pass']
    for group in report['performance']:
        rows=[r for r in profiles[group['profile']] if '-c'+str(group['concurrency'])+'-' in r['id']]
        assert len(rows)==group['requests']==96
        assert sum(r['ok'] for r in rows)==group['successful']
        assert sum(r['quality_pass'] is True for r in rows)==group['quality_pass']
        for raw_key,key in [('ttft_proxy_ms','ttft_p95_ms'),('e2e_ms','e2e_p95_ms'),('tpot_proxy_ms','tpot_p95_ms')]:
            values=sorted(r[raw_key] for r in rows if r['ok'] and r[raw_key] is not None)
            assert math.isclose(values[math.ceil(len(values)*.95)-1],group[key],rel_tol=1e-12)
    if formal_name == 'rubric3':
        assert report['rubric']=='llm-lab/3.0'
        assert report['automatic_max']==95 and report['manual_max']==5
        expected_caps={'reproduce':10,'api':10,'quality':10,'performance':15,'agent':5,'kv':15,'prefix':15,'attention':15}
        assert len(report['checks'])==8
        assert {c['id']:c['max'] for c in report['checks']}==expected_caps
        assert sum(c['points'] for c in report['checks'])==report['automatic_score']
        for case,count in [('kv',12),('prefix',96),('attention',18)]:
            check=next(c for c in report['checks'] if c['id']==case)
            assert sum(c['points'] for c in check['subchecks'])==check['points']
            assert sum(c['max'] for c in check['subchecks'])==15
            directory=formal/case
            manifest=json.loads((directory/'manifest.json').read_text())
            assert manifest['commit']==report['commit_sha'] and manifest['complete']
            assert len(read_rows(directory/'raw.jsonl'))==count
    for trace in root.glob('attention/rehearsal/*.trace.json.gz'):
        checksum=json.loads(trace.with_name(trace.name.replace('.trace.json.gz','-trace-sha256.json')).read_text())['sha256']
        assert hashlib.sha256(gzip.decompress(trace.read_bytes())).hexdigest()==checksum
    for case,count in [('kv',12),('attention',18),('prefix',96)]:
        directory=root/case/'rehearsal';rows=read_rows(directory/'raw.jsonl')
        manifest=json.loads((directory/'manifest.json').read_text())
        assert len(rows)==count and manifest['complete']
        assert manifest['correctness_pass']==all(r['ok'] for r in rows)
        if not manifest['correctness_pass']:
            assert (directory/'failure.json').exists()
    diagnostics=root/'kv-diagnostics'
    if diagnostics.exists():
        controls={}
        for precision in ('bf16','fp32'):
            directory=diagnostics/precision;rows=read_rows(directory/'raw.jsonl')
            manifest=json.loads((directory/'manifest.json').read_text())
            assert len(rows)==12 and manifest['complete']
            assert manifest['correctness_pass']==all(r['ok'] for r in rows)
            if not manifest['correctness_pass']: assert (directory/'failure.json').exists()
            controls[precision]={n:{(r['input_hash'],r['continuation_hash']) for r in rows if r['length']==n} for n in (128,512)}
            assert all(len(v)==1 for v in controls[precision].values())
        assert controls['bf16']==controls['fp32']
    print('Evidence integrity verified: hashes, 576 responses, quality, p95, profiler traces, and retained failures.')
    print('Throughput windows are not independently reconstructed by this check; see report limitations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--formal-directory',default='formal',choices=['formal','confirmation','rubric3'])
    args=parser.parse_args();verify(args.results,args.formal_directory)
