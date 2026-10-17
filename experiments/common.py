import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

class Evidence:
    def __init__(self, case, parameters):
        parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
        self.out = Path(parser.parse_args().out)
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out/'manifest.json').exists():
            raise RuntimeError('Output already contains a run; choose a new directory')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required; CPU/mock results cannot stand in for GPU')
        self.rows = []
        self.manifest = {'case': case, 'time': datetime.now(timezone.utc).isoformat(),
            'commit': os.environ.get('SOURCE_COMMIT'), 'model': os.environ['MODEL_ID'],
            'revision': os.environ['MODEL_REVISION'], 'parameters': parameters,
            'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            'software': {n: importlib.metadata.version(n) for n in ('torch','transformers','sglang')},
            'cuda': torch.version.cuda, 'complete': False}
        write(self.out/'manifest.json', self.manifest)

    def row(self, **values):
        row = {'ok': True, 'error': None, **values}
        self.rows.append(row)
        with (self.out/'raw.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def finish(self, summary):
        from .report import plot
        write(self.out/'summary.json', summary)
        self.manifest['complete'] = True
        self.manifest['correctness_pass'] = all(r['ok'] for r in self.rows)
        write(self.out/'manifest.json', self.manifest)
        plot(self.out)
        if not self.manifest['correctness_pass']:
            raise RuntimeError('Experiment completed with failed checks; see raw data')

def entry(function):
    try:
        function()
    except Exception as exc:
        # Only experiment diagnostics, never environment values or credentials.
        args = sys.argv
        if '--out' in args:
            out = Path(args[args.index('--out')+1]); out.mkdir(parents=True,exist_ok=True)
            write(out/'failure.json', {'ok': False, 'type': type(exc).__name__, 'error': str(exc)[:1000]})
        traceback.print_exc()
        raise SystemExit(1)
