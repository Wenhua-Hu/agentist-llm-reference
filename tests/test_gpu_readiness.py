"""CPU-only behavior tests for the bounded GPU readiness gate; never launches Docker."""
import os
from pathlib import Path
import subprocess

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/check-gpu.sh'
def run_gate(tmp_path,failures):
    commands={
        'sudo':'#!/bin/bash\nshift\nexec "$@"\n',
        'timeout':'#!/bin/bash\nshift 3\nexec "$@"\n',
        'sleep':'#!/bin/bash\nexit 0\n',
        'docker':'''#!/bin/bash
if [[ "$1" == rm ]]; then exit 0; fi
n=0; [[ ! -f "$PROBE_COUNT" ]] || read -r n < "$PROBE_COUNT"
n=$((n+1)); echo "$n" > "$PROBE_COUNT"
[[ "$n" -gt "$PROBE_FAILURES" ]]
'''}
    for name,text in commands.items():
        p=tmp_path/name;p.write_text(text);p.chmod(0o755)
    env={**os.environ,'PATH':str(tmp_path)+':'+os.environ['PATH'],'PROBE_COUNT':str(tmp_path/'count'),'PROBE_FAILURES':str(failures)}
    result=subprocess.run(['bash',str(SCRIPT),'test-image'],env=env,capture_output=True,text=True,timeout=5)
    return result,int((tmp_path/'count').read_text())
def test_transient_cuda_unavailability_retries_before_model_start(tmp_path):
    r,count=run_gate(tmp_path,2)
    assert r.returncode==0 and count==3
    assert 'NVML and CUDA allocation ready' in r.stdout

def test_persistent_cuda_failure_stops_at_six_attempts(tmp_path):
    r,count=run_gate(tmp_path,99)
    assert r.returncode==1 and count==6
    assert 'refusing to start' in r.stdout
