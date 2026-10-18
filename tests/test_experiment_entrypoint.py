"""Exercise the public native experiment shell contract without GPU allocation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
from server.config import MODEL, REVISION, KV_MODEL, KV_REVISION, configuration


def test_kv_entrypoint_defaults_fp32_and_preserves_explicit_fp32(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'experiments').mkdir()
    shutil.copy(root/'experiments/run.sh', tmp_path/'experiments/run.sh')
    (tmp_path/'.runtime').mkdir()
    (tmp_path/'setup.sh').write_text('#!/usr/bin/env bash\n[[ "$1" == --stop ]]\n')
    binary = tmp_path/'bin'
    binary.mkdir()
    for name, script in {
        'git': '#!/usr/bin/env bash\necho fixture-commit\n',
        'flock': '#!/usr/bin/env bash\nexit 0\n',
        'engine': '#!/usr/bin/env bash\nprintf "%s" "$KV_DTYPE" > "$DTYPE_RECORD"\nninja > "$TOOL_RECORD"\n'
    }.items():
        path = binary/name
        path.write_text(script)
        path.chmod(0o755)
    engine_bin = tmp_path/'engine-venv/bin'
    engine_bin.mkdir(parents=True)
    ninja = engine_bin/'ninja'
    ninja.write_text('#!/usr/bin/env bash\necho engine-ninja\n')
    ninja.chmod(0o755)
    (tmp_path/'.runtime/runtime.json').write_text(json.dumps({'engine_python': str(binary/'engine'),
        'engine_bin': str(engine_bin), 'gateway_bin': str(binary)}))
    env = {**os.environ, 'PATH': str(binary)+':'+os.environ['PATH'],
           'DTYPE_RECORD': str(tmp_path/'dtype'), 'TOOL_RECORD': str(tmp_path/'tool')}
    for key in ('KV_DTYPE','MODEL_ID','MODEL_REVISION'):env.pop(key, None)
    for explicit, expected in [(None, 'float32'), ('float32', 'float32')]:
        if explicit:
            env['KV_DTYPE'] = explicit
        subprocess.run(['bash', 'experiments/run.sh', '--case', 'kv', '--out', 'results/check'],
                       cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
        assert (tmp_path/'dtype').read_text() == expected
        assert (tmp_path/'tool').read_text().strip() == 'engine-ninja'


def test_fresh_shell_reuses_literal_runtime_cache_with_explicit_override(tmp_path):
    """Run the actual shell entry point with an empty environment, without a GPU."""
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'experiments').mkdir()
    shutil.copy(root/'experiments/run.sh', tmp_path/'experiments/run.sh')
    (tmp_path/'.runtime').mkdir()
    (tmp_path/'setup.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
    binary = tmp_path/'bin'; binary.mkdir()
    for name, script in {
        'git': '#!/usr/bin/env bash\necho fixture-commit\n',
        'flock': '#!/usr/bin/env bash\nexit 0\n',
        'engine': '#!/usr/bin/env python3\nimport os,json\nfrom pathlib import Path\nPath(os.environ["CACHE_RECORD"]).write_text(json.dumps(os.environ["HF_HOME"]))\n',
    }.items():
        path = binary/name; path.write_text(script); path.chmod(0o755)
    runtime = {'engine_python': str(binary/'engine'), 'engine_bin': str(binary), 'gateway_bin': str(binary)}
    marker = tmp_path/'must-not-execute'
    recorded = str(tmp_path/'cache with spaces') + ' $(touch ' + str(marker) + ') `false`'
    base_env = {'HOME': str(tmp_path/'fresh-home'), 'PATH': str(binary)+':'+os.environ['PATH'],
                'CACHE_RECORD': str(tmp_path/'cache-record')}
    for stored, explicit, expected in [
        (recorded, None, recorded),
        (recorded, str(tmp_path/'caller override'), str(tmp_path/'caller override')),
        (None, None, str(tmp_path/'fresh-home/.cache/huggingface')),
        (None, str(tmp_path/'caller only'), str(tmp_path/'caller only')),
    ]:
        (tmp_path/'.runtime/runtime.json').write_text(json.dumps({**runtime, **({'hf_home':stored} if stored else {})}))
        env = {**base_env, **({'HF_HOME':explicit} if explicit else {})}
        subprocess.run(['bash','experiments/run.sh','--case','kv','--out','results/fresh'],
                       cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
        assert json.loads((tmp_path/'cache-record').read_text()) == expected
        assert not marker.exists()


@pytest.mark.parametrize('case,model,revision',[('kv',KV_MODEL,KV_REVISION),('prefix',MODEL,REVISION),('attention',MODEL,REVISION)])
def test_case_identity_defaults_and_mismatches_fail_before_stop(tmp_path,case,model,revision):
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'experiments').mkdir()
    shutil.copy(root/'experiments/run.sh',tmp_path/'experiments/run.sh')
    (tmp_path/'.runtime').mkdir()
    (tmp_path/'setup.sh').write_text('#!/usr/bin/env bash\ntouch stopped\n')
    binary=tmp_path/'bin';binary.mkdir()
    scripts={
        'git':'#!/usr/bin/env bash\necho fixture\n',
        'flock':'#!/usr/bin/env bash\nexit 0\n',
        'engine':'#!/usr/bin/env python3\nimport os,json\nfrom pathlib import Path\nPath("identity.json").write_text(json.dumps({k:os.environ.get(k) for k in ["MODEL_ID","MODEL_REVISION","KV_DTYPE"]}))\n',
    }
    for name,script in scripts.items():
        p=binary/name;p.write_text(script);p.chmod(0o755)
    (tmp_path/'.runtime/runtime.json').write_text(json.dumps({'engine_python':str(binary/'engine'),'engine_bin':str(binary),'gateway_bin':str(binary)}))
    env={'PATH':str(binary)+':'+os.environ['PATH'],'HOME':str(tmp_path)}
    command=['bash','experiments/run.sh','--case',case,'--out','results/identity']
    for explicit in ({},{'MODEL_ID':model,'MODEL_REVISION':revision}):
        subprocess.run(command,cwd=tmp_path,env={**env,**explicit},capture_output=True,check=True)
        actual=json.loads((tmp_path/'identity.json').read_text())
        assert actual['MODEL_ID']==model and actual['MODEL_REVISION']==revision
        if case=='kv':assert actual['KV_DTYPE']=='float32'
    (tmp_path/'stopped').unlink();(tmp_path/'identity.json').unlink()
    invalid=[{'MODEL_ID':''}, {'MODEL_REVISION':''}, {'MODEL_ID':'wrong'}, {'MODEL_REVISION':'wrong'},
             {'MODEL_ID':MODEL if case=='kv' else KV_MODEL}]
    if case=='kv':invalid.append({'KV_DTYPE':'bfloat16'})
    for values in invalid:
        result=subprocess.run(command,cwd=tmp_path,env={**env,**values},capture_output=True)
        assert result.returncode==2
        assert not (tmp_path/'stopped').exists() and not (tmp_path/'identity.json').exists()


@pytest.mark.parametrize('profile',['baseline','optimized'])
def test_7b_serving_profile_and_separate_principle_identity(monkeypatch,profile):
    monkeypatch.setenv('BENCH_PROFILE',profile)
    _,cfg=configuration()
    assert MODEL=='Qwen/Qwen2.5-7B-Instruct' and REVISION=='a09a35458c702b33eeacc393d103063234e8bc28'
    assert KV_MODEL=='Qwen/Qwen2.5-1.5B-Instruct' and KV_REVISION=='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    assert cfg['dtype']=='bfloat16' and cfg['mem_fraction_static']==.8
    assert cfg['sampling_defaults']=='openai'
