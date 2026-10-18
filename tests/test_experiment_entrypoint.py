"""Exercise the public native experiment shell contract without GPU allocation."""
import json
import os
from pathlib import Path
import shutil
import subprocess


def test_kv_entrypoint_defaults_fp32_and_preserves_explicit_bf16(tmp_path):
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
    env.pop('KV_DTYPE', None)
    for explicit, expected in [(None, 'float32'), ('bfloat16', 'bfloat16')]:
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
