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
        'engine': '#!/usr/bin/env bash\nprintf "%s" "$KV_DTYPE" > "$DTYPE_RECORD"\n'
    }.items():
        path = binary/name
        path.write_text(script)
        path.chmod(0o755)
    (tmp_path/'.runtime/runtime.json').write_text(json.dumps({'engine_python': str(binary/'engine')}))
    env = {**os.environ, 'PATH': str(binary)+':'+os.environ['PATH'],
           'DTYPE_RECORD': str(tmp_path/'dtype')}
    env.pop('KV_DTYPE', None)
    for explicit, expected in [(None, 'float32'), ('bfloat16', 'bfloat16')]:
        if explicit:
            env['KV_DTYPE'] = explicit
        subprocess.run(['bash', 'experiments/run.sh', '--case', 'kv', '--out', 'results/check'],
                       cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
        assert (tmp_path/'dtype').read_text() == expected
