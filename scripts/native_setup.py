"""Rebuild and supervise a native service in an otherwise clean GPU sandbox.

An optional SGLANG_PYTHON selects a platform-published, version-checked runtime;
otherwise install the declared stack in a private venv. No global pip writes.
"""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.runtime'
EXPECTED = {'sglang': '0.5.9', 'torch': '2.9.1', 'transformers': '4.57.1',
            'sgl-kernel': '0.3.21', 'flashinfer-python': '0.6.3'}


def process_identity(pid):
    """Linux start ticks prevent signalling an unrelated reused PID."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, IndexError, ProcessLookupError):
        return None


def stop_owned():
    path = STATE / 'process.json'
    if not path.exists():
        return
    old = json.loads(path.read_text())
    pid = old['pid']
    if process_identity(pid) == old['start_ticks']:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 40
        while process_identity(pid) == old['start_ticks'] and time.monotonic() < deadline:
            time.sleep(.2)
        if process_identity(pid) == old['start_ticks']:
            raise RuntimeError('Supervisor did not exit; inspect logs before retrying')
    path.unlink(missing_ok=True)


def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def ubuntu_sources(apt_root=Path('/etc/apt')):
    """Use only existing Ubuntu source entries, without changing system sources."""
    def official(uri):
        parsed = urlsplit(uri)
        return parsed.scheme in ('http', 'https') and (parsed.hostname or '').endswith('.ubuntu.com')
    modern = apt_root / 'sources.list.d/ubuntu.sources'
    legacy = apt_root / 'sources.list'
    if modern.is_file():
        selected = []
        for block in modern.read_text().split('\n\n'):
            uris = [u for line in block.splitlines() if line.startswith('URIs:')
                    for u in line.split(':', 1)[1].split()]
            if uris and all(official(uri) for uri in uris):
                selected.append(block)
        output = STATE / 'apt-ubuntu.sources'
        content = '\n\n'.join(selected)
    elif legacy.is_file():
        selected = []
        for line in legacy.read_text().splitlines():
            if not line.lstrip().startswith(('deb ', 'deb-src ')):
                continue
            uris = [word for word in line.split() if word.startswith(('http://', 'https://'))]
            if len(uris) == 1 and official(uris[0]):
                selected.append(line)
        output = STATE / 'apt-ubuntu.list'
        content = '\n'.join(selected)
    else:
        raise RuntimeError('No existing Ubuntu apt sources found')
    if not selected:
        raise RuntimeError('No official Ubuntu entries in existing apt sources')
    output.write_text(content + '\n')
    return output


def ensure_build_prerequisites(python):
    """Triton/FlashInfer compile CPython extensions even with cached wheels."""
    metadata = json.loads(subprocess.check_output([python, '-c',
        'import json,sys,sysconfig;print(json.dumps({'
        '"version":"%s.%s"%sys.version_info[:2],'
        '"include":sysconfig.get_path("include")}))'], text=True))
    header = Path(metadata['include']) / 'Python.h'
    if header.is_file() and shutil.which('cc') and shutil.which('c++'):
        return
    if not shutil.which('apt-get'):
        raise RuntimeError('Missing Python.h or compiler; this installer requires Ubuntu apt-get')
    privilege = [] if os.geteuid() == 0 else ['sudo', '-n']
    if privilege and not shutil.which('sudo'):
        raise RuntimeError('Missing build dependencies require passwordless sudo')
    print('Installing Python development headers and native build tools for runtime JIT', flush=True)
    source = ubuntu_sources()
    apt = [*privilege, 'env', 'DEBIAN_FRONTEND=noninteractive', 'apt-get',
           '-o', 'Dir::Etc::sourcelist=' + str(source), '-o', 'Dir::Etc::sourceparts=-',
           '-o', 'Acquire::http::Timeout=20', '-o', 'Acquire::https::Timeout=20',
           '-o', 'Acquire::Retries=1', '-o', 'Dpkg::Use-Pty=0']
    run([*apt, 'update'], timeout=180)
    run([*apt, 'install', '-y', '--no-install-recommends',
         'build-essential', 'python' + metadata['version'] + '-dev'], timeout=240)
    if not header.is_file() or not shutil.which('cc') or not shutil.which('c++'):
        raise RuntimeError('Python headers/compiler still unavailable for selected runtime: ' + str(header))


def configure_runtime_environment(engine_python, gateway_python):
    """Calling venv Python does not activate its executable search path."""
    metadata = json.loads(subprocess.check_output([engine_python, '-c',
        'import json,sys;print(json.dumps({"executable":sys.executable}))'], text=True))
    # Do not resolve the Python symlink: /venv/bin/python may target /usr/bin/python.
    engine_bin = str(Path(metadata['executable']).absolute().parent)
    gateway_bin = str(Path(gateway_python).absolute().parent)
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home and Path('/usr/local/cuda/bin/nvcc').is_file():
        cuda_home = '/usr/local/cuda'
    if not cuda_home:
        nvcc = shutil.which('nvcc')
        if nvcc:
            cuda_home = str(Path(nvcc).absolute().parent.parent)
    if cuda_home and not (Path(cuda_home) / 'bin/nvcc').is_file():
        raise RuntimeError('CUDA_HOME does not contain the nvcc compiler: ' + cuda_home)
    entries = [engine_bin, gateway_bin]
    if cuda_home:
        os.environ['CUDA_HOME'] = cuda_home
        entries.append(str(Path(cuda_home) / 'bin'))
    entries.extend(os.environ.get('PATH', '/usr/bin:/bin').split(os.pathsep))
    os.environ['PATH'] = os.pathsep.join(dict.fromkeys(entries))
    return {'engine_bin': engine_bin, 'gateway_bin': gateway_bin, 'cuda_home': cuda_home}


def validate_jit_tools():
    found = {name: shutil.which(name) for name in ('cc', 'c++', 'ninja', 'nvcc')}
    missing = [name for name, path in found.items() if not path]
    if missing:
        raise RuntimeError('Missing native JIT tools after runtime setup: ' + ', '.join(missing))
    print('Native JIT tools: ' + json.dumps(found), flush=True)
    return found


def runtime_versions(python):
    code = ('import importlib.metadata as m,json;print(json.dumps({' +
            ','.join(repr(n)+':m.version('+repr(n)+')' for n in EXPECTED) + '}))')
    return json.loads(subprocess.check_output([python, '-c', code], text=True))


def validate_versions(actual):
    bad = [f'{n}: expected {v}, found {actual.get(n)}' for n, v in EXPECTED.items()
           if actual.get(n, '').split('+')[0] != v]
    if bad:
        raise RuntimeError('Runtime version mismatch: ' + '; '.join(bad))


def healthy(host, port):
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/healthz', timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def forward_logs(logs, *, max_bytes=65536):
    """Forward only new complete lines; redact secrets even across read boundaries."""
    secret = os.environ.get('API_KEY', '').encode()
    for path, state in logs.items():
        if not path.exists():
            continue
        with path.open('rb') as source:
            source.seek(state['offset'])
            new = source.read(max_bytes)
            state['offset'] = source.tell()
        data = state['pending'] + new
        lines = data.split(b'\n')
        state['pending'] = lines.pop()
        for line in lines:
            if secret:
                line = line.replace(secret, b'[REDACTED]')
            print('[' + path.stem + '] ' + line.decode('utf-8', errors='replace'), flush=True)


def main():
    os.chdir(ROOT)
    STATE.mkdir(mode=0o700, exist_ok=True)
    STATE.chmod(0o700)
    with (STATE / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if sys.argv[1:] == ['--stop']:
            stop_owned()
            return
        if sys.argv[1:]:
            raise ValueError('Only optional argument is --stop')
        for name in ('API_KEY', 'MODEL_ID', 'MODEL_REVISION', 'BENCH_PROFILE'):
            if not os.environ.get(name):
                raise ValueError(name + ' required')
        if os.environ['BENCH_PROFILE'] not in ('baseline', 'optimized'):
            raise ValueError('Invalid BENCH_PROFILE')
        host = os.environ.setdefault('HOST', '127.0.0.1')
        if host not in ('127.0.0.1', '0.0.0.0'):
            raise ValueError('HOST must be 127.0.0.1 or 0.0.0.0')
        port = int(os.environ.setdefault('PORT', '8000'))
        if not 1024 <= port <= 65535 or port == 30000:
            raise ValueError('PORT must be 1024..65535 except engine port 30000')
        os.environ.setdefault('HF_HOME', str(Path.home() / '.cache/huggingface'))
        os.environ['SOURCE_COMMIT'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        os.environ.update(RUNTIME_DIR=str(STATE), PYTHONUNBUFFERED='1', HF_HUB_DISABLE_XET='1')
        fields = ['SOURCE_COMMIT', 'HOST', 'PORT', 'API_KEY', 'MODEL_ID', 'MODEL_REVISION',
                  'BENCH_PROFILE', 'HF_HOME', 'SGLANG_PYTHON', 'MAX_RUNNING_REQUESTS',
                  'GATEWAY_SLOTS', 'GATEWAY_QUEUE', 'ATTENTION_BACKEND', 'PREFIX_CACHE']
        digest = hashlib.sha256()
        for name in fields:
            digest.update((name + '=' + os.environ.get(name, '') + '\n').encode())
        for path in sorted((ROOT / 'configs').glob('*.yaml')):
            digest.update(path.read_bytes())
        for path in (ROOT / 'requirements.lock', ROOT / 'requirements-runtime.lock'):
            digest.update(path.read_bytes())
        fingerprint = digest.hexdigest()
        record = STATE / 'process.json'
        if record.exists():
            old = json.loads(record.read_text())
            if (old['fingerprint'] == fingerprint and process_identity(old['pid']) == old['start_ticks']
                    and healthy(host.replace('0.0.0.0', '127.0.0.1'), port)):
                print('Same configuration already ready (idempotent)', flush=True)
                return
        stop_owned()
        run(['nvidia-smi', '-L'])
        gateway_python = ROOT / '.venv' / 'bin' / 'python'
        if not gateway_python.exists():
            run([sys.executable, '-m', 'venv', ROOT / '.venv'])
        run([gateway_python, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements.lock'])
        engine_python = os.environ.get('SGLANG_PYTHON')
        if not engine_python:
            engine_python = str(gateway_python)
            run([engine_python, '-m', 'pip', 'install', '--disable-pip-version-check',
                 '-r', 'requirements-runtime.lock', '-r', 'requirements-analysis.lock'])
        runtime_environment = configure_runtime_environment(engine_python, gateway_python)
        ensure_build_prerequisites(engine_python)
        jit_tools = validate_jit_tools()
        versions = runtime_versions(engine_python)
        validate_versions(versions)
        # Validate real CUDA execution, not just a driver visible to nvidia-smi.
        run([engine_python, '-c', 'import torch;assert torch.cuda.is_available();'
             'x=torch.ones(16,device="cuda");assert x.sum().item()==16;print("CUDA allocation ready")'])
        os.environ['SGLANG_PYTHON'] = engine_python
        (STATE / 'runtime.json').write_text(json.dumps({'engine_python': engine_python,
            'gateway_python': str(gateway_python), 'versions': versions, 'mode': 'native',
            **runtime_environment, 'jit_tools': jit_tools}, indent=2))
        logs = {STATE/name: {'offset': (STATE/name).stat().st_size if (STATE/name).exists() else 0,
                             'pending': b''}
                for name in ('supervisor.log', 'engine.log', 'gateway.log')}
        with (STATE / 'supervisor.log').open('a') as log:
            child = subprocess.Popen([str(gateway_python), '-m', 'server.supervisor'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
        record.write_text(json.dumps({'pid': child.pid, 'start_ticks': process_identity(child.pid),
                                    'fingerprint': fingerprint}))
        deadline = time.monotonic() + int(os.environ.get('STARTUP_TIMEOUT', '420'))
        while time.monotonic() < deadline and child.poll() is None:
            forward_logs(logs)
            if healthy(host.replace('0.0.0.0', '127.0.0.1'), port):
                forward_logs(logs)
                print('Native SGLang service ready: profile=' + os.environ['BENCH_PROFILE'], flush=True)
                return
            time.sleep(1)
        # Stop processes even on failed startup, and preserve diagnostic files.
        stop_owned()
        forward_logs(logs)
        raise RuntimeError('Service did not become ready; inspect .runtime/{engine,gateway,supervisor}.log')

if __name__ == '__main__':
    main()
