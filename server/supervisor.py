"""Own one engine and gateway process group, on a native sandbox or in Docker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .config import configuration

def engine_command(config, cache=None):
    args = [os.environ.get('SGLANG_PYTHON', sys.executable), '-m', 'sglang.launch_server',
        '--model-path', os.environ['MODEL_ID'], '--revision', os.environ['MODEL_REVISION'],
        '--served-model-name', os.environ['MODEL_ID'], '--host', '127.0.0.1', '--port', '30000',
        '--dtype', config['dtype'], '--context-length', str(config['context_length']),
        '--mem-fraction-static', str(config['mem_fraction_static']),
        '--max-running-requests', str(config['max_running_requests']),
        '--max-total-tokens', str(config['max_total_tokens']),
        '--max-prefill-tokens', str(config['max_prefill_tokens']),
        '--attention-backend', config['attention_backend'],
        '--sampling-defaults', config.get('sampling_defaults', 'openai'), '--enable-metrics',
        '--stream-interval', '1']
    if not config.get('cuda_graph', False):
        args += ['--disable-cuda-graph']
    else:
        args += ['--cuda-graph-max-bs', str(config['max_running_requests'])]
    if not (config['prefix_cache'] if cache is None else cache):
        args += ['--disable-radix-cache']
    return args

def main():
    _, cfg = configuration()
    logs = Path(os.environ.get('RUNTIME_DIR', '/runtime')); logs.mkdir(exist_ok=True)
    command = engine_command(cfg)
    (logs/'engine-command.json').write_text(json.dumps(command, indent=2))
    engine_log = (logs/'engine.log').open('a')
    gateway_log = (logs/'gateway.log').open('a')
    children = []
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        children.append(subprocess.Popen(command, stdout=engine_log, stderr=engine_log, start_new_session=True))
        children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'server.app:factory', '--factory',
            '--host', os.environ['HOST'], '--port', os.environ['PORT'], '--workers', '1', '--no-access-log'],
            stdout=gateway_log, stderr=gateway_log, start_new_session=True))
        while not stop:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('managed child exited; inspect .runtime logs')
            time.sleep(.5)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
if __name__ == '__main__':
    main()
