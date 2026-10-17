"""Runtime boundary checks; these tests do not install packages or allocate GPUs."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import pytest
from server.config import configuration
from server.supervisor import engine_command

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_setup', ROOT/'scripts/native_setup.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_optimized_engine_uses_actual_cache_and_attention_configuration(monkeypatch):
    monkeypatch.setenv('BENCH_PROFILE', 'optimized')
    monkeypatch.setenv('MODEL_ID', 'model-fixture')
    monkeypatch.setenv('MODEL_REVISION', 'revision-fixture')
    monkeypatch.setenv('SGLANG_PYTHON', '/opt/runtime/bin/python')
    _, cfg = configuration()
    command = engine_command(cfg)
    assert command[0] == '/opt/runtime/bin/python'
    assert command[command.index('--attention-backend')+1] == 'flashinfer'
    assert '--disable-radix-cache' not in command
    assert '--disable-cuda-graph' not in command
    assert command[command.index('--max-running-requests')+1] == '32'
    assert '--disable-radix-cache' in engine_command(cfg, cache=False)


def test_controlled_ablation_and_invalid_overrides(monkeypatch):
    monkeypatch.setenv('BENCH_PROFILE', 'optimized')
    monkeypatch.setenv('PREFIX_CACHE', '0')
    monkeypatch.setenv('MAX_RUNNING_REQUESTS', '8')
    monkeypatch.setenv('ATTENTION_BACKEND', 'triton')
    _, cfg = configuration()
    assert cfg['prefix_cache'] is False and cfg['max_running_requests'] == 8
    assert cfg['attention_backend'] == 'triton'
    monkeypatch.setenv('MAX_RUNNING_REQUESTS', '0')
    with pytest.raises(ValueError):
        configuration()


def test_runtime_pin_rejects_drift_but_accepts_cuda_build_suffix():
    actual = {**native.EXPECTED, 'torch': '2.9.1+cu128'}
    native.validate_versions(actual)
    with pytest.raises(RuntimeError, match='Runtime version mismatch'):
        native.validate_versions({**actual, 'sglang': '0.5.2'})


def test_stop_does_not_signal_recycled_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'STATE', tmp_path)
    path = tmp_path/'process.json'
    path.write_text(json.dumps({'pid': 123, 'start_ticks': 'old'}))
    monkeypatch.setattr(native, 'process_identity', lambda _: 'new')
    def forbidden(*_):
        raise AssertionError('must not signal reused PID')
    monkeypatch.setattr(native.os, 'kill', forbidden)
    native.stop_owned()
    assert not path.exists()


def test_stop_terminates_only_owned_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'STATE', tmp_path)
    (tmp_path/'process.json').write_text(json.dumps({'pid': 123, 'start_ticks': 'same'}))
    identities = iter(['same', None, None])
    monkeypatch.setattr(native, 'process_identity', lambda _: next(identities))
    calls = []
    monkeypatch.setattr(native.os, 'kill', lambda *args: calls.append(args))
    native.stop_owned()
    assert calls == [(123, signal.SIGTERM)]
