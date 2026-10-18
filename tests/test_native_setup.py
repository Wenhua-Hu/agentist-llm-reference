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


def test_startup_log_forwarding_redacts_split_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('API_KEY', 'secret-fixture-value')
    path = tmp_path/'engine.log'
    path.write_text('ready secret-fixture-value now\n')
    state = {path: {'offset': 0, 'pending': b''}}
    native.forward_logs(state, max_bytes=12)
    assert capsys.readouterr().out == ''
    native.forward_logs(state)
    output = capsys.readouterr().out
    assert 'secret' not in output and '[REDACTED]' in output
    native.forward_logs(state)
    assert capsys.readouterr().out == ''


def test_missing_runtime_headers_are_installed_for_selected_python(tmp_path, monkeypatch):
    include = tmp_path/'include'
    include.mkdir()
    monkeypatch.setattr(native.subprocess, 'check_output', lambda *a, **kw:
                        json.dumps({'version': '3.12', 'include': str(include)}))
    monkeypatch.setattr(native.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(native.shutil, 'which', lambda name: '/usr/bin/' + name)
    calls = []
    def install(args, **kwargs):
        calls.append(args)
        assert kwargs['timeout'] in (180, 240)
        if 'install' in args:
            (include/'Python.h').write_text('/* test header */')
    monkeypatch.setattr(native, 'ubuntu_sources', lambda: tmp_path/'ubuntu.sources')
    monkeypatch.setattr(native, 'run', install)
    native.ensure_build_prerequisites('/project/.venv/bin/python')
    assert len(calls) == 2
    assert calls[0][:4] == ['sudo', '-n', 'env', 'DEBIAN_FRONTEND=noninteractive']
    assert calls[1][-2:] == ['build-essential', 'python3.12-dev']
    assert 'Dir::Etc::sourceparts=-' in calls[0]
    assert 'Acquire::http::Timeout=20' in calls[0]
    assert 'Acquire::https::Timeout=20' in calls[0]
    assert 'Acquire::Retries=1' in calls[0]
    assert '-qq' not in calls[0]
    native.ensure_build_prerequisites('/project/.venv/bin/python')
    assert len(calls) == 2  # Repeated setup does not apt-install again.


def test_missing_headers_after_package_install_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(native.subprocess, 'check_output', lambda *a, **kw:
                        json.dumps({'version': '3.12', 'include': str(tmp_path)}))
    monkeypatch.setattr(native.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(native.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(native, 'ubuntu_sources', lambda: tmp_path/'ubuntu.sources')
    monkeypatch.setattr(native, 'run', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='still unavailable'):
        native.ensure_build_prerequisites('/unsupported/python')


def test_ubuntu_sources_exclude_cuda_and_preserve_signed_source(tmp_path, monkeypatch):
    apt = tmp_path/'apt'
    (apt/'sources.list.d').mkdir(parents=True)
    state = tmp_path/'state'
    state.mkdir()
    monkeypatch.setattr(native, 'STATE', state)
    ubuntu = 'Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\nSuites: noble noble-updates\nComponents: main universe\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg'
    cuda = 'Types: deb\nURIs: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/\nSuites: /\nComponents: main'
    source = apt/'sources.list.d/ubuntu.sources'
    source.write_text(ubuntu+'\n\n'+cuda+'\n')
    (apt/'sources.list.d/cuda.sources').write_text(cuda)
    selected = native.ubuntu_sources(apt).read_text()
    assert 'archive.ubuntu.com' in selected and 'Signed-By:' in selected
    assert 'nvidia' not in selected
    assert source.read_text() == ubuntu+'\n\n'+cuda+'\n'  # System files untouched.


def test_legacy_ubuntu_sources_filter_third_party(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'STATE', tmp_path)
    (tmp_path/'sources.list').write_text('deb http://us.archive.ubuntu.com/ubuntu focal main\n'
        'deb https://example.com/ubuntu focal main\n# deb http://security.ubuntu.com/ubuntu focal main\n')
    source = native.ubuntu_sources(tmp_path)
    assert source.suffix == '.list'
    assert source.read_text() == 'deb http://us.archive.ubuntu.com/ubuntu focal main\n'


def test_runtime_tools_resolve_with_clean_path_without_activation(tmp_path, monkeypatch):
    engine_bin = tmp_path/'engine/bin'
    gateway_bin = tmp_path/'gateway/bin'
    cuda = tmp_path/'cuda'
    for folder in (engine_bin, gateway_bin, cuda/'bin'):
        folder.mkdir(parents=True)
    for tool, folder in [('ninja', engine_bin), ('cc', gateway_bin), ('c++', gateway_bin), ('nvcc', cuda/'bin')]:
        path = folder/tool
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    # Simulate venv Python being a symlink to a system interpreter.
    (engine_bin/'python').symlink_to('/usr/bin/python3')
    monkeypatch.setenv('PATH', '/a-clean-path-with-no-venv')
    monkeypatch.setenv('CUDA_HOME', str(cuda))
    monkeypatch.setattr(native.subprocess, 'check_output', lambda *a, **kw:
                        json.dumps({'executable': str(engine_bin/'python')}))
    env = native.configure_runtime_environment(str(engine_bin/'python'), str(gateway_bin/'python'))
    assert env['engine_bin'] == str(engine_bin)
    assert os.environ['PATH'].split(os.pathsep)[:3] == [str(engine_bin), str(gateway_bin), str(cuda/'bin')]
    tools = native.validate_jit_tools()
    assert tools['ninja'] == str(engine_bin/'ninja')
    assert tools['nvcc'] == str(cuda/'bin/nvcc')


def test_missing_jit_tool_fails_before_model_start(monkeypatch):
    monkeypatch.setattr(native.shutil, 'which', lambda name: None if name == 'ninja' else '/tools/'+name)
    with pytest.raises(RuntimeError, match='ninja'):
        native.validate_jit_tools()


def test_analysis_and_complete_lock_changes_invalidate_dependency_fingerprint(tmp_path):
    for name in native.DEPENDENCY_FILES:
        (tmp_path/name).write_text('fixture==1\n')
    first = native.dependency_fingerprint(tmp_path)
    (tmp_path/'requirements-analysis.lock').write_text('fixture==2\n')
    second = native.dependency_fingerprint(tmp_path)
    assert second != first
    (tmp_path/'requirements-native-linux-py312.lock').write_text('fixture==3\n')
    assert native.dependency_fingerprint(tmp_path) != second


def test_native_closure_rejects_pin_drift_missing_and_extra_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'STATE', tmp_path)
    lock = tmp_path/'lock'
    lock.write_text('idna==3.20\nninja==1.13.2\n')
    current = {'idna': '3.20', 'ninja': '1.13.2'}
    monkeypatch.setattr(native.subprocess, 'check_output', lambda *a, **kw: json.dumps(current))
    verified = native.verify_native_packages('/python', lock)
    assert verified['package_count'] == 2
    assert 'lock_sha256' in json.loads((tmp_path/'native-dependencies.json').read_text())
    for invalid in ({'idna': '3.10', 'ninja': '1.13.2'}, {'idna': '3.20'},
                    {'idna': '3.20', 'ninja': '1.13.2', 'unexpected': '1'}):
        current = invalid
        with pytest.raises(RuntimeError, match='differs from lock'):
            native.verify_native_packages('/python', lock)


def test_direct_constraints_match_captured_native_closure():
    def pins(name):
        return {native.re.sub(r'[-_.]+', '-', line.split('==')[0]).lower(): line.split('==')[1]
                for line in (ROOT/name).read_text().splitlines() if line and not line.startswith('#')}
    full = pins('requirements-native-linux-py312.lock')
    assert len(full) == 175
    for name in native.DEPENDENCY_FILES[:-1]:
        for package, version in pins(name).items():
            assert full[package] == version, (name, package)


@pytest.mark.parametrize('profile', ['baseline', 'optimized'])
def test_both_profiles_disable_hidden_model_sampling_defaults(monkeypatch, profile):
    monkeypatch.setenv('BENCH_PROFILE', profile)
    monkeypatch.setenv('MODEL_ID', 'model-fixture')
    monkeypatch.setenv('MODEL_REVISION', 'revision-fixture')
    _, cfg = configuration()
    assert cfg['sampling_defaults'] == 'openai'
    assert cfg['repetition_penalty'] == cfg['top_p'] == 1.0
    assert cfg['top_k'] == -1 and cfg['request_temperature_default'] == 0.0
    args = engine_command(cfg)
    assert args[args.index('--sampling-defaults')+1] == 'openai'


def test_contract_setup_installs_only_gateway_and_never_starts_or_checks_gpu(tmp_path, monkeypatch):
    state=tmp_path/'runtime'
    monkeypatch.setattr(native,'STATE',state)
    monkeypatch.setattr(native.sys,'argv',['native_setup.py','--contract-test'])
    for key,value in {'API_KEY':'ordinary-test-key','MODEL_ID':'Qwen/Qwen2.5-7B-Instruct',
                      'MODEL_REVISION':'fixed-test-revision','BENCH_PROFILE':'optimized',
                      'ENGINE_BASE_URL':'http://127.0.0.1:30001'}.items():monkeypatch.setenv(key,value)
    calls=[]
    monkeypatch.setattr(native,'run',lambda args,**kwargs:calls.append(list(map(str,args))))
    monkeypatch.setattr(native.subprocess,'check_output',lambda *args,**kwargs:'fixed-test-commit\n')
    monkeypatch.setattr(native,'process_identity',lambda pid:'known-start')
    monkeypatch.setattr(native,'healthy',lambda *args:True)
    class Child:
        pid=12345
        def poll(self):return None
    monkeypatch.setattr(native.subprocess,'Popen',lambda args,**kwargs:(calls.append(list(map(str,args))) or Child()))
    native.main()
    assert os.environ['CONTRACT_TEST']=='1'
    assert json.loads((state/'runtime.json').read_text())['mode']=='contract-test'
    assert not any('nvidia-smi' in call or 'sglang.launch_server' in call for call in calls)
    installs=[call for call in calls if 'install' in call]
    assert len(installs)==1 and installs[0][-2:]==['-r','requirements.lock']
    assert 'requirements-runtime.lock' not in str(installs)
    assert any('server.supervisor' in call for call in calls)


def test_startup_logs_redact_all_three_injected_secrets(tmp_path,monkeypatch,capsys):
    for name in ('API_KEY','ADMIN_API_KEY','KEY_HASH_SECRET'):monkeypatch.setenv(name,'private-'+name+'-test')
    path=tmp_path/'gateway.log'
    path.write_text(' '.join(os.environ[n] for n in ('API_KEY','ADMIN_API_KEY','KEY_HASH_SECRET'))+'\n')
    native.forward_logs({path:{'offset':0,'pending':b''}})
    output=capsys.readouterr().out
    assert 'private-' not in output and output.count('[REDACTED]')==3
