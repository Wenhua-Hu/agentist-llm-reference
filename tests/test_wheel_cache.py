"""Exercise cache selection and a real offline pip install without GPU packages."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from pip._vendor.packaging.tags import Tag

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wheel_cache', ROOT / 'scripts/wheel_cache.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def wheel(directory, version='1.0', tag='py3-none-any', name='cache_fixture'):
    path = directory / f'{name}-{version}-{tag}.whl'
    info = f'{name}-{version}.dist-info'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'{name}/__init__.py', f'VERSION={version!r}\n')
        z.writestr(f'{info}/METADATA', f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        z.writestr(f'{info}/WHEEL', f'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n')
        z.writestr(f'{info}/RECORD', '')
    return path


def test_only_locked_compatible_wheels_selected(tmp_path):
    lock = tmp_path / 'requirements.lock'
    lock.write_text('cache-fixture==1.0\nmissing==1.0\n')
    desired = wheel(tmp_path)
    wheel(tmp_path, version='2.0')
    wheel(tmp_path, tag='cp310-cp310-win_amd64')
    wheel(tmp_path, name='not_in_lock')
    (tmp_path / 'invalid.whl').write_bytes(b'not a wheel')
    links = f'{tmp_path} https://example.invalid/cache /missing relative'
    assert cache.select_wheels(lock, links, [Tag('py3', 'none', 'any')]) == [str(desired)]


def test_target_tag_ranking_and_no_symlink_wheels(tmp_path):
    lock = tmp_path / 'requirements.lock'; lock.write_text('cache-fixture==1.0\n')
    generic = wheel(tmp_path)
    native = wheel(tmp_path, tag='cp312-cp312-manylinux_2_28_x86_64')
    tags = [Tag('cp312', 'cp312', 'manylinux_2_28_x86_64'), Tag('py3', 'none', 'any')]
    assert cache.select_wheels(lock, str(tmp_path), tags) == [str(native)]
    native.unlink(); native.symlink_to(generic)
    assert cache.select_wheels(lock, str(tmp_path), tags) == [str(generic)]
    assert cache.select_wheels(lock, '', tags) == []


def test_real_offline_install_has_exact_pin_and_no_unrequested_package(tmp_path):
    wheels = tmp_path / 'wheels'; wheels.mkdir()
    desired = wheel(wheels); wheel(wheels, version='2.0'); wheel(wheels, name='unrequested')
    lock = tmp_path / 'requirements.lock'; lock.write_text('cache-fixture==1.0\n')
    target = tmp_path / 'target'
    env = dict(os.environ, PIP_FIND_LINKS=str(wheels), PIP_CONFIG_FILE=os.devnull,
               PIP_NO_INDEX='1', PYTHONDONTWRITEBYTECODE='1')
    selected = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'scripts/wheel_cache.py'), str(lock)], env=env, text=True))
    assert selected == [str(desired)]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--no-index', '--no-deps', '--target', str(target), *selected], env=env, check=True, capture_output=True)
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(target)])}
    assert packages == {'cache_fixture': '1.0'}
