"""Select compatible, exactly pinned wheels from optional local download caches.

Run with the target venv's Python so Python/ABI/platform tags match installation.
This bootstraps before project dependencies exist, using pip's bundled packaging.
It does not import a model, resolve dependencies, or contact a package index.
"""
import json
import os
from pathlib import Path
import shlex
import sys

from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import Version


def select_wheels(lock, links, supported=None):
    pins = {}
    for line in Path(lock).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, version = line.split('==')
        name = canonicalize_name(name)
        if name in pins:
            raise ValueError('duplicate dependency pin')
        pins[name] = Version(version)
    ranks = {tag: i for i, tag in enumerate(sys_tags() if supported is None else supported)}
    selected = {}
    for link in shlex.split(links):
        # Only explicit existing local directories; never fetch a URL or scan a
        # global environment. Missing caches simply use the normal pip install.
        directory = Path(link)
        if '://' in link or not directory.is_absolute() or not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.whl')):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                name, version, _, tags = parse_wheel_filename(path.name)
            except ValueError:
                continue
            compatible = [ranks[tag] for tag in tags if tag in ranks]
            if pins.get(name) != version or not compatible:
                continue
            candidate = (min(compatible), str(path))
            if name not in selected or candidate < selected[name]:
                selected[name] = candidate
    return [selected[name][1] for name in sorted(selected)]


if __name__ == '__main__':
    print(json.dumps(select_wheels(sys.argv[1], os.environ.get('PIP_FIND_LINKS', ''))))
