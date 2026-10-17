import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

def configuration():
    profile = os.environ.get('BENCH_PROFILE', 'baseline')
    if profile not in ('baseline', 'optimized'):
        raise ValueError('unknown profile')
    return profile, yaml.safe_load((ROOT / 'configs' / (profile + '.yaml')).read_text())
