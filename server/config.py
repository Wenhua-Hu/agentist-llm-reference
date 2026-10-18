import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'

KV_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
KV_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

def configuration():
    profile = os.environ.get('BENCH_PROFILE', 'baseline')
    if profile not in ('baseline', 'optimized'):
        raise ValueError('unknown profile')
    cfg = yaml.safe_load((ROOT / 'configs' / (profile + '.yaml')).read_text())
    for env, field, maximum in [('MAX_RUNNING_REQUESTS', 'max_running_requests', 32),
                                 ('GATEWAY_SLOTS', 'gateway_slots', 64),
                                 ('GATEWAY_QUEUE', 'gateway_queue', 128)]:
        if env in os.environ:
            value = int(os.environ[env])
            if not 1 <= value <= maximum:
                raise ValueError(env + ' outside supported range')
            cfg[field] = value
    if 'PREFIX_CACHE' in os.environ:
        if os.environ['PREFIX_CACHE'] not in ('0', '1'):
            raise ValueError('PREFIX_CACHE must be 0 or 1')
        cfg['prefix_cache'] = os.environ['PREFIX_CACHE'] == '1'
    if 'ATTENTION_BACKEND' in os.environ:
        if os.environ['ATTENTION_BACKEND'] not in ('triton', 'flashinfer'):
            raise ValueError('Unsupported ATTENTION_BACKEND')
        cfg['attention_backend'] = os.environ['ATTENTION_BACKEND']
    # These are SGLang 0.5.9's effective defaults under the explicit OpenAI policy.
    # The API accepts temperature overrides; only its request default is zero.
    if cfg.get('sampling_defaults') == 'openai':
        cfg.update(repetition_penalty=1.0, top_p=1.0, top_k=-1,
                   request_temperature_default=0.0)
    return profile, cfg
