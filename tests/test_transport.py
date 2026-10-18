import json
import os
import socket
import subprocess
import sys
import time
import httpx
from server.config import MODEL

def test_real_socket_disconnect_releases_slots():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    process=subprocess.Popen([sys.executable,'-m','uvicorn','tests.mock_app:app',
        '--host','127.0.0.1','--port',str(port),'--no-access-log'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=3,trust_env=False,
            headers={'Authorization':'Bearer socket-test-key'}) as client:
            for _ in range(60):
                if process.poll() is not None:raise RuntimeError('test server failed')
                try:
                    if client.get('/healthz').status_code==200:break
                except httpx.HTTPError:time.sleep(.05)
            else:raise RuntimeError('test server did not start')
            # Set a real tenant-level limit of one: accounting reconciliation must
            # retain unknown money reservations without retaining execution slots.
            response=client.patch('/admin/tenants/tenant_benchmark',
                headers={'Authorization':'Bearer '+os.environ['ADMIN_API_KEY']},
                json={'concurrency_limit':1})
            assert response.status_code==200
            for _ in range(20):
                with client.stream('POST','/v1/chat/completions',json={
                    'model':MODEL,'messages':[{'role':'user','content':'slow'}],'stream':True}) as response:
                    assert response.status_code==200
                    for line in response.iter_lines():
                        if 'content' in line:break
                # Poll actual released ownership rather than trusting an arbitrary sleep.
                for _ in range(100):
                    state=client.get('/metrics').json()
                    if state['active']==0 and state['queued']==0:break
                    time.sleep(.01)
                assert state['active']==state['queued']==0
            for _ in range(100):
                state=client.get('/metrics').json()
                if state['counters'].get('abort_acknowledged',0)==20:break
                time.sleep(.01)
            assert state['counters'].get('abort_acknowledged',0)==20
            assert state['counters'].get('complete',0)==0
            for _ in range(80):
                usage=client.get('/v1/usage').json()
                if usage['summary']['reserved_micro_usd']==0:break
                time.sleep(.05)
            assert usage['summary']['reserved_micro_usd']==0
            assert len(usage['items'])==20
            assert all(r['status']=='adjusted_unknown' and r['input_tokens'] is None and r['cost_micro_usd'] is None for r in usage['items'])
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
