import asyncio
import json
import time
import httpx
import pytest
from fastapi.testclient import TestClient
from server.admission import Admission, Overloaded
from server.app import create_app
from server.streaming import relay, BrokenStream
from server.config import MODEL
from agent.agent import run
from agent.tools import execute

class FakeBackend:
    def __init__(self):self.aborts=[];self.healthy=True
    async def count_input(self,payload):return 1
    async def ready(self):return self.healthy
    async def close(self):pass
    async def abort(self,rid):self.aborts.append(rid)
    async def complete(self,payload,rid):
        return {'choices':[{'message':{'content':'mock'}}],'usage':{'prompt_tokens':1,'completion_tokens':1}}
    async def stream(self,payload,rid):
        return FakeResponse(['data: '+json.dumps({'choices':[{'delta':{'role':'assistant'}}]}),
            'data: '+json.dumps({'choices':[{'delta':{'content':'mock'},'finish_reason':None}]}),
            'data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1}}),
            'data: [DONE]'])

class FakeResponse:
    def __init__(self,lines):self.lines=lines
    async def aclose(self):pass
    async def aiter_lines(self):
        for line in self.lines:yield line

def payload(**extra):
    return {'model':MODEL,'messages':[{'role':'user','content':'hello'}],**extra}

def test_initial_metrics_include_zero_cancellation_counter():
    backend = FakeBackend()
    app = create_app(backend, key='initial-metrics-key')
    with TestClient(app) as client:
        assert client.get('/metrics').status_code == 401
        response = client.get('/metrics', headers={'Authorization': 'Bearer initial-metrics-key'})
        assert response.status_code == 200
        values = response.json()
        for value in (values['active'], values['queued'], values['counters']['abort_acknowledged']):
            assert type(value) is int and value == 0
        assert backend.aborts == []

def test_api_contract_and_auth():
    backend=FakeBackend();app=create_app(backend,key='secret-fixture-789')
    with TestClient(app) as client:
        assert client.get('/healthz').json()=={'status':'ready'}
        for route in ['/info','/v1/models','/metrics']:
            assert client.get(route).status_code==401
        assert client.post('/agent/run',json={}).status_code==401
        client.headers['Authorization']='Bearer secret-fixture-789'
        assert client.get('/v1/models').json()['data'][0]['id']==MODEL
        assert 'secret-fixture-789' not in client.get('/info').text
        assert client.post('/v1/chat/completions',json=payload()).status_code==200
        streamed=client.post('/v1/chat/completions',json=payload(stream=True))
        assert '[DONE]' in streamed.text
        assert client.get('/metrics').json()['active']==0
        assert client.post('/v1/chat/completions',json=payload(model='wrong')).status_code==404
        assert client.post('/v1/chat/completions',json=payload(max_tokens=-1)).status_code==422
        backend.healthy=False
        assert client.get('/healthz').status_code==503

@pytest.mark.asyncio
async def test_overload_fifo_and_twenty_cancellations():
    gate=Admission(2,2)
    await gate.acquire();await gate.acquire()
    first=asyncio.create_task(gate.acquire());second=asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    start=time.monotonic()
    with pytest.raises(Overloaded):await gate.acquire()
    assert time.monotonic()-start<1
    gate.release();await first
    assert not second.done()
    first=None;gate.release();await second
    gate.release();gate.release()
    assert gate.active==gate.queued==0
    for _ in range(20):
        await gate.acquire();await gate.acquire()
        pending=asyncio.create_task(gate.acquire());await asyncio.sleep(0)
        pending.cancel();await asyncio.gather(pending,return_exceptions=True)
        gate.release();gate.release()
    assert gate.active==gate.queued==0

@pytest.mark.asyncio
async def test_cancellation_after_slot_transfer():
    gate=Admission(1,1);await gate.acquire()
    pending=asyncio.create_task(gate.acquire());await asyncio.sleep(0)
    gate.release();pending.cancel()
    await asyncio.gather(pending,return_exceptions=True)
    assert gate.active==gate.queued==0

@pytest.mark.asyncio
@pytest.mark.parametrize('lines', [[],['data: {}'],['data: [DONE]'],['data: {"error":"failed"}']])
async def test_truncated_stream_rejected(lines):
    with pytest.raises(BrokenStream):
        async for _ in relay(FakeResponse(lines),time.monotonic()+1):pass

@pytest.mark.asyncio
async def test_stream_deadline():
    class Slow(FakeResponse):
        async def aiter_lines(self):
            await asyncio.sleep(1)
            yield 'data: {}'
    with pytest.raises(asyncio.TimeoutError):
        async for _ in relay(Slow([]),time.monotonic()+.01):pass

def test_engine_failure_and_timeout():
    class Unavailable(FakeBackend):
        async def complete(self,payload,rid):raise httpx.ConnectError('unavailable')
    app=create_app(Unavailable(),key='k')
    with TestClient(app) as client:
        assert client.post('/v1/chat/completions',headers={'Authorization':'Bearer k'},json=payload()).status_code==503
        assert app.state.gate.active==0
    class Slow(FakeBackend):
        async def complete(self,payload,rid):await asyncio.sleep(1)
    app=create_app(Slow(),key='k',config={'gateway_slots':1,'gateway_queue':1,'request_timeout':.01})
    with TestClient(app) as client:
        assert client.post('/v1/chat/completions',headers={'Authorization':'Bearer k'},json=payload()).status_code==504
        assert app.state.gate.active==0

@pytest.mark.asyncio
async def test_agent_really_calls_twice_and_preserves_model_answer():
    calls=[]
    async def infer(messages):
        calls.append(messages)
        content={'name':'lookup_order','arguments':{'order_id':'X91'}} if len(calls)==1 else {'order_id':'X91','status':'MODEL_OUTPUT','eta_days':6}
        return {'choices':[{'message':{'content':json.dumps(content)}}],'usage':{'completion_tokens':8}}
    result=await run('Find order X91',{'orders':{'X91':{'status':'shipped','eta_days':6}}},infer)
    assert len(calls)==2 and result['usage']['model_calls']==2
    assert result['answer']['status']=='MODEL_OUTPUT' # No cheating by replacing model output with facts.
    assert result['tool_calls'][0]['result']['status']=='shipped'
    assert 'shipped' not in json.dumps(calls[0])

def test_tool_whitelist():
    with pytest.raises(ValueError):execute('shell',{'command':'pwd'},{})
    with pytest.raises(ValueError):execute('lookup_order',{'order_id':'missing'},{'orders':{}})


@pytest.mark.asyncio
async def test_gateway_payload_and_sse_preserve_repeated_digits():
    from server.app import Chat
    from server.backend import Backend
    expected = json.dumps({'records': [{'id': 41550}, {'id': 69430}, {'id': 50085}]})
    messages = [{'role': 'user', 'content': 'Copy exactly 41550, 69430, 50085. Keep every digit.'}]
    captured = []
    def engine(request):
        captured.append(json.loads(request.content))
        chunks = [json.dumps({'choices': [{'delta': {'content': char}}]}) for char in expected]
        chunks += [json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}],
                               'usage': {'completion_tokens': len(expected)}}), '[DONE]']
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=''.join('data: '+chunk+'\n\n' for chunk in chunks))
    backend = Backend()
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(base_url='http://engine', transport=httpx.MockTransport(engine))
    payload = Chat(model=MODEL, messages=messages, max_tokens=384, temperature=0, stream=True).model_dump(exclude_none=True)
    response = await backend.stream(payload, 'copy-test')
    parts = []
    async for event in relay(response, time.monotonic()+2):
        data = event[5:].strip()
        if data != '[DONE]':
            parts.extend(c.get('delta', {}).get('content', '') for c in json.loads(data).get('choices', []))
    await response.aclose()
    await backend.close()
    assert ''.join(parts) == expected
    assert captured[0]['messages'] == messages
    assert captured[0]['max_tokens'] == 384 and captured[0]['temperature'] == 0
    assert captured[0]['rid'] == 'copy-test'


@pytest.mark.parametrize('profile', ['baseline', 'optimized'])
def test_info_declares_effective_neutral_sampling_policy(monkeypatch, profile):
    monkeypatch.setenv('BENCH_PROFILE', profile)
    with TestClient(create_app(FakeBackend(), key='k')) as client:
        cfg = client.get('/info', headers={'Authorization': 'Bearer k'}).json()['config']
        assert cfg['sampling_defaults'] == 'openai'
        assert cfg['repetition_penalty'] == cfg['top_p'] == 1.0
        assert cfg['top_k'] == -1 and cfg['request_temperature_default'] == 0.0


@pytest.mark.parametrize('failure,expected', [(httpx.ReadTimeout, 504),
    (httpx.ConnectTimeout, 504), (httpx.WriteTimeout, 504),
    (httpx.PoolTimeout, 504), (httpx.ConnectError, 503)])
@pytest.mark.parametrize('route,stream', [('/v1/chat/completions', False),
    ('/v1/chat/completions', True), ('/agent/run', False)])
def test_upstream_timeouts_distinct_from_transport_failure(failure, expected, route, stream):
    class Failing(FakeBackend):
        async def complete(self, payload, rid):
            raise failure('engine failure fixture')
        async def stream(self, payload, rid):
            raise failure('engine failure fixture')
    backend = Failing()
    app = create_app(backend, key='k')
    body = payload(stream=stream) if route.endswith('completions') else {
        'task': 'Find order X91', 'tool_data': {'orders': {'X91': {'status': 'shipped', 'eta_days': 6}}}}
    with TestClient(app) as client:
        result = client.post(route, headers={'Authorization': 'Bearer k'}, json=body)
        assert result.status_code == expected
        assert app.state.gate.active == app.state.gate.queued == 0
        if route.endswith('completions'):
            assert app.state.metrics.counts['timeout' if expected == 504 else 'failed'] == 1
    assert backend.aborts


@pytest.mark.parametrize('failure,code', [
    (asyncio.TimeoutError, 'request_timeout'),
    (httpx.ReadTimeout, 'upstream_timeout'),
    (httpx.ConnectError, 'stream_failure'),
    (BrokenStream, 'stream_failure'),
    (ValueError, 'stream_failure')])
def test_failure_after_stream_headers_has_precise_code(failure, code):
    class FailingStream(FakeResponse):
        async def aiter_lines(self):
            yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'partial'}}]})
            raise failure('private failure text must not be exposed')
    class Failing(FakeBackend):
        async def stream(self, payload, rid):
            return FailingStream([])
    backend = Failing()
    app = create_app(backend, key='k')
    with TestClient(app) as client:
        result = client.post('/v1/chat/completions', headers={'Authorization': 'Bearer k'},
                             json=payload(stream=True))
        assert result.status_code == 200
        assert 'partial' in result.text and '"error"' in result.text
        assert '[DONE]' not in result.text
        errors = [json.loads(line[6:])['error'] for line in result.text.splitlines() if line.startswith('data: ') and '"error"' in line]
        assert errors == [{'code': code, 'message': 'upstream stream failed'}]
        assert 'private failure text' not in result.text
        assert app.state.gate.active == app.state.gate.queued == 0
        assert app.state.metrics.counts['stream_error'] == 1
    assert backend.aborts


def test_relay_total_deadline_emits_timeout_code_and_releases_slot():
    class SlowStream(FakeResponse):
        async def aiter_lines(self):
            yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'partial'}}]})
            await asyncio.sleep(1)
    class Slow(FakeBackend):
        async def stream(self, payload, rid):
            return SlowStream([])
    backend = Slow()
    app = create_app(backend, key='k', config={
        'gateway_slots': 1, 'gateway_queue': 1, 'request_timeout': .03})
    with TestClient(app) as client:
        result = client.post('/v1/chat/completions', headers={'Authorization': 'Bearer k'},
                             json=payload(stream=True))
        assert result.status_code == 200
        assert 'request_timeout' in result.text and '[DONE]' not in result.text
        assert app.state.gate.active == app.state.gate.queued == 0
    assert backend.aborts
