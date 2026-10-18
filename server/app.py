import asyncio
from contextlib import asynccontextmanager
import importlib.metadata
import json
import os
import hmac
import time
import uuid
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from .admission import Admission, Overloaded
from control_plane.storage import Store, LedgerError
from control_plane.api import install as install_control_plane
from control_plane.meter import Counter, Recovery
from .backend import Backend
from .config import configuration, MODEL, REVISION
from .metrics import Metrics
from .streaming import relay, BrokenStream
from agent.agent import run as run_agent

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1, max_length=24000)

class Chat(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=96, ge=1, le=2048)
    temperature: float = Field(default=0, ge=0, le=2)
    stream: bool = False
    stream_options: dict | None = None

class AgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=4000)
    tool_data: dict

async def disconnect_guard(request, coro, timeout):
    job = asyncio.create_task(coro)
    stopped = asyncio.Event()
    async def monitor():
        while not stopped.is_set():
            if await request.is_disconnected():
                return
            try:
                await asyncio.wait_for(stopped.wait(), .05)
            except asyncio.TimeoutError:
                pass
    watcher = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait([job, watcher], timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if job in done:
            return job.result()
        if watcher in done:
            raise HTTPException(499, 'client disconnected')
        raise asyncio.TimeoutError()
    finally:
        stopped.set()
        if not job.done():
            job.cancel()
        await asyncio.gather(job, watcher, return_exceptions=True)

def create_app(backend=None, key=None, config=None, store=None):
    profile, default_config = configuration()
    cfg = config or default_config
    model = os.environ.get('MODEL_ID', MODEL)
    revision = os.environ.get('MODEL_REVISION', REVISION)
    backend = backend or Backend(os.environ.get("ENGINE_BASE_URL", "http://127.0.0.1:30000"))
    key = key if key is not None else os.environ.get('API_KEY', '')
    if not key:
        raise ValueError('API_KEY required')
    admin_key = os.environ.get('ADMIN_API_KEY', '')
    if admin_key and hmac.compare_digest(admin_key, key):
        raise ValueError('ADMIN_API_KEY must differ from API_KEY')
    if store is None:
        directory = os.environ.get('DATA_DIR')
        if not directory:
            raise ValueError('DATA_DIR required')
        store = Store(directory, os.environ.get('KEY_HASH_SECRET', ''), model, key)
    if not admin_key:
        print('Management API closed: ADMIN_API_KEY is not configured', flush=True)
    counter = Counter(backend, model, revision, os.environ.get('CONTRACT_TEST') == '1')
    recovery = Recovery(store, backend)
    gate = Admission(cfg['gateway_slots'], cfg['gateway_queue'])
    metrics = Metrics()
    cleanup_tasks = set()

    def abort_later(rid):
        async def clean():
            try:
                await backend.abort(rid)
                metrics.counts['abort_acknowledged'] += 1
            except Exception:
                metrics.counts['abort_failed'] += 1
        task = asyncio.create_task(clean())
        cleanup_tasks.add(task)
        task.add_done_callback(cleanup_tasks.discard)

    @asynccontextmanager
    async def lifespan(app):
        await recovery.start()
        try:
            yield
        finally:
            await recovery.close()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        await backend.close()

    app = FastAPI(lifespan=lifespan)
    app.state.store = store
    app.state.recovery = recovery
    app.state.gate = gate
    app.state.metrics = metrics

    @app.middleware('http')
    async def authentication(request, call_next):
        request.state.rid = uuid.uuid4().hex
        path = request.url.path
        if path not in ('/healthz', '/console'):
            header = request.headers.get('authorization', '')
            raw = header[7:] if header.startswith('Bearer ') else ''
            try:
                if path.startswith('/admin/'):
                    if not admin_key or not raw or not hmac.compare_digest(raw, admin_key):
                        raise LedgerError(401, 'admin_auth_required')
                else:
                    request.state.auth = store.authenticate(raw)
            except LedgerError as exc:
                return ledger_error(exc)
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.rid
        response.headers['Cache-Control'] = 'no-store'
        return response

    def ledger_error(exc):
        error = {'code': exc.code, 'message': exc.message}
        if exc.request_id:
            error['request_id'] = exc.request_id
        return JSONResponse({'error': error}, status_code=exc.status,
                            headers={'Retry-After': '60' if exc.code == 'rpm_limit' else '1'} if exc.status == 429 else {})

    @app.exception_handler(LedgerError)
    async def ledger_exception(request, exc):
        return ledger_error(exc)

    install_control_plane(app, store)

    async def begin(request, payload, route, parent=None, rid=None):
        rid = rid or request.state.rid
        count = await counter.count(payload) if 'messages' in payload else 0
        store.reserve(request.state.auth, rid, route, payload, count, payload.get('max_tokens', 0),
                      request.headers.get('idempotency-key') if not parent else None, parent)
        recovery.live.add(rid)
        return rid

    def finish(rid, usage):
        store.record_usage(rid, usage)
        store.settle(rid)

    def interrupted(rid):
        store.fail(rid)
        recovery.live.discard(rid)

    @app.get('/healthz')
    async def health():
        ready = await backend.ready()
        return JSONResponse({'status': 'ready' if ready else 'loading'}, status_code=200 if ready else 503)

    @app.get('/info')
    async def info():
        try:
            runtime_file = os.path.join(os.environ.get('RUNTIME_DIR', '.runtime'), 'runtime.json')
            if os.path.isfile(runtime_file):
                with open(runtime_file) as f:
                    engine_version = json.load(f)['versions']['sglang']
            else:
                engine_version = importlib.metadata.version('sglang')
        except importlib.metadata.PackageNotFoundError:
            engine_version = 'mock-test-only'
        return {'contract': 'llm-lab/1.0', 'model_id': model, 'model_revision': revision,
                'engine': 'sglang', 'engine_version': engine_version, 'profile': profile,
                'config': cfg, 'commit': os.environ.get('SOURCE_COMMIT', 'unknown')}

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': model, 'object': 'model'}]}

    @app.get('/metrics')
    async def counters():
        return {'active': gate.active, 'queued': gate.queued, 'counters': dict(metrics.counts)}

    async def acquire():
        try:
            await gate.acquire()
        except Overloaded:
            raise HTTPException(429, 'queue capacity reached', headers={'Retry-After': '1'})

    @app.post('/v1/chat/completions')
    async def chat(body: Chat, request: Request):
        if body.model != model:
            raise HTTPException(404, 'unknown model')
        rid = request.state.rid
        start = time.monotonic()
        deadline = start + cfg['request_timeout']
        held = False
        response = None
        handed_off = False
        status = 'failed'
        queue_ms = None
        admitted = False
        engine_started = False
        try:
            payload = body.model_dump(exclude_none=True)
            await begin(request, payload, '/v1/chat/completions')
            admitted = True
            await disconnect_guard(request, acquire(), max(.001, deadline-time.monotonic()))
            held = True
            queue_ms = (time.monotonic()-start)*1000
            store.started(rid)
            engine_started = True
            if not body.stream:
                value = await disconnect_guard(request, backend.complete(payload, rid), max(.001, deadline-time.monotonic()))
                finish(rid, value.get('usage'))
                status = 'complete'
                return value
            response = await disconnect_guard(request, backend.stream(payload, rid), max(.001, deadline-time.monotonic()))
            async def generate():
                stream_status = 'cancelled'
                stream_error_code = None
                stream_usage = None
                stream_finished = False
                try:
                    async for event in relay(response, deadline):
                        data = event[5:].strip() if event.startswith('data:') else ''
                        if data == '[DONE]':
                            finish(rid, stream_usage)
                        elif data:
                            obj = json.loads(data)
                            if any(c.get('finish_reason') is not None for c in obj.get('choices', [])):
                                stream_finished = True
                            if obj.get('usage') and stream_finished:
                                stream_usage = obj['usage']
                                store.record_usage(rid, stream_usage)
                        yield event
                    stream_status = 'complete'
                except (BrokenStream, ValueError, httpx.HTTPError, asyncio.TimeoutError, LedgerError) as exc:
                    stream_status = 'stream_error'
                    if isinstance(exc, asyncio.TimeoutError):
                        stream_error_code = 'request_timeout'
                    elif isinstance(exc, httpx.TimeoutException):
                        stream_error_code = 'upstream_timeout'
                    else:
                        stream_error_code = 'stream_failure'
                    # Headers already went out: explicit typed error, NO DONE.
                    # Only locally observed timeout exceptions receive timeout codes.
                    yield 'data: ' + json.dumps({'error': {'code': stream_error_code,
                                                          'message': 'upstream stream failed'}}) + '\n\n'
                finally:
                    gate.release()
                    interrupted(rid)
                    if stream_status != 'complete':
                        abort_later(rid)
                    metrics.event(rid, stream_status, start, queue_ms=queue_ms, error_code=stream_error_code)
                    task = asyncio.create_task(response.aclose())
                    cleanup_tasks.add(task)
                    task.add_done_callback(cleanup_tasks.discard)
            handed_off = True
            return StreamingResponse(generate(), media_type='text/event-stream', headers={
                'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        except asyncio.TimeoutError:
            status = 'timeout'
            raise HTTPException(504, 'total request deadline exceeded')
        except httpx.TimeoutException:
            status = 'timeout'
            raise HTTPException(504, 'engine request timed out')
        except httpx.HTTPStatusError as exc:
            raise HTTPException(exc.response.status_code if exc.response.status_code in (400, 413, 422, 429) else 502,
                                'engine rejected request')
        except httpx.HTTPError:
            raise HTTPException(503, 'engine unavailable')
        finally:
            if not handed_off:
                if admitted:
                    interrupted(rid)
                if held:
                    gate.release()
                if engine_started and status != 'complete':
                    abort_later(rid)
                if response is not None:
                    await response.aclose()
                metrics.event(rid, status, start, queue_ms=queue_ms)

    @app.post('/agent/run')
    async def agent(body: AgentTask, request: Request):
        rid = request.state.rid
        start = time.monotonic()
        held = False
        children = []
        admitted = False
        async def infer(messages):
            child = rid + '-' + str(len(children))
            payload = {'model': model, 'messages': messages,
                       'temperature': 0, 'max_tokens': 160, 'response_format': {'type': 'json_object'}}
            await begin(request, payload, '/agent/model_call', parent=rid, rid=child)
            children.append(child)
            try:
                store.started(child)
                value = await backend.complete(payload, child)
                finish(child, value.get('usage'))
            finally:
                interrupted(child)
            metrics.event(child, 'agent_model_call', start,
                          completion_tokens=value['usage']['completion_tokens'])
            return value
        try:
            await begin(request, body.model_dump(), '/agent/run')
            admitted = True
            await disconnect_guard(request, acquire(), 30)
            held = True
            result = await disconnect_guard(request, run_agent(body.task, body.tool_data, infer), max(.001, 40-(time.monotonic()-start)))
            finish(rid, {'prompt_tokens': 0, 'completion_tokens': 0})
            return result
        except ValueError:
            raise HTTPException(422, 'invalid model plan or tool arguments')
        except asyncio.TimeoutError:
            raise HTTPException(504, 'agent deadline exceeded')
        except httpx.TimeoutException:
            raise HTTPException(504, 'engine request timed out')
        except httpx.HTTPError:
            raise HTTPException(503, 'engine unavailable')
        finally:
            if admitted:
                interrupted(rid)
            if held:
                gate.release()
            for child in children:
                abort_later(child)
    return app

def factory():
    return create_app()
