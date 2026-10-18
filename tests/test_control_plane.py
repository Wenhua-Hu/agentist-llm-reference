"""HTTP-level contract tests against real SQLite, with a controlled inference backend."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
import os
import time
import pytest
from fastapi.testclient import TestClient
from server.app import create_app
from server.config import MODEL
from control_plane.storage import Store, LedgerError
from tests.test_service import FakeResponse

ADMIN='test-only-admin-key-independent'
BOOT='ordinary-metered-bootstrap-key'

def headers(key=ADMIN,idem=None):
    return {'Authorization':'Bearer '+key,**({'Idempotency-Key':idem} if idem else {})}
def body(**kw):
    return dict(model=MODEL,messages=[{'role':'user','content':'hello'}],max_tokens=80,**kw)
class Engine:
    def __init__(self): self.calls=[];self.aborts=[];self.delay=0;self.entered=None;self.release=None
    async def ready(self): return True
    async def close(self): pass
    async def count_input(self,payload): return 100
    async def abort(self,rid): self.aborts.append(rid)
    async def complete(self,payload,rid):
        self.calls.append(rid)
        if self.entered: self.entered.set()
        if self.release: await self.release.wait()
        if self.delay: await asyncio.sleep(self.delay)
        if payload.get('response_format'):
            if 'Tools' in payload['messages'][0]['content'] or 'lookup_order' in payload['messages'][0]['content']:
                content=json.dumps({'name':'lookup_order','arguments':{'order_id':'X91'}})
            else: content=json.dumps({'order_id':'X91','status':'shipped','eta_days':6})
        else: content='hello'
        return {'choices':[{'message':{'content':content},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':50,'prompt_tokens_details':{'cached_tokens':25}}}
    async def stream(self,payload,rid):
        value=await self.complete(payload,rid)
        return FakeResponse(['data: '+json.dumps({'choices':[{'delta':{'content':'hello'},'finish_reason':'stop'}]}),
            'data: '+json.dumps({'choices':[],'usage':value['usage']}),'data: [DONE]'])

def tenant(c,**kw):
    r=c.post('/admin/tenants',headers=headers(),json=dict(name='Alice',balance_micro_usd=100000,token_limit=100000,rpm_limit=1000,concurrency_limit=120,**kw))
    assert r.status_code==201,r.text
    return r.json()['tenant_id']
def issue(c,tid,expiry=None):
    r=c.post(f'/admin/tenants/{tid}/keys',headers=headers(),json={'name':'Agent','expires_at':expiry})
    assert r.status_code==201,r.text
    return r.json()
def usage(c,tid):return c.get('/admin/usage',headers=headers(),params={'tenant_id':tid}).json()
def make_tenant(c,**overrides):
    values={'name':'Alice','balance_micro_usd':100000,'token_limit':100000,'rpm_limit':1000,'concurrency_limit':120}
    values.update(overrides)
    r=c.post('/admin/tenants',headers=headers(),json=values)
    assert r.status_code==201,r.text
    return r.json()['tenant_id']


def test_http_isolation_key_hashes_and_console(tmp_path):
    app=create_app(Engine(),key=BOOT)
    with TestClient(app) as c:
        a=make_tenant(c);b=make_tenant(c,name='Bob')
        keys=[issue(c,t) for t in (a,a,b,b)]
        for key in keys:
            assert c.post('/v1/chat/completions',headers=headers(key['api_key']),json=body()).status_code==200
            assert c.get('/admin/tenants',headers=headers(key['api_key'])).status_code==401
            assert c.get('/v1/usage',headers=headers(key['api_key']),params={'tenant_id':b}).status_code==403
            own=c.get('/v1/usage',headers=headers(key['api_key'])).json()
            assert all(r['tenant_id']==key['tenant_id'] for r in own['items'])
        for t in (a,b):
            u=usage(c,t);assert len(u['items'])==2 and u['summary']['cost_micro_usd']==1000
        listed=c.get(f'/admin/tenants/{a}/keys',headers=headers()).text
        audit=c.get('/admin/audit',headers=headers()).text
        for key in keys:
            assert key['api_key'] not in listed+audit
            for f in app.state.store.path.parent.iterdir():
                if f.is_file(): assert key['api_key'].encode() not in f.read_bytes()
        html=c.get('/console').text
        assert c.get('/console').status_code==200 and '创建用户' in html and '发布新价格版本' in html
        assert not any(k['api_key'] in html for k in keys)
        assert 'localStorage' not in html
        assert c.get('/v1/usage',headers=headers()).status_code==401 # admin never inference authority


def test_rotate_revoke_expiry_disabled_no_engine_charge():
    engine=Engine();app=create_app(engine,key=BOOT)
    with TestClient(app) as c:
        tid=make_tenant(c);old=issue(c,tid)
        assert c.post('/v1/chat/completions',headers=headers(old['api_key']),json=body()).status_code==200
        r=c.post('/admin/keys/'+old['key_id']+'/rotate',headers=headers(),json={'expires_at':None})
        assert r.status_code==200
        new=r.json();assert new['key_id']!=old['key_id'] and new['replaced_status']=='revoked'
        assert c.post('/v1/chat/completions',headers=headers(old['api_key']),json=body()).status_code==401
        assert c.post('/v1/chat/completions',headers=headers(new['api_key']),json=body()).status_code==200
        for _ in range(2):assert c.post('/admin/keys/'+new['key_id']+'/revoke',headers=headers()).status_code==200
        assert c.post('/v1/chat/completions',headers=headers(new['api_key']),json=body()).status_code==401
        expired=issue(c,tid,(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat())
        assert c.post('/v1/chat/completions',headers=headers(expired['api_key']),json=body()).status_code==401
        active=issue(c,tid)
        assert c.patch('/admin/tenants/'+tid,headers=headers(),json={'status':'disabled'}).status_code==200
        assert c.post('/v1/chat/completions',headers=headers(active['api_key']),json=body()).status_code==403
        assert len(engine.calls)==2
        assert usage(c,tid)['items'][0]['key_id']==old['key_id']
        assert usage(c,tid)['summary']['cost_micro_usd']==1000


def test_stream_nonstream_pricing_credit_history_and_strict_fields():
    engine=Engine();app=create_app(engine,key=BOOT)
    with TestClient(app) as c:
        tid=make_tenant(c);key=issue(c,tid)['api_key']
        assert c.post('/v1/chat/completions',headers=headers(key),json=body()).status_code==200
        streamed=c.post('/v1/chat/completions',headers=headers(key),json=body(stream=True))
        assert streamed.status_code==200 and '[DONE]' in streamed.text
        u=usage(c,tid);assert [r['cost_micro_usd'] for r in u['items']]==[500,500]
        assert u['summary']['reserved_micro_usd']==0 and u['summary']['used_tokens']==300
        assert all(r['cached_tokens']==25 for r in u['items'])
        first=u['items'][0]['pricing_version']
        price=c.put('/admin/pricing',headers=headers(),json={'model':MODEL,'input_micro_usd_per_token':4,'output_micro_usd_per_token':8}).json()
        assert price['pricing_version']!=first
        c.post('/v1/chat/completions',headers=headers(key),json=body())
        u=usage(c,tid);assert [r['cost_micro_usd'] for r in u['items']]==[500,500,800]
        assert u['summary']['cost_micro_usd']==1800
        req={'delta_micro_usd':1000,'reason':'test allocation'}
        results=[c.post(f'/admin/tenants/{tid}/credits',headers=headers(idem='credit-once'),json=req).json() for _ in range(2)]
        assert results[0]==results[1]
        assert usage(c,tid)['summary']['balance_micro_usd']==100000-1800+1000
        assert c.post(f'/admin/tenants/{tid}/credits',headers=headers(idem='credit-once'),json={**req,'delta_micro_usd':2000}).status_code==409
        assert c.patch('/admin/tenants/'+tid,headers=headers(),json={'used_tokens':0}).status_code==422
        assert c.put('/admin/pricing',headers=headers(),json={'model':MODEL,'input_micro_usd_per_token':True,'output_micro_usd_per_token':8}).status_code==422
        assert c.post('/v1/chat/completions',headers=headers(key),json={**body(),'user_id':'fake','cost_micro_usd':0}).status_code==422


def test_twenty_simultaneous_idempotency_and_different_body():
    engine=Engine();engine.delay=.2;app=create_app(engine,key=BOOT)
    with TestClient(app) as c:
        tid=make_tenant(c);key=issue(c,tid)['api_key']
        def send(_):return c.post('/v1/chat/completions',headers=headers(key,'one'),json=body())
        with ThreadPoolExecutor(max_workers=20) as pool: results=list(pool.map(send,range(20)))
        assert sum(r.status_code==200 for r in results)==1
        assert all(r.status_code in (200,409) for r in results) and len(engine.calls)==1
        assert usage(c,tid)['summary']['cost_micro_usd']==500
        conflict=c.post('/v1/chat/completions',headers=headers(key,'one'),json={**body(),'max_tokens':81})
        assert conflict.status_code==409 and conflict.json()['error']['code']=='idempotency_conflict'
        again=send(0);assert again.status_code==409 and again.json()['error']['request_id']==engine.calls[0]


def test_twenty_atomic_balance_reservations_and_shared_quota():
    engine=Engine();engine.delay=.3;app=create_app(engine,key=BOOT)
    with TestClient(app) as c:
        tid=make_tenant(c,balance_micro_usd=680);keys=[issue(c,tid)['api_key'] for _ in range(2)]
        def send(i):return c.post('/v1/chat/completions',headers=headers(keys[i%2]),json=body())
        with ThreadPoolExecutor(max_workers=20) as pool: results=list(pool.map(send,range(20)))
        assert sum(r.status_code==200 for r in results)==1
        assert sum(r.status_code==402 for r in results)==19
        u=usage(c,tid);assert u['summary']['balance_micro_usd']==180 and u['summary']['reserved_micro_usd']==0
        tid2=make_tenant(c,token_limit=200);keys2=[issue(c,tid2)['api_key'] for _ in range(2)]
        assert c.post('/v1/chat/completions',headers=headers(keys2[0]),json=body()).status_code==200
        assert c.post('/v1/chat/completions',headers=headers(keys2[1]),json=body()).status_code==402


def test_rpm_and_concurrency_shared_across_keys():
    engine=Engine();app=create_app(engine,key=BOOT)
    with TestClient(app) as c:
        tid=make_tenant(c,rpm_limit=1);keys=[issue(c,tid)['api_key'] for _ in range(2)]
        assert c.post('/v1/chat/completions',headers=headers(keys[0]),json=body()).status_code==200
        limited=c.post('/v1/chat/completions',headers=headers(keys[1]),json=body())
        assert limited.status_code==429 and limited.headers['Retry-After']=='60'
        assert limited.json()['error']['code']=='rpm_limit'
        engine.delay=.2
        tid=make_tenant(c,concurrency_limit=1);keys=[issue(c,tid)['api_key'] for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            rs=list(pool.map(lambda k:c.post('/v1/chat/completions',headers=headers(k),json=body()),keys))
        assert sorted(r.status_code for r in rs)==[200,429]
        assert next(r for r in rs if r.status_code==429).json()['error']['code']=='concurrency_limit'


def test_agent_two_metered_model_calls_with_parent_idempotency():
    engine=Engine();app=create_app(engine,key=BOOT)
    with TestClient(app) as c:
        tid=make_tenant(c);key=issue(c,tid)['api_key']
        payload={'task':'Find order X91','tool_data':{'orders':{'X91':{'status':'shipped','eta_days':6}}}}
        r=c.post('/agent/run',headers=headers(key,'agent-once'),json=payload)
        assert r.status_code==200,r.text
        assert len(engine.calls)==2
        u=usage(c,tid);assert u['summary']['cost_micro_usd']==1000 and u['summary']['used_tokens']==300
        children=[x for x in u['items'] if x['route']=='/agent/model_call']
        assert len(children)==2 and children[0]['parent_request_id']==r.headers['X-Request-ID']
        assert c.post('/agent/run',headers=headers(key,'agent-once'),json=payload).status_code==409
        assert len(engine.calls)==2


def test_restart_reconciles_unknown_and_settles_journal_exactly_once(tmp_path):
    store=Store(tmp_path/'crash','x'*40,MODEL,BOOT)
    auth=store.authenticate(BOOT)
    store.reserve(auth,'unknown','/v1/chat/completions',body(),100,80)
    store.started('unknown')
    store.reserve(auth,'journal','/v1/chat/completions',body(),100,80)
    store.started('journal');store.record_usage('journal',{'prompt_tokens':100,'completion_tokens':50})
    # Reopen persisted SQLite as a new gateway process would; no graceful cleanup.
    reopened=Store(tmp_path/'crash','x'*40,MODEL,BOOT)
    with TestClient(create_app(Engine(),key=BOOT,store=reopened)) as c:
        u=c.get('/v1/usage',headers=headers(BOOT)).json()
        items={r['request_id']:r for r in u['items']}
        assert items['unknown']['status']=='adjusted_unknown'
        assert items['unknown']['input_tokens'] is items['unknown']['cost_micro_usd'] is None
        assert items['journal']['cost_micro_usd']==500
        assert u['summary']['reserved_micro_usd']==0
        audit=c.get('/admin/audit',headers=headers()).json()['items']
        assert any(a['event']=='credit_adjustment' and a['request_id']=='unknown' and a['detail']['usage_known'] is False for a in audit)
        reopened.settle('journal');assert c.get('/v1/usage',headers=headers(BOOT)).json()['summary']['cost_micro_usd']==500
    with pytest.raises(ValueError,match='differs'):Store(tmp_path/'crash','y'*40,MODEL,BOOT)


def test_missing_engine_usage_never_fabricates_free_success():
    class Missing(Engine):
        async def complete(self,payload,rid):
            result=await super().complete(payload,rid);result.pop('usage');return result
    app=create_app(Missing(),key=BOOT)
    with TestClient(app) as c:
        r=c.post('/v1/chat/completions',headers=headers(BOOT),json=body())
        assert r.status_code==502 and r.json()['error']['code']=='invalid_engine_usage'
        pending=c.get('/v1/usage',headers=headers(BOOT)).json()['items'][0]
        assert pending['status']=='pending_reconciliation' and pending['cost_micro_usd'] is None
        c.portal.call(app.state.recovery.sweep)
        final=c.get('/v1/usage',headers=headers(BOOT)).json()
        assert final['summary']['reserved_micro_usd']==0 and final['items'][0]['cost_micro_usd'] is None


def test_normal_stream_durable_usage_before_disconnect():
    app=create_app(Engine(),key=BOOT)
    with TestClient(app) as c:
        r=c.post('/v1/chat/completions',headers=headers(BOOT),json=body(stream=True))
        assert '[DONE]' in r.text
        rows=c.get('/v1/usage',headers=headers(BOOT)).json()['items']
        assert len(rows)==1 and rows[0]['cost_micro_usd']==500
        app.state.store.settle(rows[0]['request_id'])
        assert c.get('/v1/usage',headers=headers(BOOT)).json()['summary']['cost_micro_usd']==500


def test_missing_admin_closes_management_and_distinct_admin_authority(monkeypatch):
    monkeypatch.delenv('ADMIN_API_KEY')
    with TestClient(create_app(Engine(),key=BOOT)) as c:
        assert c.get('/admin/tenants',headers=headers(BOOT)).status_code==401
        assert c.get('/admin/tenants').status_code==401
        assert c.get('/v1/usage',headers=headers(BOOT)).status_code==200
        assert c.get('/console').status_code==200
    monkeypatch.setenv('ADMIN_API_KEY',BOOT)
    with pytest.raises(ValueError,match='differ'):create_app(Engine(),key=BOOT)


def test_revoked_key_admitted_agent_continues_but_new_admission_rejected(tmp_path):
    store=Store(tmp_path/'rotation','p'*40,MODEL,BOOT)
    auth=store.authenticate(BOOT)
    store.reserve(auth,'parent','/agent/run',{'task':'find'},0,0)
    store.revoke(auth['key_id'])
    # Existing orchestration may finish with the old billing identity.
    store.reserve(auth,'parent-0','/agent/model_call',body(),100,80,parent='parent')
    store.started('parent-0');store.record_usage('parent-0',{'prompt_tokens':100,'completion_tokens':50});store.settle('parent-0')
    with pytest.raises(LedgerError) as failure:store.reserve(auth,'new','/v1/chat/completions',body(),100,80)
    assert failure.value.status==401
    assert store.usage(auth['tenant_id'])['summary']['cost_micro_usd']==500
