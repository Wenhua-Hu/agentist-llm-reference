"""Explicit admin authority; browser UI uses an in-memory Bearer credential, no cookies."""
from datetime import datetime, timezone
from pathlib import Path
from fastapi import Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from typing import Literal, Annotated
from .storage import LedgerError

Money = Annotated[StrictInt, Field(ge=0, le=10**16)]
Positive = Annotated[StrictInt, Field(ge=1, le=10**9)]
class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')
class Tenant(Strict):
    name: str = Field(min_length=1,max_length=120)
    balance_micro_usd: Money
    token_limit: Money
    rpm_limit: Positive
    concurrency_limit: Annotated[StrictInt,Field(ge=1,le=120)]
class Patch(Strict):
    status: Literal['active','disabled'] | None = None
    token_limit: Money | None = None
    rpm_limit: Positive | None = None
    concurrency_limit: Annotated[StrictInt,Field(ge=1,le=120)] | None = None
class Expiry(Strict):
    expires_at: datetime | None = None
    @field_validator('expires_at')
    @classmethod
    def timezone_required(cls,value):
        if value and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError('expires_at must include a UTC offset')
        return value
    def expiry(self):
        return self.expires_at.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z') if self.expires_at else None
class Key(Expiry):
    name: str = Field(min_length=1,max_length=120)
class Credit(Strict):
    delta_micro_usd: Annotated[StrictInt,Field(ge=-10**16,le=10**16)]
    reason: str = Field(min_length=1,max_length=400)
    @field_validator('delta_micro_usd')
    @classmethod
    def nonzero(cls,value):
        if not value: raise ValueError('delta must be nonzero')
        return value
class Price(Strict):
    model: str = Field(min_length=1,max_length=200)
    input_micro_usd_per_token: Annotated[StrictInt,Field(ge=0,le=10**9)]
    output_micro_usd_per_token: Annotated[StrictInt,Field(ge=0,le=10**9)]

def install(app, store):
    @app.get('/console', response_class=HTMLResponse)
    async def console():
        return HTMLResponse((Path(__file__).resolve().parents[1]/'console/index.html').read_text(),
            headers={'Cache-Control':'no-store','Content-Security-Policy':"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",'X-Content-Type-Options':'nosniff'})
    @app.post('/admin/tenants',status_code=201)
    async def create_tenant(body:Tenant): return store.create_tenant(body.model_dump())
    @app.get('/admin/tenants')
    async def tenants(): return {'items':store.tenants()}
    @app.patch('/admin/tenants/{tid}')
    async def patch(tid:str,body:Patch): return store.patch_tenant(tid,body.model_dump(exclude_none=True))
    @app.post('/admin/tenants/{tid}/credits')
    async def credit(tid:str,body:Credit,request:Request): return store.credits(tid,body.model_dump(),request.headers.get('idempotency-key'))
    @app.post('/admin/tenants/{tid}/keys',status_code=201)
    async def issue(tid:str,body:Key): return store.issue(tid,body.name,body.expiry())
    @app.get('/admin/tenants/{tid}/keys')
    async def keys(tid:str): return {'items':store.keys(tid)}
    @app.post('/admin/keys/{kid}/rotate')
    async def rotate(kid:str,body:Expiry): return store.rotate(kid,body.expiry())
    @app.post('/admin/keys/{kid}/revoke')
    async def revoke(kid:str): return store.revoke(kid)
    @app.get('/admin/pricing')
    async def pricing(): return store.pricing()
    @app.put('/admin/pricing')
    async def price(body:Price): return store.pricing(body.model_dump())
    @app.get('/admin/usage')
    async def usage(tenant_id:str, start:str|None=Query(None,alias='from'),end:str|None=Query(None,alias='to'),cursor:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=500)):
        return store.usage(tenant_id,start,end,cursor,limit)
    @app.get('/v1/usage')
    async def own_usage(request:Request,cursor:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=500)):
        if set(request.query_params)-{'cursor','limit'}:
            raise LedgerError(403,'tenant_scope_forbidden')
        return store.usage(request.state.auth['tenant_id'],cursor=cursor,limit=limit)
    @app.get('/admin/audit')
    async def audit(tenant_id:str|None=None,cursor:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=500)):
        return store.audits(tenant_id,cursor,limit)
