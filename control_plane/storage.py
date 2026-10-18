"""SQLite transactions are the authority for every key, reservation and ledger entry.

A single gateway owns a database (an OS flock is held by the serving lifespan).
SQLite BEGIN IMMEDIATE serializes admission across keys and concurrent requests.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

ACTIVE = ('reserved', 'running', 'pending_reconciliation', 'usage_recorded')

def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')

def ident(prefix):
    return prefix + '_' + uuid.uuid4().hex

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

class LedgerError(Exception):
    def __init__(self, status, code, message=None, request_id=None):
        self.status, self.code = status, code
        self.message = message or code.replace('_', ' ')
        self.request_id = request_id

class Store:
    def __init__(self, directory, pepper, model, bootstrap_key):
        if not pepper or len(pepper.encode()) < 32:
            raise ValueError('KEY_HASH_SECRET must contain at least 32 bytes')
        self.pepper, self.model = pepper.encode(), model
        directory = Path(directory).expanduser().resolve()
        if directory in (Path('/'), Path.home(), Path(__file__).resolve().parents[1]):
            raise ValueError('DATA_DIR must be a dedicated private subdirectory')
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        self.path = directory / 'ledger.sqlite3'
        with self.connection() as db:
            db.executescript((Path(__file__).resolve().parents[1] / 'migrations/001_control_plane.sql').read_text())
        os.chmod(self.path, 0o600)
        with self.transaction() as db:
            fingerprint = self.key_hash('credential-service/1 pepper identity')
            old = db.execute("SELECT value FROM settings WHERE name='pepper_fingerprint'").fetchone()
            if old and not hmac.compare_digest(old[0], fingerprint):
                raise ValueError('KEY_HASH_SECRET differs from the persistent ledger')
            db.execute("INSERT OR IGNORE INTO settings VALUES('pepper_fingerprint',?)", (fingerprint,))
            if not db.execute('SELECT 1 FROM prices WHERE model=?', (model,)).fetchone():
                initial = json.loads((Path(__file__).resolve().parents[1]/'configs/pricing.json').read_text())
                if initial['model'] != model or any(type(initial[k]) is not int or initial[k] < 0 for k in ('input_micro_usd_per_token','output_micro_usd_per_token')):
                    raise ValueError('Invalid initial pricing configuration')
                self._price(db, model, initial['input_micro_usd_per_token'], initial['output_micro_usd_per_token'])
            # Stable bootstrap tenant, no privileged inference or exemption from billing.
            old = db.execute('SELECT tenant_id FROM api_keys WHERE key_hash=?', (self.key_hash(bootstrap_key),)).fetchone()
            if not old:
                tid = 'tenant_benchmark'
                db.execute('INSERT OR IGNORE INTO tenants VALUES(?,?,?,?,?,?,?,?)',
                           (tid, 'Benchmark', 'active', 10**15, 10**12, 100000, 120, now()))
                self._key(db, tid, 'Bootstrap benchmark', None, raw=bootstrap_key)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def key_hash(self, raw):
        return hmac.new(self.pepper, raw.encode(), hashlib.sha256).hexdigest()

    def audit(self, db, event, tenant_id=None, request_id=None, **detail):
        db.execute('INSERT INTO audit(event,tenant_id,request_id,detail,created_at) VALUES(?,?,?,?,?)',
                   (event, tenant_id, request_id, canonical(detail), now()))

    def authenticate(self, raw):
        with self.connection() as db:
            row = db.execute('SELECT k.*,t.status tenant_status FROM api_keys k JOIN tenants t USING(tenant_id) WHERE key_hash=?',
                             (self.key_hash(raw),)).fetchone()
        if not row or row['status'] != 'active' or (row['expires_at'] and row['expires_at'] <= now()):
            raise LedgerError(401, 'invalid_api_key')
        if row['tenant_status'] != 'active':
            raise LedgerError(403, 'tenant_disabled')
        return {'tenant_id': row['tenant_id'], 'key_id': row['key_id']}

    def _tenant(self, db, tid):
        row = db.execute('SELECT * FROM tenants WHERE tenant_id=?', (tid,)).fetchone()
        if not row:
            raise LedgerError(404, 'tenant_not_found')
        return dict(row)

    def _totals(self, db, tid):
        r = db.execute('SELECT COALESCE(SUM(input_tokens),0) i,COALESCE(SUM(output_tokens),0) o,COALESCE(SUM(cost_micro_usd),0) c,'
                       'COALESCE(SUM(reserved_micro_usd),0) rm,COALESCE(SUM(reserved_input+reserved_output),0) rt FROM requests WHERE tenant_id=?', (tid,)).fetchone()
        return {'input_tokens': r['i'], 'output_tokens': r['o'], 'used_tokens': r['i']+r['o'],
                'cost_micro_usd': r['c'], 'reserved_micro_usd': r['rm'], 'reserved_tokens': r['rt']}

    def tenant(self, tid):
        with self.connection() as db:
            value = self._tenant(db, tid)
            value.update(self._totals(db, tid))
        value['available_micro_usd'] = value['balance_micro_usd']-value['reserved_micro_usd']
        return value

    def tenants(self):
        with self.connection() as db:
            ids = [r[0] for r in db.execute('SELECT tenant_id FROM tenants ORDER BY created_at,tenant_id')]
        return [self.tenant(t) for t in ids]

    def create_tenant(self, values):
        tid = ident('tenant')
        with self.transaction() as db:
            db.execute('INSERT INTO tenants VALUES(?,?,?,?,?,?,?,?)',
                       (tid, values['name'], 'active', values['balance_micro_usd'], values['token_limit'], values['rpm_limit'], values['concurrency_limit'], now()))
            self.audit(db, 'tenant_created', tid)
        return self.tenant(tid)

    def patch_tenant(self, tid, values):
        with self.transaction() as db:
            self._tenant(db, tid)
            if values:
                db.execute('UPDATE tenants SET '+','.join(k+'=?' for k in values)+' WHERE tenant_id=?', (*values.values(), tid))
            self.audit(db, 'tenant_updated', tid, fields=values)
        return self.tenant(tid)

    def _key(self, db, tid, name, expiry, raw=None):
        raw = raw or 'ak_' + secrets.token_urlsafe(32)
        value = dict(key_id=ident('key'), tenant_id=tid, name=name, masked_key=raw[:6]+'…'+raw[-4:],
                     status='active', created_at=now(), expires_at=expiry)
        db.execute('INSERT INTO api_keys VALUES(?,?,?,?,?,?,?,?)',
                   (value['key_id'], tid, name, self.key_hash(raw), value['masked_key'], 'active', value['created_at'], expiry))
        self.audit(db, 'key_created', tid, key_id=value['key_id'])
        return {**value, 'api_key': raw}

    def issue(self, tid, name, expiry):
        with self.transaction() as db:
            self._tenant(db, tid)
            return self._key(db, tid, name, expiry)

    def keys(self, tid):
        with self.connection() as db:
            self._tenant(db, tid)
            return [dict(r) for r in db.execute('SELECT key_id,tenant_id,name,masked_key,status,created_at,expires_at FROM api_keys WHERE tenant_id=? ORDER BY created_at,key_id', (tid,))]

    def rotate(self, kid, expiry):
        with self.transaction() as db:
            old = db.execute('SELECT * FROM api_keys WHERE key_id=?', (kid,)).fetchone()
            if not old:
                raise LedgerError(404, 'key_not_found')
            if old['status'] != 'active':
                raise LedgerError(409, 'key_already_revoked')
            new = self._key(db, old['tenant_id'], old['name'], expiry)
            db.execute("UPDATE api_keys SET status='revoked' WHERE key_id=?", (kid,))
            self.audit(db, 'key_rotated', old['tenant_id'], replaced_key_id=kid, key_id=new['key_id'])
            return {**new, 'replaced_key_id': kid, 'replaced_status': 'revoked', 'old_status': 'revoked'}

    def revoke(self, kid):
        with self.transaction() as db:
            old = db.execute('SELECT * FROM api_keys WHERE key_id=?', (kid,)).fetchone()
            if not old:
                raise LedgerError(404, 'key_not_found')
            if old['status'] != 'revoked':
                db.execute("UPDATE api_keys SET status='revoked' WHERE key_id=?", (kid,))
                self.audit(db, 'key_revoked', old['tenant_id'], key_id=kid)
        return {'key_id': kid, 'status': 'revoked'}

    def credits(self, tid, values, idem):
        if not idem:
            raise LedgerError(400, 'idempotency_key_required')
        with self.transaction() as db:
            tenant = self._tenant(db, tid)
            old = db.execute('SELECT * FROM adjustments WHERE tenant_id=? AND idempotency_key=?', (tid, idem)).fetchone()
            if old:
                if old['body_hash'] != digest(values):
                    raise LedgerError(409, 'idempotency_conflict')
                return json.loads(old['response_json'])
            balance = tenant['balance_micro_usd'] + values['delta_micro_usd']
            if balance < self._totals(db, tid)['reserved_micro_usd']:
                raise LedgerError(402, 'insufficient_quota')
            result = dict(adjustment_id=ident('adjustment'), tenant_id=tid, **values, balance_micro_usd=balance, created_at=now())
            db.execute('UPDATE tenants SET balance_micro_usd=? WHERE tenant_id=?', (balance, tid))
            db.execute('INSERT INTO adjustments VALUES(?,?,?,?,?,?,?,?)',
                       (result['adjustment_id'], tid, values['delta_micro_usd'], values['reason'], idem, digest(values), canonical(result), result['created_at']))
            self.audit(db, 'credit_adjustment', tid, **values, adjustment_id=result['adjustment_id'])
            return result

    def _price(self, db, model, inp, out):
        value = dict(pricing_version=ident('price'), model=model, input_micro_usd_per_token=inp,
                     output_micro_usd_per_token=out, created_at=now())
        db.execute('INSERT INTO prices VALUES(?,?,?,?,?)', tuple(value[k] for k in ('pricing_version','model','input_micro_usd_per_token','output_micro_usd_per_token','created_at')))
        return value

    def pricing(self, values=None):
        with self.transaction() as db:
            if values:
                if values['model'] != self.model:
                    raise LedgerError(404, 'unknown_model')
                result = self._price(db, values['model'], values['input_micro_usd_per_token'], values['output_micro_usd_per_token'])
                self.audit(db, 'pricing_created', pricing_version=result['pricing_version'])
                return result
            return {'items': [dict(r) for r in db.execute('SELECT * FROM prices ORDER BY created_at DESC,rowid DESC')]}

    def reserve(self, auth, rid, route, body, count, max_tokens, idem=None, parent=None):
        tid, kid = auth['tenant_id'], auth['key_id']
        with self.transaction() as db:
            tenant = self._tenant(db, tid)
            if idem:
                old = db.execute('SELECT * FROM requests WHERE tenant_id=? AND route=? AND idempotency_key=?', (tid, route, idem)).fetchone()
                if old:
                    code = 'idempotency_conflict' if old['body_hash'] != digest(body) else ('request_in_progress' if old['status'] in ACTIVE else 'request_already_completed')
                    raise LedgerError(409, code, request_id=old['request_id'])
            # Recheck admission identity in this transaction: rotate/disable cannot race reserve.
            key = db.execute('SELECT status,expires_at FROM api_keys WHERE key_id=? AND tenant_id=?', (kid, tid)).fetchone()
            if parent:
                admitted = db.execute("SELECT 1 FROM requests WHERE request_id=? AND tenant_id=? AND key_id=? AND route='/agent/run' AND status IN ('reserved','running')", (parent,tid,kid)).fetchone()
                if not admitted:
                    raise LedgerError(409, 'parent_request_not_active')
            if not parent and (not key or key['status'] != 'active' or (key['expires_at'] and key['expires_at'] <= now())):
                raise LedgerError(401, 'invalid_api_key')
            if not parent and tenant['status'] != 'active':
                raise LedgerError(403, 'tenant_disabled')
            totals = self._totals(db, tid)
            price = db.execute('SELECT * FROM prices WHERE model=? ORDER BY created_at DESC,rowid DESC LIMIT 1', (self.model,)).fetchone()
            cost = count*price['input_micro_usd_per_token']+max_tokens*price['output_micro_usd_per_token']
            if tenant['balance_micro_usd']-totals['reserved_micro_usd'] < cost or totals['used_tokens']+totals['reserved_tokens']+count+max_tokens > tenant['token_limit']:
                raise LedgerError(402, 'insufficient_quota')
            if not parent:
                active = db.execute("SELECT COUNT(*) FROM requests WHERE tenant_id=? AND parent_request_id IS NULL AND status IN ('reserved','running','usage_recorded')", (tid,)).fetchone()[0]
                if active >= tenant['concurrency_limit']:
                    raise LedgerError(429, 'concurrency_limit')
                recent = db.execute('SELECT COUNT(*) FROM requests WHERE tenant_id=? AND parent_request_id IS NULL AND created_epoch>?', (tid, time.time()-60)).fetchone()[0]
                if recent >= tenant['rpm_limit']:
                    raise LedgerError(429, 'rpm_limit')
            db.execute('INSERT INTO requests(request_id,tenant_id,key_id,parent_request_id,route,model,body_hash,idempotency_key,status,pricing_version,reserved_input,reserved_output,reserved_micro_usd,created_at,created_epoch) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (rid,tid,kid,parent,route,self.model,digest(body),idem,'reserved',price['pricing_version'],count,max_tokens,cost,now(),time.time()))

    def started(self, rid):
        with self.transaction() as db:
            db.execute("UPDATE requests SET status='running',engine_started=1 WHERE request_id=? AND status='reserved'", (rid,))

    def record_usage(self, rid, usage):
        if not isinstance(usage, dict):
            raise LedgerError(502, 'invalid_engine_usage')
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if any(type(x) is not int or x < 0 for x in (inp,out)):
            raise LedgerError(502, 'invalid_engine_usage')
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', usage.get('cached_tokens', 0)) if isinstance(usage.get('prompt_tokens_details', {}),dict) else 0
        if type(cached) is not int or not 0 <= cached <= inp:
            raise LedgerError(502, 'invalid_engine_usage')
        with self.transaction() as db:
            row = db.execute('SELECT * FROM requests WHERE request_id=?', (rid,)).fetchone()
            if not row or row['status'] not in ACTIVE:
                return
            if inp != row['reserved_input'] or out > row['reserved_output']:
                raise LedgerError(502, 'engine_usage_exceeds_reservation')
            db.execute("UPDATE requests SET usage_evidence=?,status='usage_recorded' WHERE request_id=?", (canonical({'input_tokens':inp,'output_tokens':out,'cached_tokens':cached}),rid))

    def settle(self, rid):
        with self.transaction() as db:
            row = db.execute('SELECT r.*,p.input_micro_usd_per_token ip,p.output_micro_usd_per_token op FROM requests r JOIN prices p USING(pricing_version) WHERE request_id=?', (rid,)).fetchone()
            if not row or row['status'] not in ACTIVE:
                return
            if not row['usage_evidence']:
                raise LedgerError(502, 'missing_engine_usage')
            usage = json.loads(row['usage_evidence'])
            cost = usage['input_tokens']*row['ip']+usage['output_tokens']*row['op']
            db.execute('UPDATE tenants SET balance_micro_usd=balance_micro_usd-? WHERE tenant_id=?', (cost,row['tenant_id']))
            db.execute("UPDATE requests SET status='completed',input_tokens=?,output_tokens=?,cached_tokens=?,cost_micro_usd=?,reserved_input=0,reserved_output=0,reserved_micro_usd=0,finished_at=? WHERE request_id=?", (usage['input_tokens'],usage['output_tokens'],usage['cached_tokens'],cost,now(),rid))
            self.audit(db, 'usage_settled', row['tenant_id'], rid, cost_micro_usd=cost)

    def fail(self, rid, code='request_interrupted'):
        with self.transaction() as db:
            row = db.execute('SELECT * FROM requests WHERE request_id=?', (rid,)).fetchone()
            if not row or row['status'] not in ACTIVE:
                return
            if row['engine_started']:
                db.execute("UPDATE requests SET status='pending_reconciliation',error_code=? WHERE request_id=?", (code,rid))
            else:
                db.execute("UPDATE requests SET status='rejected',input_tokens=0,output_tokens=0,cached_tokens=0,cost_micro_usd=0,reserved_input=0,reserved_output=0,reserved_micro_usd=0,error_code=?,finished_at=? WHERE request_id=?", (code,now(),rid))

    def unfinished(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM requests WHERE status IN ('reserved','running','pending_reconciliation','usage_recorded')")]

    def reconcile(self, rid, reason):
        with self.connection() as db:
            row = db.execute('SELECT * FROM requests WHERE request_id=?', (rid,)).fetchone()
        if not row or row['status'] not in ACTIVE:
            return
        if row['usage_evidence']:
            self.settle(rid)
            return
        with self.transaction() as db:
            row = db.execute('SELECT * FROM requests WHERE request_id=?', (rid,)).fetchone()
            if row['status'] not in ACTIVE:
                return
            # No invented usage or cost. Release only this reservation and preserve unknowns.
            db.execute("UPDATE requests SET status='adjusted_unknown',reserved_input=0,reserved_output=0,reserved_micro_usd=0,error_code=?,finished_at=? WHERE request_id=?", (reason,now(),rid))
            self.audit(db, 'credit_adjustment', row['tenant_id'], rid, reason=reason,
                       released_micro_usd=row['reserved_micro_usd'], released_tokens=row['reserved_input']+row['reserved_output'], usage_known=False)

    def usage(self, tid, start=None, end=None, cursor=0, limit=100):
        with self.connection() as db:
            tenant = self._tenant(db, tid)
            where, args = 'tenant_id=?', [tid]
            for op,value in [('>=',start),('<',end)]:
                if value:
                    where += ' AND created_at'+op+'?'; args.append(value)
            sums = db.execute('SELECT COALESCE(SUM(input_tokens),0) i,COALESCE(SUM(output_tokens),0) o,COALESCE(SUM(cost_micro_usd),0) c,COUNT(*) n FROM requests WHERE '+where,args).fetchone()
            rows = db.execute('SELECT rowid AS cursor,request_id,tenant_id,key_id,parent_request_id,route,model,status,input_tokens,output_tokens,cached_tokens,cost_micro_usd,pricing_version,reserved_micro_usd,error_code,created_at,finished_at FROM requests WHERE '+where+' AND rowid>? ORDER BY rowid LIMIT ?', (*args,cursor,limit+1)).fetchall()
            summary = {**self._totals(db,tid), 'balance_micro_usd':tenant['balance_micro_usd']}
            summary.update(input_tokens=sums['i'],output_tokens=sums['o'],cost_micro_usd=sums['c'],requests=sums['n'])
            summary['available_micro_usd'] = summary['balance_micro_usd']-summary['reserved_micro_usd']
            return {'summary':summary,'items':[dict(r) for r in rows[:limit]],'next_cursor':rows[limit-1]['cursor'] if len(rows)>limit else None}

    def audits(self, tid=None, cursor=0, limit=100):
        with self.connection() as db:
            rows = db.execute('SELECT * FROM audit WHERE audit_id>?'+(' AND tenant_id=?' if tid else '')+' ORDER BY audit_id LIMIT ?', (cursor,*([tid] if tid else []),limit+1)).fetchall()
            items = [{**dict(r),'detail':json.loads(r['detail'])} for r in rows[:limit]]
            return {'items':items,'next_cursor':items[-1]['audit_id'] if len(rows)>limit else None}
