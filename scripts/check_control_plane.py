"""Public, non-grading checks. Mock suite starts isolated test state; live calls existing service."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import httpx

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS={
 1:('two_tenants','test_http_isolation_key_hashes_and_console'),
 2:('key_storage','test_http_isolation_key_hashes_and_console'),
 3:('rotate','test_rotate_revoke_expiry_disabled_no_engine_charge'),
 4:('revoke_expiry_disable','test_rotate_revoke_expiry_disabled_no_engine_charge'),
 5:('exact_billing','test_stream_nonstream_pricing_credit_history_and_strict_fields'),
 6:('concurrent_idempotency','test_twenty_simultaneous_idempotency_and_different_body'),
 7:('atomic_reservation','test_twenty_atomic_balance_reservations_and_shared_quota'),
 8:('shared_limits','test_rpm_and_concurrency_shared_across_keys'),
 9:('stream_cancellation','test_real_socket_disconnect_releases_slots'),
 10:('restart_reconciliation','test_restart_reconciles_unknown_and_settles_journal_exactly_once'),
 11:('price_and_credits','test_stream_nonstream_pricing_credit_history_and_strict_fields'),
 12:('live_two_tenant_agent',None)}

def stamp():return datetime.now(timezone.utc).isoformat()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--suite',choices=['mock','live'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('Output already exists; use a new path')
    started=stamp();checks=[]
    if args.suite=='mock':
        for number,(name,test) in SCENARIOS.items():
            record={'test_id':name,'scenario':number,'expected':'documented contract assertions pass','actual':None,'passed':False,'evidence':{}}
            if not test:
                record.update(skipped=True,actual='Requires existing real GPU; run --suite live')
            else:
                file='tests/test_transport.py' if number==9 else 'tests/test_control_plane.py'
                command=[sys.executable,'-m','pytest','-q',file+'::'+test]
                result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
                # Keep key-bearing failure repr out of the report; rerun named CPU test to inspect locally.
                record.update(actual='passed' if result.returncode==0 else 'failed; rerun named CPU test',passed=result.returncode==0,
                              evidence={'test':file+'::'+test,'exit_code':result.returncode,'isolated_database':True})
            checks.append(record)
    else:
        admin=os.environ.get('ADMIN_API_KEY')
        if not admin:p.error('ADMIN_API_KEY required in environment')
        for number,(name,_) in SCENARIOS.items():
            checks.append({'test_id':name,'scenario':number,'expected':'deterministic mock contract assertions' if number<12 else 'Two tenants each complete stream, nonstream and Agent with attributed bills',
                'actual':'Run --suite mock for deterministic failure/usage tests','passed':False,'skipped':True,'evidence':{}})
        record=checks[-1];record['skipped']=False
        evidence={'tenants':[]}
        try:
            with httpx.Client(base_url=args.base_url,timeout=60,trust_env=False) as client:
                ah={'Authorization':'Bearer '+admin}
                def call(method,path,**kwargs):
                    r=client.request(method,path,**kwargs);r.raise_for_status();return r.json()
                model=call('GET','/admin/pricing',headers=ah)['items'][0]['model']
                for name in ('Alice','Bob'):
                    tenant=call('POST','/admin/tenants',headers=ah,json={'name':name+' self-test '+uuid.uuid4().hex[:8],
                        'balance_micro_usd':1000000,'token_limit':100000,'rpm_limit':60,'concurrency_limit':4})
                    tid=tenant['tenant_id'];key=call('POST',f'/admin/tenants/{tid}/keys',headers=ah,json={'name':'live self-test'})
                    kh={'Authorization':'Bearer '+key['api_key']}
                    try:
                        payload={'model':model,'messages':[{'role':'user','content':'Reply with hello.'}],'max_tokens':64,'temperature':0}
                        call('POST','/v1/chat/completions',headers=kh,json=payload)
                        with client.stream('POST','/v1/chat/completions',headers=kh,json={**payload,'stream':True}) as response:
                            response.raise_for_status();data=''.join(response.iter_text())
                        if '[DONE]' not in data or '"error"' in data:raise ValueError('stream did not complete')
                        call('POST','/agent/run',headers=kh,json={'task':'Where is order X91? Return order_id, status, eta_days.',
                            'tool_data':{'orders':{'X91':{'status':'shipped','eta_days':6}}}})
                        usage=call('GET','/v1/usage',headers=kh)
                        billed=[r for r in usage['items'] if r['route']!='/agent/run']
                        if len(billed)!=4 or any(r['tenant_id']!=tid or r['key_id']!=key['key_id'] or r['status']!='completed' for r in billed):
                            raise ValueError('unexpected ledger attribution/completion')
                        if usage['summary']['cost_micro_usd']!=sum(r['cost_micro_usd'] for r in usage['items']):raise ValueError('summary mismatch')
                        evidence['tenants'].append({'tenant_id':tid,'key_id':key['key_id'],'request_ids':[r['request_id'] for r in billed],
                            'summary':usage['summary']})
                    finally:
                        call('POST',f"/admin/keys/{key['key_id']}/revoke",headers=ah,json={})
                record.update(actual='Two tenant paths and exact ledger sums passed',passed=True,evidence=evidence)
        except Exception as exc:
            record.update(actual='Live self-test failed: '+type(exc).__name__,evidence=evidence)
    result={'schema':'credential-service-test/1','official_score':False,'suite':args.suite,
        'commit_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'started_at':started,'finished_at':stamp(),'checks':checks}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    failed=sum(not c['passed'] and not c.get('skipped') for c in checks)
    print(json.dumps({'passed':sum(c['passed'] for c in checks),'failed':failed,'skipped':sum(bool(c.get('skipped')) for c in checks),'official_score':False}))
    return bool(failed)
if __name__=='__main__':sys.exit(main())
