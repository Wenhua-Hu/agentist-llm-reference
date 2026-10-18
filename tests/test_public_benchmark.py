import asyncio
import json
from pathlib import Path
import unittest
import httpx
from benchmarks.serving_benchmark import request_once, summarize, score_performance, output_valid, NEUTRAL_DECODING, decoding_matches, measure, core_contract_digest, validate_acceptance
class Tokenizer:
    def encode(self,text,add_special_tokens=False): return list(text)
def config():
    c=json.loads((Path(__file__).resolve().parents[1]/'benchmarks'/'acceptance.json').read_text())
    c.update(status='calibrated',calibration={'run_id':'calibration-run','instance_id':'calibration-gpu','provider':'daytona','commit_sha':'a'*40,'seed':1})
    c['targets']=[{'workload':'short','concurrency':8,'output_tokens_per_second_min':100,
      'ttft_p95_ms_max':200,'tpot_p95_ms_max':20,'e2e_p95_ms_max':1500,'quality_goodput_min':1,
      'slo_attainment_min':.95,'success_rate_min':.99,'quality_rate_min':.95}]
    c['targets']=[{**c['targets'][0],'workload':w,'concurrency':n} for w in ('short','long') for n in (8,16)]
    c['calibration']['independent_confirmation']={'kind':'independent_confirmation','passed':True,'run_id':'confirm-run','instance_id':'confirm-gpu','seed':2,'provider':'daytona','commit_sha':'a'*40,'calibration_run_id':'calibration-run','calibration_instance_id':'calibration-gpu','files_sha256':{'test':'a'*64},'core_contract_sha256':core_contract_digest(c)}
    return c
def manifest():
    c=config();return {**{k:c[k] for k in ('model_id','model_revision','dtype','decoding_policy')},'gpu_name':'NVIDIA GeForce RTX 4090'}
def summary():
    return {'profile':'optimized','workload':'short','concurrency':8,'requests':48,'repeats':3,
      'response_format':config()['response_format'],'decoding_policy':config()['decoding_policy'],'measurement_version':config()['measurement_version'],'usage_verified':True,'timing_complete':True,'workload_version':config()['workload_version'],
      'output_tokens_per_second':100,'ttft_p95_ms':200,'tpot_p95_ms':20,'e2e_p95_ms':1500,
      'quality_goodput':1,'slo_attainment':.95,'success_rate':.99,'quality_rate':.95}
def summaries(row=None):
    row=summary() if row is None else row
    return [{**row,'workload':w,'concurrency':n} for w in ('short','long') for n in (8,16)]
class AcceptanceTests(unittest.TestCase):
    def test_neutral_decoding_required_and_recorded(self):
        self.assertTrue(decoding_matches({'config':NEUTRAL_DECODING}, config()))
        for changed in ({}, {**NEUTRAL_DECODING,'repetition_penalty':1.1},
                        {**NEUTRAL_DECODING,'sampling_defaults':'model'},
                        {**NEUTRAL_DECODING,'repetition_penalty':True}):
            self.assertFalse(decoding_matches({'config':changed}, config()))
            m=manifest();m['decoding_policy']=changed
            self.assertEqual(score_performance(config(),summaries(),m)['points'],0)
            row=summary();row['decoding_policy']=changed
            self.assertEqual(score_performance(config(),summaries(row),manifest())['points'],0)
        c=config();c.pop('decoding_policy')
        self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
    def test_v4_scoring_requires_recorded_json_policy(self):
        from benchmarks.serving_benchmark import WORKLOAD_V4, BANK_SHA256, JSON_RESPONSE_FORMAT
        c=config();c.update(workload_version=WORKLOAD_V4,prose_bank_sha256=BANK_SHA256,response_format=dict(JSON_RESPONSE_FORMAT))
        c['calibration']['independent_confirmation']['core_contract_sha256']=core_contract_digest(c)
        rows=summaries()
        for row in rows:row.update(workload_version=WORKLOAD_V4,response_format=dict(JSON_RESPONSE_FORMAT))
        self.assertEqual(score_performance(c,rows,manifest())['points'],30)
        for row in rows:row.pop('response_format')
        self.assertEqual(score_performance(c,rows,manifest())['points'],0)
    def test_measure_refuses_inherited_model_defaults_before_inference(self):
        import tempfile
        from unittest.mock import patch
        calls=[]
        def handler(req):
            calls.append(req.url.path)
            return httpx.Response(200,json={'profile':'baseline','config':{
                **NEUTRAL_DECODING,'sampling_defaults':'model','repetition_penalty':1.1}})
        async def run():
            with tempfile.TemporaryDirectory() as d:
                client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
                with patch('benchmarks.serving_benchmark.httpx.AsyncClient',return_value=client):
                    with self.assertRaisesRegex(ValueError,'neutral decoding'):
                        await measure('http://localhost','key','baseline',config(),Tokenizer(),1,d)
        asyncio.run(run())
        self.assertEqual(calls,['/info'])

    def test_candidate_never_scores_in_production(self):
        c=config();c['calibration']['independent_confirmation']=None
        self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
        result=score_performance(c,summaries(),manifest(),allow_provisional=True)
        self.assertEqual(result['points'],30);self.assertTrue(result['provisional'])
    def test_truncated_targets_never_award_full_credit(self):
        c=config();c['targets']=c['targets'][:1]
        for mode in (False,True):
            self.assertEqual(score_performance(c,summaries(),manifest(),allow_provisional=mode)['points'],0)
    def test_receipt_cannot_confirm_changed_threshold_or_workload(self):
        for field,value in [('deadline_seconds',60),('warmup_requests',3)]:
            c=config();c[field]=value
            self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
        c=config();c['targets'][0]['ttft_p95_ms_max']+=1
        self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
        c=config();c['calibration']['independent_confirmation']['instance_id']='calibration-gpu'
        self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
    def test_boundaries_pass(self):
        r=score_performance(config(),summaries(),manifest());self.assertTrue(r['passed']);self.assertEqual(r['points'],30)
    def test_uncalibrated_fails(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'benchmarks'/'acceptance.json').read_text());c['status']='uncalibrated';c['targets']=None;self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
    def test_every_gate_fails_closed(self):
        for field,value in [('quality_rate',0),('success_rate',0),('quality_goodput',0),('output_tokens_per_second',0),
            ('ttft_p95_ms',201),('tpot_p95_ms',21),('e2e_p95_ms',1501),('slo_attainment',.94),('requests',0),
            ('repeats',1),('usage_verified',False),('timing_complete',False),('ttft_p95_ms',float('nan')),('ttft_p95_ms',None)]:
            with self.subTest(field=field,value=value):
                s=summary();s[field]=value;self.assertEqual(score_performance(config(),summaries(s),manifest())['points'],0)
    def test_identity_missing_duplicate(self):
        self.assertEqual(score_performance(config(),[],manifest())['points'],0)
        self.assertEqual(score_performance(config(),summaries()+[summary()],manifest())['points'],22.5)
        for field in ('gpu_name','model_revision','dtype','decoding_policy'):
            m=manifest();m[field]='wrong';self.assertEqual(score_performance(config(),summaries(),m)['points'],0)
    def test_thresholds_no_defaults(self):
        for value in (None,0,-1,float('nan'),float('inf')):
            c=config();c['targets'][0]['ttft_p95_ms_max']=value;self.assertEqual(score_performance(c,summaries(),manifest())['points'],0)
    def test_wrong_answer_not_useful_throughput(self):
        r=summarize([{'ok':True,'quality_pass':False,'output_tokens':1000,'ttft_ms':1,'tpot_ms':1,'e2e_ms':10,'content_events':2}],1,config()['targets'][0])
        self.assertEqual(r['output_tokens_per_second'],0);self.assertEqual(r['quality_goodput'],0);self.assertFalse(r['timing_complete'])
    def test_empty_window(self):
        for rows,seconds in (([],1),([{}],0),([{}],float('nan'))):
            with self.assertRaises(ValueError):summarize(rows,seconds)
    def test_json_correctness(self):
        self.assertTrue(output_valid('{"id":2}',{'id':2}))
        for text in ('{"id":"2"}','```json\n{"id":2}\n```','{"id":2,"extra":1}', '{"id":2.0}', '{"id":true}'):
            self.assertFalse(output_valid(text,{'id':2}))
    def test_counts_independent_no_fake_tpot(self):
        text='{"id":2}';chunks=[json.dumps({'choices':[{'delta':{'content':text}}]}),json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}],'usage':{'completion_tokens':999999}}),'[DONE]']
        async def run():
            transport=httpx.MockTransport(lambda req:httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+c+'\n\n' for c in chunks)))
            async with httpx.AsyncClient(transport=transport) as client:
                return await request_once(client,'http://localhost','key','model',{'id':'x','input_tokens':512,'messages':[],'max_tokens':96,'expected':{'id':2}},Tokenizer(),1,'test')
        row=asyncio.run(run());self.assertTrue(row['quality_pass']);self.assertEqual(row['output_tokens'],len(text));self.assertEqual(row['declared_output_tokens'],999999);self.assertIsNone(row['tpot_ms']);self.assertFalse(summarize(summaries(row),1)['timing_complete'])
    def test_explicit_sse_timeouts_stay_failed_generic_errors_not_capacity(self):
        async def run(code):
            payload={'error':{'message':'must-not-leak-secret'}}
            if code is not None:payload['error']['code']=code
            content='data: '+json.dumps({'choices':[{'delta':{'content':'{"id":2}'}}]})+'\n\n'
            content+='data: '+json.dumps(payload)+'\n\n'
            transport=httpx.MockTransport(lambda req:httpx.Response(200,headers={'content-type':'text/event-stream'},text=content))
            async with httpx.AsyncClient(transport=transport) as client:
                return await request_once(client,'http://localhost','key','model',
                    {'id':'x','input_tokens':512,'messages':[],'max_tokens':96,'expected':{'id':2}},Tokenizer(),1,'test')
        for code,error in [('request_timeout','sse_request_timeout'),('upstream_timeout','sse_upstream_timeout'),
                           ('stream_failure','sse_stream_failure'),(None,'protocol_error'),('unexpected','protocol_error')]:
            row=asyncio.run(run(code))
            self.assertEqual(row['error'],error);self.assertFalse(row['ok']);self.assertFalse(row['quality_pass'])
            self.assertEqual(row['status'],200);self.assertGreater(row['output_tokens'],0)
            self.assertNotIn('must-not-leak-secret',json.dumps(row))
    def test_diagnostic_penalty_explicit_without_changing_default(self):
        bodies=[]
        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(429)
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                item={'id':'x','input_tokens':512,'messages':[],'max_tokens':96,'expected':{'id':2}}
                await request_once(client,'http://localhost','key','model',item,Tokenizer(),1,'default')
                await request_once(client,'http://localhost','key','model',item,Tokenizer(),1,'explicit',{'repetition_penalty':1.0})
                with self.assertRaises(ValueError):
                    await request_once(client,'http://localhost','key','model',item,Tokenizer(),1,'invalid',{'max_tokens':1})
        asyncio.run(run())
        self.assertNotIn('repetition_penalty',bodies[0])
        self.assertEqual(bodies[1]['repetition_penalty'],1.0)
        self.assertEqual(bodies[0]['messages'],bodies[1]['messages'])
        self.assertEqual(bodies[0]['max_tokens'],bodies[1]['max_tokens'])
    def test_json_mode_forwarded_and_recorded_without_output_repair(self):
        bodies=[];text='```json\n{"id":2}\n```'
        def handler(request):
            bodies.append(json.loads(request.content))
            chunks=[{'choices':[{'delta':{'content':text},'finish_reason':'stop'}]}]
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                item={'id':'x','input_tokens':512,'messages':[],'max_tokens':96,'expected':{'id':2}}
                old=await request_once(client,'http://localhost','key','model',item,Tokenizer(),1,'old')
                new=await request_once(client,'http://localhost','key','model',{**item,'response_format':{'type':'json_object'}},Tokenizer(),1,'new')
                return old,new
        old,new=asyncio.run(run())
        self.assertNotIn('response_format',bodies[0]);self.assertNotIn('response_format',old)
        self.assertEqual(bodies[1],{**bodies[0],'response_format':{'type':'json_object'}})
        self.assertEqual(new['response_format'],{'type':'json_object'})
        self.assertEqual(new['text'],text);self.assertFalse(new['quality_pass'])
        self.assertEqual(new['output_tokens'],len(text))
    def test_incomplete_stream(self):
        async def run():
            transport=httpx.MockTransport(lambda req:httpx.Response(200,headers={'content-type':'text/event-stream'},text='data: '+json.dumps({'choices':[{'delta':{'content':'{"id":2}'},'finish_reason':'stop'}]})+'\n\n'))
            async with httpx.AsyncClient(transport=transport) as client:
                return await request_once(client,'http://localhost','key','model',{'id':'x','input_tokens':512,'messages':[],'max_tokens':96,'expected':{'id':2}},Tokenizer(),1,'test')
        self.assertFalse(asyncio.run(run())['ok'])
if __name__=='__main__':unittest.main()

def test_public_tool_checks_actual_profile_and_model_before_measurement():
    import pytest
    from benchmarks.serving_benchmark import validate_service_identity
    c = config()
    declared = {'engine': 'sglang', 'profile': 'baseline', 'model_id': c['model_id'],
                'model_revision': c['model_revision'], 'config': {'dtype': c['dtype'], **NEUTRAL_DECODING}}
    validate_service_identity(declared, c, 'baseline')
    for field, value in [('profile', 'optimized'), ('model_revision', 'unfrozen'), ('engine', 'mock')]:
        with pytest.raises(ValueError):
            validate_service_identity({**declared, field: value}, c, 'baseline')


def test_public_cli_marks_artifacts_unofficial(monkeypatch, tmp_path):
    import sys
    from benchmarks import serving_benchmark as bench
    c = config()
    declared = {'engine': 'sglang', 'profile': 'baseline', 'model_id': c['model_id'],
                'model_revision': c['model_revision'], 'config': {'dtype': c['dtype'], **NEUTRAL_DECODING}}
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=declared)))
    monkeypatch.setattr(bench.httpx, 'Client', lambda **kwargs: client)
    monkeypatch.setenv('API_KEY', 'test-only-key')
    monkeypatch.setattr(bench, 'load_tokenizer', lambda config, local_files_only=False: Tokenizer())
    async def no_inference(*args):
        return [], []
    monkeypatch.setattr(bench, 'measure', no_inference)
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--base-url', 'http://localhost:8000',
        '--profile', 'baseline', '--out', str(tmp_path), '--seed', '123', '--local-files-only'])
    bench.main()
    artifact = json.loads((tmp_path/'baseline-benchmark-manifest.json').read_text())
    assert artifact['official_score'] is False and artifact['official_grade'] is False
    assert artifact['measurement_location'] == 'public_self_test'
    assert artifact['acceptance']['status'] == 'calibrated'
    assert artifact['acceptance']['calibration']['independent_confirmation']['passed'] is True
    assert 'test-only-key' not in json.dumps(artifact)
