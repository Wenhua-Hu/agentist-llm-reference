import asyncio
import json
from pathlib import Path
import unittest
import httpx
from benchmarks.serving_benchmark import request_once, summarize, score_performance, output_valid, NEUTRAL_DECODING, decoding_matches, measure
class Tokenizer:
    def encode(self,text,add_special_tokens=False): return list(text)
def config():
    c=json.loads((Path(__file__).resolve().parents[1]/'benchmarks'/'acceptance.json').read_text())
    c.update(status='calibrated',calibration={'run_ids':['test-only']})
    c['targets']=[{'workload':'short','concurrency':8,'output_tokens_per_second_min':100,
      'ttft_p95_ms_max':200,'tpot_p95_ms_max':20,'e2e_p95_ms_max':1500,'quality_goodput_min':1,
      'slo_attainment_min':.95,'success_rate_min':.99,'quality_rate_min':.95}]
    return c
def manifest():
    c=config();return {**{k:c[k] for k in ('model_id','model_revision','dtype','decoding_policy')},'gpu_name':'NVIDIA GeForce RTX 4090'}
def summary():
    return {'profile':'optimized','workload':'short','concurrency':8,'requests':48,'repeats':3,
      'decoding_policy':config()['decoding_policy'],'usage_verified':True,'timing_complete':True,'workload_version':config()['workload_version'],
      'output_tokens_per_second':100,'ttft_p95_ms':200,'tpot_p95_ms':20,'e2e_p95_ms':1500,
      'quality_goodput':1,'slo_attainment':.95,'success_rate':.99,'quality_rate':.95}
class AcceptanceTests(unittest.TestCase):
    def test_neutral_decoding_required_and_recorded(self):
        self.assertTrue(decoding_matches({'config':NEUTRAL_DECODING}, config()))
        for changed in ({}, {**NEUTRAL_DECODING,'repetition_penalty':1.1},
                        {**NEUTRAL_DECODING,'sampling_defaults':'model'},
                        {**NEUTRAL_DECODING,'repetition_penalty':True}):
            self.assertFalse(decoding_matches({'config':changed}, config()))
            m=manifest();m['decoding_policy']=changed
            self.assertEqual(score_performance(config(),[summary()],m)['points'],0)
            row=summary();row['decoding_policy']=changed
            self.assertEqual(score_performance(config(),[row],manifest())['points'],0)
        c=config();c.pop('decoding_policy')
        self.assertEqual(score_performance(c,[summary()],manifest())['points'],0)
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

    def test_boundaries_pass(self):
        r=score_performance(config(),[summary()],manifest());self.assertTrue(r['passed']);self.assertEqual(r['points'],30)
    def test_uncalibrated_fails(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'benchmarks'/'acceptance.json').read_text());c['status']='uncalibrated';c['targets']=None;self.assertEqual(score_performance(c,[summary()],manifest())['points'],0)
    def test_every_gate_fails_closed(self):
        for field,value in [('quality_rate',0),('success_rate',0),('quality_goodput',0),('output_tokens_per_second',0),
            ('ttft_p95_ms',201),('tpot_p95_ms',21),('e2e_p95_ms',1501),('slo_attainment',.94),('requests',0),
            ('repeats',1),('usage_verified',False),('timing_complete',False),('ttft_p95_ms',float('nan')),('ttft_p95_ms',None)]:
            with self.subTest(field=field,value=value):
                s=summary();s[field]=value;self.assertEqual(score_performance(config(),[s],manifest())['points'],0)
    def test_identity_missing_duplicate(self):
        for rows in ([],[summary(),summary()]):self.assertEqual(score_performance(config(),rows,manifest())['points'],0)
        for field in ('gpu_name','model_revision','dtype','decoding_policy'):
            m=manifest();m[field]='wrong';self.assertEqual(score_performance(config(),[summary()],m)['points'],0)
    def test_thresholds_no_defaults(self):
        for value in (None,0,-1,float('nan'),float('inf')):
            c=config();c['targets'][0]['ttft_p95_ms_max']=value;self.assertEqual(score_performance(c,[summary()],manifest())['points'],0)
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
        row=asyncio.run(run());self.assertTrue(row['quality_pass']);self.assertEqual(row['output_tokens'],len(text));self.assertEqual(row['declared_output_tokens'],999999);self.assertIsNone(row['tpot_ms']);self.assertFalse(summarize([row],1)['timing_complete'])
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
