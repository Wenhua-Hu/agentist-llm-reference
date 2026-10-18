import copy,hashlib,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.prose_workload import Fitter,BANK_SHA256,chat_ids,split_v1
from benchmarks.serving_benchmark import WORKLOAD_V1,WORKLOAD_V3,WORKLOAD_V4,JSON_RESPONSE_FORMAT,build_workload,core_contract_digest,validate_workload_contract,measure
class WordTokenizer:
 def apply_chat_template(self,messages,**kwargs):return list(range(6+sum(1 if word=='z' else 2 for m in messages for word in m['content'].split())))
class ProseTests(unittest.TestCase):
 def test_current_v4_default_and_historical_v3_bank_are_frozen(self):
  root=Path(__file__).resolve().parents[1]/'benchmarks'
  self.assertEqual(json.loads((root/'acceptance.json').read_text())['workload_version'],WORKLOAD_V4)
  pending=json.loads((root/'acceptance-v3.pending.json').read_text())
  self.assertEqual(pending['workload_version'],WORKLOAD_V3);self.assertIsNone(pending['targets']);self.assertEqual(pending['status'],'uncalibrated')
  self.assertEqual(hashlib.sha256((root/'prose-bank.txt').read_bytes()).hexdigest(),BANK_SHA256)
  validate_workload_contract(pending);pending['targets']=[];digest=core_contract_digest(pending)
  for wrong in (None,'0'*64):
   bad={**pending,'prose_bank_sha256':wrong}
   with self.assertRaisesRegex(ValueError,'prose bank'):validate_workload_contract(bad)
   self.assertNotEqual(core_contract_digest(bad),digest)
 def test_all_rng_cases_answers_and_limits_preserved(self):
  t=WordTokenizer();state=random.getstate();zero_ids=0
  for spec in json.loads((Path(__file__).resolve().parents[1]/'benchmarks').joinpath('acceptance.json').read_text())['workloads']:
   for seed in (2026101801,2026101802,2026101803,2026101804):
    old=build_workload(t,spec,seed,8,WORKLOAD_V1);new=build_workload(t,spec,seed,8,WORKLOAD_V3)
    self.assertEqual(len(new),8)
    for a,b in zip(old,new):
     self.assertEqual({k:v for k,v in a.items() if k!='messages'},{k:v for k,v in b.items() if k!='messages'})
     self.assertEqual(a['messages'][0],b['messages'][0]);self.assertTrue(b['messages'][1]['content'].endswith(split_v1(a['messages'])[1]))
     self.assertEqual(len(chat_ids(t,b['messages'])),spec['input_tokens']);self.assertNotIn(' z z z ',b['messages'][1]['content']);self.assertNotIn('<answer>',b['messages'][1]['content'])
     records=[b['expected']] if spec['records']==1 else b['expected']['records'];zero_ids+=sum('0' in str(r['id']) for r in records)
    self.assertEqual(new,build_workload(t,spec,seed,8,WORKLOAD_V3))
  self.assertGreater(zero_ids,0);self.assertEqual(random.getstate(),state)
 def test_v4_only_adds_frozen_json_policy_without_mutating_v3(self):
  t=WordTokenizer();root=Path(__file__).resolve().parents[1]/'benchmarks'
  config=json.loads((root/'acceptance-v4.pending.json').read_text())
  self.assertEqual(config['workload_version'],WORKLOAD_V4);self.assertIsNone(config['targets'])
  self.assertEqual(config['model_id'],'Qwen/Qwen2.5-7B-Instruct')
  self.assertEqual(config['model_revision'],'a09a35458c702b33eeacc393d103063234e8bc28')
  self.assertEqual(config['dtype'],'bfloat16')
  validate_workload_contract(config);config['targets']=[];digest=core_contract_digest(config)
  for wrong in (None,{}, {'type':'text'}, {'type':'json_object','extra':True}):
   bad={**config,'response_format':wrong}
   with self.assertRaisesRegex(ValueError,'response_format'):validate_workload_contract(bad)
   self.assertNotEqual(core_contract_digest(bad),digest)
  bad={**config};bad.pop('response_format')
  with self.assertRaisesRegex(ValueError,'response_format'):validate_workload_contract(bad)
  for spec in config['workloads']:
   for seed in range(2026101805,2026101809):
    v3=build_workload(t,spec,seed,8,WORKLOAD_V3);v4=build_workload(t,spec,seed,8,WORKLOAD_V4)
    self.assertEqual(len(v4),8)
    for a,b in zip(v3,v4):
     self.assertEqual(b,{**a,'response_format':JSON_RESPONSE_FORMAT})
     self.assertNotIn('response_format',a)
     self.assertEqual(len(chat_ids(t,b['messages'])),spec['input_tokens'])
 def test_tampered_bank_and_impossible_fit_fail_without_inference(self):
  t=WordTokenizer()
  with self.assertRaisesRegex(ValueError,'bank hash'):Fitter(t,b'modified text')
  f=Fitter(t,(Path(__file__).resolve().parents[1]/'benchmarks').joinpath('prose-bank.txt').read_bytes());spec={'id':'short','records':1,'input_tokens':512,'max_output_tokens':96};original=build_workload(t,spec,23,1)[0];unchanged=copy.deepcopy(original)
  with self.assertRaises(ValueError):f.fit(original['messages'],10)
  self.assertEqual(original,unchanged)
 def test_measure_rejects_invalid_v3_before_http(self):
  import asyncio
  config=json.loads((Path(__file__).resolve().parents[1]/'benchmarks').joinpath('acceptance-v3.pending.json').read_text());config.pop('prose_bank_sha256')
  async def run():
   with tempfile.TemporaryDirectory() as d,patch('benchmarks.serving_benchmark.httpx.AsyncClient') as client:
    with self.assertRaisesRegex(ValueError,'prose bank'):await measure('http://localhost','key','optimized',config,WordTokenizer(),42,d)
    client.assert_not_called()
  asyncio.run(run())
if __name__=='__main__':unittest.main()
