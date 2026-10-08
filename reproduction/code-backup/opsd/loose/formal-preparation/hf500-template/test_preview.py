"""Synthetic temporary negative fixtures only; no actual future receipt produced."""
import json,sys,tempfile,unittest,subprocess,hashlib
from pathlib import Path
PREVIEW=Path(__file__).resolve().parents[1]/'hf500-formal2999-periodic64-seven-source-preview'
sys.path.insert(0,str(PREVIEW))
from formal_completion_check import verify
from training_evidence import COMMIT,digest,ONE,ZERO
from periodic_validation import verify_periodic_validations
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.base=Path(self.t.name);self.run=self.base/'runs/opsd-formal-2999';self.run.mkdir(parents=True)
  self.receipt=self.base/'receipt.json';self.roles={};self.expected={**{str(i):7000 for i in range(6)},'6':9713}
  self.periodic={'val/qa_count/all':64}
  for source,n in [('nq',10)]+[(str(i),9) for i in range(6)]:self.periodic['val/qa_count/'+source]=n;self.periodic['val/evaluated_count/'+source]=n
  for source in ('nq','all')+tuple(str(i) for i in range(6)):
   for kind in ('f1','em','answer_format','format_rate'):self.periodic[f'val/{kind}/{source}']=.123456
  self.raw=['BASELINE_EVALUATION_METRICS_JSON='+json.dumps(self.periodic),'step:0 - '+self.summary(self.periodic)]
  for step in range(1,501):
   metrics={'actor/optimizer_steps':10}
   for rank in range(8):
    for key in ONE:metrics[f'{key}/rank_{rank}']=1
    for key in ZERO:metrics[f'{key}/rank_{rank}']=0
    for key,value in {'actor/samples':320,'actor/optimizer_steps':10,'teacher/cuda_update_entries_cumulative_per_rank':step,'actor/supervised_tokens':1,'actor/grad_norm_mean':1,'actor/opsd_loss_mean':1}.items():metrics[f'{key}/rank_{rank}']=value
   if step%50==0:
    self.raw.append('BASELINE_EVALUATION_METRICS_JSON='+json.dumps(self.periodic));metrics.update(self.periodic)
   self.raw.append('step:'+str(step)+' - '+self.summary(metrics))
  metrics={'val/qa_count/all':51713}
  for source,count in self.expected.items():metrics['val/qa_count/'+source]=count;metrics['val/evaluated_count/'+source]=count
  for source in set(self.expected)|{'all'}:
   for kind in ('f1','em','answer_format','format_rate'):metrics[f'val/{kind}/{source}']=.5
  self.raw.extend(['BASELINE_EVALUATION_METRICS_JSON='+json.dumps(metrics),'SEARCH_R1_FINAL_METRICS_JSON='+json.dumps(metrics),'step:501 - '+self.summary(metrics)])
  self.put('raw_log','\n'.join(self.raw));self.put('source_commit',COMMIT);self.put('slurm_terminal','JobId=2999 JobState=COMPLETED ExitCode=0:0 NodeList=h100-10 NumNodes=1 NumCPUs=192 TresPerNode=gres:gpu:h100:8');self.put('exit','0');self.put('finished','FORMAL_500_OUTERS_AND_FINAL_EVALUATION_EXITED');self.put('expected_final_counts',json.dumps(self.expected))
  self.put('completeness',json.dumps(dict(passed=True,source_commit=COMMIT,actual_outer_updates=500,final_questions=51713,final_per_source_counts=self.expected,checkpoint_steps=[5,100,200,300,400,500],fresh_hf_reload_pending=True)))
  cp=self.run/'output/checkpoints/opsd-gpu-formal500/actor/global_step_500';cp.mkdir(parents=True)
  for n in ('config.json','tokenizer.json'): (cp/n).write_text('{}')
  for i in range(3):(cp/f'{i}.safetensors').write_text('synthetic not real HF')
  (cp/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{str(i):'0.safetensors' for i in range(435)}}));self.seal()
 def summary(self,metrics):return ' - '.join(k+':'+format(v,'.3f') for k,v in metrics.items())
 def tearDown(self):self.t.cleanup()
 def put(self,key,text):
  p=self.run/key;p.write_text(text);self.roles[key]={'path':str(p),'sha256':digest(p)}
 def seal(self):
  review={'accepted':True,'source_commit':COMMIT,'formal_job':'2999','reviewer':'synthetic unit fixture','report':'synthetic','bound_evidence_sha256':{k:v['sha256'] for k,v in self.roles.items() if k!='independent_review'}}
  self.put('independent_review',json.dumps(review));self.receipt.write_text(json.dumps(dict(source_commit=COMMIT,formal_job='2999',evidence=self.roles)))
 def check(self):return verify(self.receipt,self.base)
 def test_valid_synthetic_structure_only(self):self.assertEqual(self.check()['actual_outer_updates'],500)
 def test_missing_terminal(self):
  self.put('slurm_terminal','JobId=2999 ExitCode=0:0');self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_running_not_complete(self):
  self.put('slurm_terminal','JobId=2999 JobState=RUNNING ExitCode=0:0');self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_wrong_training_node(self):
  self.put('slurm_terminal','JobId=2999 JobState=COMPLETED ExitCode=0:0 NodeList=h100-11 NumNodes=1 NumCPUs=192 TresPerNode=gres:gpu:h100:8');self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_incomplete_hf_index(self):
  p=self.run/'output/checkpoints/opsd-gpu-formal500/actor/global_step_500/model.safetensors.index.json';p.write_text('{"weight_map": {}}')
  with self.assertRaises(AssertionError):self.check()
 def test_failed_exit(self):
  self.put('exit','1');self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_499_updates(self):
  self.put('raw_log','\n'.join(self.raw[:-2]+self.raw[-1:]));self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_self_pass_without_raw(self):
  self.put('raw_log','');self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_wrong_source(self):
  self.put('source_commit','a'*40);self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_missing_final(self):
  self.put('raw_log','\n'.join(self.raw[:-1]));self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_duplicate_final(self):
  self.put('raw_log','\n'.join(self.raw+self.raw[-1:]));self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_bad_hash(self):
  self.roles['raw_log']['sha256']='0'*64;self.seal()
  with self.assertRaises(AssertionError):self.check()
 def test_review_unbound(self):
  self.put('raw_log','\n'.join(self.raw)+'\n');self.receipt.write_text(json.dumps(dict(source_commit=COMMIT,formal_job='2999',evidence=self.roles)))
  with self.assertRaises(AssertionError):self.check()
 def test_preview_frozen_hf_bytes_and_gate(self):
  old=PREVIEW.parents[1]/'deployment-bundles/hf-runtime-8e60976d364a-pilot-2994-reviewed-47cab4b02661'
  for n in ('checkpoint_reload.py','resource_sample.py'):self.assertEqual((old/n).read_bytes(),(PREVIEW/n).read_bytes())
  shell=(PREVIEW/'reload_job.sh').read_text();self.assertIn('exit 78',shell);self.assertIn('OPSD_FORMAL_COMPLETION_RECEIPT',shell);self.assertIn('global_step_500',shell);self.assertNotIn('job_2994',shell);self.assertNotIn('afterok',shell)
  subprocess.run(['bash','-n',str(PREVIEW/'reload_job.sh')],check=True)
if __name__=='__main__':unittest.main()
