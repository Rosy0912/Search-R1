import unittest,pathlib,importlib.util,json,copy,math
B=pathlib.Path(__file__).resolve().parent;CWD=B.parent.parent;s=importlib.util.spec_from_file_location('w',B/'final3292-readonly.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
REF=CWD/'work/e1-300-stabilityfix-r2/step5-extract.py';PREFIX=CWD/'work/e1-300-recovery3231/recovery-inputs/train-prefix1-25.log'
def synthetic():
 raw=PREFIX.read_bytes();d=w.records(raw,'BAPO_TRAIN_METRICS_JSON=')[0]['event'];r=[x['event']for x in w.records(raw,'ACTOR_RANK_UPDATE_JSON=')if x['event']['scheduler_last_epoch']==1];train=[];ranks=[]
 for step in range(1,301):
  e=copy.deepcopy(d);e['step']=step;m=e['metrics'];m.update({'bapo/actual_outer_updates':step,'bapo/stage_validation':.5,'bapo/stalled':step-1,'bapo/early_stage':int(step<6),'actor/lr':1e-6*min(step/114,1),'actor/lr_before_outer':1e-6*min((step-1)/114,1),'actor/scheduler_outer_steps':step});train.append(e)
  for old in r:
   x=copy.deepcopy(old);x.update(scheduler_last_epoch=step,lr_before=m['actor/lr_before_outer'],lr_after=m['actor/lr']);x['metrics'].update({'actor/lr':m['actor/lr'],'actor/lr_before_outer':m['actor/lr_before_outer'],'actor/scheduler_outer_steps':step});ranks.append(x)
 return train,ranks
class T(unittest.TestCase):
 def test_actual25_prefix_oracle(self):
  a,b=w.scope_functions(REF);raw=PREFIX.read_bytes();d=[x['event']for x in w.records(raw,'BAPO_TRAIN_METRICS_JSON=')];r=[x['event']for x in w.records(raw,'ACTOR_RANK_UPDATE_JSON=')];self.assertEqual(len(b(d,r)),25);self.assertIsInstance(a(d,r),dict)
 def test_300_cap(self):
  a,b=w.scope_functions(REF);d,r=synthetic();a(d,r);self.assertEqual(len(b(d,r)),300);self.assertEqual(r[-1]['lr_after'],1e-6)
 def test_unsaturated_lr_rejected(self):
  a,_=w.scope_functions(REF);d,r=synthetic();r[-1]['lr_after']=1e-6*300/114
  with self.assertRaises(ValueError):a(d,r)
 def test_missing_duplicate_rank(self):
  a,_=w.scope_functions(REF);d,r=synthetic();r[-1]=copy.deepcopy(r[-2])
  with self.assertRaises(ValueError):a(d,r)
 def test_nonfinite_raw_mini(self):
  a,_=w.scope_functions(REF);d,r=synthetic();r[-1]['metrics']['actor/pg_loss'][0]=float('nan')
  with self.assertRaises(ValueError):a(d,r)
 def test_driver_wrong_reduction(self):
  _,b=w.scope_functions(REF);d,r=synthetic();d[-1]['metrics']['actor/pg_loss']+=1
  with self.assertRaises(ValueError):b(d,r)
 def test_terminal_running_or_wrongjob(self):
  raw=b'JobId=3292 JobName=bapo-instruct300-formal300 JobState=COMPLETED ExitCode=0:0 DerivedExitCode=0:0 Restarts=0 NodeList=h100-5 NumNodes=1 TresPerNode=gres:gpu:h100:8 JOB_GRES=gpu:h100:8 TRES=cpu=32,node=1,gres/gpu=8 RunTime=01:02:03'
  self.assertEqual(w.terminal(raw)['reserved_seconds'],3723)
  self.assertEqual(w.terminal(raw.replace(b'TRES=cpu=32,node=1,gres/gpu=8',b'TRES=cpu=192,mem=512G,node=1,billing=192'))['gpu_count'],8)
  for bad in [raw.replace(b'COMPLETED',b'RUNNING'),raw.replace(b'3292',b'3287'),raw+b' Restarts=0',raw.replace(b'gres/gpu=8',b'gres/gpu=7')]:
   with self.assertRaises(ValueError):w.terminal(bad)
 def finalraw(self,mutate=None):
  d={'val/qa_count/all':51713}
  for src,count in w.COUNTS.items():
   d['val/qa_count/'+src]=count;d['val/evaluated_count/'+src]=count
  for src in ['all']+list(w.COUNTS):
   for key in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):d['val/'+key+'/'+src]=.5
  if mutate:mutate(d)
  j=json.dumps(d);return ('BASELINE_EVALUATION_METRICS_JSON='+j+'\nSEARCH_R1_FINAL_METRICS_JSON='+j+'\n').encode()
 def test_final_actual_aggregate_scope(self):self.assertFalse(w.final_metrics(self.finalraw())['per_answer_recomputed'])
 def test_final_missing_counts_and_wrong_weighted(self):
  for mut in [lambda d:d.update({'val/qa_count/nq':64}),lambda d:d.update({'val/f1/all':.8}),lambda d:d.update({'val/em/all':float('nan')})]:
   with self.assertRaises(ValueError):w.final_metrics(self.finalraw(mut))
 def test_duplicate_final_and_jsonkeys(self):
  raw=self.finalraw()
  with self.assertRaises(ValueError):w.final_metrics(raw+raw)
  with self.assertRaises(ValueError):w.strictjson('{"x":1,"x":2}')
if __name__=='__main__':unittest.main()
