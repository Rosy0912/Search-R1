import json,tempfile,unittest
from pathlib import Path
from verify_formal_prerequisites import *
def raw(count):
 lines=[]
 for step in range(1,count+1):
  v={'actor/optimizer_steps':10}
  for rank in range(8):
   data={**dict.fromkeys(ONE,1),**dict.fromkeys(ZERO,0),'actor/samples':320,'actor/optimizer_steps':10,'teacher/cuda_update_entries_cumulative_per_rank':step,'actor/supervised_tokens':12,'actor/grad_norm_mean':.1,'actor/opsd_loss_mean':.01}
   v.update({f'{k}/rank_{rank}':x for k,x in data.items()})
  lines.append(f'step:{step} - '+' - '.join(f'{k}:{x}' for k,x in v.items()))
 return '\n'.join(lines)
def valmetrics():
 d={'val/qa_count/all':64,'val/qa_count/nq':40,'val/qa_count/triviaqa':24,'val/evaluated_count/nq':40,'val/evaluated_count/triviaqa':24}
 for name in ('all','nq','triviaqa'):
  for k in ('f1','em','answer_format','format_rate'):d[f'val/{k}/{name}']=.5
 return d
def pilot_validations(metrics=None):
 v=valmetrics() if metrics is None else metrics
 payload='BASELINE_EVALUATION_METRICS_JSON='+json.dumps(v)
 summary=' - '.join(f'{k}:{x}' for k,x in v.items())
 steps=raw(5).splitlines()
 return payload+'\nstep:0 - '+summary+'\n'+'\n'.join(steps[:-1])+'\n'+payload+'\n'+steps[-1]+' - '+summary
class Tests(unittest.TestCase):
 def test_good_all8_five(self):self.assertEqual(verify_outers(raw(5),5)['ranks'],8)
 def test_good_all8_500(self):verify_outers(raw(500),500)
 def test_reject_duplicate_step(self):
  with self.assertRaises(AssertionError):verify_outers(raw(5)+'\n'+raw(1),5)
 def test_reject_missing_step(self):
  with self.assertRaises(AssertionError):verify_outers(raw(4),5)
 def test_reject_each_audit(self):
  keys=ONE+ZERO+('actor/samples','actor/optimizer_steps','teacher/cuda_update_entries_cumulative_per_rank','actor/supervised_tokens','actor/grad_norm_mean')
  for k in keys:
   with self.subTest(k=k):
    bad=raw(5).replace(f'{k}/rank_7:',f'missing_{k}/rank_7:')
    with self.assertRaises(KeyError):verify_outers(bad,5)
 def test_reject_old_counter(self):
  with self.assertRaises(AssertionError):verify_outers(raw(5).replace('teacher/cuda_update_entries_cumulative_per_rank/rank_0:2','teacher/cuda_update_entries_cumulative_per_rank/rank_0:1'),5)
 def test_reject_duplicate_metric(self):
  with self.assertRaises(AssertionError):verify_outers(raw(1)+' - actor/samples/rank_0:320',1)
 def test_strict_json(self):
  for s in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}','{"x":[Infinity]}'):
   with self.subTest(s=s),self.assertRaises(AssertionError):loads(s)
 def test_escape_symlink(self):
  with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as other:
   base=Path(d);external=Path(other,'x');external.write_text('x');p=base/'link';p.symlink_to(external)
   with self.assertRaises(AssertionError):own_file(str(p),base)
 def test_self_flags_cannot_replace_roles(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);p=base/'receipt.json';p.write_text(json.dumps(dict(source_commit=COMMIT,pilot_job='2960',reload_job='2962',completed_actual_outer_updates=5,fresh_hf_reload_passed=True,independent_acceptance_passed=True,evidence={})))
   with self.assertRaises(AssertionError):verify_receipt(p,base)
 def test_manifest_missing_names(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);(base/'tools/formal').mkdir(parents=True);(base/'tools/formal/source-manifest.json').write_text(json.dumps(dict(source_commit=COMMIT,runtime_sources={},formal_sources={})))
   with self.assertRaises(AssertionError):verify_deployed(base)
 def test_manifest_real_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);data={'source_commit':COMMIT,'runtime_sources':{},'formal_sources':{}}
   for field,names,directory in (('runtime_sources',RUNTIME_NAMES,'runtime'),('formal_sources',FORMAL_NAMES,'formal')):
    (base/'tools'/directory).mkdir(parents=True,exist_ok=True)
    for name in names:
     p=base/'tools'/directory/name;p.write_text(name);data[field][name]=digest(p)
   (base/'tools/formal/source-manifest.json').write_text(json.dumps(data));verify_deployed(base)
   (base/'tools/runtime/checkpoint_reload.py').write_text('altered')
   with self.assertRaises(AssertionError):verify_deployed(base)
class CompletionTests(unittest.TestCase):
 def test_completed_slurm(self):verify_slurm('JobId=2960 JobState=COMPLETED ExitCode=0:0','2960')
 def test_not_completed(self):
  for state,exit in (('RUNNING','0:0'),('PENDING','0:0'),('CANCELLED','0:15'),('COMPLETED','1:0')):
   with self.subTest(state=state),self.assertRaises(AssertionError):verify_slurm(f'JobId=2960 JobState={state} ExitCode={exit}','2960')
 def test_good_pilot_both_validations(self):verify_pilot_validations(pilot_validations())
 def test_missing_step0_or_step5_validation(self):
  for step in (0,5):
   text=pilot_validations();lines=text.splitlines()
   lines=[line.split(' - val/',1)[0] if line.startswith(f'step:{step} - ') else line for line in lines]
   with self.subTest(step=step),self.assertRaises((KeyError,AssertionError)):verify_pilot_validations('\n'.join(lines))
 def test_bad_counts_or_metrics(self):
  for key,value in (('val/qa_count/all',63),('val/qa_count/nq',39),('val/evaluated_count/nq',39),('val/f1/all',float('nan')),('val/em/nq',float('inf')),('val/format_rate/triviaqa',1.1)):
   v=valmetrics();v[key]=value
   with self.subTest(key=key),self.assertRaises(AssertionError):verify_pilot_validations(pilot_validations(v))
 def test_json_without_step_bound_summary(self):
  with self.assertRaises(AssertionError):verify_pilot_validations(pilot_validations().replace('step:5 - ','step:4 - '))
 def test_missing_json(self):
  with self.assertRaises(AssertionError):verify_pilot_validations('\n'.join(x for x in pilot_validations().splitlines() if not x.startswith('BASELINE_EVALUATION_METRICS_JSON=')))
if __name__=='__main__':unittest.main()
