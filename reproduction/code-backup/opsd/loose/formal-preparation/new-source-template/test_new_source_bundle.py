import copy,json,sys,tempfile,unittest,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from micro_evidence import validate_capacity_record,verify_micro_evidence
from render_new_pin import render
from render_staged import pilot,hf,replace_once
from finalize_reviewed import finalize
from verify_formal_prerequisites import RUNTIME_NAMES
EVIDENCE=HERE.parents[1]/'opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2960_checkpoint_capacity_reviewer_raw.json'
RAW=json.loads(EVIDENCE.read_text())
SUMMARY=RAW['micro-capacity.json']['value'];RECORD=RAW['micro-2.json']['value']
class Tests(unittest.TestCase):
 def test_actual_e43_fixture_only(self):validate_capacity_record(SUMMARY,RECORD,'2960',SUMMARY['source_commit'],2)
 def test_mutated_all8_rejected(self):
  mutations=[('world_size',7),('input_width',4096),('response_width',2048),('optimizer_steps',1),('minimum_required_free_fraction',.1),('passed',False),('allocator_backend','cudaMallocAsync'),('student_master_parameter_dtypes',['torch.bfloat16'])]
  for key,val in mutations:
   r=copy.deepcopy(RECORD);r['ranks'][7][key]=val
   with self.subTest(key=key),self.assertRaises(AssertionError):validate_capacity_record(SUMMARY,r,'2960',SUMMARY['source_commit'],2)
 def test_old_source_self_flags_rejected(self):
  with self.assertRaises(AssertionError):validate_capacity_record(SUMMARY,RECORD,'2960','a'*40,2)
  with self.assertRaises(AssertionError):validate_capacity_record(SUMMARY,RECORD,'9999',SUMMARY['source_commit'],2)
 def test_missing_rank_teacher_adam_headroom(self):
  changes=[lambda r:r['ranks'].pop(),lambda r:r['ranks'][0]['teacher_active_runtime'].update(attention='sdpa'),lambda r:r['ranks'][0]['optimizer_step_records'][1].update(tracked_parameter_delta=0),lambda r:r['ranks'][0].update(estimated_peak_device_used_bytes=r['ranks'][0]['gpu_total_bytes']),lambda r:r['ranks'][0]['decoder_forward_pre_hook_counts'].update({'0':2})]
  for change in changes:
   r=copy.deepcopy(RECORD);change(r)
   with self.assertRaises(AssertionError):validate_capacity_record(SUMMARY,r,'2960',SUMMARY['source_commit'],2)
 def test_declared_micro4_cannot_reuse_micro2(self):
  with self.assertRaises(AssertionError):validate_capacity_record(SUMMARY,RECORD,'2960',SUMMARY['source_commit'],4)
 def profile(self):
  snapshots=HERE.parents[1]/'runtime-20261003'
  return dict(source_commit='a'*40,pilot_job='9991',reload_job='9992',pilot_run_name='opsd-acceptance-9991',reload_run_name='opsd-hf-reload-9992',pilot_checkpoint_experiment='opsd-new-fresh5',picked_local_micro=4,reload_wrapper_sha256=hashlib.sha256((snapshots/'reload_job.sh').read_bytes()).hexdigest(),completed_prior_jobs=['2960','2962','2984','2987'],production_math_sources={n:hashlib.sha256((HERE.parents[1]/'opsd'/n).read_bytes()).hexdigest() for n in ('search_r1/baselines/actor.py','search_r1/baselines/distillation.py')},runtime_snapshot_directory=str(snapshots),runtime_sources={n:hashlib.sha256((snapshots/n).read_bytes()).hexdigest() for n in RUNTIME_NAMES})
 def test_unfilled_rejected(self):
  with tempfile.TemporaryDirectory() as d,self.assertRaises(AssertionError):render(HERE/'source-profile.UNFILLED.json',Path(d)/'out')
 def test_preview_isolated_no_deploy(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'profile.json';profile=self.profile()
   # Pilot stage must work without ANY future acceptance/job/micro declaration.
   source={k:v for k,v in profile.items() if k not in ('pilot_job','reload_job','pilot_run_name','reload_run_name','pilot_checkpoint_experiment','picked_local_micro')}
   p.write_text(json.dumps(source));po=Path(d)/'pilot';pm=pilot(p,po)
   before={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in po.iterdir() if x.is_file()}
   hp={**source,'pilot_job':'9991','pilot_runtime_manifest':str(po/'source-manifest.json')};p.write_text(json.dumps(hp));ho=Path(d)/'hf';hm=hf(p,ho)
   formal={**source,'pilot_job':'9991','reload_job':'9992','picked_local_micro':4,'pilot_runtime_manifest':str(po/'source-manifest.json'),'hf_runtime_manifest':str(ho/'source-manifest.json')}
   p.write_text(json.dumps(formal));out=Path(d)/'formal';m=render(p,out)
   assert m['picked_local_micro']==4 and m['picked_global_micro']==32
   submission=(out/'submit_formal.sh').read_text()
   assert '--dependency' not in submission and '--kill-on-invalid-dep' not in submission
   assert m['formal_sources']['submit_formal.sh']==hashlib.sha256((out/'submit_formal.sh').read_bytes()).hexdigest()
   assert 'OPSD_ACCEPTANCE_RECEIPT' in submission and 'OPSD_STAGE_MANIFEST_SHA256' in submission
   verifier=(out/'verify_formal_prerequisites.py').read_text()
   assert "else:verify_slurm(slurm_text,job)" in verifier
   assert "verify_reload(loads(paths['reload_result'].read_text())" in verifier
   assert 'capacity_micro >= 4' in (out/'formal_supervisor.py').read_text()
   assert 'verify_micro_evidence(review,pilot' in (out/'verify_formal_prerequisites.py').read_text()
   for stage in (po,ho,out):
    for path in stage.rglob('*.py'):compile(path.read_text(),str(path),'exec')
   for path in (po/'gpu_job_entry.sh',ho/'reload_job.sh',out/'formal_entry.sh'):
    assert 'exit 78' in path.read_text();subprocess.run(['bash','-n',str(path)],check=True)
   assert 'formal' not in (po/'gpu_job_entry.sh').read_text()
   assert 'source_bundle_check.py' in (ho/'reload_job.sh').read_text()
   assert before=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in po.iterdir() if x.is_file()}
   assert hashlib.sha256((po/'gpu_teacher_probe-factor-dtype.py').read_bytes()).hexdigest()=='e98dbb92850d4e5b89a17209e209f1c7bcbc9dcf3712981d2b1b6e98ddf044dc'
   with self.assertRaises(AssertionError):pilot(p,po)
 def test_dependency_removal_fails_closed_on_template_change(self):
  target=' --dependency=afterok:2962 --kill-on-invalid-dep=yes '+chr(92)+'\n'
  template=(HERE.parent/'submit_formal.sh').read_text()
  assert template.count(target)==1
  assert '--dependency' not in replace_once(template,target,'')
  for broken in (template.replace(target,''),template+target):
   with self.assertRaises(AssertionError):replace_once(broken,target,'')
 def test_formal_requires_actual_ids_and_micro(self):
  with tempfile.TemporaryDirectory() as d:
   profile=self.profile();profile.pop('pilot_job');p=Path(d)/'p.json';p.write_text(json.dumps(profile))
   with self.assertRaises(KeyError):render(p,Path(d)/'out')
   profile=self.profile();profile.pop('picked_local_micro');p.write_text(json.dumps(profile))
   with self.assertRaises(KeyError):render(p,Path(d)/'out2')
 def test_snapshot_byte_tampering_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   profile=self.profile();profile['runtime_sources']['job.sh']='0'*64;p=Path(d)/'p.json';p.write_text(json.dumps(profile))
   with self.assertRaises(AssertionError):pilot(p,Path(d)/'out')
class YAMLTests(unittest.TestCase):
 def test_actual_yaml_micro_and_hash_binding(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);candidate=root/'micro-2.json';candidate.write_text(json.dumps(RECORD))
   summary=copy.deepcopy(SUMMARY);selected=next(x for x in summary['candidates'] if x['local_micro']==2);selected.update(aggregate_json=str(candidate),aggregate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
   sp=root/'micro-capacity.json';sp.write_text(json.dumps(summary))
   config={'actor_rollout_ref':{'model':{'path':'/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets/qwen2.5-3b'},'actor':{'ppo_micro_batch_size':16,'ppo_mini_batch_size':256}},'trainer':{'total_training_steps':6,'smoke_only':True,'val_before_train':True,'test_freq':5},'data':{'val_batch_size':64,'train_batch_size':512},'baselines':{'distillation':{'teacher_path':'/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets/qwen2.5-3b','teacher_device':'cuda','loss_checkpointing':True},'local_runtime_only':True,'local_runtime_host':'h100-10'},'retriever':{'url':'http://127.0.0.1:18084/retrieve'}}
   cp=root/'config.yaml';cp.write_text(json.dumps(config))
   def review():return dict(picked_local_micro=2,picked_global_micro=16,auxiliary_bound_artifacts={role:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for role,p in (('capacity_summary',sp),('capacity_candidate',candidate),('pilot_config',cp))})
   verify_micro_evidence(review(),root,'2960',SUMMARY['source_commit'],2)
   config['actor_rollout_ref']['actor']['ppo_micro_batch_size']=32;cp.write_text(json.dumps(config))
   with self.assertRaises(AssertionError):verify_micro_evidence(review(),root,'2960',SUMMARY['source_commit'],2)
   config['actor_rollout_ref']['actor']['ppo_micro_batch_size']=16;cp.write_text(json.dumps(config));r=review();cp.write_text('tampered')
   with self.assertRaises(AssertionError):verify_micro_evidence(r,root,'2960',SUMMARY['source_commit'],2)

class FixtureCompatibilityTests(unittest.TestCase):
 def test_two_frozen_chunks_have_supervision(self):
  mask=[1]*25;mask[8:16]=[0]*8
  counts=[sum(mask[i:i+16]) for i in range(0,25,16)]
  self.assertEqual(counts,[8,9]);self.assertEqual([i for i,n in enumerate(counts) if n],[0,1])
 def test_all_empty_chunk_is_a_different_fixture(self):
  mask=[0]*16+[1]*9;counts=[sum(mask[i:i+16]) for i in range(0,25,16)]
  self.assertEqual(counts,[0,9]);self.assertNotEqual([i for i,n in enumerate(counts) if n],[0,1])



class ExternalPinTests(unittest.TestCase):
 profile=Tests.profile
 def stage(self,root):
  profile=self.profile();p=root/'profile.json';p.write_text(json.dumps(profile));out=root/'preview';pilot(p,out)
  import importlib.util
  spec=importlib.util.spec_from_file_location('pinned_checker',out/'source_bundle_check.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return out,module
 def test_valid_missing_malformed_external_pin(self):
  with tempfile.TemporaryDirectory() as d:
   out,module=self.stage(Path(d));pin=hashlib.sha256((out/'source-manifest.json').read_bytes()).hexdigest()
   module.check_manifest_pin(out,{'OPSD_STAGE_MANIFEST_SHA256':pin})
   for value in ('','0'*63,'z'*64,'0'*64):
    with self.subTest(value=value),self.assertRaises(AssertionError):module.check_manifest_pin(out,{'OPSD_STAGE_MANIFEST_SHA256':value})
   with self.assertRaises(AssertionError):module.check_manifest_pin(out,{})
 def test_joint_file_and_manifest_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   out,module=self.stage(Path(d));pin=hashlib.sha256((out/'source-manifest.json').read_bytes()).hexdigest()
   file=out/'job.sh';file.write_text(file.read_text()+'\n# accidental change\n')
   mp=out/'source-manifest.json';m=json.loads(mp.read_text());m['file_sources']['job.sh']=hashlib.sha256(file.read_bytes()).hexdigest();m['runtime_sources']['job.sh']=m['file_sources']['job.sh'];mp.write_text(json.dumps(m))
   with self.assertRaises(AssertionError):module.check_manifest_pin(out,{'OPSD_STAGE_MANIFEST_SHA256':pin})
 def test_enable_reviewed_regenerates_new_immutable_output(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);preview,module=self.stage(root);pin=hashlib.sha256((preview/'source-manifest.json').read_bytes()).hexdigest();before={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in preview.iterdir() if x.is_file()}
   final=root/'final';m=finalize(preview,pin,final)
   assert m['preview_blocked'] is False and m['directory_name']!=json.loads((preview/'source-manifest.json').read_text())['directory_name']
   assert 'exit 78' not in (final/'gpu_job_entry.sh').read_text()
   assert before=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in preview.iterdir() if x.is_file()}
   with self.assertRaises(AssertionError):finalize(preview,pin,final)
 def test_missing_duplicate_critical_template_rejected(self):
  for text in ('absent','target target'):
   with self.assertRaises(AssertionError):replace_once(text,'target','replacement')
  assert replace_once('one target','target','replacement')=='one replacement'


class FormalExternalPinTests(unittest.TestCase):
 def make(self,root):
  profile=Tests().profile();p=root/'p.json';p.write_text(json.dumps(profile));po=root/'pilot';pilot(p,po)
  profile['pilot_runtime_manifest']=str(po/'source-manifest.json');p.write_text(json.dumps(profile));ho=root/'hf';hf(p,ho)
  profile['hf_runtime_manifest']=str(ho/'source-manifest.json');p.write_text(json.dumps(profile));out=root/'formal';m=render(p,out)
  import importlib.util,shutil
  own=root/'ownroot';directory=own/'tools'/m['formal_directory_name'];directory.parent.mkdir(parents=True);shutil.copytree(out,directory)
  spec=importlib.util.spec_from_file_location('formal_checker',directory/'verify_formal_prerequisites.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return own,directory,module
 def test_formal_valid_missing_wrong_external_pin(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   own,folder,module=self.make(Path(d));pin=hashlib.sha256((folder/'source-manifest.json').read_bytes()).hexdigest()
   with patch.dict('os.environ',{'OPSD_STAGE_MANIFEST_SHA256':pin},clear=True):module.load_reviewed_formal_manifest(own)
   for value in ('','0'*64,'z'*64):
    with patch.dict('os.environ',{'OPSD_STAGE_MANIFEST_SHA256':value},clear=True),self.subTest(value=value),self.assertRaises(AssertionError):module.verify_deployed(own)
   assert 'OPSD_STAGE_MANIFEST_SHA256="$OPSD_STAGE_MANIFEST_SHA256"' in (folder/'submit_formal.sh').read_text()
   assert ': "${OPSD_STAGE_MANIFEST_SHA256:?' in (folder/'formal_entry.sh').read_text()
   assert 'verify_deployed()' in (folder/'collect_receipt.py').read_text()
 def test_formal_joint_file_manifest_tamper(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   own,folder,module=self.make(Path(d));mp=folder/'source-manifest.json';pin=hashlib.sha256(mp.read_bytes()).hexdigest()
   f=folder/'formal_job.sh';f.write_text(f.read_text()+'\n# accidental modification\n');m=json.loads(mp.read_text());m['formal_sources'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest();mp.write_text(json.dumps(m))
   with patch.dict('os.environ',{'OPSD_STAGE_MANIFEST_SHA256':pin},clear=True),self.assertRaises(AssertionError):module.verify_deployed(own)
if __name__=='__main__':unittest.main()
