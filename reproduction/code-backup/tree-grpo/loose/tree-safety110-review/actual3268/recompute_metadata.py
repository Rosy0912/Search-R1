import json,hashlib,math,os
from pathlib import Path
from omegaconf import OmegaConf
W=Path.cwd();O=W/'work/tree-safety110-review/actual3268';P=W/'work/tree-safety110-a632ba9-v2';m=json.loads((P/'manifest.json').read_bytes());v=json.loads((O/'METADATA1.json').read_bytes());f=v['files'];H=lambda b:hashlib.sha256(b).hexdigest();config='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b'
for row in f.values():assert H(row['text'].encode())==row['sha256']
assert f['resolved.yaml']['sha256']==f['output/entry-audit/entry-config.yaml']['sha256']==config
c=OmegaConf.create(f['resolved.yaml']['text']);entry=json.loads(f['output/entry-audit/entry.jsonl']['text']);assert entry['actual_config_sha256']==entry['config_sha256']==config and entry['commit']==m['code_commit'] and entry['actual_data_identity']==m['common_data_identity'] and entry['run_id']=='tree-safety300-3268'
assert c.data.train_batch_size==640 and c.actor_rollout_ref.actor.ppo_mini_batch_size==384 and c.actor_rollout_ref.actor.ppo_micro_batch_size==16
assert c.baselines.holdout.manifest_sha256==m['common_data_identity'] and c.actor_rollout_ref.ref.model_path==c.actor_rollout_ref.model.path and c.trainer.total_training_steps==301 and c.trainer.acceptance_step==115
assert c.actor_rollout_ref.rollout.sampling_consistency.enabled and c.actor_rollout_ref.rollout.sampling_consistency.enforce is False
stage=f['stage.log']['text'];installer=[s for s in stage.splitlines() if 'TREE_VALIDATION_COST_INSTALLED_JSON=' in s];assert len(installer)==1
s=installer[0].split('TREE_VALIDATION_COST_INSTALLED_JSON=',1)[1];decoder=json.JSONDecoder();ins,_=decoder.raw_decode(s);assert ins['job_id']=='3268' and ins['calibration_only']=='1' and ins['trainer_source_sha256']==m['files']['source/repo/verl/trainer/ppo/ray_trainer.py'] and ins['observer_source_sha256']==m['files']['validation_cost_hooks_instruct300.py'] and ins['output'].endswith('/tree-safety300-3268/output') and isinstance(ins['pid'],int)
r=json.loads(f['retrieval-gate.json']['text']);bpath=W/'outputs/evidence/shared-original-gpu-2854.json';b=json.loads(bpath.read_bytes());assert r['baseline_sha256']==H(bpath.read_bytes()) and r['source_sha256']==m['external_files']['/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/probe_tree_shared_retrieval.py'];assert r['asset_identity_verified'] and r['actual_asset_identity']['index']==b['asset_identity']['index'] and r['actual_asset_identity']['corpus']==b['asset_identity']['corpus']
fixed={};measures=[];maxdelta=0
for q in r['measurements']:
 k=q['k'];n=q['queries'];rows=q['results'];key=(q.get('label','fixed14'),k);assert len(rows)==n and q['wall_seconds']>0
 if q['repeat']==0:fixed[key]=rows
 else:assert fixed[key]==rows
 for i,a in enumerate(rows):
  assert len(a['corpus_ids'])==len(a['scores'])==len(a['contents_sha256'])==k and len(set(a['corpus_ids']))==k and all(math.isfinite(z) for z in a['scores'])
  if n==14 or 'repeated' in key[0]:
   ref=b['results'][i%14][str(k)];assert a['corpus_ids']==ref['corpus_ids'] and a['contents_sha256']==ref['contents_sha256'];maxdelta=max(maxdelta,max(abs(x-y) for x,y in zip(a['scores'],ref['scores'])))
 measures.append(dict(label=key[0],k=k,n=n,repeat=q['repeat'],wall_seconds=q['wall_seconds']))
assert r['over_512_rows']==513
samples=[json.loads(s) for s in f['resource-samples.jsonl']['text'].splitlines()];assert all(x['job_id']=='3268' and not x.get('event') and len(x['gpus'])==8 for x in samples)
for x in samples:
 for g in x['gpus']:assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
out=dict(status='ACTUAL_STARTUP_ENTRY_RETRIEVAL_GUARD_PREFIX_PASS_ONLY',config_sha256=config,actual_ray_installer=ins,fresh_retrieval_reference_equal_ID_and_content=True,max_matched_score_delta=maxdelta,measurements=measures,guard_prefix_samples=len(samples),guard_last_epoch=samples[-1]['epoch'],sampled_total_gpu_peak_GiB=max(g['total_used_memory_bytes'] for x in samples for g in x['gpus'])/2**30,actual_final_cleanup_accepted=False,scope='Startup prefix only; no complete zero-write observation or terminal claim')
(O/'STARTUP_METADATA_PROOF.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='measurements'}))
