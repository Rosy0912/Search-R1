"""Independently compare downloaded allocated inputs; no remote/process/GPU calls."""
from pathlib import Path
import json,hashlib,re,yaml
B=Path(__file__).resolve().parent;E=B.parent/'evidence';ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');RUN=ROOT/'runs/bapo-instruct300-hardware-3239';SOURCE='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((E/'recovery3231-hardware3239-allocation-files.json').read_text())
for n,x in a.items():
 assert x['path']==str(RUN/n);raw=x['text'].encode();assert len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256']
m=json.loads(a['recovery-manifest.json']['text']);assert m['target_job_id']==3239 and m['target_source_commit']==SOURCE and m['target_run_root']==str(RUN)
assert m['step']==m['actual_outer_updates']==25 and m['next_step']==26 and m['world_size']==8 and m['old_source_commit']=='39cdcdd1404213b897bb0de99cc52d72707682f0'
c=json.loads((E/'formal3231-checkpoint25-fullhash-cpu-candidate.json').read_text());old=Path(m['old_run_root']);bind=[x for r in m['ranks']for x in (r['state'],r['report'])]+[m['driver'],m['driver_report'],m['algorithm']]+m['hf_files'];assert len(bind)==31
assert {x['path']:(x['sha256'],x['bytes'])for x in bind}=={x['path']:(x['sha256'],x['bytes'])for x in c['files']};assert c['generation_rng_in_old_checkpoint']is False and c['sampler_cursor']==12800
rawroot=E/'formal3231-fault-raw-20261005T0918/runs/bapo-instruct300-formal300-3231';cfg=rawroot/'resolved-config.yaml';assert m['resolved_config']['sha256']==h(cfg)==c['config_sha256'];config=yaml.safe_load(cfg.read_text())
assert config['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1 and config['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and config['trainer']['total_training_steps']==301
assert config['baselines']['bapo']['stage_validation_size']==64 and config['data']['train_batch_size']==512 and config['actor_rollout_ref']['actor']['ppo_mini_batch_size']==128 and config['actor_rollout_ref']['actor']['ppo_micro_batch_size']==16
prefix=B/'recovery-inputs/train-prefix1-25.log';assert h(prefix)==m['train_prefix']['sha256']and prefix.stat().st_size==m['train_prefix']['bytes'];assert (rawroot/'train.log').read_bytes()[:prefix.stat().st_size]==prefix.read_bytes()
steps=[json.JSONDecoder().raw_decode(l.split('BAPO_TRAIN_METRICS_JSON=',1)[1].lstrip())[0]['step']for l in prefix.read_text().splitlines()if'BAPO_TRAIN_METRICS_JSON='in l];assert steps==list(range(1,26))
assert a['source-commit.txt']['text'].strip()==a['cpu-gate-source.txt']['text'].strip()==SOURCE
for name,local in [('cpu-gate.json','actual-source-linux-cpu-record.json'),('tools-manifest.json','tools-manifest.json'),('launch-approval.json','launch-approval.json')]:assert a[name]['sha256']==h(B/local)
cpu=json.loads(a['cpu-gate.json']['text']);assert cpu['passed']==395 and cpu['failed']==cpu['skipped']==cpu['exit_code']==0 and cpu['sys_prefix']==str(ROOT/'env')
for stage in ('before','after'):assert 'search_r1.baselines.recovery'in cpu['import_origins'][stage]['modules']
origin={}
for label in ('before','after'):
 marker='CPU_IMPORT_ORIGINS_'+label.upper()+'_JSON=';rows=[json.loads(l.split(marker,1)[1])for l in a['runtime-import-origins.log']['text'].splitlines()if marker in l];assert len(rows)==1;origin[label]=rows[0];assert rows[0]['source_commit']==SOURCE and rows[0]['repo']==str(ROOT/'repo')
assert origin['before']['functions']==origin['after']['functions']
repo=B.parent/'bapo-e1-300-recovery'
for stage in ('before','after'):
 for mod,x in origin[stage]['modules'].items():assert x['sha256']==h(repo/Path(x['path']).relative_to(ROOT/'repo'))
 assert origin[stage]['functions']==cpu['import_origins']['before']['functions']
slurm=(E/'recovery3231-hardware3239-live-slurm.txt').read_text()
for k,v in [('JobId','3239'),('JobName','bapo-instruct300-hardware'),('JobState','RUNNING'),('Restarts','0'),('NodeList','h100-5')]:assert re.search(r'(?:^|\s)'+k+'='+re.escape(v)+r'(?:\s|$)',slurm)
print(json.dumps(dict(status='PASS_INDEPENDENT_ACTUAL_ALLOCATED_VERIFY_ONLY_INPUT_REVIEW_NOT_GPU',job_id=3239,source_commit=SOURCE,run_root=str(RUN),manifest_sha256=a['recovery-manifest.json']['sha256'],checkpoint_bound_files=31,old_rng_gap_acknowledged=True,cpu_passed=395,tools_count=27,actual_gpu=False)))
