import pathlib,json,subprocess,datetime,os,csv,io,hashlib,sys,time
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')
commit='0553adec146537e07b84377ce31cba41a54f37cd'
sys.path.insert(0,str(base/'repo'))
from reproduction.common8.native_allocation import fields
from reproduction.common8.shared_slurm import AUTHORIZATION,authorization
from reproduction.common8.submit_mixed_native import publish_exclusive
assert (base/'preparation/asset-binding.json').is_file()
for n in ['shared-slurm-authorization.json','native-submit-intent.json','native-submit-receipt.json']:
 assert not (base/'preparation'/n).exists(),n+' already exists: inspect instead of replay'
origin=pathlib.Path('/mnt/data/shaoshitong/search-g1-speed-resume5-c7785c0-20261012')
saved=json.loads((origin/'preparation/job3379-terminal-full-sha-v1.json').read_text());terminal=fields(saved['job'])
assert terminal['JobState']=='FAILED' and terminal['ExitCode']=='1:0'
assert saved['original_training_pids_gone'] and saved['own_cgroup_pids']==[]
assert json.loads((origin/'logs/cleanup-status.json').read_text())['cleaned'] is True
video=fields(subprocess.check_output(['scontrol','show','job','-o','3371'],text=True))
assert video['JobState']=='RUNNING' and video['NodeList']=='h100-8'
gpus=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True))))
gpus=sorted([[int(r[0]),r[1].strip()] for r in gpus]);assert [r[0] for r in gpus]==list(range(8))
a=dict(authorization=AUTHORIZATION,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base=str(base),physical_indices=list(range(8)),physical_gpu_uuids=[r[1] for r in gpus],training_world_size=8,actor_micro_batch_per_rank=2,vllm_gpu_memory_utilization=.1,cpu_request=96,memory_request='512G',preserve_foreign_jobs=True,source_completed_outer=5,allowed_native_jobs={'3371':{k:video[k] for k in ['JobId','NodeList','StartTime','Command']}},scope='Explicit human shared scheduling and reduced extraneous startup checks; terminal3379 preserved, video3371 retained; actual new GPU restore pending')
publish_exclusive(base/'preparation/shared-slurm-authorization.json',a)
a,sha=authorization(base/'preparation/shared-slurm-authorization.json',base)
v=video
p=subprocess.run([str(base/'env/bin/python'),'-B','-m','reproduction.common8.submit_mixed_native','--base',str(base),'--source-commit',commit,'--source-manifest',str(base/'preparation/published-source-files.json'),'--restore-receipt','/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/diagnostics/checkpoint5-cpu-validation-once-v1/RESULT.json','--execution-profile','native8_bf16_varlen_fp32_master','--shared-authorization',str(base/'preparation/shared-slurm-authorization.json')],cwd=base/'repo',env={'PATH':'/usr/local/bin:/usr/bin:/bin','PYTHONPATH':str(base/'repo'),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},capture_output=True,text=True,timeout=45)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_failed3379_terminal=terminal,video_preserved=v,authorization_sha256=sha,submission_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
assert p.returncode==0
