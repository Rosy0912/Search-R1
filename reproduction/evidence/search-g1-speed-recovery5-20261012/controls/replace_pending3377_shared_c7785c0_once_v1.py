import pathlib,json,subprocess,datetime,os,csv,io,hashlib,sys,time
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-speed-resume5-c7785c0-20261012')
commit='c7785c0e6334a28d4af70df045da9f48caf0ca77'
sys.path.insert(0,str(base/'repo'))
from reproduction.common8.native_allocation import fields
from reproduction.common8.shared_slurm import AUTHORIZATION,authorization
from reproduction.common8.submit_mixed_native import publish_exclusive
assert (base/'preparation/asset-binding.json').is_file()
for n in ['shared-slurm-authorization.json','pending3377-replacement-intent-v1.json','pending3377-replacement-receipt-v1.json','native-submit-intent.json','native-submit-receipt.json']:
 assert not (base/'preparation'/n).exists(),n+' already exists: inspect instead of replay'
job=fields(subprocess.check_output(['scontrol','show','job','-o','3377'],text=True))
assert job['JobState']=='PENDING' and job['RunTime']=='00:00:00' and job['JobId']=='3377' and job['JobName']=='search-g1-mixed200'
assert job['Command']=='/mnt/data/shaoshitong/search-g1-speed-resume5-653712b-20261012/preparation/launch-mixed-native.sh'
video=fields(subprocess.check_output(['scontrol','show','job','-o','3371'],text=True))
assert video['JobState']=='RUNNING' and video['NodeList']=='h100-8'
gpus=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True))))
gpus=sorted([[int(r[0]),r[1].strip()] for r in gpus]);assert [r[0] for r in gpus]==list(range(8))
a=dict(authorization=AUTHORIZATION,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base=str(base),physical_indices=list(range(8)),physical_gpu_uuids=[r[1] for r in gpus],training_world_size=8,actor_micro_batch_per_rank=2,vllm_gpu_memory_utilization=.1,cpu_request=96,memory_request='512G',preserve_foreign_jobs=True,source_completed_outer=5,allowed_native_jobs={'3371':{k:video[k] for k in ['JobId','NodeList','StartTime','Command']}},scope='Explicit human shared scheduling; own pending3377 replaced, video3371 retained; actual GPU work pending')
publish_exclusive(base/'preparation/shared-slurm-authorization.json',a)
a,sha=authorization(base/'preparation/shared-slurm-authorization.json',base)
publish_exclusive(base/'preparation/pending3377-replacement-intent-v1.json',dict(authorization=AUTHORIZATION,utc=a['observed_utc'],object='own pending job3377 only',scope='Replace pending request with nativeCPU/memory allocation and authorized actualshared8; no stopped running training or video',old_job=job,video=video,source_commit=commit,authorization_sha256=sha))
r=subprocess.run(['scancel','3377'],capture_output=True,text=True,timeout=15);assert r.returncode==0,(r.stdout,r.stderr)
terminal=fields(subprocess.check_output(['scontrol','show','job','-o','3377'],text=True));assert terminal['JobState']=='CANCELLED' and terminal['RunTime']=='00:00:00',terminal
v=fields(subprocess.check_output(['scontrol','show','job','-o','3371'],text=True));assert all(v[k]==video[k] for k in ['JobId','StartTime','NodeList','Command']) and v['JobState']=='RUNNING'
publish_exclusive(base/'preparation/pending3377-replacement-receipt-v1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scancel_returncode=r.returncode,terminal=terminal,video_preserved=v))
p=subprocess.run([str(base/'env/bin/python'),'-B','-m','reproduction.common8.submit_mixed_native','--base',str(base),'--source-commit',commit,'--source-manifest',str(base/'preparation/published-source-files.json'),'--restore-receipt','/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/diagnostics/checkpoint5-cpu-validation-once-v1/RESULT.json','--execution-profile','native8_bf16_varlen_fp32_master','--shared-authorization',str(base/'preparation/shared-slurm-authorization.json')],cwd=base/'repo',env={'PATH':'/usr/local/bin:/usr/bin:/bin','PYTHONPATH':str(base/'repo'),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},capture_output=True,text=True,timeout=45)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_pending_terminal=terminal,video_preserved=v,authorization_sha256=sha,submission_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
assert p.returncode==0
