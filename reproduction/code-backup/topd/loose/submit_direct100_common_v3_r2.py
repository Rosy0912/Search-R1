import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
src=r'''
import pathlib,hashlib,json,tarfile,subprocess,os,time,socket,sys
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
D=R/'logs/r1-standard32b-grpo100-common-v3-r2';record=R/'logs/r1-standard32b-grpo100-common-v3-r2-submission.json'
if record.exists():print(json.dumps({'existing':json.loads(record.read_text()),'scope':'NO_DUPLICATE'}));raise SystemExit(0)
q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True,timeout=15);assert not any('r1-32b-grpo' in x for x in q.splitlines()),q
packages={'search-r1-32b-grpo100-source-v6':'7364140285a20b87da94235389d9c56cbad6799d9602eb71343d17adbb8774b3','search-r1-32b-grpo100-tools-direct-v3':'90086e7a70ebb8b589021f40c2c6c3f86b19307526b0b1c76732f6cc797d9304','qwen25-32b-window-safety-direct-own-sm-v1':'904a4a057cc2ef30d0df864ac7d3c21c610a699fa747b23f91447732eae8cc3e'}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for name,digest in packages.items():
 A=R/'tools'/(name+'.tar.gz');assert sha(A)==digest
 if not (R/'tools'/name).exists():
  with tarfile.open(A) as tf:
   for m in tf.getmembers():assert not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts and not m.issym() and not m.islnk()
   tf.extractall(R/'tools')
T=R/'tools/search-r1-32b-grpo100-tools-direct-v3';S=R/'tools/search-r1-32b-grpo100-source-v6';W=R/'tools/qwen25-32b-window-safety-direct-own-sm-v1'
sys.dont_write_bytecode=True
for root in (T,S):
 for cache in root.rglob('__pycache__'):
  import shutil
  shutil.rmtree(cache)
C=R/'assets/common-instruct-0b002b';assert sha(C/'manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
prior=json.loads((R/'logs/r1-standard32b-grpo100-direct-v2-r1/static-config.json').read_text())
prior['train_file']=str(C/'train.parquet');prior['val_file']=str(C/'stage.parquet')
sys.path.insert(0,str(T));from prepare import derive
c=derive(prior,'train100',D);c['tools']=str(T);c['candidate']=str(S);c['support']['path']=str(T/'owner_support.py');c['retriever_command'][3]=str(T/'retrieval_server.py')
c['source_manifest']={'path':str(S/'r1-files.json'),'sha256':sha(S/'r1-files.json')};c['tools_manifest']={'path':str(T/'files.json'),'sha256':sha(T/'files.json')}
c['telemetry_policy']={'directory':str(W),'runtime_sha256':sha(W/'runtime_safety.py'),'guard_sha256':sha(W/'startup_guard.py'),'manifest_sha256':sha(W/'files.json'),'actual_independent_native_review':None}
c['input_bindings']=[{'path':str(C/f),'sha256':sha(C/f)} for f in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet')]
c['common_data']={'manifest_sha256':sha(C/'manifest.json'),'directory':str(C),'train_rows':169028,'validation_rows':512,'stage_rows':64,'final_rows':51713,'validation_file':str(C/'validation.parquet'),'final_file':str(C/'final.parquet')}
# Actual row metadata is read from the copied parquet, never reconstructed from counts.
code="import pyarrow.parquet as pq,json,sys;print(json.dumps({n:{'rows':pq.ParquetFile(sys.argv[1]+'/'+n+'.parquet').metadata.num_rows,'columns':pq.ParquetFile(sys.argv[1]+'/'+n+'.parquet').schema.names} for n in ('train','validation','stage','final')}))"
rows=json.loads(subprocess.check_output([str(R/'env/bin/python'),'-c',code,str(C)],text=True))
assert {n:r['rows'] for n,r in rows.items()}=={'train':169028,'validation':512,'stage':64,'final':51713};c['actual_common_parquet_metadata']=rows
c['training_command'].append('+actor_rollout_ref.ref.entropy_coeff=0')
from supervise import verify_recipe
verify_recipe(c)
assert not D.exists();D.mkdir(mode=0o700);(D/'static-config.json').write_text(json.dumps(c,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',R1_CONFIG=str(D/'static-config.json'),R1_TOOLS=str(T),R1_PREFLIGHT=c['preflight_source']['path'])
rec={'status':'UNKNOWN_BEFORE_SUBMISSION','epoch':time.time(),'directory':str(D),'source_manifest':c['source_manifest']['sha256'],'common_data_manifest':c['common_data']['manifest_sha256'],'expected_outer':100,'micro_global':32,'rank_micro':4,'mini_global':128,'optimizer_per_outer_rank':20,'micro_per_outer_rank':80,'loss_agg_mode':'sequence_mean','entropy_coeff':0,'warmup_outer':57}
with record.open('x') as f:json.dump(rec,f);f.flush();os.fsync(f.fileno())
r=subprocess.run(['sbatch','--parsable',str(T/'production.sbatch')],env=env,capture_output=True,text=True,timeout=30);rec.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:
 jid=r.stdout.strip().split(';')[0];assert jid.isdecimal();rec.update(status='SUBMITTED',job_id=int(jid))
else:rec['status']='SUBMISSION_FAILED'
p=record.with_suffix('.tmp');p.write_text(json.dumps(rec,indent=2)+'\n');os.replace(p,record);print(json.dumps(rec));raise SystemExit(r.returncode)
'''
r=send(src,attempts=1,timeout=180)
pathlib.Path('outputs/standard-r1-common-v3-r2-submission-result.log').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());print(r.stderr.decode());sys.exit(r.returncode)
