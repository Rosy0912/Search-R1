python3 - <<'PY'
import pathlib,hashlib,json,tarfile,os,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd'); tools=r/'tools'; arc=tools/'g1-initial-token-replay-v2.tar.gz'; target=tools/'g1-initial-token-replay-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(arc)=='a18bf440f2304c4f64f80aef1e1e18d6bb17a5e345fa8d08b8972058e01c2843'
with tarfile.open(arc,'r:gz') as t:
 members=t.getmembers();assert len(members)==6
 for m in members:assert m.isfile() and pathlib.PurePosixPath(m.name).parts[0]==target.name and len(pathlib.PurePosixPath(m.name).parts)==2
 if not target.exists():t.extractall(tools)
assert sha(target/'files.json')=='ed81bec6e4ef3f5b137f8ca41617171aaba2801778bb3a9318214a15b920802c'
manifest=json.loads((target/'files.json').read_text())
assert {p.name for p in target.iterdir()}==set(manifest['files'])|{'files.json'}
for name,v in manifest['files'].items():assert (target/name).stat().st_size==v['size'] and sha(target/name)==v['sha256']
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  assert not any(str(tools/f'g1-initial-token-replay-{v}/replay.py').encode() in args for v in ('v1','v2')),f'Existing replay PID {p.name}'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
prefix=r/'logs/g1-initial3124-original-token-grader-replay-v2'
output=pathlib.Path(str(prefix)+'.json');progress=pathlib.Path(str(prefix)+'-progress.jsonl');raw=pathlib.Path(str(prefix)+'-raw.log');identity=pathlib.Path(str(prefix)+'-invocation.json')
for p in (output,progress,raw,identity):assert not p.exists(),f'Refuse existing {p}'
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
allowed=sorted(os.sched_getaffinity(0))[:2];assert len(allowed)==2
cmd=[str(r/'env/bin/python'),str(target/'replay.py'),'--directory',str(r/'logs/g1-initial3102-r4r4-allocation-evidence/initial/outer-0'),'--original-dump',str(tools/'g1-staged-calibration-producer-v3/original/ts_probe_dump.py'),'--output',str(output),'--progress',str(progress)]
with raw.open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:os.sched_setaffinity(0,allowed))
 proc=pathlib.Path('/proc')/str(p.pid)
 record={'scope':'READ_ONLY_CPU_OLD_PARTIAL_A_NOT_CALIBRATION_APPROVAL','pid':p.pid,'stat':(proc/'stat').read_text(),'cgroup':(proc/'cgroup').read_text(),'uid':proc.stat().st_uid,'cpu_affinity':sorted(os.sched_getaffinity(p.pid)),'cuda_visible_devices':'','thread_limits':2,'started_epoch':time.time(),'archive_sha256':sha(arc),'manifest_sha256':sha(target/'files.json'),'files':manifest['files'],'command':cmd,'budget_seconds':1800}
 with identity.open('x') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps({'started':record}),flush=True)
 try:rc=p.wait(timeout=1800)
 except subprocess.TimeoutExpired:p.kill();p.wait();raise
 print(json.dumps({'returncode':rc,'output':str(output),'output_sha256':sha(output) if output.exists() else None,'progress_sha256':sha(progress) if progress.exists() else None,'raw_sha256':sha(raw),'receipt':json.loads(output.read_text()) if output.exists() else None}),flush=True)
 raise SystemExit(rc)
PY
