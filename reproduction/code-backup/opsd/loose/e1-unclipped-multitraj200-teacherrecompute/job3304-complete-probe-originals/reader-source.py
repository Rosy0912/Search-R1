from pathlib import Path
import os,stat,json,hashlib,base64,datetime,socket
LIMIT=20*1024*1024
TOTAL_LIMIT=128*1024*1024
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unique(raw):
 def pairs(items):
  out={}
  for k,v in items:
   need(k not in out,'Duplicate JSON key');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sig(s):return (s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read_exact(fd,size):
 out=[];left=size
 while left:
  b=os.read(fd,min(left,1024*1024));need(bool(b),'File shortened during read');out.append(b);left-=len(b)
 return b''.join(out)
def member(p,uid=0):
 need(p.is_absolute() and p.resolve(strict=True)==p,'Nonphysical member')
 first=p.lstat();need(stat.S_ISREG(first.st_mode) and first.st_uid==uid and first.st_size<LIMIT,'Wrong file type/UID/size')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);need(sig(opened)==sig(first),'Member replaced before open')
  raw=read_exact(fd,opened.st_size)
  after=os.fstat(fd);last=p.lstat()
  need((opened.st_dev,opened.st_ino,opened.st_uid)==(after.st_dev,after.st_ino,after.st_uid)==(last.st_dev,last.st_ino,last.st_uid),'Member identity changed')
  need(stat.S_ISREG(last.st_mode) and after.st_size<LIMIT and last.st_size<LIMIT,'Member exceeded bound/type')
  if p.suffix=='.log':
   need(after.st_size>=opened.st_size and last.st_size>=opened.st_size,'Log was truncated')
   os.lseek(fd,0,os.SEEK_SET);again=read_exact(fd,opened.st_size);need(again==raw,'Log prefix changed')
   final=os.fstat(fd);path_final=p.lstat()
   need((opened.st_dev,opened.st_ino,opened.st_uid)==(final.st_dev,final.st_ino,final.st_uid)==(path_final.st_dev,path_final.st_ino,path_final.st_uid),'Log identity changed after prefix check')
   need(final.st_size>=opened.st_size and path_final.st_size>=opened.st_size and final.st_size<LIMIT and path_final.st_size<LIMIT,'Log bound/truncation changed')
   last=path_final;scope='VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED'
  else:
   need(sig(opened)==sig(after)==sig(last),'JSON/raw member changed during read');scope='STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF'
  return dict(base64=base64.b64encode(raw).decode(),sha256=sha(raw),bytes=len(raw),uid=opened.st_uid,physical=str(p),device=opened.st_dev,inode=opened.st_ino,stat_before=list(sig(opened)),stat_after=list(sig(last)),reading_scope=scope)
 finally:os.close(fd)
def collect(request,required,optional):
 need(os.getuid()==0 and socket.gethostname()=='h100-10','Own native node UID0 reader required')
 root=Path(request['run_root']);need(root.resolve(strict=True)==root and root.stat().st_uid==0,'Own physical run required')
 rows={}
 for name in required:
  rows[name]=member(root/name)
 ready=unique(base64.b64decode(rows['approval-ready.json']['base64']))
 bound=unique(base64.b64decode(rows['bindings/manifest.json']['base64']))
 need(type(ready['version']) is int and ready['version']==1 and type(ready['uid']) is int and ready['uid']==0,'Wrong ready identity types')
 need(ready['job_id']==request['job_id'] and ready['source_commit']==request['source_commit'] and ready['run_root']==str(root),'Wrong actual ready source/job/run')
 need(rows['bindings/manifest.json']['sha256']==ready['binding_sha256'],'Ready binding SHA differs')
 need(bound['job_id']==request['job_id'] if 'job_id' in bound else True,'Foreign binding job')
 need(bound['source_commit']==request['source_commit'] and bound['run_root']==str(root),'Wrong run binding')
 roles={'allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest'}
 need(set(bound['files'])==roles,'Wrong binding roles')
 for role,pin in bound['files'].items():
  name='bindings/'+role+'.json';need(pin['path']==str(root/name) and pin['sha256']==rows[name]['sha256'],'Manifest role SHA/path differs')
 need(rows['bindings/public.json']['sha256']==ready['public_key_sha256']==PUBLIC_SHA,'Public key binding differs')
 src=unique(base64.b64decode(rows['bindings/source.json']['base64']))
 alloc=unique(base64.b64decode(rows['bindings/allocation.json']['base64']))
 need(src['clean'] is True and src['source_commit']==request['source_commit'] and src['repo']==request['repo'] and src['source_files']==request['source_files'] and src['formal_manifest_sha256']==request['formal_sha'],'Wrong actual source binding')
 need(type(alloc['uid']) is int and alloc['uid']==0 and alloc['job_id']==request['job_id'] and alloc['hostname']=='h100-10','Wrong actual allocation binding')
 missing=[]
 for name in optional:
  p=root/name
  if os.path.lexists(p):rows[name]=member(p)
  else:missing.append(name)
 need(sum(r['bytes'] for r in rows.values())<TOTAL_LIMIT,'Snapshot exceeds total bound')
 return dict(version=1,kind=SCOPE,scope=SCOPE,job_id=request['job_id'],source_commit=request['source_commit'],run_root=str(root),hostname=socket.gethostname(),uid=os.getuid(),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,missing_optional=missing)

PUBLIC_SHA='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
SCOPE='BOUNDED_OWN_RUN_PROBE_BYTES_SNAPSHOT_NOT_ACCEPTANCE_OR_TERMINAL_PROOF'
print(json.dumps(collect({'job_id': '3304', 'source_commit': 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6', 'run_root': '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304', 'repo': '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/repo-fcecbea5d939', 'formal_sha': '71d2501d45120c546d5a94cc9add1a60fb7afbf70dd4ed7a5333e51955ab1028', 'source_files': {'reproduction/baselines/e1-multitraj200/microbatch_probe.py': '20290c781a557214152d3fc55c91ee5b0dde6db91f24d7932b692110be90f73f', 'reproduction/baselines/e1_import_bootstrap.py': '49e8b145748876c1ce4efd408f9206bb09e0d886ff2d77e7c364e662fb80718f', 'reproduction/baselines/run_opsd_e1_multitraj200.sh': '7112bbf83608be1c0b3c3ebf4dcd4753c2f0f4a19ebbfad419658949fbfdabd5', 'reproduction/h20/retriever_local.py': 'a78f0b493c538dddc6619e0e462995fd9945c25a187c2165e99d4c10622ce459', 'search_r1/baselines/actor.py': 'bbf16e04df0ce5d887d674311be3875220bbffe21cbfa2188b3f9833366c1222', 'search_r1/baselines/checkpoint_gate.py': '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16', 'search_r1/baselines/controller.py': '29b4e9c8bc76a5d4e7f956246ffa7907ad685a2eaa7fed68701dba4a8a9f0bf6', 'search_r1/baselines/distillation.py': 'a43c4d70259cff416cafb6fbd423c7d8aaff0328a65ef732d8afc6989270b2d7', 'search_r1/baselines/e1_profile.py': 'c79e67e7baef3a3f82c79e1ed3e3c4ec3d7b8077bdbc0ab01c4aa97b5e653c9b', 'search_r1/baselines/full_state.py': 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6', 'search_r1/baselines/main.py': '2d4b8eb0fcc307c0ea74713505d53c673d55ac8b7e72a74e03ee21e6abd802c1', 'search_r1/baselines/preflight.py': 'cc9445265a3835e1623c0de9d95380717f0e44d4a303d45e3fe1f559431e1605', 'search_r1/baselines/rollout.py': '4b3501a1eafc10ed4ae7abd76c5a8f8bc9b90c839ab748b6d522295c934a4e5a', 'search_r1/baselines/trajectory_archive.py': '88277e24453058e8e62ae2e13b2df94dc1b35110618ac44381a4b5f579f17455', 'search_r1/search/retrieval_server.py': 'b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052', 'verl/trainer/config/ppo_opsd_e1_multitraj200.yaml': 'd017061a138b0f90c0197754bfdf05c0ea46cb76b088ffb1594df44d7ddd878f', 'verl/trainer/ppo/ray_trainer.py': 'f78dc2bebdf6b42b94d6845cd77cd1d723c2b6e33446f25295baf01f7e8d159d', 'verl/utils/torch_functional.py': 'd46ab49f849b2be950e3e20a844a7747294b842cde0ad0abe49ae4ea31ab1635', 'verl/workers/fsdp_workers.py': 'aeb49c89d968fbd0f186e61efb151d6525ae1018000a63e4860b6e81b57356f7'}, 'runtime_binding_sha256': '8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228', 'local_submission_pins': {'intent': 'cbc018c1d9dae753db2df87307179d2e208648bd1ec7a0190841e12ca5519bba', 'original': '5e7d84724664f71a9951b1a591aa6b959b13755d572add0e1bda31dd88442b14', 'actual_job_id': 'dc44e9190c4d2175c9954084c5ba83e60d19a71b0c32c861dba031077172f793'}},('approval-ready.json', 'bindings/manifest.json', 'bindings/public.json', 'bindings/allocation.json', 'bindings/data_manifest.json', 'bindings/environment.json', 'bindings/import_closure.json', 'bindings/model_manifest.json', 'bindings/source.json', 'bindings/tool_manifest.json'),('bindings/gpu-init-approval.json', 'formal-wrapper.log', 'retriever.log', 'evidence/hardware-exit.json', 'evidence/hardware-torchrun.log', 'evidence/service.json', 'evidence/http-top3-24.json', 'evidence/hardware-rank-0.json', 'evidence/hardware-rank-1.json', 'evidence/hardware-rank-2.json', 'evidence/hardware-rank-3.json', 'evidence/hardware-rank-4.json', 'evidence/hardware-rank-5.json', 'evidence/hardware-rank-6.json', 'evidence/hardware-rank-7.json', 'evidence/capacity-micro4/exit.json', 'evidence/capacity-micro4/torchrun.log', 'evidence/capacity-micro4/micro-4.json', 'evidence/capacity-micro4/micro-4-rank-0.json', 'evidence/capacity-micro4/micro-4-rank-1.json', 'evidence/capacity-micro4/micro-4-rank-2.json', 'evidence/capacity-micro4/micro-4-rank-3.json', 'evidence/capacity-micro4/micro-4-rank-4.json', 'evidence/capacity-micro4/micro-4-rank-5.json', 'evidence/capacity-micro4/micro-4-rank-6.json', 'evidence/capacity-micro4/micro-4-rank-7.json', 'evidence/capacity-micro4/old-logprob-rank-0.json', 'evidence/capacity-micro4/old-logprob-rank-1.json', 'evidence/capacity-micro4/old-logprob-rank-2.json', 'evidence/capacity-micro4/old-logprob-rank-3.json', 'evidence/capacity-micro4/old-logprob-rank-4.json', 'evidence/capacity-micro4/old-logprob-rank-5.json', 'evidence/capacity-micro4/old-logprob-rank-6.json', 'evidence/capacity-micro4/old-logprob-rank-7.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-0.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-1.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-2.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-3.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-4.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-5.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-6.json', 'evidence/capacity-micro4/teacher-logits-recompute-rank-7.json')),sort_keys=True))
