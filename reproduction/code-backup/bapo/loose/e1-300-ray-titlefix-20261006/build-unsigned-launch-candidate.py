"""Local unsigned candidate builder; never deploys, signs, submits or grants permission."""
import argparse,hashlib,json,pathlib,shutil
SOURCE='484c0b30fa73fa5354bdb3452f007134ec5db13f'
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006'
CONFIG='4a985fce68eaa90548eea96f1c9613cb3e3d37784a93958c0566cc0ac5ff1064'
DATA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 p=pathlib.Path(p);need(p.is_file()and not p.is_symlink(),'Ordinary input required');return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(v):return(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def binding(p,remote):return dict(path=ROOT+'/'+remote,bytes=pathlib.Path(p).stat().st_size,sha256=sha(p))
def build(tools,cpu,raw,config):
 tools=pathlib.Path(tools);raw=pathlib.Path(raw);need(tools.is_dir()and not tools.is_symlink(),'Ordinary tools directory required');entries=list(tools.iterdir());need(all(p.is_file()and not p.is_symlink()for p in entries),'Exact allregular tools required');hashes={p.name:sha(p)for p in sorted(entries)};need(len(hashes)==35,'Exactly approved final35 required')
 need(sha(config)==CONFIG,'Actual fixedsource config differs');need(sha(cpu)=='5ba4ec90f9295b5d8dc05f5d9866f45971e326b3d2b892061cc08740462143c8','Independent derived CPU record differs')
 c=json.loads(pathlib.Path(cpu).read_text());need(c['source_commit']==SOURCE and c['sys_prefix']==ROOT+'/env'and c['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE','CPU identity/scope differs')
 for key,value in [('passed',395),('skipped',0),('failed',0),('exit_code',0)]:need(type(c[key])is int and c[key]==value,'Actual second CPU result differs')
 need(c['first_observation_exit_verified']is False and c['wall_seconds']is None,'Cannot invent first exit or end-to-end walltime')
 need(c['test_launcher_sha256']==hashes['cpu-tests.py'],'Actual tested launcher differs')
 deployed=json.loads((raw/'logs__tools-manifest.json').read_text());need(deployed['source_commit']==SOURCE and len(deployed['files'])==35,'Original deployed tool snapshot differs');need({k:v for k,v in hashes.items()if deployed['files'].get(k)!=v}=={'verify_launch.py':'28cb841903152fe08f80b63c60fd1e9292a82de936fdff79a22c91893bdc4df0','test_freeze_scalability.py':'0379154834cf3b342422153eb7fe65320408ec7ab184edea6226746fcb56fa09'},'Only declared durablegate/strict exec-race test deltas allowed')
 log=raw/'logs__cpu-gate-durable.log';exitp=raw/'logs__cpu-gate-durable-exit.json';runtime=raw/'logs__runtime-executable.json';exitrecord=json.loads(exitp.read_text());rt=json.loads(runtime.read_text())
 need(c['log_sha256']==exitrecord['log_sha256']==sha(log)and c['exit_record_sha256']==sha(exitp),'CPU original log/exit binding differs')
 need(type(exitrecord['exit_code'])is int and exitrecord['exit_code']==0 and exitrecord['source_commit']==SOURCE and exitrecord['training_submitted']is False,'Actual exit witness differs')
 need(rt['source_commit']==SOURCE and rt['root']==ROOT and rt['sys_prefix']==ROOT+'/env'and rt['host']=='h100-5'and rt['scope']=='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE','Runtime original record differs');need(sha(runtime)=='95c6253ef905cafa8fb93c6e18e6743b6aa3e5b65521bbf6a8dcc5f74104ec20','Runtime original bytes differ')
 manifest=dict(source_commit=SOURCE,files=hashes)
 approval=dict(status='CANDIDATE_REQUIRES_INDEPENDENT_E1_LAUNCH_REVIEW',scope='UNSIGNED_LOCAL_CANDIDATE_NOT_LAUNCH_OR_GPU_PERMISSION',source_commit=SOURCE,root=ROOT,tools_file_count=35,tools_manifest_sha256=hashlib.sha256(encode(manifest)).hexdigest(),tool_files=hashes,config_yaml_sha256=CONFIG,data_manifest_sha256=DATA,model_manifest_sha256=MODEL,cpu_gate_record_sha256=sha(cpu),cpu_expected_passed=395,cpu_gate_log=binding(log,'logs/cpu-gate-durable.log'),cpu_gate_exit_record=binding(exitp,'logs/cpu-gate-durable-exit.json'),runtime_executable=binding(runtime,'logs/runtime-executable.json'),independent_review=dict(path=ROOT+'/logs/final-launch-independent-review.md',sha256=None,pending_independent_review=True),actual_gpu=False,actual_gpu_accepted=False,real_hardware_restore_accepted=False,production_training_permission_signed=False,launch_authorized=False,guardian_actual_linux_acceptance_pending=True)
 approval['shared_input_bindings']=dict(data=dict(path=ROOT+'/common-data/manifest.json',sha256=DATA),model=dict(path=ROOT+'/assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json',sha256=MODEL),source_config=dict(path=ROOT+'/repo/verl/trainer/config/ppo_bapo_e1_300.yaml',sha256=CONFIG),shared_inputs_verifier=dict(path=ROOT+'/tools/verify_shared_inputs.py',sha256=hashes['verify_shared_inputs.py']),complete25_prefix=dict(path=ROOT+'/recovery-inputs/train-prefix1-25.log',bytes=959681,sha256='57b05f48a27ab5fd5c81e3e524f71569c6675f8c3ca234ac105b0c677e2c3dc0',actual_newroot_preparation_pending=True),expected_existing_budget=dict(total_outer=300,scheduler_warmup=114,stage_questions=64,complete_restore_step=25,next_step=26))
 need({p.name:sha(p)for p in tools.iterdir()}==hashes,'Tools changed during build');return manifest,approval,{k:hashes[k]for k in ('submit.sh','hardware.sbatch','formal300.sbatch')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',required=True);p.add_argument('--cpu-record',required=True);p.add_argument('--raw',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);need(not out.exists(),'Fresh output directory required');need(pathlib.Path(ROOT)not in out.resolve().parents and out.resolve()!=pathlib.Path(ROOT),'Never write actual targetROOT');m,r,h=build(a.tools,a.cpu_record,a.raw,a.config);out.mkdir(parents=True)
 for name,obj in [('tools-manifest.json',m),('launch-approval.UNSIGNED-CANDIDATE.json',r),('approved-launch-hashes.UNSIGNED-CANDIDATE.json',h)]:
  with(out/name).open('xb')as f:f.write(encode(obj))
 for src,name in [(pathlib.Path(a.cpu_record),'cpu-gate.json'),(pathlib.Path(a.raw)/'source-expected.txt','cpu-gate-source.txt'),(pathlib.Path(a.raw)/'logs__cpu-gate-durable.log','cpu-gate-durable.log'),(pathlib.Path(a.raw)/'logs__cpu-gate-durable-exit.json','cpu-gate-durable-exit.json'),(pathlib.Path(a.raw)/'logs__runtime-executable.json','runtime-executable.json'),(pathlib.Path(a.raw)/'logs__tools-manifest.json','tools-manifest.DEPLOYED35-BEFORE-DURABLE-GATE.json')]:
  with(out/name).open('xb')as f:f.write(src.read_bytes())
 print(json.dumps(dict(status='UNSIGNED_NOT_AUTHORIZATION',output=str(out.resolve()),files={p.name:sha(p)for p in out.iterdir()})))
if __name__=='__main__':main()
