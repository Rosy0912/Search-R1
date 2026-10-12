"""Recheck fixed source with explicit test-only Linux CPU adaptations."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
assert sys.platform=='linux' and os.uname().nodename=='h100-5','CPU acceptance requires target Linux h100-5'
repo=ROOT/'repo'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
names=['baselines/test_baseline_alignment_contract.py','baselines/test_baseline_aligned_actor.py','baselines/test_baseline_alignment_protocol_data.py','baselines/test_raw_sampler_diagnostic.py','baselines/test_algorithms.py','baselines/test_bapo_fixes.py','baselines/test_fresh200_policy_safety.py','baselines/test_sequence_mean_actor.py','baselines/test_actor_current_device.py','baselines/test_fresh200_driver_config_transport.py','baselines/test_fresh200_contract.py','baselines/test_fresh200_skip_contract.py','baselines/test_fresh200_driver_fit.py','baselines/test_fresh200_gate_skip.py','baselines/test_baseline_aligned_first5_gate.py','baselines/test_aligned_outer_commit.py','baselines/test_training_em.py','baselines/test_baseline_alignment_common_goal.py','baselines/test_prebuilt_holdout.py','reproduction/test_baseline_aligned_generation.py','reproduction/test_baseline_aligned_validation.py','reproduction/test_baseline_raw_sampling_capture.py','reproduction/test_baseline_retrieval_identity.py']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'tools/deployment')+os.pathsep+str(repo))
identity=json.loads(subprocess.check_output([str(ROOT/'env/bin/python'),'-B','-c','import sys,os,json;print(json.dumps(dict(prefix=sys.prefix,executable=sys.executable,host=os.uname().nodename,platform=sys.platform)))'],cwd=repo,env=env,text=True))
assert identity['prefix']==str(ROOT/'env') and identity['host']=='h100-5' and identity['platform']=='linux','wrong actual CPU environment'
command=[str(ROOT/'env/bin/python'),'-B','-m','pytest','-q','-p','aligned_linux_cpu_plugin']+['tests/'+name for name in names]
log=ROOT/'logs/target-linux-cpu-compatible.log';record=ROOT/'logs/target-linux-cpu-compatible.json'
assert not log.exists() and not record.exists(),'existing CPU observation: inspect rather than rerun'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('x')as f:
 p=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=480)
value={'host':os.uname().nodename,'actual_python':identity,'source':SOURCE,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':p.returncode,'command':command,'env_prefix':str(ROOT/'env'),'log':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'tail':log.read_text()[-2500:],'test_fixture_adaptations':['actual-reviewed-tests namespace binding','single CPU ToyHF oracle uses existing naive CE backend; restored afterwards'],'scope':'ACTUAL_TARGET_LINUX_CPU_WITH_EXPLICIT_TEST_ONLY_COMPATIBILITY_NOT_GPU_TRAINING'}
with record.open('x')as f:json.dump(value,f,indent=2)
print(json.dumps(value));raise SystemExit(p.returncode)
