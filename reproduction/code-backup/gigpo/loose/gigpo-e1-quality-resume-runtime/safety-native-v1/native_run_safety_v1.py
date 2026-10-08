"""Owned allocation for quality-selected ten-Adam continuation validation; no policy300 training."""
import fcntl,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime_resume_v1 import ROOT,require,sha,cgroup_scope,OwnedProcesses,gpu_sample
binding=Path(sys.argv[1])
require(len(sys.argv)==3 and sha(binding)==sys.argv[2],'Native binding expected SHA mismatch')
value=json.loads(binding.read_text())
require(value['status']=='REVIEWED_FOR_NATIVE_ONLY','Native-only independent deployment review required')
require(value['validation_scope']=='GIGPO_POLICY_SAFETY_NATIVE_ONLY' and value['allowed_safety_modes']==['native_fixture'],'Wrong new safety native-only scope')
require(value['native_test_threshold_sets']==[[100.,1.]],'Native diagnostic threshold binding differs')
require(value['restore_step']==25 and value['restore_manifest_sha256']=='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4','Not the complete healthy25 source')
os.environ.update(GIGPO_POLICY_NATIVE_BINDING=str(binding),GIGPO_POLICY_NATIVE_BINDING_SHA256=sys.argv[2],SEARCH_R1_BASELINE_COMMIT=value['code_commit'])
for item in value['evidence']:require(sha(item['path'])==item['sha256'],'Native independent review or evidence changed')
from omegaconf import OmegaConf
from search_r1.baselines.quality_resume import selected_origin
origin=selected_origin(OmegaConf.create({'trainer':{'quality_resume':value['quality_selection_control']}}))
os.environ.update(GIGPO_QUALITY_CHECKPOINT=origin['checkpoint'],GIGPO_QUALITY_MANIFEST_SHA256=origin['manifest'],GIGPO_QUALITY_RESTORE_STEP=str(origin['step']))
require(Path(os.environ['GIGPO_RESUME_REPO']).resolve()==Path(value['repo']).resolve(),'Native repo differs from binding')
require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==value['code_commit'],'Source commit mismatch')
for item in value['files']:require(sha(item['path'])==item['sha256'],'Native source/helper changed')
require(socket.gethostname().split('.')[0]=='h100-8','Wrong node')
job=os.environ['SLURM_JOB_ID']
scheduler=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
require('Requeue=0' in scheduler.split() and 'Restarts=0' in scheduler.split(),'Native restart/requeue forbidden')
require(os.environ['SLURM_JOB_NODELIST']=='h100-8' and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8,'Not full8 allocation')
from search_r1.baselines.preflight import validate_inherited_gpu_leases,gpu_snapshot,check_selected_gpus
validate_inherited_gpu_leases(8)
lock=(ROOT/'gpu-job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
os.environ.update(NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_NVLS_ENABLE='0')
run=ROOT/'runs'/f'gigpo-e1-policy-safety-native-job{job}';run.mkdir()
os.environ['CPATH']='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10'
os.environ['NUMERIC_FIXTURE_OUTPUT']=str(run/'gpu-records')
for key in list(os.environ):
    if key.startswith('SEARCH_R1_BASELINE_AUDIT'):os.environ.pop(key,None)
scope,relative=cgroup_scope(job);owned=OwnedProcesses(scope,relative)
inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
(ROOT/'tools/expected-gpu-uuids.json').write_text(json.dumps([g['uuid'] for g in selected]))
import torch,transformers,search_r1,inspect
import search_r1.baselines.recovery as actual_recovery
require(inspect.getfile(actual_recovery)==str(Path(os.environ['GIGPO_RESUME_REPO'])/'search_r1/baselines/recovery.py'),'Native imported wrong source')
result=dict(runtime_identity=dict(executable=sys.executable,prefix=sys.prefix,python=sys.version,torch=torch.__version__,transformers=transformers.__version__,recovery_file=inspect.getfile(actual_recovery)),status='FAILED',validation_scope='GIGPO_POLICY_SAFETY_NATIVE_ONLY',restore_step=origin['step'],restore_manifest_sha256=origin['manifest'],expected_optimizer_steps_per_outer=10,expected_gradient_micro_batches_per_rank=160,global_row_denominator=256,production_denominator_per_mini=32,job=job,cgroup=str(scope),code_commit=value['code_commit'],binding_sha256=sha(binding),initial_gpu=gpu_sample())
def interrupted(sig,frame):raise RuntimeError('Own native fixture interrupted')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'guard-resources.jsonl')],run/'guard.log')
    deadline=time.monotonic()+30
    while not (run/'guard-resources.ready').exists():
        require(guard.poll() is None and time.monotonic()<deadline,'Resource guard failed');time.sleep(1)
    cpu=owned.spawn([sys.executable,'-m','pytest','tests/baselines','tests/test_recovery.py','tests/reproduction/test_final_checkpoint.py',str(Path(__file__).parent/'test_safety_audit_v1.py'),'-q','-p','no:cacheprovider'],run/'cpu.log',env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    while cpu.poll() is None:require(guard.poll() is None,'Own guard failed');time.sleep(1)
    require(cpu.returncode==0,'Native CPU regression failed')
    oracle=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=26381',str(Path(__file__).parent/'native_gpu_safety_oracle_v1.py')],run/'gpu-oracle.log',env=dict(os.environ,NUMERIC_FIXTURE_OUTPUT=str(run/'oracle-records')))
    while oracle.poll() is None:require(guard.poll() is None,'Own guard failed');time.sleep(1)
    require(oracle.returncode==0,'Ten-Adam/FSDP/HF/state-replay/global256 row oracle failed')
    oracles=[json.loads((run/f'oracle-records/rank-{i}.json').read_text()) for i in range(8)]
    require(all(r['status']=='PASS' and r['rank']==i and r['sequence_pg_kl_dp8_micro2_equal'] and r['production_denominator_per_mini']==32 and r['global_row_denominator']==256 and r['production_manager_probe'] and r['dual_clip']==3. and r['single_invalid_rank_synchronous_refusal'] and r['no_extra_adam_after_refusal'] and r['raw_joint_exp_not_computed'] and r['monitor_guards']==140 and r['optimizer_steps_at5']==50 and r['optimizer_steps_at6']==60 and all(x['optimizer_steps']==10 and x['micro_batches']==160 for x in r['records']) for i,r in enumerate(oracles)),'Incomplete actual native ten-Adam oracle')
    phases = []
    for phase in ('A', 'B'):
        env = dict(os.environ, NUMERIC_FIXTURE_PHASE=phase)
        gpu=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=26381',str(Path(__file__).parent/'native_gpu_safety_restore_v1.py')],run/f'gpu-{phase}.log',env=env)
        while gpu.poll() is None:require(guard.poll() is None,'Own guard failed');time.sleep(1)
        require(gpu.returncode==0,'Official3B chosen-checkpoint full restore or ten-mini replay failed')
        records=[json.loads((run/f'gpu-records/rank-{phase}-{i}.json').read_text()) for i in range(8)]
        require(all(r['status']=='PASS' and r['rank']==i and r['actual_outer_updates']==0 and r['full_saved_state_equal'] and r['optimizer_steps_after']==int(os.environ['GIGPO_QUALITY_RESTORE_STEP'])*20+10 and r['native_mini_optimizer_steps']==10 and r['gradient_micro_batches']==160 and r['monitor_guard_count']==20 and r['dual_clip']==3. and r['raw_joint_exp_not_computed'] and r['cuda_diagnostic_checks']==['negative_joint800_exact_dualclip_zero_grad','real_global_empty_rank_exact_quantile','token_k3_tail_refusal'] for i,r in enumerate(records)),'Missing actual official3B records')
        # New safety schema: bounded Dual-Clip PG, no old unbounded-ratio scaling proof.
        for record in records:
            require(record['safety_scope']=='native_fixture_not_production_calibration','Wrong diagnostic scope')
            require(all(math.isfinite(float(n)) and float(n)<=1.000001 for n in record['grad_norms_after_clip']),'Invalid actual clipped norms')
        phases.append(records)
    for first, replayed in zip(*phases):
        for field in ('restored_identity', 'next_mini_identity'):
            left={k:v for k,v in first[field].items() if k not in ('pid', 'start_ticks')}
            right={k:v for k,v in replayed[field].items() if k not in ('pid', 'start_ticks')}
            require(left==right,'Independent fresh-process full tensor identity replay differs: '+field)
        require(first['next_mini_identity']['pid']!=replayed['next_mini_identity']['pid'],'Replay was not a newly spawned process')
        require(first['python_numpy_rng_sha256']==replayed['python_numpy_rng_sha256'],'Full RNG replay differs')
    result.update(safety_scope='native_fixture_not_production_calibration',dual_clip=3.,production_thresholds_not_calibrated=True,fixtures_passed=True,cpu_exit=cpu.returncode,gpu_exit=gpu.returncode,phases=phases,oracle_records=oracles,actual_outer_updates=0)

except BaseException as error:result['error']=repr(error);raise
finally:
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        owned.cleanup()
        snapshot=gpu_sample()
        require(not snapshot['processes'],'Residual GPU compute after own native cleanup')
        if result.get('fixtures_passed'):result['status']='PASS'
    finally:
        result.update(cleanup_residual_states=getattr(owned,'cleanup_residual',{}),final_gpu=gpu_sample())
        (run/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
