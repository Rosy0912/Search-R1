"""Bounded recheck of unchanged idle/lease checks after retriever warmup."""
import ast,hashlib,json,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
B=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');O=B/'outputs/training-preparation';T=B/'runtime/tools/topd-vllm-teacher-resume5-20261013'
HELPER='''"""Repeat the same inherited lease checks until warmup utilization settles."""
import time
def wait_for_lease(preflight, expected=4, timeout=30., interval=1., clock=time.monotonic, sleep=time.sleep):
    deadline=clock()+timeout;retries=0
    while True:
        try:
            preflight.validate_inherited_gpu_leases(expected)
            return retries
        except RuntimeError as error:
            if 'exceeds the 15% idle thresholds' not in str(error) or clock()>=deadline:raise
            retries+=1
            print('TEACHER_PREWARM_IDLE_RECHECK retries='+str(retries),flush=True)
            sleep(interval)
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert '713681|FAILED|1:0' in subprocess.check_output(['sacct','-j','713681','-n','-P','--format=JobID,State,ExitCode'],text=True)
    assert not any('topd-alpha1' in s for s in subprocess.check_output(['squeue','-u','mou176','-h','-o','%i|%j'],text=True).splitlines())
    lines=(B/'runtime/logs/topd-hpc3-713681/evidence/teacher.log').read_text().splitlines()
    snapshot=next(json.loads(l.split('=',1)[1]) for l in lines if l.startswith('BASELINE_GPU_OCCUPANCY_JSON='))
    assert snapshot['inventory']['2']['utilization']==19 and snapshot['inventory']['2']['memory_used']/snapshot['inventory']['2']['memory_total']<.15
    scope={};exec(compile(HELPER,'lease-wait','exec'),scope);wait=scope['wait_for_lease']
    class P:
        def __init__(self,errors):self.errors=list(errors);self.calls=0
        def validate_inherited_gpu_leases(self,n):
            assert n==4;self.calls+=1
            if self.errors:
                x=self.errors.pop(0)
                if x:raise x
    clock=[0.];now=lambda:clock[0]
    sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
    p=P([RuntimeError('Allocated GPU 2 exceeds the 15% idle thresholds'),None]);assert wait(p,clock=now,sleep=sleep)==1 and p.calls==2
    p=P([RuntimeError('foreign descriptor')])
    try:wait(p,clock=now,sleep=sleep)
    except RuntimeError as e:assert str(e)=='foreign descriptor' and p.calls==1
    else:raise AssertionError('Unrelated lease error swallowed')
    p=P([RuntimeError('exceeds the 15% idle thresholds')]*5)
    try:wait(p,timeout=2,clock=now,sleep=sleep)
    except RuntimeError:assert p.calls==3
    else:raise AssertionError('Idle wait not bounded')
    path=T/'topd_vllm_teacher/lease_wait.py';assert not path.exists();path.write_text(HELPER)
    p=T/'topd_vllm_teacher/daemon.py';s=p.read_text()
    old="spec.loader.exec_module(pre);pre.validate_inherited_gpu_leases(4)";assert old in s
    s=s.replace(old,"spec.loader.exec_module(pre)\n    from .lease_wait import wait_for_lease\n    lease_idle_retries=wait_for_lease(pre,4)")
    s=s.replace('first_expected_outer=a.start_outer,pid=','first_expected_outer=a.start_outer,lease_idle_retries=lease_idle_retries,pid=');ast.parse(s);p.write_text(s)
    c=json.loads((O/'hpc3-topd200-alpha1-resume5-teacher-sequence-fixed-20261013.json').read_text());p=T/'files.json';d=json.loads(p.read_text());d['files']={str(f.relative_to(T)):{'size':f.stat().st_size,'sha256':sha(f)} for f in T.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts};p.write_text(json.dumps(d,sort_keys=True,indent=2));c['teacher_backend_manifest']['sha256']=sha(p)
    config=O/'hpc3-topd200-alpha1-resume5-teacher-settle-fixed-20261013.json';config.write_text(json.dumps(c,indent=2))
    sys.path.insert(0,c['tools']);from student_binding import check;check(c)
    report={'status':'IDENTICAL_LEASE_CHECK_BOUNDED_IDLE_SETTLE_TEST_PASS','epoch':time.time(),'failed_job':713681,'new_optimizer_updates':0,'observed_gpu_utilization':19,'idle_threshold_unchanged':15,'bounded_wait_seconds':30,'CPU_test_cases':['transient_idle_recheck_then_pass','unrelated_error_not_swallowed','persistent_busy_timeout'],'config':str(config),'configuration_sha256':sha(config),'weights_and_objective_changed':False}
    (O/'alpha1-teacher-idle-settle-repair.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
