"""Bind the frozen teacher's strict request sequence to authorized recovery."""
import ast,hashlib,json,pathlib,shutil,subprocess,sys,time
sys.dont_write_bytecode=True
B=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');R=B/'runtime';O=B/'outputs/training-preparation'
OLD=R/'tools/topd-vllm-teacher-v3';NEW=R/'tools/topd-vllm-teacher-resume5-20261013'
L=R/'tools/topd-alpha1-launch-resume5-20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=subprocess.check_output(['sacct','-j','713661','-n','-P','--format=JobID,State,ExitCode'],text=True);assert '713661|FAILED|1:0' in a
    assert not any('topd-alpha1' in s for s in subprocess.check_output(['squeue','-u','mou176','-h','-o','%i|%j'],text=True).splitlines())
    evidence=R/'logs/topd-hpc3-713661/evidence'
    assert not list((evidence/'actor-resource-audit').glob('actor-updates-rank*-outer6.json'))
    err=json.loads((evidence/'teacher/first-error.json').read_text());assert err['outer_step']==1 and err['message']=='Teacher request authority/order/math'
    if NEW.exists():raise RuntimeError('Teacher recovery backend already prepared; inspect rather than overwrite')
    shutil.copytree(OLD,NEW,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    p=NEW/'topd_vllm_teacher/daemon.py';s=p.read_text()
    assert "step=1;sampler=" in s and s.count(';a=p.parse_args()')==1
    s=s.replace(';a=p.parse_args()',";p.add_argument('--start-outer',type=int,choices=range(1,201),default=1);a=p.parse_args()")
    s=s.replace('step=1;sampler=','step=a.start_outer;sampler=')
    s=s.replace("status='FROZEN_BF16_TP4_WEIGHTS_RESIDENT_KV_RELEASED',pid=", "status='FROZEN_BF16_TP4_WEIGHTS_RESIDENT_KV_RELEASED',first_expected_outer=a.start_outer,pid=")
    tree=ast.parse(s)
    # Execute the exact authority/order condition with the resumed sequence;
    # accept 6 then7 and reject old, skipped, unauthorized and gold requests.
    condition=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='Teacher request authority/order/math' for x in ast.walk(n)))
    executable=compile(ast.fix_missing_locations(ast.Module(body=[condition],type_ignores=[])),'teacher-order-gate','exec')
    for expected in [6,7]:
        valid={'capability':'fixture','outer_step':expected,'teacher_path':'/fixture','temperature':1.,'targets':None}
        exec(executable,{'meta':valid,'cap':'fixture','step':expected,'teacher':pathlib.Path('/fixture')})
        for key,val in [('outer_step',expected-1),('outer_step',expected+1),('capability','wrong'),('targets',['gold']),('temperature',.5)]:
            invalid=dict(valid);invalid[key]=val
            try:exec(executable,{'meta':invalid,'cap':'fixture','step':expected,'teacher':pathlib.Path('/fixture')})
            except ValueError:pass
            else:raise AssertionError('Teacher order/authority gate relaxed')
    p.write_text(s)
    p=L/'runtime/supervise.py';s=p.read_text()
    old="'--gpu-memory-utilization',str(c['teacher_gpu_memory_utilization'])]"
    assert old in s
    s=s.replace(old,"'--gpu-memory-utilization',str(c['teacher_gpu_memory_utilization']),'--start-outer',str(c.get('teacher_start_outer',1))]")
    ast.parse(s);p.write_text(s)
    c=json.loads((O/'hpc3-topd200-alpha1-resume5-20261013.json').read_text())
    c['teacher_backend']=str(NEW);c['teacher_backend_manifest']['path']=str(NEW/'files.json');c['teacher_start_outer']=6
    assert c['resume_authorization']['next_outer_step']==c['teacher_start_outer']
    for key in ['teacher_backend_manifest','tools_manifest','launcher_manifest']:
        p=pathlib.Path(c[key]['path']);d=json.loads(p.read_text());d['files']={str(f.relative_to(p.parent)):{'size':f.stat().st_size,'sha256':sha(f)} for f in p.parent.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts};p.write_text(json.dumps(d,sort_keys=True,indent=2));c[key]['sha256']=sha(p)
    config=O/'hpc3-topd200-alpha1-resume5-teacher-sequence-fixed-20261013.json';config.write_text(json.dumps(c,indent=2))
    sys.path.insert(0,str(L/'runtime'));from student_binding import check;check(c)
    report={'status':'TEACHER_AUTHORIZED_RECOVERY_SEQUENCE_BOUND_AND_EXACT_GATE_TEST_PASS','epoch':time.time(),'failed_job':713661,'new_optimizer_updates':0,'teacher_start_outer':6,'original_backend_untouched':str(OLD),'backend':str(NEW),'config':str(config),'config_sha256':sha(config),'gate_cases_passed':12,'weights_and_objective_changed':False,'GPU_resumed_outer_not_yet_verified':True}
    (O/'alpha1-teacher-resume-repair.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
