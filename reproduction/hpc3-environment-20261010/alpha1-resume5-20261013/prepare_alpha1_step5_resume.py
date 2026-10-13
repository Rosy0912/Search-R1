"""Prepare the explicitly authorized alpha=1 step5 recovery; never submit."""
import ast, hashlib, json, pathlib, shutil, subprocess, sys, time
sys.dont_write_bytecode = True
BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
ROOT = BASE/'runtime'
OUT = BASE/'outputs/training-preparation'
OLD = ROOT/'tools/topd-alpha1-launch-validation-repair-20261013'
NEW = ROOT/'tools/topd-alpha1-launch-resume5-20261013'
CKPT = ROOT/'logs/topd-hpc3-712204/evidence/checkpoints/actor/global_step_5'

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    jobs=subprocess.check_output(['squeue','-u','mou176','-h','-o','%i|%j|%T'],text=True)
    assert not any('topd-alpha1' in s for s in jobs.splitlines()), jobs
    assert '712204|FAILED|1:0' in subprocess.check_output(['sacct','-j','712204','-n','-P','--format=JobID,State,ExitCode'],text=True)
    receipt=json.loads((CKPT/'resume-complete.json').read_text())
    assert receipt['status']=='ALL_FOUR_RANKS_AND_DRIVER_RESUME_STATE_VERIFIED' and receipt['outer_step']==5
    records=receipt['ranks']+[receipt['driver']]
    verified=[]
    for r in records:
        p=pathlib.Path(r['path']);assert p.parent==CKPT and not p.is_symlink()
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        verified.append({'path':str(p),'bytes':r['bytes'],'sha256':r['sha256']})
        print(json.dumps({'fresh_verified_file':p.name}),flush=True)
    if NEW.exists():raise RuntimeError('Prepared resume launcher already exists; inspect it instead of overwriting')
    shutil.copytree(OLD,NEW,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in NEW.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.sbatch','.sh'):
            s=p.read_text().replace(str(OLD),str(NEW)).replace('$TASK_ROOT/tools/topd-alpha1-launch-validation-repair-20261013','$TASK_ROOT/tools/topd-alpha1-launch-resume5-20261013')
            if p.suffix=='.py':ast.parse(s)
            p.write_text(s)
    p=NEW/'runtime/topd_resume_state.py';s=p.read_text()
    s=s.replace("            state=torch.load(pathlib.Path(checkpoint)/('resume-rank%d.pt'%rank),map_location='cpu',weights_only=False)","""            root=pathlib.Path(checkpoint)
            complete=json.loads((root/'resume-complete.json').read_text())
            if complete['status']!='ALL_FOUR_RANKS_AND_DRIVER_RESUME_STATE_VERIFIED' or complete['outer_step']!=5:raise RuntimeError('Wrong authorized recovery point')
            proof=next(x for x in complete['ranks'] if x['rank']==rank)
            path=root/('resume-rank%d.pt'%rank)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            if path.stat().st_size!=proof['bytes'] or h.hexdigest()!=proof['sha256']:raise RuntimeError('Actual rank recovery bytes differ')
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['outer_step']!=5 or len(state['resource_updates'])!=5:raise RuntimeError('Wrong recovery outer')""")
    marker='            torch.distributed.barrier()\n        return result'
    assert marker in s
    s=s.replace(marker,"""            if not equal_tree({n:p.detach().cpu() for n,p in params.items()},state['model_local']):raise RuntimeError('Restored full local model differs')
            if not equal_tree(cpu_tree(self.actor_optimizer.state_dict()),state['optimizer']):raise RuntimeError('Restored Adam differs')
            if not equal_tree(self.actor_lr_scheduler.state_dict(),state['scheduler']):raise RuntimeError('Restored scheduler differs')
            if not equal_tree(self._resource_updates,state['resource_updates']):raise RuntimeError('Restored actual update history differs')
            if not equal_tree(random.getstate(),state['python_rng']) or not equal_tree(np.random.get_state(),state['numpy_rng']) or not equal_tree(torch.get_rng_state(),state['torch_rng']) or not equal_tree(torch.cuda.get_rng_state(),state['cuda_rng']):raise RuntimeError('Restored worker RNG differs')
            from pipeline import atomic
            record={'status':'ACTUAL_GPU_WORKER_STEP5_FULL_RESTORE_EQUALITY_PASS','rank':rank,'outer_step':5,'checkpoint':str(root),'checkpoint_sha256':proof['sha256'],'model_local_equal':True,'adam_equal':True,'scheduler_equal':True,'rng_equal':True,'actual_update_history_equal':True,'optimizer_steps':100,'scheduler_last_epoch':self.actor_lr_scheduler.last_epoch}
            atomic(pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY']).parent/('resume-loaded-rank%d.json'%rank),record)
            torch.distributed.barrier()
        return result""")
    old="        order=state['data_order'];trainer.global_steps=state['next_outer_step'];trainer._topd_resume_batch=state['next_batch']"
    assert old in s
    s=s.replace(old,"""        order=state['data_order']
        if state['outer_step']!=5 or state['next_outer_step']!=6 or state['next_batch']!=5 or len(order)!=len(trainer.train_dataset) or sorted(order)!=list(range(len(trainer.train_dataset))):raise RuntimeError('Recovery driver data position/order differs')
        trainer.global_steps=state['next_outer_step'];trainer._topd_resume_batch=state['next_batch']""")
    old="        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])"
    assert old in s
    s=s.replace(old,old+"""
        if not equal_tree(random.getstate(),state['python_rng']) or not equal_tree(np.random.get_state(),state['numpy_rng']) or not equal_tree(torch.get_rng_state(),state['torch_rng']):raise RuntimeError('Restored driver RNG differs')
        from pipeline import atomic
        atomic(pathlib.Path(os.environ['TOPD_RUN_DIRECTORY'])/'resume-loaded-driver.json',{'status':'ACTUAL_DRIVER_STEP5_RECOVERY_POSITION_RNG_PASS','outer_step':5,'next_outer_step':6,'next_batch':5,'train_rows':len(order),'order_sha256':hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),'checkpoint':str(checkpoint),'rng_equal':True})""")
    ast.parse(s);p.write_text(s)
    p=NEW/'runtime/train_entry.py';s=p.read_text();assert '   evaluate_stage(self,0)' in s
    s=s.replace('   evaluate_stage(self,0)',"   if not self.config.trainer.get('topd_resume_checkpoint',None):evaluate_stage(self,0)")
    ast.parse(s);p.write_text(s)
    p=NEW/'runtime/supervise.py';s=p.read_text()
    marker=" root=pathlib.Path(c['evidence_root']);root.mkdir(mode=0o700,exist_ok=False)"
    assert marker in s
    s=s.replace(marker,marker+"""
 if c.get('resume_authorization'):
  origin=pathlib.Path(c['resume_authorization']['prior_outer_receipt'])
  previous=json.loads(origin.read_text());first=[x for x in previous['records'] if x['outer_step']<=5]
  if [x['outer_step'] for x in first]!=list(range(1,6)):raise RuntimeError('Actual recovery lineage 1..5 missing')
  atomic(root/'actual-outer-updates.json',{'actual_completed_outer_steps':list(range(1,6)),'records':first,'export_step':5,'checkpoint_directory':c['resume_authorization']['checkpoint'],'scope':'ACTUAL_PRIOR_RUN_OUTERS_1_TO_5_USER_AUTHORIZED_RECOVERY_LINEAGE','resume_lineage':c['resume_authorization']})
  atomic(root/'resume-authorization.json',c['resume_authorization'])""")
    ast.parse(s);p.write_text(s)
    c=json.loads((OUT/'hpc3-topd200-alpha1-validation-repair-PREPARED-NOT-SUBMITTED.json').read_text())
    c=json.loads(json.dumps(c).replace(str(OLD),str(NEW)))
    c['training_overrides'] += ['+actor_rollout_ref.topd_resume_checkpoint='+str(CKPT),'+trainer.topd_resume_checkpoint='+str(CKPT)]
    c['experiment_arm']='TOPD_ALPHA_1_USER_AUTHORIZED_STEP5_FULL_STATE_RESUME'
    c['required_protocol']['fresh_initialization']=False
    c['resume_authorization']={'human_choice':'从已校验的第5轮恢复，重算第6轮起（推荐）','recorded_epoch':time.time(),'prior_job':712204,'checkpoint':str(CKPT),'outer_step':5,'next_outer_step':6,'prior_outer_receipt':str(ROOT/'logs/topd-hpc3-712204/evidence/actual-outer-updates.json'),'original_official_initialization':True,'recompute_outers_6_to_50_authorized':True,'fresh_sha_verified_files':verified}
    for key in ['tools_manifest','launcher_manifest']:
        p=pathlib.Path(c[key]['path']);d=json.loads(p.read_text());d['files']={str(f.relative_to(p.parent)):{'size':f.stat().st_size,'sha256':sha(f)} for f in p.parent.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts};p.write_text(json.dumps(d,sort_keys=True,indent=2));c[key]['sha256']=sha(p)
    config=OUT/'hpc3-topd200-alpha1-resume5-20261013.json';config.write_text(json.dumps(c,indent=2))
    sys.path.insert(0,str(NEW/'runtime'))
    from supervise import verify_recipe
    from student_binding import check
    assert verify_recipe(c)==200;check(c)
    result={'epoch':time.time(),'status':'USER_AUTHORIZED_ALPHA1_STEP5_RECOVERY_PREPARED_FRESH_SHA_VERIFIED_NOT_SUBMITTED','config':str(config),'launcher':str(NEW),'checkpoint':str(CKPT),'files':verified,'GPU_restore_tested':False,'configuration_sha256':sha(config)}
    (OUT/'alpha1-resume5-preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
