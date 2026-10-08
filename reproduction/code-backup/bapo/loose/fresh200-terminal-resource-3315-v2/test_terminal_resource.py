import copy
import unittest
import terminal_resource as v


def slurm():
    fields=dict(UserId='root(0)',ReqNodeList='h100-5',JobId='3315',JobState='COMPLETED',ExitCode='0:0',Restarts='0',Requeue='0',NodeList='h100-5',NumNodes='1',BatchHost='h100-5',
        Comment=v.COMMENT,Command=str(v.ROOT/'tools/formal200.sbatch'),WorkDir=str(v.ROOT),SuspendTime='None',SecsPreSuspend='0',
        StartTime='2026-10-07T01:00:00',EndTime='2026-10-08T02:00:00',RunTime='1-01:00:00',TRES='gres/gpu=8')
    return dict(job_id=3315,returncode=0,stdout=' '.join(k+'='+val for k,val in fields.items()))


def guard():return dict(status='PASS_ACTUAL_GUARD_JOURNAL_REPLAY',source_commit=v.SOURCE,job_id=3315,run_root=str(v.RUN),
    unfinished_recovery=False,abort_events=0,allocated_uuids=['GPU-'+str(i)for i in range(8)],samples=2,recovery_samples=3,
    freeze_events=8,safe_resume_events=1,prelaunch_recoveries=0,cost_wall_seconds=60)


def cleanup():return dict(own_marker=str(v.RUN),stopped_pids=[100],remaining=[])


class TerminalTests(unittest.TestCase):
    def test_reserved_gpu_hours_not_utilization(self):
        out=v.terminal_slurm(slurm());self.assertEqual(out['reserved_gpu_hours'],200);self.assertEqual(out['elapsed_seconds'],90000)
    def test_running_refused(self):
        r=slurm();r['stdout']=r['stdout'].replace('COMPLETED','RUNNING')
        with self.assertRaises(ValueError):v.terminal_slurm(r)
    def test_failure_refused(self):
        r=slurm();r['stdout']=r['stdout'].replace('ExitCode=0:0','ExitCode=1:0')
        with self.assertRaises(ValueError):v.terminal_slurm(r)
    def test_restart_or_requeue_refused(self):
        for key in ('Restarts','Requeue'):
            r=slurm();r['stdout']=r['stdout'].replace(key+'=0',key+'=1')
            with self.assertRaises(ValueError):v.terminal_slurm(r)
    def test_elapsed_mismatch_refused(self):
        r=slurm();r['stdout']=r['stdout'].replace('1-01:00:00','1-00:00:00')
        with self.assertRaises(ValueError):v.terminal_slurm(r)
    def test_foreign_job_comment_refused(self):
        r=slurm();r['stdout']=r['stdout'].replace(v.NONCE,'foreign')
        with self.assertRaises(ValueError):v.terminal_slurm(r)
    def test_gpu_count_conflict(self):
        r=slurm();r['stdout']=r['stdout'].replace('gpu=8','gpu=1')
        with self.assertRaises(ValueError):v.terminal_slurm(r)
    def test_cleanup_good_synthetic_scope(self):
        out=v.check_cleanup(cleanup(),{100:{123}},guard(),{'original':'0','cleanup':'0'},{'cleanup':b'','gpu_exit':b''})
        self.assertEqual(out['stopped_pid_epochs'],{'100':123})
    def test_cleanup_foreign_or_remaining(self):
        for item in ({'own_marker':'foreign'}, {'remaining':[{'pid':100}]}):
            c=cleanup();c.update(item)
            with self.assertRaises(ValueError):v.check_cleanup(c,{100:{123}},guard(),{'original':'0','cleanup':'0'},{'cleanup':b'','gpu_exit':b''})
    def test_cleanup_missing_reused_pid_epoch(self):
        for epochs in ({},{100:{123,124}}):
            with self.assertRaises(ValueError):v.check_cleanup(cleanup(),epochs,guard(),{'original':'0','cleanup':'0'},{'cleanup':b'','gpu_exit':b''})
    def test_unfinished_or_nonfinite_guard(self):
        for change in ({'unfinished_recovery':True},{'cost_wall_seconds':float('nan')},{'samples':0}):
            g=guard();g.update(change)
            with self.assertRaises(ValueError):v.check_cleanup(cleanup(),{100:{123}},g,{'original':'0','cleanup':'0'},{'cleanup':b'','gpu_exit':b''})
    def test_duplicate_json_and_nonfinite(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):v.strict_json(raw)
    def test_monitor_nonfinite(self):
        with self.assertRaises(ValueError):v.historical_epochs([], [{'time':float('inf'),'processes':[]}])
    def test_gpu_exit_uuid_set(self):
        uuids=guard()['allocated_uuids'];raw=('index, uuid, utilization.gpu [%]\n'+'\n'.join('%d, %s, 0 %%'%(i,u)for i,u in enumerate(uuids))).encode()
        self.assertFalse(v.check_gpu_exit(raw,uuids)['foreign_process_absence_inferred'])
        with self.assertRaises(ValueError):v.check_gpu_exit(raw.replace(b'GPU-0',b'GPU-9'),uuids)
    def test_cleanup_failure_not_hidden(self):
        with self.assertRaises(ValueError):v.check_cleanup(cleanup(),{100:{123}},guard(),{'original':'0','cleanup':'1'},{'cleanup':b'','gpu_exit':b''})

    def test_actual_userid_missing_wrong_rejected(self):
        valid=slurm();self.assertEqual(v.terminal_slurm(valid,0)['allocation_uid'],0)
        for text in (valid['stdout'].replace('UserId=root(0) ','') ,valid['stdout'].replace('root(0)','foreign(99)'),valid['stdout'].replace('root(0)','0')):
            with self.assertRaises(ValueError):v.terminal_slurm(dict(valid,stdout=text),0)
    def test_requested_node_and_tres_per_node_conflict(self):
        r=slurm()
        for text in (r['stdout'].replace('ReqNodeList=h100-5','ReqNodeList=h100-6'),r['stdout']+' TresPerNode=gres/gpu:1'):
            with self.assertRaises(ValueError):v.terminal_slurm(dict(r,stdout=text),0)
    def test_guard_exact_full_journal_binding(self):
        journal=b'{"synthetic_test":1}\n';g=guard()
        g.update(raw_path=str(v.RUN/'resource-watch.jsonl'),raw_sha256=v.digest(journal),own_uid=0)
        terminal=v.terminal_slurm(slurm(),0)
        v.check_replay_binding(g,journal,terminal,0)
        for change in ({'raw_sha256':v.digest(b'other journal\n')},{'raw_path':'/foreign/resource-watch.jsonl'},{'own_uid':99}):
            bad=dict(g,**change)
            with self.assertRaises(ValueError):v.check_replay_binding(bad,journal,terminal,0)
        with self.assertRaises(ValueError):v.check_replay_binding(g,journal,terminal,99)

    def test_actual_lease_owner_counterexample(self):
        uuids=guard()['allocated_uuids'];records=[dict(uuid=u,fd=i,inode=1000+i,exclusive_lock_record='%d: FLOCK ADVISORY WRITE 100 00:01:%d 0 EOF'%(i,1000+i))for i,u in enumerate(uuids)]
        collection=dict(status='ACTUAL_FIRST5_COLLECTION_REQUIRES_INDEPENDENT_REVIEW',source_commit=v.SOURCE,job_id=3315,run_root=str(v.RUN),
            leases=dict(guard=dict(pid=100,start_ticks=123,selected_env={'BAPO_RUN_ID':str(v.RUN),'SLURM_JOB_ID':'3315'}),leases=records,
            verification='read_only_proc_fd_inode_and_proc_locks_no_flock_mutation'))
        self.assertEqual(v.check_leases(collection,uuids)['guard_pid'],100)
        collection['leases']['leases'][0]['exclusive_lock_record']='0: FLOCK ADVISORY WRITE 999 00:01:1000 0 EOF'
        with self.assertRaises(ValueError):v.check_leases(collection,uuids)


if __name__=='__main__':unittest.main()
