"""One existing-ledger GPU read plus exact new3ecc preclaim-failure metadata.

No project process signals, model payloads, remote writes, or launch retries.
The original A9 reader rechecks all historical project births and inventories
only the assigned six GPUs; the extension separately binds new3ecc evidence.
"""
import a800_actor_probe_gpu_ownership_read as reader

EXTRA = r'''
    stage='new3ecc_existing_preclaim_metadata'
    newbase='/data/shaoshitong/Search-G1-a800-fresh200-76398c90bc554e7cbd81d465a547eb71'
    expected_new=dict(run_id='3ecc1a1afd9948d18d1d5d670c547b51',
        execution_id='6fd0219e4f56467b938ad3e549ad8ad4',source_commit=p['source_commit'])
    evidence=dict(base=newbase,identity=expected_new,files=[],launch_allowed=False,
        remote_write_attempted=False,trainer_started=False,complete_outers=0,
        processed_opportunities=0,successful_adam=0,skips=0)
    names=('run-manifest.json','deployment-launch-receipt.json','deployment-pipeline.log')
    def existing_new_bytes(name):
        need(name in names,'new_metadata_path_scope')
        fd_directory=open_directory(newbase+'/logs')
        try:
            fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd_directory)
            try:
                info=os.fstat(fd)
                need(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid()
                     and info.st_size<=FILE_LIMIT,'new_metadata_type_owner_size')
                raw=b''
                while True:
                    remaining();chunk=os.read(fd,min(65536,FILE_LIMIT+1-len(raw)))
                    if not chunk:break
                    raw+=chunk;need(len(raw)<=FILE_LIMIT,'new_metadata_limit')
                need(stable_stat(info)==stable_stat(os.fstat(fd)) and len(raw)==info.st_size,'new_metadata_changed')
            finally:os.close(fd)
        finally:os.close(fd_directory)
        evidence['files'].append(dict(relative='logs/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        return raw
    current=json.loads(existing_new_bytes('run-manifest.json'))
    launch=json.loads(existing_new_bytes('deployment-launch-receipt.json'))
    need(all(current[k]==launch[k]==value for k,value in expected_new.items()),'new_preclaim_identity')
    need(current['execution_host']==launch['hostname']==before_node['hostname'],'new_preclaim_host')
    need(current['completed_updates']==0 and current['training_world_size']==4,'new_zero_state')
    need(launch['deployment_id']=='76398c90bc554e7cbd81d465a547eb71','new_deployment_identity')
    need(launch['pid']==2592914 and launch['start_ticks']==602487016 and launch['sid']==launch['pid'],'new_pipeline_birth_binding')
    prior=before.get(launch['pid'],dict(pid=launch['pid'],read_status='not_observed_before'))
    actual=birth(launch['pid'])
    gone=(actual['read_status']=='absent_or_exited' or
          actual['read_status']=='read' and
          (actual['start_ticks']!=launch['start_ticks'] or actual['state'] in ('Z','X','x')))
    evidence['pipeline_identity']=dict(pid=launch['pid'],start_ticks=launch['start_ticks'],sid=launch['sid'])
    evidence['pipeline_before']=prior;evidence['pipeline_after']=actual
    evidence['pipeline_exited_verified']=gone;need(gone,'new_pipeline_not_terminal')
    log=existing_new_bytes('deployment-pipeline.log').decode('utf-8',errors='replace')
    lines=[line.strip() for line in log.splitlines()
           if line.strip().startswith('AssertionError: Authorized local GPUs are occupied:')]
    need(len(lines)==1 and len(lines[0])<=8192,'actual_native_occupancy_error_required')
    evidence['actual_startup_error']=lines[0]
    need('reproduction/common8/native_allocation.py' in log and 'in admit_local' in log,'actual_native_gate_trace_required')
    absent={name:not (os.path.lexists(newbase+'/logs/'+name)) for name in
        ('pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json',
         'native-allocation-admission.json','pipeline-status.json','actual-updates.jsonl',
         'actual-attempts.jsonl','search-g1-common8-training.log')}
    evidence['pretrainer_artifacts_absent']=absent;need(all(absent.values()),'new_pretrainer_artifact_present')
    evidence['all_A9_registered_births_exited']=all(
        record['after']['read_status']=='absent_or_exited' or
        record['after']['read_status']=='read' and
        (record['after']['start_ticks']!=record['start_ticks'] or record['after']['state'] in ('Z','X','x'))
        for job in out['jobs'] for record in job['registered_identities'])
    evidence['new_checkpoint_inventory_empty']=not os.listdir(newbase+'/checkpoints')
    evidence['scope']='actual new3ecc pretrainer occupancy failure; A9historical-ledger ownership remains separately bound'
    evidence['observed_unix']=time.time()
    out['new3ecc_preclaim_failure']=evidence
'''


if __name__ == '__main__':
    marker="except BaseException as error:\n    # Never print source JSON"
    assert reader.REMOTE_BODY.count(marker)==1
    reader.REMOTE_BODY=reader.REMOTE_BODY.replace(marker,EXTRA+'\n'+marker)
    raise SystemExit(reader.main())
