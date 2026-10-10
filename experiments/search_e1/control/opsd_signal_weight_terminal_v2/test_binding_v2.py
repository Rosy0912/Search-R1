"""Exact new metadata topology fixtures, not production/file/tensor acceptance."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import capture as c


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.graph={}
        def ref(path,sha,value,bytecount=123):
            r=dict(path=str(path),sha256=sha,bytes=bytecount);self.graph[str(path)]=(value,r);return r
        self.ref=ref;oldroot=c.BASE/'source/training-00fec71/opsd_signal_weight_v1'
        newroot=c.BASE/'source/evaluation-recovery-31e2d74/opsd_signal_weight_eval_recovery_v2'
        helper=Path(c.__file__).resolve().parent
        own=dict(path=str(helper/'source-manifest.json'),sha256='9'*64)
        pub=dict(path='/fixture-own-publication',sha256='8'*64)
        self.own,self.pub=own,pub
        groups=[];configrefs=[]
        for variant in c.VARIANTS:
            cfg=dict(variant=variant,study_id='fixed-study',run_id=c.G_POS_RUN if variant=='G_POS' else 'fixture-'+variant,
                source_commit=c.OLD_COMMIT,owner=dict(uid=1001,physical_gpus=[4,5,6,7]),
                paths=dict(python='/original/python'),helper_source_manifest=dict(path=str(oldroot/'source-manifest.json'),sha256=c.OLD_MANIFEST))
            cr=ref(c.BASE/('group-'+variant)/'receipts/config.json',c.CONFIG_SHAS[variant],cfg)
            configrefs.append(cr);groups.append(dict(variant=variant,config=cr))
        study=dict(study_base=str(c.BASE),study_id='fixed-study',source_commit=c.OLD_COMMIT,
            helper_root=str(oldroot),helper_source_manifest=dict(path=str(oldroot/'source-manifest.json'),sha256=c.OLD_MANIFEST),
            publication=dict(path='/original-pub',sha256='7'*64),groups=groups)
        sr=ref(c.BASE/'receipts/study-manifest.json',c.STUDY_SHA,study)
        failure_root=c.BASE/'group-G_POS/runs/evaluations/outer-15'
        preserved_root=c.BASE/'preserved/fixture-failed15'
        cpref=ref(c.BASE/'group-G_POS/runs/checkpoints/update-00000150-outer-00014/manifest.json',c.CP150_SHA,
            dict(accepted=True,run_id=c.G_POS_RUN,config_sha256=c.CONFIG_SHAS['G_POS'],optimizer_updates=150,
                 cumulative_pair_exposure=4568,cursor=dict(outer_index=14,next_pair_cursor=308,usable_pair_count=308)))
        phase=dict(phase='evaluation-running',child_exit_evidence=[dict(actual_wait_returncode=1)])
        phaseref=ref(c.BASE/'group-G_POS/runs/phase-exits/fixture.json','6'*64,phase)
        ctr=dict(actual_wait_returncode=1,controller=dict(pid=1554467,starttick=687295549))
        ctrref=ref(c.BASE/'sequence-attempts/fixture/actual-controller-exit.json','5'*64,ctr)
        metadata=[]
        for i,path in enumerate([c.BASE/'group-G_POS/runs/status.json',c.BASE/'group-G_POS/runs/failure-actual-exits.json',
                                c.BASE/'study-status.json',c.BASE/'group-G_POS/receipts/scoped-cpu-processes.json']):
            original=dict(path=str(path),sha256=str(i+1)*64,bytes=123)
            pr=ref(preserved_root.with_name(preserved_root.name+'.metadata')/('%02d-'%i+path.name),original['sha256'],dict(fixture=i))
            metadata.append(dict(original=original,preserved=pr))
        failure=dict(schema='opsd-signal-first15-original-failure-recovery-metadata-not-an-execution-request-v1',
            variant='G_POS',run_id=c.G_POS_RUN,completed_outers=15,actual_optimizer_updates=150,cumulative_pair_exposure=4568,
            native_endpoint_accepted=False,missing_original_wave1_actual_wait_evidence=True,
            original_evaluator_actual_parent_wait_returncode=1,original_controller_actual_supervisor_wait_returncode=1,
            original_driver_actual_same_pane_exit_returncode=1,same_server_socket_and_pane=dict(pane=dict(session_id='$45',pane_id='%45',pane_dead='1',pane_dead_status='1')),
            eight_old_worker_processes_actual_absent=[dict(pid=10+i,starttick=20+i,state='ABSENT',wait_returncode='UNKNOWN') for i in range(8)],
            raw_checkpoint=cpref,original_native_evaluator_parent_terminal=[dict(reference=phaseref,value=phase)],
            original_group_controller_terminal=[dict(reference=ctrref)],failed_status=dict(reference=metadata[0]['original']),
            failure_actual_exits=dict(reference=metadata[1]['original']))
        fr=ref(c.BASE/'receipts/evaluation-recovery-failure-metadata-31e2d74.json',c.FAILURE_SHA,failure)
        source=dict(path=str(newroot/'source-manifest.json'),sha256=c.REPAIR_MANIFEST)
        recoverypub=dict(path=str(c.BASE/'receipts/evaluation-recovery-publication-31e2d74.json'),sha256='a'*64)
        fileold=dict(path=str(failure_root/'execution/request.json'),sha256='b'*64,bytes=45)
        filenew=dict(path=str(preserved_root/'execution/request.json'),sha256='b'*64,bytes=45)
        preservation=dict(schema='opsd-signal-failed-evaluation-preservation-v2',accepted=True,formal_endpoint_accepted=False,
            failure_metadata=fr,resume_checkpoint=cpref,source_manifest=source,publication=recoverypub,original_root=str(failure_root),
            preserved_root=str(preserved_root),actual_failed_evaluator_wait=1,actual_failed_controller_wait=1,
            actual_failed_supervisor_wait=1,original_worker_unknown_waits_not_promoted=True,old_absolute_references_must_resolve_with_this_mapping=True,
            files=[dict(original=fileold,preserved=filenew)],file_count=1,metadata_files=metadata,metadata_file_count=4)
        pr=ref(c.BASE/'receipts/evaluation15-failure-preserved-31e2d74.json',c.PRESERVATION_SHA,preservation)
        repair=dict(schema='opsd-signal-weight-evaluation-recovery-manifest-v2',accepted=True,
            scope='evaluation_hold_retry_and_existing_named_outer_old_fp32',source_commit=c.REPAIR_COMMIT,source_manifest=source,
            supervisor_run_id=c.MARKER,study_id='fixed-study',study_manifest=sr,original_configs=configrefs,helper_root=str(newroot),
            publication=recoverypub,evaluation_failure=fr,preservation=pr,resume_checkpoint=cpref)
        rr=ref(c.BASE/'receipts/evaluation-recovery-input-31e2d74.json',c.REPAIR_SHA,repair)
        argv=['/original/python',str(newroot/'entry.py'),'--mode','study','--recovery-ref',rr['path'],'--recovery-sha256',rr['sha256'],
              '--','--study-manifest',sr['path'],'--study-manifest-sha256',sr['sha256'],'--resume']
        script=('\n'.join(['#!/usr/bin/env bash','set -euo pipefail',"export CUDA_VISIBLE_DEVICES=''",'export PYTHONDONTWRITEBYTECODE=1',
            'export PYTHONUNBUFFERED=1','export E1_RUN_ID='+c.MARKER,'exec '+' '.join(argv)])+'\n').encode()
        scriptref=dict(path=str(c.BASE/'receipts/launch-evaluation-recovery-31e2d74.sh'),sha256=c.LAUNCH_SHA,bytes=len(script))
        inputpub=dict(accepted=True,branch='search-e1',input_publication_commit=c.INPUT_COMMIT,remote_readback_sha=c.INPUT_COMMIT,
            deployed_source_commit=c.REPAIR_COMMIT,files=[dict(path='experiments/search_e1/configs/signal-weight-4gpu-20261010/evaluation-hold-recovery-launch-31e2d74/'+name,
            sha256=p['sha256'],bytes=p['bytes'],remote_equal=True) for name,p in [('recovery-input.json',rr),('launch-evaluation-recovery.sh',scriptref)]])
        ipr=ref('/fixture-inputpub','c'*64,inputpub)
        log=str(c.BASE/'logs/evaluation-recovery-31e2d74.log')
        request=dict(schema='opsd-signal-weight-terminal-request-v2',accepted=True,owner=c.OWNER,supervisor_run_id=c.MARKER,
            study_manifest=sr,recovery_manifest=rr,launch_script=scriptref,launch_input_publication=ipr,log_path=log,
            tmux=dict(session=c.SESSION,session_id='$46',pane_id='%46',pane_pid=c.OWNER['pid'],server_pid=c.SERVER_PID,
                start_command="bash -lc 'tmux set-option -w remain-on-exit on; exec bash "+scriptref['path']+' > '+log+" 2>&1'"))
        self.requestref=ref('/fixture-request','d'*64,request)
        self.request,self.repair,self.failure,self.preservation,self.inputpub=request,repair,failure,preservation,inputpub
        self.script,self.scriptref=script,scriptref

    def bound(self,reference):
        value,proof=self.graph[reference['path']]
        c.require(c.same_ref(reference,proof),'Fixture exact pinned SHA mismatch')
        return copy.deepcopy(value),copy.deepcopy(proof)

    def check(self):
        def read(path,expected=None):
            c.require(str(path)==self.scriptref['path'] and expected==c.LAUNCH_SHA,'Fixture wrong launch bytes')
            return self.script,copy.deepcopy(self.scriptref)
        with patch.object(c,'bound',side_effect=self.bound),patch.object(c,'read',side_effect=read),patch.object(c,'tree',return_value={'fixture_only':True}):
            return c.load_inputs(self.requestref,self.own,self.pub)

    def reject(self):
        with self.assertRaises(c.Hold):self.check()

    def test_fixed_new_metadata_topology_and_prior_failure_accept(self):
        value,proof,study,configs=self.check();self.assertEqual(value['owner'],c.OWNER)
        self.assertEqual(proof['prior_failed_evaluation']['prior_actual_waits'],[1,1,1])
        self.assertEqual(proof['prior_failed_evaluation']['orphan_worker_waits_remain_unknown'],8)
        self.assertFalse(proof['prior_failed_evaluation']['failed_tree_payload_fresh_SHA'])
        self.assertEqual(len(configs),4)
    def test_old_request_schema_rejected(self):self.request['schema']='opsd-signal-weight-terminal-request-v1';self.reject()
    def test_old_supervisor_rejected(self):self.request['owner']=dict(pid=1554337,uid=1001,starttick=687293806);self.reject()
    def test_old_pane_cannot_replace_new_pane(self):self.request['tmux']['pane_id']='%45';self.reject()
    def test_old_marker_rejected(self):self.request['supervisor_run_id']='signal-study-recovery-old';self.reject()
    def test_wrong_recovery_scope_rejected(self):self.repair['scope']='named_outer_old_fp32_before_setter_only';self.reject()
    def test_wrong_recovery_commit_rejected(self):self.repair['source_commit']='e'*40;self.reject()
    def test_wrong_recovery_manifest_rejected(self):self.repair['source_manifest']['sha256']='e'*64;self.reject()
    def test_input_readback_commit_mismatch_rejected(self):self.inputpub['remote_readback_sha']='f'*40;self.reject()
    def test_input_source_commit_mismatch_rejected(self):self.inputpub['deployed_source_commit']='f'*40;self.reject()
    def test_failed_remote_input_readback_rejected(self):self.inputpub['files'][0]['remote_equal']=False;self.reject()
    def test_other_foreground_log_rejected(self):self.request['log_path']='/log-other';self.reject()
    def test_another_birth_command_rejected(self):self.request['tmux']['start_command']+='; true';self.reject()
    def test_old_driver_wait_one_cannot_be_zeroed(self):self.failure['original_driver_actual_same_pane_exit_returncode']=0;self.reject()
    def test_old_pane_one_cannot_be_zeroed(self):self.failure['same_server_socket_and_pane']['pane']['pane_dead_status']='0';self.reject()
    def test_unknown_old_worker_wait_cannot_be_zeroed(self):self.failure['eight_old_worker_processes_actual_absent'][0]['wait_returncode']=0;self.reject()
    def test_failed512_cannot_be_marked_formal(self):self.preservation['formal_endpoint_accepted']=True;self.reject()
    def test_missing_mutable_failure_copy_rejected(self):self.preservation['metadata_files'].pop();self.reject()
    def test_wrong_old_to_new_mapping_bytes_rejected(self):self.preservation['files'][0]['preserved']['bytes']=46;self.reject()
    def test_wrong_actual_parent_wait_rejected(self):self.failure['original_native_evaluator_parent_terminal'][0]['value']['child_exit_evidence'][0]['actual_wait_returncode']=0;self.reject()
    def test_wrong_cp_or_run_rejected(self):
        cp=self.graph[self.repair['resume_checkpoint']['path']][0];cp['run_id']='other-run';self.reject()


if __name__=='__main__':unittest.main()
