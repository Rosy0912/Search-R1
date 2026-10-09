"""Synthetic tiny CPU monitor fixtures: counter, identity and evidence semantics."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_signal_monitor_test_module',HERE/'monitor.py')
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
production=Path(os.environ.get('OPSD_SIGNAL_TEST_PRODUCTION_HELPER',HERE.parent/'opsd_signal_weight_v1'))
cp=types.ModuleType('_signal_monitor_bound_checkpoint_fixture');cp.__file__=str(production/'checkpoint.py')
exec(compile((production/'checkpoint.py').read_bytes(),cp.__file__,'exec'),cp.__dict__)


class MonitorContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.output=self.root/'runs';self.output.mkdir()
        runtime=self.put('runtime-ref.json',{'fixture':True});source=self.put('source-ref.json',{'fixture':True})
        initial=self.put('initial-ref.json',{'fixture':True})
        self.cfg=dict(run_id='a'*32,_config_sha256='b'*64,variant='G_POS',precision={'mode':'bf16'},
            runtime_manifest=runtime,source_model_manifest=source,common_initial_adapter_manifest=initial,
            scientific={'global_batch':32},paths={'output_root':str(self.output)})

    def put(self,name,value):
        return cp.exclusive_json(self.root/name,value)

    def bytes(self,name,raw=b'synthetic bytes only'):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return cp.fileproof(path)

    def checkpoint(self,outer,K,cursor,updates,exposure,base_updates=0):
        directory=self.output/'checkpoints'/('update-%08d-outer-%05d'%(updates,outer));directory.mkdir(parents=True)
        ranks=[dict(rank=i,file=self.bytes(str((directory/('rank-%d-state.pt'%i)).relative_to(self.root)),('rank'+str(i)).encode())) for i in range(4)]
        optimizer=self.bytes(str((directory/'optimizer-shared.pt').relative_to(self.root)))
        files=[self.bytes(str((directory/'adapter/adapter_config.json').relative_to(self.root)),b'{}'),
            self.bytes(str((directory/'adapter/adapter_model.safetensors').relative_to(self.root)),b'not-real-tensors; identity-only fixture')]
        dummy=self.put('input-%d.json'%outer,{'identity':outer})
        receipts=[];exposed=0
        for number in range(1,math_ceil(cursor)+1):
            actual=min(32,cursor-exposed);metrics=[]
            for rank in range(4):
                pairs=[dict(pair_index=i,valid_tokens=2,raw_loss=.3,post_loss=.3,weighted_post_loss=.3,pair_weight=1.)
                    for i in range(exposed+rank,exposed+actual,4)]
                metrics.append(dict(rank=rank,pairs=pairs))
            receipts.append(self.put('updates/outer-%d-mini-%d.json'%(outer,number),dict(schema='opsd-online-real-update-v1',
                accepted=True,run_id=self.cfg['run_id'],outer_index=outer,optimizer_updates=base_updates+number,
                global_actual_pairs=actual,all_rank_actual_token_metrics=metrics)))
            exposed+=actual
        value=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=self.cfg['run_id'],config_sha256=self.cfg['_config_sha256'],
            contract=cp.science_contract(self.cfg),runtime_identity=self.cfg['runtime_manifest']['sha256'],
            source_model_identity=self.cfg['source_model_manifest']['sha256'],common_initial_adapter=self.cfg['common_initial_adapter_manifest'],
            outer_old_adapter=self.cfg['common_initial_adapter_manifest'],all_rank_scaler_state=[{'enabled':False}]*4,
            optimizer_updates=updates,cumulative_pair_exposure=exposure,complete_rank_count=4,equal_optimizer_steps=True,
            ranks=ranks,optimizer_storage='shared_exact_four_rank_optimizer_v1',shared_optimizer=optimizer,
            all_rank_optimizer_tensor_sha256=['c'*64]*4,adapter={'path':str(directory/'adapter'),'files':files},
            all_rank_adapter_sha256=['d'*64]*4,all_rng=['python','numpy','torch_cpu','cuda_all'],scheduler={'kind':'none','warmup_updates':0},
            cursor={'outer_index':outer,'next_pair_cursor':cursor,'usable_pair_count':K},
            training_input={k:dummy for k in ('outer_manifest','pairs','order','encoding')},sampling_model_identity={'sha256':'e'*64},
            outer_update_receipts=receipts)
        return self.put(str((directory/'manifest.json').relative_to(self.root)),value)

    def actual_exit(self,phase='training-running',rc=0):
        exits=[dict(identity={'pid':42,'uid':1001,'starttick':99,'run_id':self.cfg['run_id']},actual_wait_returncode=rc)]
        actual=self.put('exits/'+phase+'-rc-'+str(rc)+'.json',dict(run_id=self.cfg['run_id'],config_sha256=self.cfg['_config_sha256'],phase=phase,child_exit_evidence=exits))
        return dict(accepted=True,actual_terminal=actual,child_exit_evidence=exits,gpu_release={'all_owned_release':{'accepted':True}})

    def completed_outer(self,K=33,P=35):
        checkpoint=self.checkpoint(0,K,K,math_ceil(K),K)
        outer=self.output/'outers/outer-00000';outer.mkdir(parents=True)
        self.put(str((outer/'outer-manifest.json').relative_to(self.root)),dict(outer_index=0,run_id=self.cfg['run_id'],config_sha256=self.cfg['_config_sha256'],
            P=P,previous_complete_checkpoint=None,cumulative_optimizer_updates_before=0))
        completion=self.put(str((outer/'training/outer-complete.json').relative_to(self.root)),dict(schema='opsd-online-outer-complete-v1',accepted=True,
            run_id=self.cfg['run_id'],outer_index=0,P=P,K=K,optimizer_updates_delta=math_ceil(K),optimizer_updates=math_ceil(K),
            cumulative_pair_exposure=K,checkpoint=checkpoint))
        summary=dict(accepted=True,variant='G_POS',run_id=self.cfg['run_id'],config_sha256=self.cfg['_config_sha256'],outer_index=0,displayed_iteration=1,
            train_correct=1000,train_denominator=2560,train_em=1000/2560,train_post_update=False,all5_wrong=7,P=P,K=K,
            optimizer_updates=math_ceil(K),optimizer_updates_delta=math_ceil(K),actual_outer_unweighted_post_loss=.3 if K else None,
            actual_outer_weighted_post_loss=.3 if K else None,weight={'mean':1. if K else None})
        self.put(str((outer/'summary.json').relative_to(self.root)),summary)
        self.put(str((outer/'trainer-logs/attempt-00000-exit.json').relative_to(self.root)),self.actual_exit())
        return checkpoint,completion

    def test_native_outer_and_partial_next_adam_are_separate_and_tail_exact(self):
        self.completed_outer();self.checkpoint(1,65,32,3,65,base_updates=2)
        value=monitor.train_progress(self.cfg,cp)
        self.assertEqual(value['native_completed_outers'],1);self.assertEqual(value['controller_accepted_completed_outers'],1)
        self.assertEqual(value['completed_chain_optimizer_updates'],2);self.assertEqual(value['completed_chain_K'],33)
        latest=value['latest_committed_checkpoint'];self.assertEqual(latest['optimizer_updates'],3)
        self.assertEqual(latest['next_pair_cursor'],32);self.assertEqual(latest['usable_pairs'],65)
        self.assertTrue(latest['payload_fresh_sha']);self.assertFalse(latest['optimizer_tensor_numeric_deserialization'])
        self.assertEqual(value['train_em_rows'][0]['train_denominator'],2560)

    def test_zero_usable_outer_counts_one_outer_and_zero_adam(self):
        self.completed_outer(K=0,P=0);value=monitor.train_progress(self.cfg,cp)
        self.assertEqual(value['native_completed_outers'],1);self.assertEqual(value['completed_chain_optimizer_updates'],0)
        self.assertEqual(value['latest_committed_checkpoint']['committed_minis'],0)

    def test_pending_checkpoint_never_becomes_committed(self):
        pending=self.output/'checkpoints/update-00009999-outer-00000.pending-fixture';pending.mkdir(parents=True)
        (pending/'manifest.json').write_text('{"accepted":true}')
        value=monitor.train_progress(self.cfg,cp);self.assertIsNone(value['latest_committed_checkpoint'])

    def test_latest_payload_corruption_rejected_without_tensor_deserialization(self):
        checkpoint,_=self.completed_outer();body=json.loads(Path(checkpoint['path']).read_text())
        Path(body['shared_optimizer']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):monitor.train_progress(self.cfg,cp)

    def test_nonzero_wait_evidence_does_not_accept_controller_outer(self):
        self.completed_outer();path=self.output/'outers/outer-00000/trainer-logs/attempt-00000-exit.json'
        path.unlink();self.put(str(path.relative_to(self.root)),self.actual_exit(rc=1))
        with self.assertRaisesRegex(monitor.EvidenceError,'Unknown/nonzero'):monitor.train_progress(self.cfg,cp)

    def test_bad_actual_tail_and_2560_denominator_are_rejected(self):
        checkpoint,_=self.completed_outer();body=json.loads(Path(checkpoint['path']).read_text());ref=body['outer_update_receipts'][-1]
        update=json.loads(Path(ref['path']).read_text());update['global_actual_pairs']=32
        Path(ref['path']).write_text(json.dumps(update));body['outer_update_receipts'][-1]=monitor.proof(ref['path'])
        Path(checkpoint['path']).write_text(json.dumps(body))
        with self.assertRaises(monitor.EvidenceError):monitor.committed_updates(body,0)
        bad=dict(accepted=True,run_id=self.cfg['run_id'],config_sha256=self.cfg['_config_sha256'],variant='G_POS',outer_index=0,
            train_denominator=33,train_post_update=False)
        with self.assertRaisesRegex(monitor.EvidenceError,'Full2560'):monitor.compact_train(bad,0,self.cfg)

    def test_absent_or_pid_reuse_is_never_exit_zero(self):
        owner={'pid':2,'uid':1001,'starttick':8,'run_id':'current'}
        gate=types.SimpleNamespace(process=lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
        value=monitor.identity_state(owner,gate);self.assertEqual(value['state'],'ABSENT');self.assertEqual(value['wait_returncode'],'UNKNOWN')
        gate.process=lambda pid:dict(owner,starttick=9)
        with self.assertRaisesRegex(monitor.EvidenceError,'reused'):monitor.identity_state(owner,gate)

    def test_partial_512_counts_complete_lines_and_rejects_duplicate_wrong_rank(self):
        directory=self.root/'evaluation';path=directory/'worker-0/results.jsonl';path.parent.mkdir(parents=True)
        request={'run_id':'a'*32,'evaluation_id':'f'*32}
        row=dict(validation_index=0,run_id=request['run_id'],evaluation_id=request['evaluation_id'],status='complete',primary=True,
            protocol_em=0,protocol_valid_final_action=False,failure_class='final_invalid',search_calls=0)
        path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,validation_index=8))+'\n{"partial":')
        progress=monitor.partial_slots(directory,request);self.assertEqual(progress['saved_complete_primary_rows'],2)
        self.assertEqual(progress['trailing_partial_lines'],1);self.assertTrue(progress['partial_not_a_final_metric'])
        path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
        with self.assertRaisesRegex(monitor.EvidenceError,'duplicated'):monitor.partial_slots(directory,request)
        path.write_text(json.dumps(dict(row,validation_index=1))+'\n')
        with self.assertRaisesRegex(monitor.EvidenceError,'logical rank'):monitor.partial_slots(directory,request)

    def test_full_512_requires_actual_parent_wait_and_reconstructed_strict_counts(self):
        directory=self.output/'evaluations/zero';execution=directory/'execution';execution.mkdir(parents=True)
        request=dict(run_id=self.cfg['run_id'],config={'sha256':self.cfg['_config_sha256']},completed_outer=0,evaluation_id='f'*32)
        self.put(str((execution/'request.json').relative_to(self.root)),request);all_rows=[]
        for rank in range(8):
            rows=[]
            for index in range(rank,512,8):
                hit=int(index%2==0);rows.append(dict(validation_index=index,run_id=request['run_id'],evaluation_id=request['evaluation_id'],
                    status='complete',primary=True,protocol_em=hit,protocol_valid_final_action=True,
                    failure_class=None if hit else 'valid_gold_mismatch',search_calls=1))
            all_rows.extend(rows);path=execution/('worker-'+str(rank))/'results.jsonl';path.parent.mkdir()
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.put(str((path.parent/'complete.json').relative_to(self.root)),dict(accepted=True,evaluation_id=request['evaluation_id'],rank=rank,primary_rows=64))
        overall=dict(hits=256,denominator=512,em=.5,final_invalid=0,valid_gold_mismatch=256,no_executed_search=0)
        summary=self.put(str((execution/'summary.json').relative_to(self.root)),dict(accepted=True,evaluation_id=request['evaluation_id'],completed_outer=0,primary_slots=512,overall=overall,datasets={'fixture':'not a benchmark result'}))
        results=execution/'results.jsonl';results.write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
        child=self.put(str((directory/'child-complete.json').relative_to(self.root)),dict(accepted=True,run_id=self.cfg['run_id'],variant='G_POS',evaluation_id=request['evaluation_id'],
            config={'sha256':self.cfg['_config_sha256']},completed_outer=0,primary_slots=512,actual_child_exit_and_release=True,
            child_exit_evidence=[{'actual_wait_returncode':0}]*8,summary=summary,results=cp.fileproof(results)))
        parent=self.put(str((directory/'parent-complete.json').relative_to(self.root)),dict(accepted=True,run_id=self.cfg['run_id'],config_sha256=self.cfg['_config_sha256'],
            completed_outer=0,child=child,exit_proof=self.actual_exit('evaluation-running')))
        value=monitor.evaluation_progress(self.cfg,0)
        self.assertEqual(value['state'],'ACCEPTED_512_WITH_ACTUAL_PARENT_WAIT_RELEASE');self.assertEqual(value['overall']['hits'],256)
        wrong=json.loads(Path(summary['path']).read_text());wrong['overall']['hits']=257;wrong['overall']['em']=257/512
        Path(summary['path']).write_text(json.dumps(wrong))
        body=json.loads(Path(child['path']).read_text());body['summary']=monitor.proof(summary['path']);Path(child['path']).write_text(json.dumps(body))
        value=json.loads(Path(parent['path']).read_text());value['child']=monitor.proof(child['path']);Path(parent['path']).write_text(json.dumps(value))
        with self.assertRaisesRegex(monitor.EvidenceError,'strict counts'):monitor.evaluation_progress(self.cfg,0)

    def test_resource_read_ignores_foreign_other_cards_but_rejects_foreign_owned_card(self):
        uuids=['gpu-'+str(i) for i in range(4)]
        cfg=copy.deepcopy(self.cfg);cfg['owner']={'gpu_uuids':uuids};cfg['gates']={'retriever_admission':self.put('oracle.json',{'accepted':True})}
        configs={v:(dict(cfg,variant=v),{}) for v in monitor.VARIANTS}
        cards=[dict(index=i,uuid='foreign-'+str(i),actual_free_mib=0) for i in range(4)]
        cards+=[dict(index=i+4,uuid=u,actual_free_mib=40000) for i,u in enumerate(uuids)]
        apps=[dict(pid=9,gpu_uuid='foreign-0',used_mib=49000),dict(pid=7,gpu_uuid=uuids[-1],used_mib=738)]
        gate=types.SimpleNamespace(inventory=lambda:(cards,apps),retriever_admission=lambda *a:dict(owner={'pid':7,'uid':1001,'starttick':1},health={'ready':True},numeric_gate={'accepted_fixture':True}))
        result=monitor.resource_snapshot(configs,{v:None for v in monitor.VARIANTS},gate)
        self.assertEqual(len(result['devices']),4);self.assertEqual(len(result['compute_applications']),1)
        apps.append(dict(pid=100,gpu_uuid=uuids[0],used_mib=1))
        with self.assertRaisesRegex(monitor.EvidenceError,'foreign/unknown'):monitor.resource_snapshot(configs,{v:None for v in monitor.VARIANTS},gate)
        apps.pop();cards[-1]['actual_free_mib']=6000
        with self.assertRaisesRegex(monitor.EvidenceError,'direct memory.free'):monitor.resource_snapshot(configs,{v:None for v in monitor.VARIANTS},gate)

    def test_log_markers_do_not_leak_question_text_or_claim_exit_failure(self):
        path=self.output/'evaluation/worker.log';path.parent.mkdir(parents=True)
        path.write_text('RuntimeError: private_question_gold_secret\nnormal progress\nHOLD_RUNNING_GATE\n')
        value=monitor.log_exceptions(self.cfg,self.root);self.assertEqual(value['exception_or_hold_lines'],2)
        self.assertNotIn('private_question_gold_secret',json.dumps(value));self.assertTrue(value['private_text_omitted'])

    def test_immutable_audit_never_overwrites_and_unknown_read_is_explicit(self):
        path=self.root/'audit.json';monitor.write_audit(path,{'accepted':True,'scope':'tiny fixture'})
        original=path.read_bytes()
        with self.assertRaises(FileExistsError):monitor.write_audit(path,{'accepted':False})
        self.assertEqual(path.read_bytes(),original)
        unknown=monitor.guarded(lambda:monitor.current(self.root/'missing.json'))
        self.assertFalse(unknown['accepted']);self.assertEqual(unknown['state'],'UNKNOWN_OR_READ_RACE')

    def test_source_publication_and_305_actual_code_files_checked(self):
        helper=self.root/'helper';helper.mkdir();py=self.bytes('helper/synthetic.py',b'# code identity fixture\n')
        manifest=self.put('helper/source-manifest.json',{'files':{'synthetic.py':py}})
        runtime_files={}
        for index in range(305):
            name='file-%03d.py'%index;runtime_files[name]=self.bytes('runtime/'+name,b'# original runtime fixture\n')
        runtime=self.put('runtime-manifest.json',{'files':runtime_files})
        model=self.put('model-manifest.json',dict(accepted=True,source_grpo_step=100,
            files=[self.bytes('source100/payload-%d'%i,('identity'+str(i)).encode()) for i in range(12)]))
        cfg=copy.deepcopy(self.cfg);cfg.update(study_id='new-study',source_commit='f'*40,helper_source_manifest=manifest,
            runtime_manifest=runtime,source_model_manifest=model,common_cohort_manifest=self.put('cohort.json',{'fixture':True}),
            train_identity=self.bytes('data/train.jsonl',b'{}\n'),validation_identity=self.bytes('data/validation.jsonl',b'{}\n'))
        cfg['paths'].update(helper_root=str(helper),runtime=str(self.root/'runtime'))
        configs={v:(dict(cfg,variant=v),{}) for v in monitor.VARIANTS}
        publication=self.put('published.json',dict(accepted=True,branch='search-e1',source_commit='f'*40,
            remote_readback_sha='f'*40,source_manifest_sha256=manifest['sha256']))
        study=dict(publication=publication,helper_source_manifest=manifest,helper_root=str(helper),source_commit='f'*40,study_id='new-study')
        result,_=monitor.source_identity(study,configs,publication)
        self.assertEqual(result['runtime_fresh_sha_files'],305);self.assertEqual(result['source100_payloads'],12)
        self.assertIn('Current length checks only',result['source100_payload_scope'])
        (helper/'synthetic.py').write_text('# hot source mutation\n')
        with self.assertRaisesRegex(monitor.EvidenceError,'File SHA differs'):monitor.source_identity(study,configs,publication)

    def test_whole_snapshot_not_started_and_dead_controller_wait_remains_unknown(self):
        uuids=['gpu-'+str(i) for i in range(4)];oracle=self.put('oracle-gate.json',{'accepted':True})
        refs={};configs={}
        for variant in monitor.VARIANTS:
            cfg=copy.deepcopy(self.cfg);cfg.update(schema='opsd-signal-weight-config-v1',variant=variant,study_id='new-study')
            cfg['owner']=dict(uid=1001,physical_gpus=[4,5,6,7],gpu_uuids=uuids)
            cfg['scientific'].update(outer_iterations=25,world_size=4,micro_batch=1,gradient_accumulation_full_batch_per_rank=8,
                original_eight_rank_accumulation=4,learning_rate=1e-6,component_clip=10,loss_reduction='trajectory_mean')
            cfg['paths'].update(output_root=str(self.root/variant/'runs'),receipts=str(self.root/variant/'receipts'))
            cfg['gates']=dict(retriever_admission=oracle,memory_binding_path=str(self.root/variant/'receipts/memory-binding.json'),
                scoped_cpu_processes_path=str(self.root/variant/'receipts/scoped-cpu-processes.json'))
            refs[variant]=self.put(variant+'/config.json',cfg);configs[variant]=cfg
        publication=self.put('source-publication.json',{'fixture':True})
        study=self.put('study.json',dict(study_id='new-study',study_base=str(self.root),groups=[dict(variant=v,config=refs[v]) for v in monitor.VARIANTS],
            publication=publication,helper_root='/synthetic-helper'))
        gate=types.SimpleNamespace(inventory=lambda:([dict(index=i+4,uuid=u,actual_free_mib=40000) for i,u in enumerate(uuids)],
            [dict(pid=7,gpu_uuid=uuids[-1],used_mib=738)]),
            retriever_admission=lambda *a:dict(owner={'pid':7,'uid':1001,'starttick':1},health={'ready':True},numeric_gate={'fixture':True}),
            process=lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
        with patch.object(monitor,'STUDY',str(self.root)),patch.object(monitor.os,'getuid',return_value=1001),\
             patch.dict(monitor.os.environ,{'CUDA_VISIBLE_DEVICES':''}),\
             patch.object(monitor,'source_identity',return_value=({'fixture_scope':'No hardware acceptance'},{})),\
             patch.object(monitor,'load_bound_module',side_effect=lambda h,m,name:gate if name=='gates' else cp):
            value=monitor.snapshot(study,refs,publication)
            self.assertTrue(value['accepted']);self.assertEqual(value['common_zero']['result']['state'],'NOT_STARTED')
            self.assertTrue(all(g['phase']=='NOT_STARTED' for g in value['groups'].values()))
            cfg=configs['G_POS'];path=Path(cfg['paths']['output_root'])/'status.json'
            self.put(str(path.relative_to(self.root)),dict(phase='SAMPLING',run_id=cfg['run_id'],config_sha256=refs['G_POS']['sha256'],
                owner={'pid':999,'uid':1001,'starttick':9,'run_id':cfg['run_id']},known_children=[]))
            failure=monitor.snapshot(study,refs,publication)
            self.assertFalse(failure['accepted']);self.assertIn('CONTROLLER_ABSENT_WAIT_UNKNOWN',[i['kind'] for i in failure['issues']])
            self.assertEqual(failure['groups']['G_POS']['controller']['result']['wait_returncode'],'UNKNOWN')
            wrong=copy.deepcopy(refs);wrong['G_NEG']['sha256']='0'*64
            with self.assertRaisesRegex(monitor.EvidenceError,'Explicit group ref differs'):monitor.snapshot(study,wrong,publication)


def math_ceil(value):return (value+31)//32
if __name__=='__main__':unittest.main()
