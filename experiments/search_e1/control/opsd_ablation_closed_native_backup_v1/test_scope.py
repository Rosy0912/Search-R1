"""Two CPU closure fixtures only; no models/archives/Node/scientific tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).absolute().parent
ns=dict(__name__='_native_fixture',__file__=str(HERE/'source-operation.py'))
exec(compile((HERE/'source-operation.py').read_bytes(),str(HERE/'source-operation.py'),'exec'),ns)
b=ns['primitives'](HERE.parent/'opsd_ablation_outer_backup_v2/backup.py')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b['canonical'](value)+b'\n')
    return b['fileproof'](path)


def fixture(root,variant):
    ns['N']=root;S,T=ns['roots'](variant);S.mkdir(parents=True);T.mkdir(parents=True)
    run='1'*32;versions=ns['versions'](variant);owner=dict(pid=950,uid=1005,starttick=900,run_id=run)
    config_path=T.parent/'evaluation-config.json'
    cfg=dict(schema='opsd-ablation-native512-eval-config-v1',run_id=run,study_id=ns['STUDY_ID'],variant=variant,
        evaluation_role='group',paths=dict(base=str(S.parent),output_root=str(S),python='/fixture/python',helper_root=str(root/'source/eval-stats-v2-72e1e12/opsd_ablation_eval_v1')),
        owner=dict(gpu_uuids=['fixture-GPU-%d'%rank for rank in range(8)]),
        scientific=dict(versions=versions,validation_questions=512,total_primary_slots=512*len(versions)))
    cp=save(config_path,cfg)
    inputs=[];models={};phases={}
    for vi,version in enumerate(versions):
        stage=S/version;stage.mkdir();exit_rows=[]
        for rank in range(8):
            identity=dict(pid=1000+100*vi+rank,uid=1005,starttick=910+100*vi+rank,ppid=owner['pid'],run_id=run)
            exit_rows.append(dict(identity=identity,actual_wait_returncode=0))
            worker=stage/('worker-%d'%rank);worker.mkdir();records=[]
            for slot,index in enumerate(range(rank,512,8)):
                records.append(dict(status='complete',primary=True,run_id=run,version=version,rank=rank,slot=slot,
                    validation_index=index,question_id='fixture-id-%d'%index,row_semantic_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                    dataset='nq' if index<256 else 'hotpotqa',protocol_em=index%2))
            path=worker/'results.jsonl';path.write_bytes(b''.join(b['canonical'](r)+b'\n' for r in records));rp=b['fileproof'](path)
            complete=dict(accepted=True,status='COMPLETE',primary_rows=64,rank=rank,physical_gpu=rank,rank_seed=42,
                version=version,run_id=run,variant=variant,study_id=ns['STUDY_ID'],process=identity,
                gpu_uuid=cfg['owner']['gpu_uuids'][rank],
                privileged_reference=False,no_training=True,config=cp,results=rp,
                source=dict(manifest=dict(sha256='a'*64)),science=dict(manifest=dict(sha256='b'*64)))
            wp=save(worker/'complete.json',complete);inputs.extend([wp,rp])
        phase=dict(accepted=True,release=dict(accepted=True),run_id=run,version=version,expected_worker_count=8,
            actual_worker_count=8,child_exit_evidence=exit_rows,actual_exit_codes_never_inferred=True)
        pp=save(stage/'phase-complete.json',phase)
        tp=save(stage/'actual-terminal.json',dict(run_id=run,version=version,child_exit_evidence=exit_rows,all_actual_wait_zero=True))
        phases[version]=dict(phase=pp,terminal=tp);inputs.append(pp)
        models[version]=dict(overall=dict(questions=512,strict_hits=256),
            datasets={d:dict(questions=256,strict_hits=128) for d in ('nq','hotpotqa')})
        if version=='raw10_fp16':
            # Exact producer cache is excluded, while CPU merge receipts stay.
            model=stage/'cpu-merge/model';model.mkdir(parents=True)
            (model/'model.safetensors').write_bytes(b'fixture-model-must-not-be-archived')
            save(stage/'cpu-merge/cpu-merge-complete.json',dict(accepted=True))
    summary=dict(schema='opsd-ablation-native512-summary-v1',accepted=True,run_id=run,study_id=ns['STUDY_ID'],variant=variant,
        evaluation_role='group',primary_slots=512*len(versions),models=models,inputs=inputs,
        identities=dict(source_model_identity='a'*64,runtime_identity='b'*64))
    sp=save(S/'summary.json',summary)
    cc=save(S/'controller-complete.json',dict(accepted=True,run_id=run,config=cp,primary_slots=512*len(versions),summary=sp,release=dict(accepted=True)))
    accepted=save(T/'result-accepted.json',dict(accepted=True,config=cp,controller_completion=cc,summary=sp))
    command=[cfg['paths']['python'],'-B',str(Path(cfg['paths']['helper_root'])/'controller.py'),'--config',cp['path'],'--config-sha256',cp['sha256']]
    owner['argv']=command
    pe=save(T/'actual-child-exit.json',dict(run_id=run,actual_wait_returncode=0,signal_termination_not_exit0=False,identity=owner,command=command))
    pr=save(T/'actual-child-release.json',dict(accepted=True))
    pins=dict(config=cp,summary=sp,controller=cc,accepted=accepted,parent_exit=pe,parent_release=pr)
    return dict(schema='opsd-ablation-closed-native-scope-v1',accepted=True,study_id=ns['STUDY_ID'],variant=variant,run_id=run,
                versions=versions,pins=pins,phases=phases,parent_identity={k:owner[k] for k in ('pid','uid','starttick')})


class Scope(unittest.TestCase):
    def test_closed_single_and_dual_phase_original_interfaces(self):
        for variant,slots in [('M',512),('P',1024)]:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                scope=fixture(Path(tmp).resolve(),variant);result=ns['closure'](b,scope)
                self.assertEqual(result['primary_slots'],slots);self.assertEqual(len(result['workers']),8*len(scope['versions']))
                self.assertEqual(result['actual_native_parent_wait_returncode'],0)
                paths,excluded=ns['enumerate_files'](b,scope)
                self.assertFalse(any(p.suffix=='.safetensors' for p in paths))
                self.assertEqual(len(excluded),int(variant=='P'))

    def test_wrong_run_and_real_nonzero_parent_or_worker_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=fixture(Path(tmp).resolve(),'M')
            wrong=dict(scope,run_id='2'*32)
            with self.assertRaises(ValueError):ns['closure'](b,wrong)
            path=Path(scope['pins']['parent_exit']['path']);original=json.loads(path.read_bytes())
            failed=dict(original,actual_wait_returncode=2);scope['pins']['parent_exit']=save(path,failed)
            with self.assertRaises(ValueError):ns['closure'](b,scope)
            scope['pins']['parent_exit']=save(path,original)
            stage=Path(scope['phases']['raw10_bf16']['phase']['path']);phase=json.loads(stage.read_bytes())
            phase['child_exit_evidence'][0]['actual_wait_returncode']=1
            scope['phases']['raw10_bf16']['phase']=save(stage,phase)
            terminal=Path(scope['phases']['raw10_bf16']['terminal']['path']);t=json.loads(terminal.read_bytes())
            t['child_exit_evidence']=phase['child_exit_evidence'];scope['phases']['raw10_bf16']['terminal']=save(terminal,t)
            with self.assertRaises(ValueError):ns['closure'](b,scope)


if __name__=='__main__':unittest.main()
