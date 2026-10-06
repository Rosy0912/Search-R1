"""CPU receipt-schema fixtures; simulated reports are not native GPU proof."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from search_r1.baselines import fresh200_gate as gate
from search_r1.baselines.full_state import state_digest
from search_r1.baselines.fresh200_driver import Fresh200Progress


def test_full_firstfive_receipt_replays_actual_skips_and_incomplete_writes(tmp_path):
    run=tmp_path/'runs'/'fixture';directory=run/'checkpoints'/'outer5';directory.mkdir(parents=True)
    source='a'*40;pid=123;progress=Fresh200Progress();attempts=[];events=[];paths=[]
    for outer,status,writes,processed,postabort in [(1,'ABORTED',3,3,True),(1,'COMPLETE_WITH_SKIPS',18,20,False),
            (2,'SKIPPED',0,20,False),(2,'COMPLETE_WITH_SKIPS',19,20,False),
            (3,'COMPLETE',20,20,False),(4,'COMPLETE',20,20,False),(5,'COMPLETE_WITH_SKIPS',17,20,False)]:
        attempt_id=progress.begin_attempt(outer)
        complete=status in ('COMPLETE','COMPLETE_WITH_SKIPS')
        ranks=[dict(rank=rank,actual_outer=outer,attempt_id=attempt_id,status=status,optimizer_steps=writes,
            expected_optimizer_steps=20,optimizer_opportunities_total=20,processed_opportunities=processed,
            skipped_opportunities=processed-writes,unprocessed_opportunities=20-processed,
            update_complete=complete,attempt_complete=processed==20 and not postabort,postabort=postabort) for rank in range(8)]
        progress.record(ranks,outer)
        path=run/'diagnostics'/f'attempt_{attempt_id}'/'all-rank-write-ledger.json';path.parent.mkdir(parents=True)
        event=dict(attempt_id=attempt_id,candidate_outer=outer,driver_pid=pid,source_commit=source,
            status=status,accepted_complete_outer=complete,rank_write_ledger=ranks,
            progress=progress.state_dict(),ledger_path=str(path))
        path.write_text(json.dumps(event));paths.append(path);attempts.append(event)
        events.append(dict(step=outer,attempt_id=attempt_id,accepted_complete_outer=complete,
            metrics={'actor/optimizer_steps':writes,**({'bapo/actual_outer_updates':outer}if complete else {})}))
    # 94 accepted writes plus 3 retained abort writes; no requirement of 100.
    assert progress.actual_adam_writes_per_rank==[97]*8
    source_file=run/'source-commit.txt';source_file.write_text(source);paths.append(source_file)
    driver=dict(driver_pid=pid,global_step=5,next_global_step=6,actual_outer_updates=5,progress=progress.state_dict())
    driver_path=directory/'driver.pt';torch.save(driver,driver_path);paths.append(driver_path)
    reload_path=directory/'driver-reload.json'
    reload_path.write_text(json.dumps(dict(status='EXACT_DRIVER_RELOAD_OK',step=5,path=str(driver_path),sha256=gate.digest(driver_path),pid=pid,state_digest=state_digest(driver))))
    paths.append(reload_path);reports=[]
    for rank in range(8):
        path=directory/'ranks'/f'rank_{rank}.pt';path.parent.mkdir(exist_ok=True);torch.save(dict(cpu_fixture=True),path)
        report=dict(rank=rank,status='EXACT_FULL_STATE_RELOAD_OK',step=5,world_size=8,scheduler_last_epoch=5,
            model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=1,optimizer_state_entries=1,
            path=str(path),sha256=gate.digest(path),pid=1000+rank,state_digest='b'*64)
        report_path=path.with_suffix('.json');report_path.write_text(json.dumps(report));paths.extend([path,report_path]);reports.append(report)
    with patch.object(gate,'validate_gate_config',return_value=(200,57)):
        manifest_path=gate.write_manifest(paths,directory,{},source,run,5,reports,progress.state_dict())
    manifest=json.loads(manifest_path.read_text())
    prefix=''.join('FRESH200_ATTEMPT_JSON='+json.dumps(attempt)+'\nBAPO_TRAIN_METRICS_JSON='+json.dumps(event)+'\n'for attempt,event in zip(attempts,events))
    (run/'train.log').write_text(prefix);snapshot=tmp_path/'step5-prefix.log';snapshot.write_text(prefix)
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(scope='CPU_RECEIPT_FIXTURE_NOT_NATIVE_ACCEPTANCE')))
    evidence=manifest['files']+[dict(path=str(manifest_path),sha256=gate.digest(manifest_path)),dict(path=str(snapshot),sha256=gate.digest(snapshot))]
    receipt=dict(manifest,status='PASS_INDEPENDENT_FRESH200_STEP5_REVIEW',checkpoint_manifest_sha256=gate.digest(manifest_path),evidence=evidence,
        independent_review=dict(path=str(review),sha256=gate.digest(review)),
        train_log_prefix=dict(live_path=str(run/'train.log'),snapshot_path=str(snapshot),bytes=snapshot.stat().st_size,sha256=gate.digest(snapshot)))
    receipt_path=tmp_path/'release.json';receipt_path.write_text(json.dumps(receipt))
    with patch.object(gate,'validate_first_sampling_evidence',return_value={'attempt_id':2,'actual_outer':1}),patch.object(gate,'validate_gate_config',return_value=(200,57)),patch.object(gate.subprocess,'check_output',side_effect=lambda args,**kwargs:source+'\n'if 'rev-parse'in args else ''):
        gate.validate_release(receipt_path,manifest_path,tmp_path,expected_driver_pid=pid)
        with pytest.raises(ValueError,match='reload not proven'):
            gate.validate_release(receipt_path,manifest_path,tmp_path,expected_driver_pid=456)
        bad=copy.deepcopy(receipt);bad['attempt_accounting']['actual_adam_writes_per_rank'][0]=100
        receipt_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='identity mismatch'):
            gate.validate_release(receipt_path,manifest_path,tmp_path,expected_driver_pid=pid)
