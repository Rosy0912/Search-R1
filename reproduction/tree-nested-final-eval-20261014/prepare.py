"""Future remote-only checkpoint archival and explicit handoff binding.

All modes are read/receipt actions. No mode stops or starts training/evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import time
from contract import (ROOT,RUN,PACKAGE,TRAIN_SOURCE,TRAIN_UNIT,CONFIG,CONFIG_SHA,NATIVE,HF,OUTPUT,
    HANDOFF_STATUS,require,read_json,sha,training_metadata,original_terminal,verify_archive,effective_segments,restore_and_fault_evidence)
from eval_merge import asset_manifest,atomic_immutable,identities


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def archive():
    package,native,native_sha,ranks = training_metadata()
    # A committed complete manifest is published after all native/HF writers.
    # Retain the current persistent remote checkpoint in place; never copy it to Mac.
    checkpoint_files = {str(NATIVE):native_sha,str(native['driver']['path']):native['driver']['sha256']}
    checkpoint_files.update({r['path']:r['sha256'] for r in native['worker_states']})
    checkpoint_files.update({str(HF/name):digest for name,digest in native['hf_identity']['files'].items()})
    from search_r1.baselines.select_teacher import model_manifest
    require(model_manifest(HF)==native['hf_identity'],'complete HF200 shard/tokenizer inventory differs')
    inventory = {}
    for name,expected in checkpoint_files.items():
        path = Path(name);require(sha(path)==expected,'complete saved file hash differs: '+name)
        with path.open('rb') as stream:os.fsync(stream.fileno())
        inventory[name] = dict(sha256=expected,bytes=path.stat().st_size)
    for directory in {NATIVE.parent,HF}:
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    require(sha(NATIVE)==native_sha,'native manifest changed during archival')
    proof=dict(status='NESTED_NATIVE_HF200_DURABLE_ARCHIVE_VERIFIED',training_source_commit=TRAIN_SOURCE,
        training_run=str(RUN),actual_completed_outer=200,config_sha256=CONFIG_SHA,
        native_manifest=pin(NATIVE),native_identity=native['identity'],
        actual_optimizer_calls_per_rank=ranks['actual_writes_per_rank'],training_attempts=ranks['training_attempts'],
        effective_parent_outer1_included=True,failed_parent_original_outer2_included=False,
        failed_parent_ba9_outer6_included=False,failed_eb6_bootstrap_included=False,
        inherited_completed_outers=5,current_child_completed_outers=195,
        effective_training_segments=effective_segments(),restore_and_fault_evidence=restore_and_fault_evidence(),
        hf_checkpoint=str(HF),hf_identity=native['hf_identity'],files=inventory,
        native_state_deserialized=False,all_files_hash_verified=True,all_files_fsynced=True,
        persistent_storage_realpath=str(RUN.resolve()),independent_cold_reload_performed=False)
    atomic_immutable(OUTPUT/'archive-proof.json',proof)
    assets=asset_manifest(CONFIG,HF)
    require({r['name']:r['sha256'] for r in assets['model_files']}==native['hf_identity']['files'],
            'asset manifest uses the same HF files despite its distinct aggregate hash encoding')
    atomic_immutable(OUTPUT/'asset-manifest.json',assets)
    prefix=dict(rows=[],**identities(assets),actual_trained_outer=200,training_source_commit=TRAIN_SOURCE,
        native_manifest_sha256=native_sha,exact_executor_resume=False)
    path=OUTPUT/'bound-prefix-empty.json';atomic_immutable(path,prefix)
    atomic_immutable(str(path)+'.done.json',dict(sha256=sha(path),count=0,actual_trained_outer=200,identity=identities(assets)))
    template=dict(status='WAITING_FOR_EXPLICIT_FINAL_ONLY_HANDOFF_AUTHORITY_AND_TRUE_TERMINAL',
        training_source_commit=TRAIN_SOURCE,training_run=str(RUN),training_unit=TRAIN_UNIT,
        actual_training_completed_outer=200,archive=pin(OUTPUT/'archive-proof.json'),
        asset_manifest=pin(OUTPUT/'asset-manifest.json'),prefix=pin(path),score_output=str(OUTPUT/'scores'),
        exact_executor_resume=False,training_stop_or_resume_by_wrapper=False,
        authority=dict(scope='only_final_evaluation_after_completed_training200',human_instruction='',original_instruction_time=''))
    atomic_immutable(OUTPUT/'handoff-template.json',template)
    print(json.dumps(dict(status=proof['status'],archive=pin(OUTPUT/'archive-proof.json'),
        handoff_authorized=False,training_signals_sent=False),sort_keys=True))


def capture_terminal():
    verify_archive(read_json(OUTPUT/'archive-proof.json'))
    terminal=original_terminal();result=read_json(RUN/'run-result.json')
    require(result['status']=='FAILED' and result['code_commit']==TRAIN_SOURCE
            and result['actual_progress_evidence']['actual_completed_outer_from_attempt_results']==200
            and terminal['own_cgroup_live_descendants_empty'] is True
            and result['actual_progress_evidence']['completed_actual_optimizer_calls_per_rank']
                ==read_json(OUTPUT/'archive-proof.json')['actual_optimizer_calls_per_rank'],
            'only a genuinely interrupted final evaluation after trained200 can be handed off')
    cost_paths=[RUN/'run-result.json',RUN/'termination-cost.json',RUN/'supervisor-entry-result.json',
                *sorted((RUN/'output/validation-cost').glob('validation-*.json'))]
    cost_pins=[pin(path) for path in cost_paths if path.is_file()]
    record=dict(unit=TRAIN_UNIT,systemd=terminal,original_run_result=pin(RUN/'run-result.json'),
        actual_completed_training_outer=200,normal_exit_claimed=False,original_final_evaluation_complete=False,
        training_signals_sent_by_this_wrapper=False,observed_epoch=time.time(),
        original_cleanup_error_retained=result.get('cleanup_error'),
        original_cleanup_reported_success=result.get('own_cgroup_live_residual_empty') is True,
        partial_final_and_terminal_cost_inputs_retained=cost_pins,cost_independently_accepted=False)
    atomic_immutable(OUTPUT/'original-terminal.json',record)
    print(json.dumps(pin(OUTPUT/'original-terminal.json'),sort_keys=True))


def bind_handoff(args):
    require(args.explicit_final_only_authority and args.human_instruction and args.instruction_time,
            'a human instruction and its original time must explicitly authorize the final-only handoff')
    value=read_json(OUTPUT/'handoff-template.json')
    value.update(status=HANDOFF_STATUS,original_terminal=pin(OUTPUT/'original-terminal.json'),
        original_run_result=pin(RUN/'run-result.json'))
    value['authority']=dict(scope='only_final_evaluation_after_completed_training200',
        human_instruction=args.human_instruction,original_instruction_time=args.instruction_time)
    if args.prior_evaluator_terminal:
        value['prior_evaluator_terminal']=pin(Path(args.prior_evaluator_terminal))
    from contract import verify_handoff_value
    # Validate before publishing the authorization credential; this does not signal anything.
    path=Path(args.handoff_output) if args.handoff_output else OUTPUT/'handoff-authorized.json'
    verify_handoff_value(value,path)
    atomic_immutable(path,value)
    print(json.dumps(dict(handoff=pin(path),training_signals_sent=False,evaluation_launched=False),sort_keys=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['archive','capture-terminal','bind-handoff'])
    p.add_argument('--human-instruction');p.add_argument('--instruction-time')
    p.add_argument('--explicit-final-only-authority',action='store_true');p.add_argument('--prior-evaluator-terminal')
    p.add_argument('--handoff-output',help='Distinct own OUTPUT/handoff-authorized-ID.json for an explicit failed-evaluator continuation')
    args=p.parse_args()
    if args.mode=='archive':archive()
    elif args.mode=='capture-terminal':capture_terminal()
    else:bind_handoff(args)

if __name__=='__main__':main()
