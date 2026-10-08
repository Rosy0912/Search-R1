"""CPU-only full200 bookkeeping. Terminal rc alone is not scientific completion."""
import argparse
import json
import math
from pathlib import Path
import time
try:
    from .utils import *
except ImportError:
    from utils import *


def totals(completions):
    require(len(completions) == 200, 'NOT_READY: require200 completed outer rounds')
    updates = exposure = 0
    for index, value in enumerate(completions):
        require(value['outer_index'] == index and value['run_id'] == TRAIN_RUN and value['accepted'] is True,
                'Rejected/duplicate/out-of-order outer')
        P, K = value['P'], value['K']
        require(type(P) is int and type(K) is int and 0 <= K <= P <= 512, 'Invalid native P/K')
        delta = math.ceil(K/32); updates += delta; exposure += K
        require(value['optimizer_updates_delta'] == delta and value['optimizer_updates'] == updates
                and value['cumulative_pair_exposure'] == exposure and value['scheduler_advanced_updates'] == 0,
                'Actual Adam/exposure continuity differs (including tail/K0)')
        require(value['empty_outer_no_adam'] is (K == 0), 'K0/no-Adam declaration differs')
    return dict(outer_iterations=200, optimizer_updates=updates, cumulative_pair_exposure=exposure,
                training_questions=200*512, sampled_trajectory_slots=200*2560)


def committed_exit(exit_proof, count, phase, history):
    require(exit_proof.get('accepted') is True and len(exit_proof['child_exit_evidence']) == count, 'Missing committed actual child exits')
    evidence = exit_proof['child_exit_evidence']; identities = {(x['identity']['pid'],x['identity']['uid'],x['identity']['starttick']) for x in evidence}
    require(len(identities) == count and all(x.get('actual_wait_returncode') == 0 for x in evidence), 'Committed child nonzero/unknown rc')
    value, ref = checked_json(exit_proof['wait_recovery'])
    require(value.get('accepted') is True and value['run_id'] == TRAIN_RUN and value['config_sha256'] == TRAIN_CONFIG_SHA
            and value['child_phase'] == phase and value['child_exit_evidence'] == evidence, 'Wrong durable committed phase proof')
    actual, actual_ref = checked_json(value['actual_terminal'])
    require(actual['run_id'] == TRAIN_RUN and actual['config_sha256'] == TRAIN_CONFIG_SHA
            and actual['child_phase'] == phase and actual['accepted_all_wait_zero'] is True
            and actual['child_exit_evidence'] == evidence, 'Actual wait-returncodes are not persisted')
    commands = []
    for identity in evidence:
        key = tuple(identity['identity'][k] for k in ('pid','uid','starttick'))
        require(key in history and history[key]['phase'] == phase, 'Committed child missing from native history')
        commands.append(history[key]['command'])
    require(value['gpu_release']['all_owned_release']['accepted'] is True,
            'Committed phase descendant release was not accepted')
    fileproof(value['events']['path'], value['events']['sha256'])
    return dict(phase=ref, actual_terminal=actual_ref, commands=commands)


def audit_final(config, final_ref, total):
    """Consume independently executed, published CPU reader proof; hash actual states."""
    ref = config['training'].get('final_state_audit')
    require(ref is not None, 'NOT_READY: final independent actual-state CPU audit missing')
    audit, proof = checked_json(ref); execution, ep = checked_json(config['training']['final_state_execution'])
    require(execution.get('exit_code') == 0 and execution['stdout_sha256'] == proof['sha256']
            and execution['target_checkpoint'] == final_ref and execution['expected_outer'] == 199
            and execution['expected_updates'] == total['optimizer_updates'], 'Independent final execution target/rc differs')
    reader_source=fileproof(config['training']['final_reader_source']['path'],config['training']['final_reader_source']['sha256'])
    publication,_=checked_json(config['training']['final_reader_publication'])
    require(execution['helper_sha256']==reader_source['sha256'] and execution['source_commit']==publication['source_commit']
            and publication.get('accepted') is True and publication.get('remote_readback_sha')==publication['source_commit'],
            'Executed reader source/publication differs')
    require(any(row.get('equal') is True and row.get('sha256')==reader_source['sha256']
                and row.get('bytes')==reader_source['bytes'] and row['path'].endswith('/read_checkpoint_cpu.py')
                for row in publication['files']), 'Independent reader was not remotely readback equal')
    command=execution['command']
    for key,value in {'--config':config['training']['config']['path'],'--config-sha256':TRAIN_CONFIG_SHA,
            '--checkpoint':final_ref['path'],'--checkpoint-sha256':final_ref['sha256'],
            '--expected-updates':str(total['optimizer_updates']),'--expected-outer':'199'}.items():
        require(command.count(key)==1 and command[command.index(key)+1]==value,'Independent actual reader argv differs')
    require(command.count('--baseline-checkpoint')==command.count('--baseline-sha256')==1,'Explicit permanent baseline missing')
    baseline_ref=dict(path=command[command.index('--baseline-checkpoint')+1],sha256=command[command.index('--baseline-sha256')+1])
    baseline,_=checked_json(baseline_ref)
    require(baseline['run_id']==TRAIN_RUN and baseline['config_sha256']==TRAIN_CONFIG_SHA and baseline['optimizer_updates']==5
            and audit['adapter']['baseline_manifest']['sha256']==baseline_ref['sha256'],'Final audit baseline is not original permanent CP5')
    require(audit.get('schema') == 'opsd-online-explicit-checkpoint-independent-cpu-audit-v1' and audit.get('accepted') is True
            and audit['run_id'] == TRAIN_RUN and audit['config']['sha256'] == TRAIN_CONFIG_SHA
            and audit['checkpoint']['sha256'] == final_ref['sha256'] and audit['optimizer_updates'] == total['optimizer_updates']
            and audit['cursor']['outer_index'] == 199 and audit['cumulative_pair_exposure'] == total['cumulative_pair_exposure'],
            'Final CPU audit targets another checkpoint/count/run')
    cp, cp_proof = checked_json(final_ref)
    require(len(audit['ranks']) == len(cp['ranks']) == 8 and audit['adapter']['all_finite'] is True
            and audit['adapter']['actual_tensor_count'] == 504 and 0 < audit['adapter']['changed'] <= 504,
            'Final actual-state/adapter audit is incomplete')
    require({r['rank'] for r in audit['ranks']} == set(range(8))
            and all(r['actual_adam_steps'] == [total['optimizer_updates']]
                    and r['actual_state_parameters'] == 504 and r['finite_optimizer_tensors'] == 1512
                    and r['full_actual_moments_and_groups_equal_rank0'] is True for r in audit['ranks'])
            and len({r['optimizer_digest'] for r in audit['ranks']}) == 1, 'Final actual moments/groups/step proof incomplete')
    require(all(r['file']['sha256'] == next(a['file']['sha256'] for a in cp['ranks'] if a['rank']==r['rank'])
                for r in audit['ranks']), 'Fresh final state files differ from independent readback')
    actual_states = [fileproof(r['file']['path'], r['file']['sha256']) for r in cp['ranks']]
    require(len({(r['device'],r['inode']) for r in actual_states}) == 8, 'Final state files aliased')
    return dict(audit=proof, execution=ep, checkpoint=cp_proof, actual_state_fresh_sha=actual_states,
                scope='Independent actual-state audit plus fresh current8 state-byte SHA; no model forward')


def cpu_merges(config):
    train,_=original_config(config);indexed={}
    for path in sorted((Path(train['paths']['base'])/'receipts/cpu-merge-recovery').glob('*/parent-complete.json')):
        parent,pp=checked_json(fileproof(path));require(parent['accepted'] is True and parent['run_id']==TRAIN_RUN
            and parent['config_sha256']==TRAIN_CONFIG_SHA,'CPU merge parent identity differs')
        child,child_ref=checked_json(parent['child']);request,rp=checked_json(child['request']);terminal,tp=checked_json(parent['actual_terminal'])
        require(request['schema']=='opsd-online-cpu-merge-request-v1' and request['run_id']==TRAIN_RUN
            and request['config']['sha256']==TRAIN_CONFIG_SHA and child['accepted'] is True and child['run_id']==TRAIN_RUN
            and child['config_sha256']==TRAIN_CONFIG_SHA and terminal['actual_wait_returncode']==0
            and all(child['owner'][k]==terminal['identity'][k] for k in ('pid','uid','starttick')),
            'CPU merge actual child exit/request identity differs')
        require(parent['parent_cpu']==dict(cuda_visible_devices='',torch_imported=False)
            and parent['release']['all_owned_release']['accepted'] is True,'CPU merge parent CUDA/release not accepted')
        device=child['device_proof'];require(device['before']==device['after']==dict(cuda_initialized=False,visible_cuda_devices=0,cuda_visible_devices='')
            and device['original_function_called_once'] is True and device['original_bf16_method_and_arguments_unchanged'] is True,'CPU merge actual method/device proof differs')
        for key in ('peft_parameters','merged_parameters'):
            require(len(device[key])==1 and device[key][0]['device_types']==['cpu'] and device[key][0]['parameter_tensors']>0,'CPU model parameter device not verified')
        checked_json(request['source_manifest']);publication,_=checked_json(request['publication'])
        require(publication['accepted'] is True and publication['source_manifest_sha256']==request['source_manifest']['sha256']
            and publication['source_commit']==publication['remote_readback_sha'],'CPU merge source not published')
        fileproof(parent['child_log']['path'],parent['child_log']['sha256']);fileproof(parent['events']['path'],parent['events']['sha256'])
        key=(str(Path(request['output']).resolve()),request['checkpoint']['sha256'])
        require(key not in indexed,'Duplicate canonical CPU merge completion')
        indexed[key]=dict(parent=pp,child=child_ref,request=rp,terminal=tp,result=child['result'])
    return indexed


def native_summaries(config,cp,manifest,complete,previous):
    # Fresh compile the same independently published reader's stdlib summary validator.
    source=config['training']['final_reader_source'];reader_module=fresh_module('results_actual_summary_reader',source['path'],source['sha256'])
    require(source['sha256']=='7b26007cc4b0a79930bea938cf8e902ba6773613afcb690b492c2fd47cf37e03','Unexpected native CPU reader implementation')
    reader=reader_module.Reader();encoding,ep=reader.json(cp['training_input']['encoding']);order,op=reader.json(cp['training_input']['order']);K=complete['K']
    require(cp['training_input']==complete['training_input'] and encoding['P']==complete['P'] and encoding['K']==K
        and encoding['skipped']==complete['skipped'] and complete['P']==K+sum(complete['skipped'].values()) and all(type(n) is int and n>=0 for n in complete['skipped'].values())
        and len(encoding['rows'])==K and all(isinstance(x,str) and len(x)==64 for x in encoding['rows'])
        and encoding['sha256']==object_sha(encoding['rows']) and order['encoded_sha256']==encoding['sha256']==cp['cursor']['encoded_sha256']
        and op['sha256']==cp['cursor']['order_sha256'] and order['outer_index']==manifest['outer_index']
        and sorted(order['pair_indices'])==list(range(K)),'Native encoding/order/complete scope differs')
    if K:
        train,_=original_config(config);prior=checked_json(previous)[0] if previous else None
        summaries,last,traces=reader_module.validate_updates(cp,manifest,order['pair_indices'],reader,train,prior)
        require(sum(x['global_actual_pairs'] for x in summaries)==K
            and sum(x['global_valid_tokens'] for x in summaries)==complete['global_valid_tokens'],'Native complete token/exposure total differs')
    else:
        require(cp['outer_update_receipts']==[] and complete['global_valid_tokens']==0,'K0 contains optimizer/token updates');summaries=[];traces=[]
    reader.finish()
    return dict(encoding=ep,order=op,summaries=summaries,token_trace_count=len(traces),
        token_scope='Every committed native summary fresh SHA; historic token logs metadata/inventory only; final independent audit reconstructs latest tail')


def validate(config):
    started = time.time(); train, train_proof = original_config(config)
    require(train['scientific']['outer_iterations'] == 200, 'Wrong original scientific outer budget')
    terminal, terminal_ref = checked_json(config['training']['terminal'])
    require(terminal.get('schema') == 'opsd-online-controller-terminal-v1' and terminal.get('accepted') is True
            and terminal['run_id'] == TRAIN_RUN and terminal['config_sha256'] == TRAIN_CONFIG_SHA
            and terminal['actual_exit_code'] == terminal['controller']['actual_exit_code'] == 0
            and terminal['owned_release']['accepted_no_owned_or_opaque'] is True, 'NOT_READY: real terminal rc0/release missing')
    root = Path(train['paths']['output_root']); history_rows, history_ref = rows(fileproof(root/'child-history.jsonl'))
    history = {(x['identity']['pid'],x['identity']['uid'],x['identity']['starttick']):x for x in history_rows}
    core, qa, runtime = science(config); helpers, helpers_ref = frozen_helpers(config); sampling = helpers['sampling'];merges=cpu_merges(config)
    cursor = dict(cycle=0, offset=0); previous = None; updates = 0; completions = []; audits = []; committed_ids = set()
    require({p.name for p in (root/'outers').glob('outer-*') if p.is_dir()} == {f'outer-{i:05d}' for i in range(200)},
            'NOT_READY: missing/unexpected actual outer directory')
    for index in range(200):
        outer = root/'outers'/f'outer-{index:05d}'
        manifest, mr = checked_json(fileproof(outer/'outer-manifest.json'))
        complete, cr = checked_json(fileproof(outer/'training/outer-complete.json'))
        sample, sr = checked_json(fileproof(outer/'sampling-complete.json'))
        require(manifest['run_id'] == TRAIN_RUN and manifest['config_sha256'] == TRAIN_CONFIG_SHA
                and manifest['outer_index'] == index and manifest['cumulative_optimizer_updates_before'] == updates
                and manifest['runtime_identity'] == train['runtime_manifest']['sha256']
                and manifest['source_model_identity'] == train['source_model_manifest']['sha256']
                and manifest['previous_complete_checkpoint'] == previous and manifest['explicit_fresh'] is (previous is None),
                'Outer input/checkpoint/source identity chain differs')
        questions, qr = rows(manifest['questions']); expected, selection = sampling.cohort(train, cursor)
        require(questions == expected and manifest['train_cursor'] == selection, 'Actual512 TRAIN stream differs or validation was mixed')
        rollouts, rr = rows(manifest['rollouts']); pairs, pr = rows(manifest['pairs'])
        predicted_pairs, stats = sampling.verify_rollouts(rollouts, questions, core)
        require(pairs == predicted_pairs and stats == manifest['sampling_stats'] == sample['stats']
                and manifest['P'] == stats['P'] == complete['P'], 'Strict native mining/scoring/P differs')
        model = manifest['sampling_model_identity']
        require(sample['schema'] == 'opsd-online-sampling-complete-v1' and sample['accepted'] is True
                and sample['run_id'] == TRAIN_RUN and sample['rollouts'] == manifest['rollouts']
                and sample['pairs'] == manifest['pairs'] and sample['sampling_model_identity'] == model
                and model['checkpoint'] == previous and model['optimizer_updates'] == updates,
                'Sampler did not use this outer latest raw checkpoint')
        cache_scope = 'raw-source100'
        if previous:
            prior_cp, _ = checked_json(previous)
            require(model['adapter_tensor_sha256'] == prior_cp['all_rank_adapter_sha256'][0], 'Sampler adapter tensor identity differs')
            merge, merge_ref = checked_json(fileproof(outer/'merged-model.json'))
            require(all(merge.get(k)==model.get(k) for k in ('kind','path','sha256','source_model_identity',
                    'checkpoint','adapter_tensor_sha256','optimizer_updates','files')), 'Recorded merge semantic identity differs')
            merge_key=(str(Path(model['path']).resolve()),previous['sha256'])
            require(merge_key in merges and merges[merge_key]['result']==merge,'Canonical model lacks exact completed CPU-boundary receipt')
            if Path(model['path']).exists():
                for f in model['files']: fileproof(f['path'], f['sha256'])
                cache_scope = 'still-present full-byte SHA checked'
            else:
                tombstone, tp = checked_json(fileproof(outer/'merged-cache-pruned.json'))
                require(tombstone['schema'] == 'opsd-online-rebuildable-merge-cache-pruned-v1' and tombstone['run_id'] == TRAIN_RUN
                        and tombstone['config_sha256'] == TRAIN_CONFIG_SHA and tombstone['merge_receipt']['sha256'] == merge_ref['sha256']
                        and tombstone['original_files'] == merge['files'], 'Pruned historical cache lacks original-byte evidence')
                cache_scope = 'original at-pruning SHA evidence; cache not re-read/rebuilt'
        exits = committed_exit(sample['exit_proof'], 8, 'sampling-running', history)
        require(len(sample['workers']) == 8, 'Sampling workers incomplete')
        for rank, ref in enumerate(sample['workers']):
            worker, _ = checked_json(ref)
            require(worker['accepted'] is True and worker['rank'] == rank and worker['rows'] == 320
                    and worker['run_id'] == TRAIN_RUN and worker['config_sha256'] == TRAIN_CONFIG_SHA
                    and worker['sampling_model_identity'] == model['sha256'], 'Sampling worker identity differs')
            fileproof(worker['rollouts']['path'], worker['rollouts']['sha256'])
            fileproof(worker['generation_trace']['path'], worker['generation_trace']['sha256'])
            sampling.verify_http_trace(worker['http_trace'], rank, model['sha256'], worker['http_requests'])
        training_exits = []
        for path in sorted((outer/'trainer-logs').glob('attempt-*-exit.json')):
            value, er = checked_json(fileproof(path))
            if value.get('accepted') is not True:continue
            try:
                checked = committed_exit(value, 1, 'training-running', history)
                command = checked['commands'][0]
                require('--outer-manifest' in command and command[command.index('--outer-manifest')+1] == mr['path']
                        and command[command.index('--outer-manifest-sha256')+1] == mr['sha256'], 'Trainer exit not bound to this input')
            except (Hold, KeyError): raise Hold('Accepted trainer attempt has corrupt or mismatched durable exit proof')
            training_exits.append(dict(receipt=er, **checked))
        require(training_exits, 'Committed outer lacks a real successful bound trainer wait receipt')
        cp, cp_ref = checked_json(complete['checkpoint']); K = complete['K']
        require(cp['schema'] == 'opsd-online-checkpoint-v1' and cp['accepted'] is True and cp['run_id'] == TRAIN_RUN
                and cp['config_sha256'] == TRAIN_CONFIG_SHA and cp['cursor']['outer_index'] == index
                and cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'] == K
                and cp['training_input']['outer_manifest']['sha256'] == mr['sha256']
                and cp['runtime_identity']==manifest['runtime_identity'] and cp['source_model_identity']==manifest['source_model_identity']
                and cp['optimizer_updates']==complete['optimizer_updates'] and cp['cumulative_pair_exposure']==complete['cumulative_pair_exposure']
                and cp['complete_rank_count']==8 and cp['equal_optimizer_steps'] is True, 'Outer complete checkpoint cursor/input differs')
        summaries=native_summaries(config,cp,manifest,complete,previous)
        committed_ids.update(tuple(x['identity'][k] for k in ('pid','uid','starttick')) for x in sample['exit_proof']['child_exit_evidence'])
        for checked in training_exits:
            phase, _ = checked_json(checked['phase']); committed_ids.update(tuple(x['identity'][k] for k in ('pid','uid','starttick')) for x in phase['child_exit_evidence'])
        completions.append(complete); updates = complete['optimizer_updates']; previous = complete['checkpoint']; cursor = selection['after']
        audits.append(dict(outer_index=index, manifest=mr, completion=cr, sampling=sr, questions=qr,
                           rollouts=rr, pairs=pr, P=complete['P'], K=K, cache_scope=cache_scope,
                           sampling_exit=exits, successful_training_exits=training_exits, checkpoint=cp_ref,native_updates=summaries,
                           cpu_merge=None if index==0 else {k:v for k,v in merges[merge_key].items() if k!='result'}))
    total = totals(completions)
    require(config['models']['final199']['checkpoint'] == previous and config['models']['cp89']['checkpoint']['sha256'] == CP89_SHA,
            'Evaluation final model is not actual outer199 raw checkpoint')
    require(terminal['actual_optimizer_updates'] == total['optimizer_updates'] and terminal['latest_checkpoint'] == previous,
            'Terminal status/final count chain differs')
    final = audit_final(config, previous, total)
    discarded = [dict(identity={k:x['identity'][k] for k in ('pid','uid','starttick')}, phase=x['phase'],
                      returncode='NOT_INFERRED_FROM_ABSENCE; use its own preserved attempt evidence')
                 for key,x in history.items() if key not in committed_ids]
    return dict(schema='opsd-online-results-completion200-v1', accepted=True, run_id=TRAIN_RUN,
                training_config=train_proof, terminal=terminal_ref, totals=total, final_checkpoint=previous,
                final_actual_state_audit=final, outers=audits, child_history=history_ref,
                noncommitted_history=discarded, no_historical_failed_attempt_rc_rewritten=True,
                runtime_manifest=runtime['manifest'], original_helpers=helpers_ref['manifest'],
                started_unix=started, ended_unix=time.time(), no_gpu_or_inference=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--config-sha256',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    config, _=checked_json(dict(path=a.config,sha256=a.config_sha256))
    try: result=validate(config)
    except Exception as error:
        ready=all(config.get('training',{}).get(k) for k in ('terminal','final_state_audit','final_state_execution','final_reader_source','final_reader_publication')) and config.get('models',{}).get('final199',{}).get('checkpoint')
        decision='NOT_READY' if not ready or 'NOT_READY' in str(error) else 'UNKNOWN' if isinstance(error,OSError) else 'REJECTED_EVIDENCE'
        result=dict(schema='opsd-online-results-completion200-v1',accepted=False,decision=decision,error_type=type(error).__name__,reason=str(error),training_not_stopped=True,not_a_training_failure_verdict=True)
    atomic_json(a.output,result,exclusive=True); print(json.dumps(dict(accepted=result['accepted'],output=a.output)))
    return 0 if result['accepted'] else 2


if __name__=='__main__': raise SystemExit(main())
