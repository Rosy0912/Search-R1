"""Verify explicit CPU evidence, then call the original replay in this same PID."""
import argparse
import json
import os
from pathlib import Path
import sys
try:
    from .common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, read, checked_file,
                         same, exclusive, bind_helper, execution_packet)
except ImportError:
    from common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, read, checked_file,
                        same, exclusive, bind_helper, execution_packet)

ENDPOINTS = {('source100', 0)} | {(v, t) for v in VARIANTS for t in (15, 25)}


def published(spec):
    manifest, mp = read(spec['source_manifest']); publication, pp = read(spec['publication'])
    require(publication.get('accepted') is True and publication['branch'] == 'search-e1'
            and publication['source_commit'] == publication['remote_readback_sha']
            and len(publication['source_commit']) == 40 and publication['source_manifest_sha256'] == mp['sha256'],
            'Replay gate/source publication or readback missing')
    root = Path(spec['helper_root']).resolve()
    require({str(p.relative_to(root)) for p in root.rglob('*.py')} == {n for n in manifest['files'] if n.endswith('.py')},
            'Published replay helper Python inventory changed')
    for name, entry in manifest['files'].items():
        relative = Path(name); require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe helper source path')
        checked_file(dict(path=str(root / relative), sha256=entry['sha256'], bytes=entry['bytes']))
    return dict(root=str(root), manifest=mp, publication=pp)


def coverage_packet(packet, study, strong, native, configs, finals, endpoints):
    """Actual assembler CLI stdout -> genuine coverage report; finalfalse is allowed."""
    cli, execution = execution_packet(dict(packet, output=packet['actual_stdout']))
    require(cli.get('accepted') is True and same(cli['output'], packet['output']), 'Coverage CLI points to another artifact')
    value, proof = read(cli['output'])
    require(value['schema'] == 'opsd-signal-final-coverage-v1' and value['accepted'] is True
            and value['coverage_accepted'] is True and value['complete_groups'] == 4
            and value['complete_outers'] == 100 and value['train_trajectory_slots'] == 256000
            and value['formal_native512_slots'] == 4608 and value['final_checkpoints'] == finals
            and same(value['study_manifest'], study) and same(value['study_terminal'], strong)
            and same(value['native_study_completion'], native) and value['torch_imported'] is False,
            'Initial100-outer/4608-slot coverage is partial or belongs to another terminal')
    require(type(value['final_accepted']) is bool and cli['final_accepted'] == value['final_accepted'],
            'Coverage CLI/report final flag differs')
    request, request_ref = read(value['request'])
    require(same(request['study_manifest'], study) and same(request['study_terminal'], strong)
            and same(request['native_study_completion'], native)
            and [g['variant'] for g in request['groups']] == list(VARIANTS)
            and all(same(g['config'], configs[g['variant']]) for g in request['groups']),
            'Actual coverage request does not bind original four configs')
    actual_endpoints = {(r['child']['path'], r['child']['sha256']) for r in value['evaluations']}
    require(len(value['evaluations']) == 9 and actual_endpoints == endpoints,
            'Initial coverage evaluated another nine raw endpoints')
    require(len(value['outer_rows']) == 100
            and {(r['variant'], r['outer_index']) for r in value['outer_rows']} == {(v, i) for v in VARIANTS for i in range(25)},
            'Initial coverage lacks explicit100 unique outer rows')
    # Final requirements include this replay's heldout analysis; requiring finaltrue here would deadlock.
    return dict(report=proof, actual_execution=execution, request=request_ref,
                final_accepted=value['final_accepted'], final_acceptance_not_required_for_replay=True)


def preflight(request, original_entry, actual_argv):
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Independent replay gate requires authorized CPU UID/empty CUDA/no Torch')
    require(request['schema'] == 'opsd-signal-posthoc-replay-gate-request-v1', 'Wrong replay gate request')
    source = published(request['gate_source'])
    require(Path(source['root']) / 'gate.py' == Path(__file__).resolve(), 'Actual new gate entry is outside its published source')
    plan, plan_ref = read(request['old_plan']); require(plan['schema'] == 'opsd-signal-heldout-replay-plan-v1', 'Wrong original replay plan')
    old = dict(helper_root=plan['helper_root'], source_manifest=plan['source_manifest'], publication=plan['publication'])
    original_source = published(old)
    entry = str(Path(plan['helper_root']).resolve() / 'replay.py')
    require(str(Path(original_entry).resolve()) == entry and original_entry == entry and entry in actual_argv,
            'Actual CLI must contain the exact bound --original-replay-entry path for original worker ownership guard')
    require(os.environ.get('E1_RUN_ID') == plan['analysis_run_id'], 'Actual independent replay run marker differs from original plan')
    require(same(plan['four_group_terminal'], request['native_study_completion']), 'Original plan must retain native terminal schema')
    study, study_ref = read(request['study_manifest']); strong, strong_ref = read(request['strong_terminal'])
    native, native_ref = read(request['native_study_completion'])
    require(str(Path(study['study_base']).resolve()) == STUDY_ROOT and [g['variant'] for g in study['groups']] == list(VARIANTS),
            'Wrong original four-arm study')
    require(strong['schema'] == 'opsd-signal-weight-driver-terminal-v1' and strong['accepted'] is True
            and strong['actual_supervisor_wait_returncode'] == 0
            and strong['driver_exit_evidence']['source'] == 'same-retained-tmux-pane'
            and strong['driver_exit_evidence']['pane_dead'] == 1 and strong['driver_exit_evidence']['pane_dead_status'] == 0
            and strong['driver_exit_evidence']['no_exit_code_inferred_from_absence'] is True
            and strong['owned_gpu_release']['accepted'] is True
            and same(strong['study_manifest'], study_ref) and same(strong['native_study_completion'], native_ref),
            'Strong supervisor terminal is not the actual same-pane wait0/release result')
    require(native['accepted'] is True and native['completed_groups'] == 4 and native['completed_outers'] == 100
            and native['full_training_em_rows'] == 100 and same(native['study_manifest'], study_ref)
            and [g['variant'] for g in native['groups']] == list(VARIANTS), 'Native completion not all four25 groups')
    require(set(request['configs']) == set(VARIANTS) and set(request['final_checkpoints']) == set(VARIANTS),
            'Four original configs and actual raw final checkpoints required')
    refs = request['configs']; configs = {}
    for group in study['groups']:
        variant = group['variant']; config, cfg = read(refs[variant]); configs[variant] = config
        require(same(group['config'], cfg) and config['variant'] == variant and config['study_id'] == study['study_id']
                and config['helper_source_manifest']['sha256'] == ORIGINAL_SHA, 'Replay group config/study/source differs')
        cp, cp_ref = read(request['final_checkpoints'][variant])
        require(cp['schema'] == 'opsd-online-checkpoint-v1' and cp['accepted'] is True
                and cp['run_id'] == config['run_id'] and cp['config_sha256'] == cfg['sha256']
                and cp['cursor']['outer_index'] == 24 and cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count']
                and cp['complete_rank_count'] == 4 and cp['all_rank_adapter_sha256'] == [cp['all_rank_adapter_sha256'][0]] * 4,
                'Replay final is not this arm actual committed raw outer24 checkpoint')
        captured = next(g for g in strong['groups'] if g['variant'] == variant)
        declared = next(g for g in native['groups'] if g['variant'] == variant)
        require(same(captured['config'], cfg) and same(captured['actual_terminal'], declared['actual_terminal'])
                and same(captured['train_ledger'], declared['summary']) and captured['actual_controller_wait_returncode'] == declared['actual_wait_returncode'] == 0,
                'Strong/native actual group controller wait chains differ')
    require({(r['path'], r['sha256']) for r in plan['configs']} == {(r['path'], r['sha256']) for r in refs.values()}
            and len(plan['configs']) == 4 and same(plan['bank'], request['bank']), 'Original replay plan configs/frozen bank changed')
    bank, bank_ref = read(request['bank'])
    require(bank['schema'] == 'opsd-signal-fixed-common-zero-bank-v1' and bank['accepted'] is True
            and bank['study_id'] == study['study_id'] and bank['frozen_before_all_group_15_25_outputs'] is True
            and all(same(bank['configs'][v], refs[v]) for v in VARIANTS)
            and same(bank['validation_identity'], configs['G_POS']['validation_identity']),
            'Original frozen B bank/config/data identity differs; cannot refreeze dynamically')
    endpoint_refs = request['endpoints']; expected_refs = {(r['path'], r['sha256']) for r in endpoint_refs}
    require(len(endpoint_refs) == len(expected_refs) == 9
            and {(r['path'], r['sha256']) for r in plan['endpoints']} == expected_refs and len(plan['endpoints']) == 9,
            'Original plan must bind the exact same nine endpoints')
    seen = set()
    for ref in endpoint_refs:
        endpoint, proof = read(ref); key = (endpoint['version'], endpoint['completed_outer']); t = key[1]
        config = configs['G_POS' if t == 0 else key[0]]
        require(key in ENDPOINTS and key not in seen and endpoint['schema'] == 'opsd-signal-native512-complete-v1'
                and endpoint['accepted'] is True and endpoint['primary_slots'] == 512 and endpoint['actual_child_exit_and_release'] is True
                and endpoint['run_id'] == config['run_id'] and endpoint['config']['sha256'] == refs[config['variant']]['sha256'],
                'Only original accepted formal raw zero/15/25 endpoints may be replayed')
        if t:
            cp, cp_ref = read(endpoint['raw_checkpoint']['manifest'])
            require(endpoint['raw_checkpoint']['kind'] == 'raw_end_of_outer'
                    and cp['cursor']['outer_index'] == t - 1 and cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count']
                    and endpoint['input_identity']['expected_adapter_tensor_sha256'] == cp['all_rank_adapter_sha256'][0],
                    'Replay endpoint is cache/best/another raw update')
            if t == 25: require(same(cp_ref, request['final_checkpoints'][key[0]]), 'Replay final raw endpoint differs from coverage final')
        else:
            require(same(proof, bank['common_zero_evaluation']), 'Replay common-zero endpoint differs from frozen bank')
        seen.add(key)
    require(seen == ENDPOINTS, 'Nine endpoint keys incomplete')
    coverage = coverage_packet(request['initial_coverage'], study_ref, strong_ref, native_ref, refs,
                               request['final_checkpoints'], expected_refs)
    output = Path(request['replay_output']).resolve(); audit_path = Path(request['preflight_output']).resolve()
    require(output.is_relative_to(Path(STUDY_ROOT) / 'analysis') and audit_path.is_relative_to(Path(STUDY_ROOT) / 'analysis')
            and output != audit_path and not audit_path.is_relative_to(output) and not output.exists() and not audit_path.exists(),
            'Replay/output audit must be independent new remote analysis paths; do not precreate replay output')
    require('torch' not in sys.modules, 'Preflight unexpectedly imported Torch')
    return dict(schema='opsd-signal-posthoc-replay-preflight-v1', accepted=True, study_id=study['study_id'],
                old_plan=plan_ref, strong_terminal=strong_ref, native_study_completion=native_ref,
                initial_coverage=coverage, original_source=original_source, gate_source=source,
                bank=bank_ref, configs=refs, final_checkpoints=request['final_checkpoints'],
                original_replay_entry=entry, same_process_required=True, new_samples_or_training=0)


def run(request_ref, original_entry, actual_argv):
    request, rp = read(request_ref); evidence = preflight(request, original_entry, actual_argv)
    evidence['request'] = rp; audit_ref = exclusive(request['preflight_output'], evidence)
    old_plan, _ = read(request['old_plan'])
    replay = bind_helper(old_plan['helper_root'], old_plan['source_manifest'], ('replay',))['replay']
    require(Path(replay.__file__).resolve() == Path(original_entry), 'Isolated original replay source path changed')
    before = os.getpid()
    # Direct call only: no subprocess parent, monkeypatch, extra sampling, or scientific function replacement.
    result = replay.run(request['old_plan'], request['replay_output'])
    require(os.getpid() == before and 'torch' not in sys.modules, 'Replay wrapper process/device boundary changed')
    complete, _ = read(result)
    require(complete['accepted'] is True and same(complete['plan'], request['old_plan'])
            and complete['posthoc_responses'] == 4608 and complete['new_samples_or_training'] == 0
            and complete['actual_workers_exit_and_release'] is True and complete['final_release']['accepted'] is True,
            'Original replay did not actually complete all4608 rows and release')
    return dict(accepted=True, preflight=audit_ref, complete=result, same_process_pid=before,
                new_samples_or_training=0, scientific_functions_patched=False)


def main():
    parser = argparse.ArgumentParser()
    for name in ('request', 'request-sha256', 'original-replay-entry'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = run(dict(path=args.request, sha256=args.request_sha256), args.original_replay_entry, sys.argv)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
