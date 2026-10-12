"""Read-only existing3360 evidence gate; refuses incomplete/missing evidence.

Uses existing run-result, terminal, validation, native and cold-HF receipt schemas.
Does not cancel, edit, resume, rerun or remove predecessor artifacts.
"""
from pathlib import Path
from production_contract import ROOT, read_bound, require, validation_reports


def verify_predecessor(package):
    spec = package['predecessor_gate']
    run = ROOT/'runs/tree-branpo-fresh200-3360'
    require(spec['job_id']=='3360' and spec['run_id']==run.name
            and spec['run_root']==str(run) and spec['mutations_allowed'] is False,
            'Exact preserved3360 predecessor gate required')
    old_package = ROOT/'instruct300/candidates'/('branpo-fresh200-'+spec['source_commit'])/'manifest.json'
    old, package_sha, _ = read_bound(old_package, spec['package_manifest_sha256'])
    require(old['code_commit']==spec['source_commit'], 'Predecessor source differs')
    result, result_sha, _ = read_bound(run/'run-result.json')
    require(result['job_id']=='3360' and result['code_commit']==spec['source_commit']
            and result['status']=='ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE'
            and result['training_body_exit_code']==0 and result['actual_completed_outer']==200,
            'Wait for3360 terminal exit0/200; never cancel it')
    terminal, terminal_sha, _ = read_bound(run/'output/train200-terminal.json', result['terminal_evidence_sha256'])
    require(terminal['status']=='FRESH_BRANPO200_JSON_ACCOUNTING_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE'
            and terminal['rank_accounting']['completed_outer']==200
            and terminal['final']['evaluated_rows']==51713 and terminal['final']['phase']=='final',
            'Predecessor terminal accounting incomplete')
    final = validation_reports(run, 200, final=True, package=old, config_hash=spec['config_sha256'])
    require(final==terminal['final'], 'Preserved full51713 actual validation differs')
    native_path = run/'checkpoints/training_state/global_step_200/manifest.json'
    native, native_sha, _ = read_bound(native_path)
    ids = native['identity']
    require(native['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
            and ids['job_id']=='3360' and ids['completed_step']==200
            and ids['code_commit']==spec['source_commit'] and ids['config_sha256']==spec['config_sha256'],
            'Preserved native200 checkpoint identity differs')
    hf_path = run/'checkpoints/actor/global_step_200'
    require(Path(native['hf_checkpoint']).resolve()==hf_path.resolve(), 'Predecessor HF path differs')
    for relative in native['hf_identity']['files']:
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts
                and (hf_path/path).is_file() and not (hf_path/path).is_symlink(),
                'Predecessor published HF member missing')
    cold_path = run/'output/final-hf-acceptance/identity-bound-final-hf.json'
    require(spec['final_hf_receipt']==str(cold_path), 'Bound predecessor cold receipt required')
    cold, cold_sha, _ = read_bound(cold_path)
    require(cold['status']=='JOB3360_FINAL_HF_COLD_RELOAD_IDENTITY_BOUND'
            and cold['job_id']=='3360' and cold['source_commit']==spec['source_commit']
            and cold['config_sha256']==spec['config_sha256']
            and cold['package_manifest_sha256']==package_sha
            and cold['native_manifest_sha256']==native_sha
            and cold['terminal_receipt_sha256']==result_sha
            and cold['terminal_evidence_sha256']==terminal_sha
            and cold['hf_identity_sha256']==native['hf_identity']['sha256']
            and cold['input_hashes_before']==cold['input_hashes_after']
            and cold['actual_cpu_model_cold_reload'] is True
            and cold['actual_tokenizer_cold_reload'] is True
            and cold['all_parameters_finite'] is True and cold['shape_keys_full_checked'] is True,
            'Predecessor actual cold-HF preservation evidence incomplete')
    read_bound(cold_path.parent/'full-HF-cold-reload.json', cold['full_hf_report_sha256'])
    read_bound(cold_path.parent/'tokenizer-cold-reload.json', cold['tokenizer_report_sha256'])
    return dict(job_id='3360', completed_outer=200, final_rows=51713,
                source_commit=spec['source_commit'], predecessor_package_sha256=package_sha,
                result_sha256=result_sha, terminal_sha256=terminal_sha,
                final_validation_sha256=final['sha256'], native_manifest_sha256=native_sha,
                cold_hf_receipt_sha256=cold_sha, predecessor_mutations=False)
