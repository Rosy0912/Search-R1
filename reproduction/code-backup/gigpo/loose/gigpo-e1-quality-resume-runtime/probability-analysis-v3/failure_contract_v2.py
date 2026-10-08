"""Read only existing failed-job originals; never infer zero Adam from missing ready."""
import json
import os
from pathlib import Path


DIAGNOSTIC_KINDS={'compute_log_prob','forward_log_prob','reference_log_prob','probability_probe',
                  'driver_outer','config_boundary','forward_probability_diagnostics'}
STRUCTURAL_ERRORS=(
    'Generation prefix differs from full-trajectory old-policy prefix; preserve and investigate',
    'Retokenized action differs from sampled tokens; preserve and investigate',
    'Missing/different actual sampling structural identity: ',
)


def audit_zero_update_rows(rows,source,run_id,require):
    """Known forward diagnostics are legal; optimizer/gradient rows are not zero-update proof."""
    for row in rows:
        require(row.get('commit',row.get('source_commit'))==source and row.get('run_id')==run_id,
                'Failure audit source/run identity differs')
        require(row.get('kind') not in ('actor_outer','actor_outer_begin','optimizer_step','actor_update'),
                'Actual actor update record contradicts zero-Adam failure diagnostic')
        require(not any(key in row for key in ('optimizer_steps','optimizer_steps_completed','micro_index',
                    'gradient_micro_batches','grad_norms','mini_index')),
                'Gradient/update audit semantics cannot establish zero Adam')
        require(row.get('kind') in DIAGNOSTIC_KINDS,'Unknown audit semantics cannot establish zero Adam')


def validate_failure_context(args,binding,manifest,read_json,sha,require,root,source,restore):
    """Bind collector and independent review to actual already-written artifacts."""
    context=read_json(args.failure_context,args.failure_context_sha)
    review=read_json(args.failure_review,args.failure_review_sha)
    run=Path(args.run)
    require(all(Path(path).parent==root/'logs' for path in (args.failure_context,args.failure_review)),
            'Failure collector/review must be actual target logs originals')
    require(context['schema']==1 and context['scope']=='ACTUAL_CALIBRATION_ZERO_ADAM_STRUCTURAL_FAILURE'
            and context['job_id']==args.job and context['run']==args.run and context['run_id']==run.name
            and context['code_commit']==source and context['binding_sha256']==args.binding_sha
            and context['material_manifest_sha256']==args.manifest_sha,
            'Failure diagnostic source/job/run/binding/material identity differs')
    require(review['status']=='FAILED_CALIBRATION_ZERO_ADAM_DIAGNOSTICS_ONLY'
            and review['scope']==context['scope'] and review['job_id']==args.job
            and review['code_commit']==source and review['run']==args.run and review['run_id']==run.name
            and review['binding_sha256']==args.binding_sha and review['material_manifest_sha256']==args.manifest_sha
            and review['failure_context_sha256']==args.failure_context_sha
            and all(type(review[key]) is int and review[key]==0 for key in
                    ('actual_adam_writes','actual_gradient_micro_batches','retained_training_updates_contributed'))
            and review['zero_adam_audit_environment_verified'] is True
            and review['post_sampling_live_pause_available'] is False
            and review['probability_approval_generated'] is False,
            'No exact independent zero-Adam failure review; not a normal ready analysis')
    names={'terminal','result','resolved_config','ray_environment','calibration_log','cgroup_release'}
    originals=context['originals']
    require(set(originals)==names and len({r['path'] for r in originals.values()})==len(names),
            'Failure terminal/resource/environment original inventory incomplete or duplicate')
    for name,record in originals.items():
        path=Path(record['path'])
        require(path.is_absolute() and '..' not in path.parts and path.is_file()
                and not path.is_symlink() and path.resolve()==path
                and (path.parent==root/'logs' or path.is_relative_to(run))
                and type(record['bytes']) is int and path.stat().st_size==record['bytes']
                and sha(path)==record['sha256'],'Changed/missing actual failure original: '+name)
    require(originals['result']['path']==str(run/'run-result.json')
            and originals['resolved_config']['path']==str(run/'resolved-config.yaml')
            and originals['ray_environment']['path']==str(run/'ray-audit-environment-verification.json')
            and originals['calibration_log']['path']==str(run/'calibration.log'),
            'Failed-job originals outside exact actual locations')
    terminal=json.loads(Path(originals['terminal']['path']).read_text())['fields']
    require(terminal['JobId']==args.job and terminal['NodeList']=='h100-8' and terminal['JobState']=='FAILED'
            and terminal['ExitCode']!='0:0' and str(terminal['Requeue'])==str(terminal['Restarts'])=='0',
            'No actual single failed terminal')
    result=json.loads((run/'run-result.json').read_text())
    require(result['status']=='FAILED' and result['job_id']==args.job and result['run_id']==run.name
            and result['source_commit']==source and result['binding_sha256']==args.binding_sha
            and result['restore_manifest_sha256']==restore and type(result['retained_training_updates_contributed']) is int
            and result['retained_training_updates_contributed']==0
            and result['cleanup_residual_states']=={} and result['final_gpu']['processes']==[],
            'Failed run result/clean resource identity contradicts diagnostic scope')
    release=json.loads(Path(originals['cgroup_release']['path']).read_text())
    jobroot='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_'+args.job
    require(release['job']==args.job and release['node']=='h100-8' and len(release['paths'])==2
            and {r['path'] for r in release['paths']}=={jobroot,jobroot+'/step_0'}
            and all(r['exists'] is False and r['errno']==2 and r['ancestor_readable'] is True
                    and r['readable_ancestor']=='/sys/fs/cgroup/freezer/slurm_h100-8' for r in release['paths']),
            'Actual failed-job and original scope release missing')
    # Exact source call order puts structural refusal before wait/first update.
    log=(run/'calibration.log').read_text()
    require('admit_alignment' in log and 'wait_probability_acceptance' in log
            and any('ValueError: '+message in log for message in STRUCTURAL_ERRORS),
            'Not the exact pre-Adam structural refusal call path')
    environment=json.loads((run/'ray-audit-environment-verification.json').read_text())
    expected=dict(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'audit'),SEARCH_R1_BASELINE_RUN_ID=run.name,
        SEARCH_R1_BASELINE_COMMIT=source,SEARCH_R1_BASELINE_CONFIG_SHA256=originals['resolved_config']['sha256'],
        SEARCH_R1_BASELINE_DATA_IDENTITY=binding['common_manifest_sha256'])
    require(environment['driver']==expected and len(environment['workers'])==8
            and {r['rank'] for r in environment['workers']}==set(map(str,range(8)))
            and all(r['environment']==expected and r['imports']['verl.workers.actor.dp_actor']==
                    str(Path(binding['repo'])/'verl/workers/actor/dp_actor.py') for r in environment['workers']),
            'Actual full8 Ray audit-env/source-import verification differs')
    observed=context['audit_path_observations']
    expected_paths={str(run/relative) for rank in range(8) for relative in
        (f'audit/{rank}.jsonl',f'audit/numeric/micro-{rank}.jsonl',f'audit/policy-safety/rank-{rank}.jsonl')}
    require(len(observed)==24 and {r['path'] for r in observed}==expected_paths,
            'Missing/duplicate actual eight-rank actor/numeric/policy audit path observations')
    for record in observed:
        path=Path(record['path'])
        if record['stat_outcome']=='ENOENT':
            ancestor=Path(record['readable_ancestor'])
            require(record['errno']==2 and record['ancestor_readable'] is True and ancestor.is_relative_to(run/'audit')
                    and path.is_relative_to(ancestor) and ancestor.is_dir() and os.access(ancestor,os.R_OK|os.X_OK),
                    'Absent audit path has no actually readable own ancestor')
            try:path.stat()
            except FileNotFoundError:pass
            else:raise ValueError('Audit file appeared after saved absence observation')
        else:
            require(record['stat_outcome']=='PRESENT' and path.is_file() and not path.is_symlink()
                    and path.resolve()==path and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],
                    'Actual failure audit original differs')
            audit_zero_update_rows([json.loads(line) for line in path.read_text().splitlines()],source,run.name,require)
    require(not (run/'probability-identity-ready.json').exists()
            and not (run/'probability-identity-approval.json').exists()
            and not (run/'calibration-update.json').exists(),'Failed structural diagnostic has ready/approval/update artifacts')
    require('post_sampling_pause' not in manifest,'Failure context cannot masquerade as normal live pause')
    import yaml
    config=yaml.safe_load((run/'resolved-config.yaml').read_text())
    return dict(status='ACTUAL_TERMINAL_ZERO_ADAM_STRUCTURAL_FAILURE_DIAGNOSTICS_ONLY',job_id=args.job,
        terminal_state='FAILED',actual_adam_writes=0,actual_gradient_micro_batches=0,
        live_pause_available=False,independent_review_sha256=args.failure_review_sha,
        original_roles_sha256={k:r['sha256'] for k,r in originals.items()},audit_observations_count=24,
        production_thresholds_selected=False,probability_approval_generated=False),config
