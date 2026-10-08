"""Prospectively frozen, explicit two-outer measurement of the new science.

This schema never reads or reinterprets the old constant300 diagnostic schema.
Loading a plan authenticates bindings, not native execution or healthy training.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from reproduction.common8.formal_contract import (
    FORMAL_MATHEMATICS,MEASUREMENT_ROLE,assert_formal_run,assert_scientific_run,
    verify_scientific_config,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes,document_sha256,_strict_json
from reproduction.common8.verify_data import digest

ROLE=MEASUREMENT_ROLE
OUTER_COUNT=2
WRITES=40
MONITORS=80
PLAN_PATH='logs/warmup57-measurement-plan.json'
PARENT_PATH='preparation/measurement-parent-run-manifest.json'
METHOD_PATH='preparation/warmup57-measurement-method.json'
METHOD_ROLE='warmup57_measurement_method'
PHASE_BUDGET=dict(outer_count=2,initial_logical_update=0,initial_optimizer_writes=0,
    optimizer_writes_per_rank=40,monitors_per_rank=80,global_monitor_count=80,
    before_evaluation_step=0,after_evaluation_step=2,fixed_validation_count=512,
    fixed_validation_source_counts=dict(nq=256,hotpotqa=256),checkpoint_step=2,
    final_evaluation_in_this_phase=False,updates_counted_as_production=0)
THRESHOLDS=dict(dual_clip=3.0,epsilon=0.2,target_old_policy_kl=0.01,
    kl_log_bound=30.0,negative_tail_limit=None)
HEALTH=dict(method='paired_fixed512_absolute_allowed_drop',
    allowed_drop=dict(weighted_em=0.0,weighted_f1=0.0,valid_action_rate=0.0))
TAIL_METHOD='maximum_of_all_80_accepted_global_monitors'
INITIALIZATION=dict(actor='official_revision',reference='fixed_official_revision',adam_state='empty',
                    scheduler_completed_outer=0,rng='fresh',sampler='fresh',inherited_training_state=False)
PREREQUISITES=tuple(['logs/native-allocation-admission.json','logs/eight-rank-admission.json',
    'logs/retrieval-admission.json','logs/retrieval-http-admission.json',
    'logs/probe-coexistence-admission.json','logs/evaluator-release-admission.json',
    'logs/initial-probes-accepted.json','artifacts/live/manifest.json',
    'logs/model-smoke-initial/complete.json','logs/model-smoke-coexistence/complete.json']+
    [f'logs/model-smoke-{phase}/rank-{rank}.json' for phase in ('initial','coexistence') for rank in range(8)])


def need(ok,message):
    if not ok:raise ValueError('Invalid warmup57 measurement plan: '+message)


def _regular(path,base):
    path=Path(path);base=Path(base).resolve()
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and
         path.is_file() and path.is_relative_to(base),'Unbound regular owned file')
    need(path.stat().st_uid==os.getuid(),'Foreign evidence owner')
    return path


def _read(path):return _strict_json(Path(path).read_bytes())


def scientific_identity(run):
    assert_scientific_run(run)
    need(re.fullmatch('[0-9a-f]{32}',run['run_id']) is not None,'Invalid run UUID')
    need(re.fullmatch('[0-9a-f]{40}',run['source_commit']) is not None,'Invalid source commit')
    for name in ('source_sha256','initial_model_sha256','input_sha256'):
        values=run[name]
        need(isinstance(values,dict) and bool(values),'Missing '+name)
        for path,sha in values.items():
            need(isinstance(path,str) and not Path(path).is_absolute() and '..' not in Path(path).parts and
                 isinstance(sha,str) and re.fullmatch('[0-9a-f]{64}',sha) is not None,'Bad hash path '+name)
    need(run['input_sha256']['manifest.json']==run['budget']['common_manifest_sha256'],'Wrong common manifest')
    need(len(run['calibration_question_ids'])==len(set(run['calibration_question_ids']))==300,'Wrong initial probe subset')
    return dict(model_revision=run['model_revision'],initial_model_sha256=copy.deepcopy(run['initial_model_sha256']),
        input_sha256=copy.deepcopy(run['input_sha256']),source_commit=run['source_commit'],
        source_sha256=copy.deepcopy(run['source_sha256']),calibration_question_ids=copy.deepcopy(run['calibration_question_ids']),
        mathematics_sha256=document_sha256(FORMAL_MATHEMATICS),budget_sha256=document_sha256(run['budget']))


def verify_parent(run,parent):
    assert_scientific_run(run,expected_role=ROLE);assert_formal_run(parent)
    need(run['run_id']!=parent['run_id'] and run['measurement_parent_run_id']==parent['run_id'],
         'Measurement needs an actual separate fresh formal intent')
    need(scientific_identity(run)==scientific_identity(parent),'Parent scientific graph differs')
    return parent


def verify_method(method,run):
    need(type(method['schema_version']) is int and method['schema_version']==2 and method['role']==METHOD_ROLE,
         'Explicit prospective method required')
    need(type(method['created_unix']) in (int,float) and math.isfinite(method['created_unix']) and
         0<method['created_unix']<=run['started_unix'],'Method must predate this prepared measurement')
    need(method['phase_budget']==PHASE_BUDGET and method['health_rule']==HEALTH and
         method['negative_tail_method']==TAIL_METHOD and method['candidate_thresholds']==THRESHOLDS,
         'Prospective two-outer/zero-drop/max80 method differs')
    need(method['mathematics_sha256']==document_sha256(FORMAL_MATHEMATICS),'Wrong prospective science')
    return method


def _unstarted(base,run):
    from reproduction.common8.freeze_calibration_plan import _unstarted as old_unstarted
    old_unstarted(base,run)
    for pattern in ('scheduler-rank-*.jsonl','warmup57-measurement-optimizer-rank-*.json',
                    'warmup57-measurement-initial-optimizer-rank-*.json',
                    'warmup57-measurement-inputs-*.json','warmup57-measurement-groups-*.json'):
        need(not list((base/'logs').glob(pattern)),'Existing actual/partial measurement evidence')
    need(not (base/'logs/warmup57-measurement-segment.json').exists(),'Existing measured segment')


def load_plan(base,run,expected_slurm_job_id=None):
    base=Path(base).resolve();assert_scientific_run(run,expected_role=ROLE)
    path=_regular(base/PLAN_PATH,base);raw=path.read_bytes();plan=_strict_json(raw)
    sha=hashlib.sha256(raw).hexdigest()
    need(sha==run['warmup_measurement_plan_sha256'],'Plan bytes differ from the run')
    need(plan['schema_version']==2 and type(plan['schema_version']) is int and plan['role']==ROLE and
         plan['run_id']==run['run_id'] and plan['parent_production_run_id']==run['measurement_parent_run_id'],
         'Wrong typed phase/run/parent')
    need(plan['identity']==scientific_identity(run) and plan['mathematics']==FORMAL_MATHEMATICS and
         plan['phase_budget']==PHASE_BUDGET and plan['candidate_thresholds']==THRESHOLDS,'Wrong measurement science')
    need(plan['production_admission'] is False and plan['healthy_checkpoint_accepted'] is False,'Plan is not health admission')
    need(plan['initialization']==INITIALIZATION and plan['initial_probe_version']==run['initial_probe_version'] and
         isinstance(plan['initial_probe_version'],str) and plan['initial_probe_version'].startswith(run['run_id']+':0:'),
         'Wrong original initialization or real initial0 probe identity')
    need(type(plan['created_unix']) in (int,float) and math.isfinite(plan['created_unix']) and
         plan['created_unix']>=run['started_unix'],'Plan must be frozen after this measurement was prepared')
    need(expected_slurm_job_id is None or plan['slurm_job_id']==expected_slurm_job_id,'Wrong current native job')
    need(isinstance(plan['slurm_job_id'],str) and re.fullmatch('[1-9][0-9]*',plan['slurm_job_id']) is not None,'Invalid job')
    need(digest(_regular(base/PARENT_PATH,base))==plan['parent_manifest_sha256']==run['measurement_parent_manifest_sha256'],
         'Actual archived parent changed')
    verify_parent(run,_read(base/PARENT_PATH))
    method_path=_regular(base/METHOD_PATH,base)
    need(digest(method_path)==plan['method_sha256']==run['measurement_method_sha256'],'Prospective method bytes changed')
    method=verify_method(_read(method_path),run)
    need(plan['method']==method,'Frozen method differs from original bytes')
    need(set(plan['prerequisite_sha256'])==set(PREREQUISITES),'Incomplete original native/probe/HTTP/model prerequisite graph')
    for name,sha_value in plan['prerequisite_sha256'].items():
        need(digest(_regular(base/name,base))==sha_value,'Actual prerequisite changed: '+name)
    return plan,sha


def freeze(base,environ=None):
    """Actual native/probe/model checks after preparation and before launch/Adam."""
    from omegaconf import OmegaConf
    from reproduction.common8.configuration import resolved
    from reproduction.common8.freeze_calibration_plan import _actual_prerequisites,_reward_fingerprint
    base=Path(base).resolve();environ=dict(os.environ if environ is None else environ)
    manifest=base/'logs/run-manifest.json';original=manifest.read_bytes();run=_strict_json(original)
    assert_scientific_run(run,expected_role=ROLE);_unstarted(base,run)
    need(not (base/PLAN_PATH).exists() and 'warmup_measurement_plan_sha256' not in run,'Already frozen')
    parent=verify_parent(run,_read(base/PARENT_PATH));method=verify_method(_read(base/METHOD_PATH),run)
    actual,config,_=resolved(base);need(actual==run,'Manifest changed during actual config resolve')
    document=OmegaConf.to_container(config,resolve=True);config_proof=verify_scientific_config(document,run,base)
    job_id=environ['SLURM_JOB_ID'];_actual_prerequisites(base,run,document,config_proof['resolved_config_sha256'],job_id)
    reward=_reward_fingerprint(base,run,environ)
    prerequisite={name:digest(_regular(base/name,base)) for name in PREREQUISITES}
    plan=dict(schema_version=2,role=ROLE,run_id=run['run_id'],parent_production_run_id=parent['run_id'],
        slurm_job_id=job_id,identity=scientific_identity(run),mathematics=copy.deepcopy(FORMAL_MATHEMATICS),
        phase_budget=copy.deepcopy(PHASE_BUDGET),candidate_thresholds=copy.deepcopy(THRESHOLDS),
        parent_manifest_sha256=digest(base/PARENT_PATH),method_sha256=digest(base/METHOD_PATH),
        method=method,initial_reward_fingerprint=reward,initial_probe_version=run['initial_probe_version'],
        resolved_config_sha256=config_proof['resolved_config_sha256'],prerequisite_sha256=prerequisite,
        initialization=copy.deepcopy(INITIALIZATION),
        production_admission=False,healthy_checkpoint_accepted=False,created_unix=time.time())
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.freeze_calibration_plan import _replace_manifest
    write_once(base/PLAN_PATH,plan)
    need(manifest.read_bytes()==original,'Manifest changed before freezing claim')
    run['warmup_measurement_plan_sha256']=digest(base/PLAN_PATH);_replace_manifest(manifest,original,run)
    load_plan(base,run,expected_slurm_job_id=job_id)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(freeze(a.base),allow_nan=False))
