"""Bind fresh child identities under native allocation; never GPU admission."""
import argparse
import json
import os
import socket
import subprocess
from pathlib import Path

from reproduction.a800.owned_processes import publish
from reproduction.common8.native_allocation import fields
from reproduction.common8.policy_calibration_plan import (
    DISCOVERY_ROLE, CONFIRMATION_ROLE, _identity, canonical_bytes,
)
from reproduction.common8.prepare_run import prepare
from reproduction.common8.freeze_calibration_plan import _unstarted
from reproduction.common8.stage_retrieval import memory_bytes
from reproduction.common8.verify_data import digest


def native_preparation():
    """Actual job reservation before CPU model/data reading, not CUDA proof."""
    job_id = os.environ['SLURM_JOB_ID']
    job = fields(subprocess.check_output(['scontrol', 'show', 'job', '-o', job_id], text=True))
    assert job['JobId'] == job_id and job['JobState'] == 'RUNNING'
    assert job['NodeList'] == socket.gethostname().split('.')[0] == 'h100-8'
    assert job['NumNodes'] == '1' and job['TresPerNode'] in ('gres:gpu:8', 'gres:gpu:h100:8')
    assert memory_bytes(job['TRES']) >= 512 * 2**30
    return job_id


def fresh_parent(parent, discovery):
    """Keep the production intent at official0, distinct from every child."""
    manifest = parent / 'logs/run-manifest.json'
    run = json.loads(manifest.read_text())
    assert run.get('policy_calibration_role', 'production') == 'production'
    _identity(run)
    _unstarted(parent, run)
    if 'policy_calibration_discovery_base' in run:
        assert run['policy_calibration_discovery_base'] == str(discovery)
    else:
        run['policy_calibration_discovery_base'] = str(discovery)
        publish(manifest, run)
    return run


def prepare_child(base, parent, role, model, official_manifest, commit, discovery=None):
    """Preserve separate fresh manifests, exact products, and original parent bytes.

    Ordinary prepare_run does all original model/data/hash reading. These added
    fields identify a child; they do not create any ready/accepted/model marker.
    Existing child attempts are never replayed or silently overwritten.
    """
    base, parent = Path(base).resolve(), Path(parent).resolve()
    assert base != parent and role in (DISCOVERY_ROLE, CONFIRMATION_ROLE)
    assert role == DISCOVERY_ROLE or discovery is not None, 'Confirmation requires an explicit discovery base'
    discovery = base if role == DISCOVERY_ROLE else Path(discovery).resolve()
    assert role == DISCOVERY_ROLE or discovery not in (base, parent)
    job_id = native_preparation()
    assert not (base / 'logs/run-manifest.json').exists(), 'Inspect existing child, never repeat preparation'
    parent_path = parent / 'logs/run-manifest.json'
    if not parent_path.exists():
        assert role == DISCOVERY_ROLE, 'Confirmation requires the original prepared production parent'
        prepare(parent, model, official_manifest, commit)
    parent_run = fresh_parent(parent, discovery)
    child = prepare(base, model, official_manifest, commit)
    assert child['run_id'] != parent_run['run_id']
    assert canonical_bytes(_identity(child)) == canonical_bytes(_identity(parent_run))
    if role == CONFIRMATION_ROLE:
        # The later production pipeline must use this precise independent
        # confirmation. Never infer its directory or substitute another child.
        key = 'policy_calibration_confirmation_base'
        if key in parent_run:
            assert parent_run[key] == str(base), 'Production parent already binds another confirmation'
        else:
            parent_run[key] = str(base)
            publish(parent_path, parent_run)
    original = parent_path.read_bytes()
    archive = base / 'preparation/parent-run-manifest.json'
    with archive.open('xb') as stream:
        stream.write(original); stream.flush(); os.fsync(stream.fileno())
    child.update(parent_production_run_id=parent_run['run_id'],
                 policy_calibration_role=role,
                 policy_calibration_parent_manifest_sha256=digest(archive),
                 policy_calibration_slurm_job_id=job_id,
                 policy_calibration_discovery_base=str(discovery))
    publish(base / 'logs/run-manifest.json', child)
    return child


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True); p.add_argument('--parent-base', required=True)
    p.add_argument('--role', required=True, choices=(DISCOVERY_ROLE, CONFIRMATION_ROLE))
    p.add_argument('--model', required=True); p.add_argument('--official-manifest', required=True)
    p.add_argument('--source-commit', required=True); p.add_argument('--discovery-base')
    args = p.parse_args()
    result = prepare_child(args.base, args.parent_base, args.role, args.model,
                           args.official_manifest, args.source_commit, args.discovery_base)
    print(json.dumps(dict(run_id=result['run_id'], role=result['policy_calibration_role'],
                         parent_production_run_id=result['parent_production_run_id'],
                         production_admission=False, actual_grpo_updates=0)))
