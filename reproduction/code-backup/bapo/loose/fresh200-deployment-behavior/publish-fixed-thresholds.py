"""Bind user-selected thresholds to clean source; never claim native acceptance."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006')

def publish(goal_path, expected_goal_sha256):
    if sys.flags.optimize:
        raise RuntimeError('optimized Python forbidden')
    assert sys.platform == 'linux' and os.uname().nodename == 'h100-5'
    assert sys.prefix == str(ROOT / 'env') and ROOT.resolve() == ROOT
    repo = ROOT / 'repo'
    source = (ROOT / 'source-expected.txt').read_text().strip()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == source
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    sys.path.insert(0,str(repo))
    from search_r1.baselines.fresh200_contract import critical_source_hashes, verify_fixed_policy_safety_artifact
    import search_r1.baselines.fresh200_contract as contract
    assert Path(contract.__file__).resolve() == repo / 'search_r1/baselines/fresh200_contract.py'
    assert re.fullmatch(r'[0-9a-f]{64}', expected_goal_sha256)
    goal = Path(goal_path)
    assert goal.is_absolute() and goal.resolve() == goal and ROOT in goal.parents
    goal_raw = goal.read_bytes()
    goal_hash = hashlib.sha256(goal_raw).hexdigest()
    assert goal_hash == expected_goal_sha256
    human = json.loads(goal_raw)
    assert human['threadId'] == '01a0fcbd-41d8-7910-a410-372ce5c8870c'
    assert human['status'] == 'active' and isinstance(human['objective'],str) and human['objective'].strip()
    assert type(human['updatedAt']) is int and human['updatedAt'] >= 1791283394
    document = dict(protocol='BAPO_FRESH200_USER_FIXED_SAFETY_V1',
        scope='USER_FIXED_THRESHOLDS_NOT_NATIVE_ACCEPTANCE_NOT_EMPIRICAL_CALIBRATION',
        source_commit=source,source_sha256=critical_source_hashes(),
        thresholds=dict(dual_clip=3,sampled_old_policy_kl=.2,negative_advantage_tail_fraction=.2,
                        negative_tail_ratio_boundary=3,kl_log_bound=30.,on_guard='skip_and_continue'),
        monitor_scopes=['full_frozen_rollout','full_current_minibatch'],
        model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        data_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
        human_goal_sha256=goal_hash)
    path = ROOT / 'logs/user-fixed-policy-thresholds.json'
    raw = (json.dumps(document,sort_keys=True,indent=2)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    config = dict(enabled=True,source_commit=source,dual_clip=3,kl_log_bound=30.,
        target_old_policy_kl=.2,negative_tail_limit=.2,threshold_origin='user_fixed',
        skip_on_policy_guard=True,fixed_threshold_artifact=str(path),
        fixed_threshold_artifact_sha256=hashlib.sha256(raw).hexdigest())
    verify_fixed_policy_safety_artifact(config)
    output = ROOT / 'logs/frozen-policy-safety-config.json'
    with output.open('x') as stream:
        json.dump(config,stream,sort_keys=True,indent=2)
    print(json.dumps(dict(config=config,scope=document['scope'],human_goal_sha256=goal_hash)))

if __name__ == '__main__':
    assert len(sys.argv) == 3
    publish(sys.argv[1],sys.argv[2])
