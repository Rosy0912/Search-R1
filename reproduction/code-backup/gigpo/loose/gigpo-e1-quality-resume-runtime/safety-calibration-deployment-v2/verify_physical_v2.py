"""Read-only target deployment proof; imports target source without allocating CUDA."""
import inspect
import json
import subprocess
import sys
from pathlib import Path

from deployment_contract_v2 import ROOT, SOURCE, HELPERS, CONTROLS, validate_candidate, require, sha, read_bound, write_once


def metadata(path):
    p = Path(path)
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino)


def main():
    candidate_path, expected_sha, output = sys.argv[1:]
    require(sha(candidate_path) == expected_sha, 'Candidate bytes changed')
    c = json.loads(Path(candidate_path).read_text())
    validate_candidate(c)
    inherited = read_bound(c['inherited_physical'])
    old = {r['path']: r for r in inherited['metadata']}
    actual, hashed = [], []
    for item in c['files']:
        p = Path(item['path'])
        m = metadata(p)
        require(m['bytes'] == item['bytes'], 'Candidate file bytes changed: ' + str(p))
        if str(p).startswith((c['repo'] + '/', str(HELPERS) + '/', str(CONTROLS) + '/')):
            require(sha(p) == item['sha256'], 'New source/helper SHA changed: ' + str(p))
            hashed.append(str(p))
        else:
            require(m == old[str(p)], 'Inherited model/data/env/retrieval metadata changed: ' + str(p))
        actual.append(m)
    evidence = []
    for item in c['evidence']:
        p = Path(item['path'])
        require(sha(p) == item['sha256'], 'Evidence SHA changed: ' + str(p))
        evidence.append(dict(path=str(p), bytes=p.stat().st_size, sha256=item['sha256']))
    require(subprocess.check_output(['git', '-C', c['repo'], 'rev-parse', 'HEAD'], text=True).strip() == SOURCE,
            'Source commit differs')
    require(not subprocess.check_output(['git', '-C', c['repo'], 'status', '--porcelain'], text=True).strip(), 'Dirty target source')
    require(sys.executable == str(ROOT / 'env/bin/python'), 'Wrong target runtime')
    import torch
    import search_r1.baselines.policy_safety as safety
    import search_r1.baselines.recovery as recovery
    require(torch.__version__ == '2.4.0+cu121' and not torch.cuda.is_initialized(), 'Wrong runtime or unexpected CUDA initialization')
    require(inspect.getfile(safety) == c['repo'] + '/search_r1/baselines/policy_safety.py'
            and inspect.getfile(recovery) == c['repo'] + '/search_r1/baselines/recovery.py', 'Wrong import source')
    from prepare_candidate_v2 import preview_config
    preview = read_bound(c['expected_config_preview'])
    actual_config = preview_config(c, Path(c['repo']) / 'verl/trainer/config')
    require(actual_config == preview['resolved'], 'Actual target Hydra config differs from bound local preview')
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(actual_config['actor_rollout_ref']['model']['path'])
    require(isinstance(tokenizer.pad_token_id, int) and not torch.cuda.is_initialized(), 'Missing actual official tokenizer identity')
    write_once(output, dict(status='PHYSICAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',
        candidate_sha256=expected_sha, code_commit=SOURCE, executable=sys.executable, torch=torch.__version__,
        metadata=actual, evidence=evidence, files_count=len(actual), evidence_count=len(evidence),
        newly_hashed_paths=hashed, inherited_physical_sha256=c['inherited_physical']['sha256'],
        runtime_all_file_sha_before_gpu_required=True, cuda_initialized=False,
        target_resolved_config=actual_config, config_preview_sha256=c['expected_config_preview']['sha256'],
        pad_token_id=tokenizer.pad_token_id,
        resource_guard=c['resource_guard'], resource_ownership=c['resource_ownership']))
    print(json.dumps(dict(output=output, sha256=sha(output), cuda_initialized=False)))


if __name__ == '__main__':
    main()
