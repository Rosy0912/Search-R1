"""Submit the already deployed immutable V14 packet once; never retries unknown."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import ssh_retained18 as ssh

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def main():
    spec = json.loads((HERE/'deployment-v14.json').read_text())
    if '/current200-retained18-speed-v14-' not in spec['package'] or '/SUBMISSION_ONCE-retained18-current200-speed-v14-' not in spec['once_intent']:
        raise RuntimeError('Only a newly materialized V14 packet may be submitted')
    if spec.get('remote_submitted') is not False:
        raise RuntimeError('Local specification already submitted or unknown; inspect actual once journal')
    arguments = ['env', '-u', 'SLURM_JOB_ID', '-u', 'SLURM_JOB_NODELIST', '-u', 'SLURM_STEP_ID',
        'CUDA_VISIBLE_DEVICES=', 'PYTHONOPTIMIZE=0',
        'PYTHONPATH='+spec['package']+':/dataset/shaoshitong/search-r1-rl/tree-grpo/tools',
        '/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python', spec['package']+'/submit_once.py']
    for key, name in [('package', 'package'), ('manifest-sha', 'manifest_sha'),
                      ('release-path', 'release_path'), ('release-sha', 'release_sha'),
                      ('script', 'script_path'), ('script-sha', 'script_sha')]:
        arguments += ['--'+key, spec[name]]
    try:
        result = ssh.run(shlex.join(arguments), timeout=95)
        record = dict(observed_at=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, source_commit=spec['source_commit'],
            manifest_sha256=spec['manifest_sha'], no_blind_retry=True)
    except Exception as error:
        record = dict(observed_at=datetime.now(timezone.utc).isoformat(), status='SUBMISSION_RESPONSE_UNKNOWN',
            source_commit=spec['source_commit'], manifest_sha256=spec['manifest_sha'],
            observation_error=repr(error), no_blind_retry=True)
        spec['remote_submitted'] = None
        (HERE/'deployment-v14.json').write_text(json.dumps(spec, indent=2)+'\n')
        (WORKSPACE/'outputs/tree-fresh200/RETAINED18_SPEED_V14_SUBMISSION_RESPONSE_20261008.json').write_text(
            json.dumps(record, indent=2)+'\n')
        raise
    output = WORKSPACE/'outputs/tree-fresh200/RETAINED18_SPEED_V14_SUBMISSION_RESPONSE_20261008.json'
    output.write_text(json.dumps(record, indent=2)+'\n')
    if result.returncode != 0:
        spec['remote_submitted'] = None
        (HERE/'deployment-v14.json').write_text(json.dumps(spec, indent=2)+'\n')
        raise RuntimeError('Submission result failed or ambiguous; inspect actual once journal before further action')
    final = json.loads(result.stdout)
    if final.get('status') != 'SUBMITTED' or final['code_commit'] != spec['source_commit']:
        raise RuntimeError('Submission response differs from immutable source')
    spec.update(remote_submitted=True, job_id=final['job_id'], run_id=final['run_id'])
    (HERE/'deployment-v14.json').write_text(json.dumps(spec, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
