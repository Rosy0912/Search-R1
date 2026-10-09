"""Frozen-source fresh200 admission. Unbound candidate: no launch is authorized."""
from pathlib import Path
import argparse, hashlib, json, os, shlex, subprocess, sys, time
ROOT = Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE = '080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
HARDWARE_RUN = ROOT / 'runs/hardware-aligned-em200-13e07285c97044fd97974d94f53fea1b-3345'
# Root must independently review actual hardware and explicit formal authorization,
# bind their exact receipt bytes, review/publish the final tools, then install.
HARDWARE_REVIEW_SHA256 = 'fea2f12a1bbc54008cdc4176d2156105e2988681c435738451dde44200b6d49f'
FORMAL_AUTHORIZATION_SHA256 = '7475bbfb06ec6d1d31589fc24e3ea725466b1a91320b1540692f6e72db83cc24'
FORMAL_FILES = {'formal.sbatch': '1ac3e2a317488593d18f0fdecb130c068a30e8987540cbd67f8739980c9f81cd', 'formal_worker.sh': 'f65e3d8cde4aa935d0fe941da11840c003dc6d381e81901f59a4c622cd7efcb7', 'formal_inner.sh': '671625f8b8671518144a09feba280a58dceb303a1eccc399a68810b00f263cbc', 'formal_driver.py': '6c32119ec6e0267c869f9fb035ce584f6cc867bab88583b4e20fd3e2e14c943a', 'ray_cli_compat.py': 'a6ea15b2ac65ff347ca574ed3af9688c22777723d6a2ddbd7b88f33455f6ddc3', 'formal_service_proof.py': '704909275113fe079c287e8652c07ef000e4cfa06f7cf904781b93dc9bed6a96', 'bind_retrieval_proof.py': '3ae4f0095b775716b0f2d3180b513906a7be51837d1f2786b3ec72447733704d'}  # root-reviewed immutable shell/driver/shim hashes; not inferred
COMMON = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data')
COMMON_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
DATA_SHA = 'e5e47bdfbfdffcc9a5cf0feccc39e52f32aa093fa039f7464e9baa5422d21d15'
ORDER_SHA = 'ff2d15592968ceaf7667489d61b9172c2c53f81d82cb7f4ff10fd7aeb5285bc8'
FORMAL_FILE_NAMES = {'formal.sbatch', 'formal_worker.sh', 'formal_inner.sh', 'formal_driver.py', 'ray_cli_compat.py', 'formal_service_proof.py', 'bind_retrieval_proof.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'canonical ordinary evidence required: ' + str(path))
    require(path.stat().st_size <= 64 * 1024 * 1024, 'unexpected oversized evidence')
    return path.read_bytes()


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'canonical evidence file required')
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def bound_json(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'unbound independent acceptance/authorization: submission forbidden')
    raw = raw_file(path)
    require(hashlib.sha256(raw).hexdigest() == expected, 'independent receipt bytes differ')
    return json.loads(raw)


def validate_hardware():
    review = bound_json(ROOT / 'logs/hardware-independent-acceptance.json', HARDWARE_REVIEW_SHA256)
    require(review.get('status') == 'PASS_INDEPENDENT_ALIGNED_HARDWARE_REVIEW' and review.get('scope') == 'ACTUAL_8RANK_SYNTHETIC_AND_LIVE_RETRIEVAL_NOT_TRAINING', 'hardware submission/CPU receipt is not acceptance')
    require(review.get('source') == SOURCE and review.get('root') == str(ROOT) and review.get('run') == str(HARDWARE_RUN) and review.get('job') == '3345', 'foreign hardware identity')
    names = {f'numeric-fixture/rank-{r}.json' for r in range(8)} | {'live-retrieval-acceptance.json', 'capture-completed-before-live.json', 'retrieval-golden.json', 'GPU-GOLDEN-CAPTURE-CLAIM.json', 'own-retriever-birth.json', 'own-process-cleanup.json', 'original-exit-code.txt', 'cleanup-exit-code.txt', 'numeric-fixture.log', 'retrieval-golden.log', 'retriever.log', 'hardware.log', 'resource-watch.jsonl', 'gpu-exit.csv', 'source-commit.txt'} | {str(ROOT / 'scheduler-head' / n) for n in ('hardware-submit-intent.json', 'hardware-submit-receipt.json', 'hardware-sbatch-response.json')}
    require(isinstance(review.get('files'), dict) and names <= set(review['files']), 'full actual hardware evidence required')
    for name, digest in review['files'].items():
        path = HARDWARE_RUN / name
        require(ROOT in path.parents and path.resolve() == path, 'foreign evidence path')
        require(sha(path) == digest, 'actual hardware evidence changed: ' + name)
    terminal = review.get('scheduler_terminal', {})
    require(terminal.get('authority') == 'scontrol' and terminal.get('job') == '3345' and terminal.get('state') == 'COMPLETED' and terminal.get('exit_code') == '0:0', 'actual scheduler terminal evidence absent')
    terminal_path = terminal.get('path')
    require(isinstance(terminal_path, str) and terminal_path in review['files'] and review['files'][terminal_path] == terminal.get('sha256'), 'terminal record is not hash-bound')
    require(raw_file(HARDWARE_RUN / 'source-commit.txt').strip() == SOURCE.encode(), 'hardware source record differs')
    submitted = json.loads(raw_file(ROOT / 'scheduler-head/hardware-submit-receipt.json'))
    require(submitted.get('source') == SOURCE and submitted.get('job') == '3345' and submitted.get('run') == str(HARDWARE_RUN) and submitted.get('token') == '13e07285c97044fd97974d94f53fea1b', 'actual submission receipt differs')
    require(len(review.get('allocated_gpu_uuids', [])) == 8 and len(set(review['allocated_gpu_uuids'])) == 8 and all(isinstance(u, str) and u.startswith('GPU-') for u in review['allocated_gpu_uuids']), 'actual eight UUID allocation proof absent')
    require(review.get('resource_watch_healthy') is True and review.get('leases_verified') is True, 'independent resource/lease acceptance absent')
    for rank in range(8):
        numeric = json.loads(raw_file(HARDWARE_RUN / f'numeric-fixture/rank-{rank}.json'))
        require(numeric.get('status') == 'PASS' and numeric.get('rank') == rank and numeric.get('world') == 8 and numeric.get('source_commit') == SOURCE and numeric.get('sys_prefix') == str(ROOT / 'env') and numeric.get('scope') == 'ACTUAL_8RANK_SYNTHETIC_GPU_ORACLES_NOT_TRAINING', 'numeric hardware record differs')
        require(any(r.get('case') == 'ordinary_PPO_GPU_value_gradient_FP64_oracle_no_dualclip_no_ratio20_clamp' and r.get('overflow_nonfinite_rejected') is True for r in numeric['records']), 'ordinary PPO hardware oracle absent')
    live = json.loads(raw_file(HARDWARE_RUN / 'live-retrieval-acceptance.json'))
    require(live.get('source') == SOURCE and live.get('run') == str(HARDWARE_RUN) and live.get('job') == '3345' and live.get('scope') == 'ACTUAL_LOCAL_CUDA_E5_GPU_FAISS_ARCHIVE_GOLDEN_PARITY_NOT_TRAINING_ACCEPTANCE', 'actual live hardware record differs')
    require({r['uuid'] for r in live['leases']} == set(review['allocated_gpu_uuids']), 'accepted live UUID leases differ')
    require(live['result'].get('verified_probes') == 12 and live['result'].get('gpu_identity_verified') is True, 'actual retrieval parity absent')
    cleanup = json.loads(raw_file(HARDWARE_RUN / 'own-process-cleanup.json'))
    require(cleanup.get('own_marker') == str(HARDWARE_RUN) and cleanup.get('remaining') == [], 'hardware cleanup not complete')
    for name in ('original-exit-code.txt', 'cleanup-exit-code.txt'):
        require(raw_file(HARDWARE_RUN / name).strip() == b'0', 'hardware phase/cleanup failed')
    return review


def validate_admission():
    # Deliberately first: candidate cannot contact Slurm or claim a RUN while unbound.
    hardware = validate_hardware()
    authority = bound_json(ROOT / 'logs/formal-launch-authorization.json', FORMAL_AUTHORIZATION_SHA256)
    require(authority.get('status') == 'AUTHORIZED_SINGLE_FRESH200_ALIGNED_LAUNCH' and authority.get('scope') == 'EXPLICIT_FORMAL_LAUNCH_AUTHORIZATION_NOT_COMPLETED_TRAINING', 'formal authorization absent')
    require(authority.get('source') == SOURCE and authority.get('root') == str(ROOT) and authority.get('hardware_review_sha256') == HARDWARE_REVIEW_SHA256 and authority.get('same_process_first5_gate') is True, 'formal authorization identity differs')
    require(sys.platform == 'linux' and os.uname().nodename == 'h100-5' and sys.prefix == str(ROOT / 'env'), 'actual new Linux interpreter required')
    require(ROOT.resolve() == ROOT, 'canonical new ROOT required')
    for name in ('runs', 'logs', 'tools', 'tools/deployment', 'repo', 'env', 'scheduler-head'):
        path = ROOT / name
        require(path.is_dir() and path.resolve() == path, 'canonical own directories required')
    require(subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'source080 required')
    require(not subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'status', '--porcelain'], text=True).strip(), 'dirty production source')
    require(set(FORMAL_FILES) == FORMAL_FILE_NAMES, 'unbound formal tools forbidden')
    for name, expected in FORMAL_FILES.items():
        require(sha(ROOT / 'tools/deployment' / name) == expected, 'formal tool bytes differ: ' + name)
    sys.path.insert(0, str(ROOT / 'tools'))
    import resource_watch
    require(Path(resource_watch.__file__).resolve() == ROOT / 'tools/resource_watch.py', 'foreign watcher')
    resource_watch.load_runtime_anchor(ROOT / 'runs/formal-submit-anchor', verify_current=True)
    require(sha(COMMON / 'manifest.json') == COMMON_SHA, 'common169028 manifest changed')
    require(sha(ROOT / 'inputs/baseline-aligned-data.json') == DATA_SHA and sha(ROOT / 'inputs/baseline-observed-order.json') == ORDER_SHA, 'actual aligned data/order changed')
    return hardware


def verify_submission(run, timeout=60):
    deadline = time.monotonic() + timeout
    path = ROOT / 'scheduler-head/formal-submit-receipt.json'
    while not path.exists() and not path.is_symlink():
        require(time.monotonic() < deadline, 'formal submission result remains unknown: no claim/start')
        time.sleep(.2)
    receipt = json.loads(raw_file(path))
    require(receipt.get('source') == SOURCE and receipt.get('root') == str(ROOT) and receipt.get('job') == os.environ['SLURM_JOB_ID'] and receipt.get('run') == str(run), 'formal job/token/RUN differs from actual single submission')
    require(receipt.get('hardware_review_sha256') == HARDWARE_REVIEW_SHA256 and receipt.get('formal_authorization_sha256') == FORMAL_AUTHORIZATION_SHA256 and receipt.get('files') == FORMAL_FILES, 'formal submission acceptance/tool bindings differ')
    require(receipt.get('scope') == 'ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION', 'wrong submission scope')
    require(run.name == 'formal-aligned-em200-' + str(receipt['token']) + '-' + str(receipt['job']), 'formal token differs')


def claim(run):
    validate_admission()
    run = Path(run)
    job = os.environ['SLURM_JOB_ID']
    require(job.isdigit() and run.parent == ROOT / 'runs' and run.resolve() == run and run.name.startswith('formal-aligned-em200-') and run.name.endswith('-' + job), 'exact independent same-job formal RUN required')
    verify_submission(run)
    sys.path.insert(0, str(ROOT / 'repo'))
    from search_r1.baselines import aligned_startup
    require(Path(aligned_startup.__file__).resolve() == ROOT / 'repo/search_r1/baselines/aligned_startup.py', 'foreign startup')
    aligned_startup.claim_new_run(run, SOURCE)  # exactly once, before any services/nonbootstrap RUN output
    with (run / 'source-commit.txt').open('x') as handle:
        handle.write(SOURCE + '\n'); handle.flush(); os.fsync(handle.fileno())


def environment(run):
    validate_admission()
    run = Path(run)
    require(run.parent == ROOT / 'runs' and run.resolve() == run and run.is_dir(), 'canonical claimed formal RUN required')
    verify_submission(run, timeout=0)
    claim_record = json.loads(raw_file(run / '.aligned-startup-claim.json'))
    require(claim_record['source_commit'] == SOURCE, 'fresh source080 claim required')
    return {'BAPO_FRESH200_MODEL': str(ROOT / 'assets/qwen2.5-3b-instruct-aa8e725'), 'BAPO_FRESH200_DATA': str(COMMON), 'BAPO_FRESH200_DATA_MANIFEST_SHA256': COMMON_SHA, 'BAPO_FRESH200_RUN': str(run), 'BAPO_ALIGNED_DATA_MANIFEST': str(ROOT / 'inputs/baseline-aligned-data.json'), 'BAPO_ALIGNED_DATA_SHA256': DATA_SHA, 'BAPO_ALIGNED_ORDER_MANIFEST': str(ROOT / 'inputs/baseline-observed-order.json'), 'BAPO_ALIGNED_ORDER_SHA256': ORDER_SHA, 'BAPO_ALIGNED_RETRIEVAL_MANIFEST': str(HARDWARE_RUN / 'retrieval-golden.json'), 'BAPO_ALIGNED_RETRIEVAL_SHA256': sha(HARDWARE_RUN / 'retrieval-golden.json'), 'BAPO_FRESH200_RETRIEVER_URL': 'http://127.0.0.1:18083/retrieve'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['verify', 'claim', 'environment']); parser.add_argument('--run'); args = parser.parse_args()
    if args.action == 'verify':
        validate_admission(); print('FORMAL_BOUND_ADMISSION_PASS_NOT_TRAINING_COMPLETION')
    else:
        require(args.run is not None, 'RUN required')
        if args.action == 'claim': claim(args.run)
        else:
            for key, value in environment(args.run).items(): print('export ' + key + '=' + shlex.quote(value))


if __name__ == '__main__': main()
