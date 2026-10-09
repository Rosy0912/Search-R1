"""Execute source080 production main once, retaining its same-process first5 gate."""
from pathlib import Path
import os, subprocess, sys
from formal_admission import ROOT, SOURCE, environment, require


def main():
    run = Path(os.environ['BAPO_RUN_ID'])
    expected = environment(run)
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'Ray/driver startup environment differs')
    require(raw_claim(run), 'source080 fresh claim required')
    sys.path.insert(0, str(ROOT / 'repo'))
    from search_r1.baselines import preflight
    require(Path(preflight.__file__).resolve() == ROOT / 'repo/search_r1/baselines/preflight.py', 'foreign production preflight')
    preflight.validate_inherited_gpu_leases(os.environ['CUDA_VISIBLE_DEVICES'].split(','), require_guard=True)
    from formal_service_proof import validate_services, verify_ray_connection
    validate_services()
    require(os.environ.get('RAY_ADDRESS') == '127.0.0.1:16383', 'own fixed Ray head required')
    executable = str(ROOT / 'env/bin/python')
    args = ['--config-name', 'ppo_bapo_baseline_aligned_production', 'baselines.algorithm_source_commit=' + SOURCE, 'hydra.run.dir=' + str(run / 'hydra'), 'trainer.default_hdfs_dir=null', 'trainer.val_before_train=false']
    # Configuration-only Hydra command does not run main/model/update.
    with (run / 'resolved-config.yaml').open('x') as output:
        subprocess.run([executable, '-B', '-m', 'search_r1.baselines.main'] + args + ['--cfg', 'job', '--resolve'], cwd=ROOT / 'repo', stdout=output, check=True)
    # Actual source080 retriever admission before final birth recheck.
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT/'repo/verl/trainer/config'),version_base=None):
        config=compose(config_name='ppo_bapo_baseline_aligned_production',overrides=args[2:])
    OmegaConf.resolve(config)
    preflight.validate_assets_and_retriever(config)
    validate_services()
    verify_ray_connection()
    validate_services()
    os.chdir(ROOT / 'repo')
    os.execv(executable, [executable, '-B', '-m', 'search_r1.baselines.preflight', '--', executable, '-B', '-m', 'search_r1.baselines.main'] + args)


def raw_claim(run):
    import json
    from formal_admission import raw_file
    record = json.loads(raw_file(run / '.aligned-startup-claim.json'))
    return record.get('source_commit') == SOURCE and record.get('scope') == 'same-experiment-pending20-stage-only-resume'


if __name__ == '__main__': main()
