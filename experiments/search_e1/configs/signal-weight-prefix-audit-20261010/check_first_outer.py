"""Actual CPU integration of published final assembler on the first outer only.

No final/heldout claim, Torch/model, GPU/HTTP, signals or production writes.
The caller preserves compact stdout; original scoring and token math are reused.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types


def pinned(ref):
    p = Path(ref['path'])
    assert p.is_absolute() and p.is_file() and not p.is_symlink()
    first = p.stat(); raw = p.read_bytes(); last = p.stat()
    assert all(getattr(first, k) == getattr(last, k) for k in
               ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    assert hashlib.sha256(raw).hexdigest() == ref['sha256']
    assert 'bytes' not in ref or len(raw) == ref['bytes']
    return raw


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--request', required=True)
    args.add_argument('--request-sha256', required=True)
    a = args.parse_args()
    assert os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert 'torch' not in sys.modules
    request = json.loads(pinned(dict(path=a.request, sha256=a.request_sha256)))
    assert request['schema'] == 'opsd-signal-first-outer-assembler-smoke-v1'
    spec = request['completion_source']; root = Path(spec['helper_root'])
    assert str(root) == ('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
                         '/source/completion-7d9fea2/opsd_signal_weight_completion_v1')
    manifest = json.loads(pinned(spec['source_manifest']))
    assert spec['source_manifest']['sha256'] == 'b36aff538310bbd92c947e790e32254ce3267e9f1f62531aa4729a74bc719163'
    pub = json.loads(pinned(spec['publication']))
    assert pub['accepted'] and pub['source_commit'] == pub['remote_readback_sha'] == '7d9fea2333d85716cce1bd08c3d10ee0f3248e28'
    assert pub['source_manifest_sha256'] == spec['source_manifest']['sha256']
    raw = pinned(dict(path=str(root / 'common.py'), **manifest['files']['common.py']))
    boot = types.ModuleType('signal_prefix_bootstrap_common')
    boot.__file__ = str(root / 'common.py')
    exec(compile(raw, boot.__file__, 'exec'), boot.__dict__)
    assembler = boot.bind_helper(str(root), spec['source_manifest'], ('assemble',))['assemble']
    config, cfg = boot.read(request['config']); config['_config_sha256'] = cfg['sha256']
    assert config['variant'] == 'G_POS' and len(request['outers']) == 1
    assert request['outers'][0]['outer_index'] == 0
    native = boot.bind_helper(config['paths']['helper_root'], config['helper_source_manifest'],
                             ('checkpoint', 'sampling', 'training_metrics', 'scoring', 'evaluation'))
    gates = native['evaluation'].gates
    gates.publication_admission(config, gates.scoped(config, gates.bound(config['gates']['publication'])))
    native['scoring'].verify_scoring(config)
    core = importlib.import_module('search_e1.core')
    row, summary, cursor = assembler.outer(request['outers'][0], config, cfg, native, core,
                                           None, 0, 0, dict(cycle=0, offset=0))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(schema='opsd-signal-first-outer-assembler-smoke-result-v1', accepted=True,
        full_study_complete=False, formal_heldout_slots=0, config=cfg, checkpoint=row['checkpoint'],
        actual_optimizer_updates=row['optimizer_updates'], actual_pair_exposures=row['K'],
        valid_token_rows=row['token']['valid_token_rows'], native_train_correct=row['train_correct'],
        native_train_denominator=2560, source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        final_assembler_source=spec, torch_imported=False, no_new_training_generation=True,
        scope='One real original outer: source/core, full2560 strict scoring/mining, cursor, actualwait0, checkpoint and token interface only'),
        sort_keys=True))


if __name__ == '__main__':
    main()
