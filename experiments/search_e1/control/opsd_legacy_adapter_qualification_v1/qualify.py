"""Pinned optional historical L adapter-only qualification; stdout, CPU, no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
B = '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008'
N = '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008'
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
READER_SHA = '7b26007cc4b0a79930bea938cf8e902ba6773613afcb690b492c2fd47cf37e03'
PINS = {
    'config': (B + '/receipts/config.json', 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'),
    'target': (B + '/runs/checkpoints/update-00000099-outer-00009/manifest.json', '476c1b1003d8d8f4cc4711c462b013ed249bfbddc6d4a5235686f5e2ff946ad4'),
    'baseline': (B + '/runs/checkpoints/update-00000000-outer-00000/manifest.json', 'b8ed9fb8cf1112252cca6f66d28579d79980c84a22a47300492095828f38833a'),
    'common_initial': (N + '/receipts/common-initial-adapter.json', '641e98442cdd995fe6aa571a95956b4281de81ad4db510e7795b76087a21b654'),
    'common_cohort': (N + '/receipts/common-cohorts.json', '9791c62cee985d1d31633d323e63a7b36c265bcd08742d4705b6997fcbaa28a3'),
}
QUALIFICATION_SHA = '43692b7e67ec26e2645a5f1deda84c23e77ab32bd91ecd4e0f3ddee87871d425'
QUALIFICATION_SOURCE_SHA = '157b61361af152663c4e496cd1b86c3f55a054073310c044e7c7d3366dc999be'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned_inputs(value):
    require(value['schema'] == 'optional-L-adapter-only-inputs-v1', 'Explicit optional L inputs required')
    for name, (path, digest) in PINS.items():
        require(value[name]['path'] == path and value[name]['sha256'] == digest, 'Pinned historical identity differs: ' + name)
    require(value['reader']['sha256'] == READER_SHA
            and value['qualification']['sha256'] == QUALIFICATION_SHA
            and value['qualification_source']['sha256'] == QUALIFICATION_SOURCE_SHA,
            'Exact published reader and accepted cohort qualification required')


def raw_checked(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and '..' not in path.parts and all(not p.is_symlink() for p in (path, *path.parents)),
            'Canonical non-symlink input required')
    before = path.stat()
    keys = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with path.open('rb') as handle:
        require(keys(os.fstat(handle.fileno())) == keys(before), 'Opened input identity differs')
        raw = handle.read()
        require(keys(os.fstat(handle.fileno())) == keys(before), 'Input changed while reading')
    require(keys(path.stat()) == keys(before) and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Input content/identity differs')
    require('bytes' not in ref or len(raw) == ref['bytes'], 'Input bytes differ')
    return raw


def validate_tensors(tensors, torch):
    require(len(tensors) == 504, 'Expected exactly 504 adapter tensors')
    require(all(t.dtype == torch.float32 and t.device.type == 'cpu' and torch.isfinite(t).all().item()
                for t in tensors.values()), 'Actual adapter must be finite CPU FP32')


def qualify(inputs, native):
    pinned_inputs(inputs)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit empty CUDA visibility required')
    reader = native['Reader']()
    config, config_ref = reader.json(inputs['config']); config['_config_sha256'] = config_ref['sha256']
    require(config['run_id'] == RUN and config['paths']['output_root'] == B + '/runs', 'Historical run/root differs')
    modules, dependency_proof = native['bind_dependencies'](config, reader)
    qualification, qp = reader.json(inputs['qualification'])
    evidence, ep = reader.json(inputs['qualification_source'])
    require(qualification['schema'] == 'legacy-forward-L-fresh-metadata-qualification-v1'
            and qualification['actual_ssh_returncode'] == 0
            and qualification['all_ten_ordered_512_question_lists_match'] is True
            and qualification['all_ten_before_after_boundary_skips_match'] is True
            and qualification['raw_source_evidence']['sha256'] == ep['sha256'], 'Prior actual cohort qualification differs')
    require(evidence['checkpoint_manifest']['path'] == inputs['target']['path']
            and evidence['checkpoint_manifest']['sha256'] == inputs['target']['sha256'], 'Qualified historical endpoint differs')
    cohort, cohort_ref = reader.json(inputs['common_cohort'])
    require(qualification['common_cohort_sha256'] == cohort_ref['sha256']
            and cohort['schema'] == 'opsd-ablation-common-cohorts-v1' and cohort['outer_count'] == 10
            and cohort['questions_per_outer'] == 512
            and all(cohort[k]['sha256'] == config[k]['sha256'] for k in ('train_identity', 'validation_identity')),
            'Qualified common cohort/data identity differs')
    source, source_ref = reader.json(config['source_model_manifest'])
    source_files = source.get('files', source.get('source_files'))
    require(isinstance(source_files, list) and len(source_files) == 12, 'Expected twelve source100 identity refs')
    require(len({r['path'] for r in source_files}) == 12, 'Repeated source100 identity')
    source_metadata = []
    for ref in source_files:
        native['path_under'](ref['path'], config['paths']['model'])
        source_metadata.append(dict(expected=ref, current_metadata=reader.metadata(ref)))
    runtime, runtime_ref = reader.json(config['runtime_manifest'])
    target, target_ref = reader.json(inputs['target']); baseline, baseline_ref = reader.json(inputs['baseline'])
    native['checkpoint_identity'](target, inputs['target'], config, 99, 9)
    native['checkpoint_identity'](baseline, inputs['baseline'], config, 0, 0)
    require([Path(r['path']).name for r in target['adapter']['files']] == ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
            and {Path(r['path']).name for r in baseline['adapter']['files']} == {'README.md', 'adapter_config.json', 'adapter_model.safetensors'},
            'Exact three-file adapter inventory required')
    common, common_ref = reader.json(inputs['common_initial'])
    require(common['accepted'] is True and common['tensors'] == 504 and common['dtype'] == 'float32'
            and common['source_checkpoint_run'] == RUN
            and common['source_model_identity'] == source_ref['sha256'], 'Common initial identity differs')
    baseline_model = next(r for r in baseline['adapter']['files'] if Path(r['path']).name == 'adapter_model.safetensors')
    require(all(baseline_model[k] == common['adapter_model'][k] for k in ('path', 'sha256', 'bytes')), 'Common initial is not exact baseline CP0')
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA already initialized')
    current, current_file, current_digest = native['load_adapter'](target, reader, modules['trainer'], torch)
    prior, prior_file, prior_digest = native['load_adapter'](baseline, reader, modules['trainer'], torch)
    validate_tensors(current, torch); validate_tensors(prior, torch)
    require(prior_digest == common['adapter_tensor_sha256'], 'Actual common initial tensor digest differs')
    require(sum('lora_B' in k and torch.count_nonzero(t).item() == 0 for k, t in prior.items()) == common['zero_B_tensors'] == 252,
            'Common initial must have exactly 252 zero B tensors')
    delta = native['adapter_delta'](current, prior, torch)
    require(not torch.cuda.is_initialized(), 'Adapter-only read initialized CUDA')
    remembered = reader.finish()
    return dict(schema='optional-L-adapter-only-qualification-v1', accepted=True, checked_unix=time.time(),
        historical_run_id=RUN, label='L_historical_forward_outer10_Adam99', config=config_ref,
        target=target_ref, baseline=baseline_ref, common_initial=common_ref, common_cohort=cohort_ref,
        prior_qualification=qp, prior_qualification_source=ep, helper_source=dependency_proof,
        source_model_manifest=source_ref, source12_expected_and_current_metadata=source_metadata,
        runtime_manifest=runtime_ref, source100_weight_bytes_fresh_hashed=False,
        target_adapter_files=[reader.seen[r['path']] for r in target['adapter']['files']],
        baseline_adapter_files=[reader.seen[r['path']] for r in baseline['adapter']['files']],
        adapter_tensors=504, adapter_finite_fp32=True, actual_tensor_sha256=current_digest,
        common_actual_tensor_sha256=prior_digest, change_vs_common_initial=delta,
        remembered_files=remembered, cuda_initialized=False, full_Adam_or_RNG_read=False,
        scope='Adapter-only content/finite/change qualification; no optimizer, RNG restore, model, tokenizer, generation or native512 effect.',
        mandatory_seven_groups_or_5632_modified=False, optional_does_not_block_seven=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(); inputs = json.loads(raw_checked(dict(path=args.manifest, sha256=args.manifest_sha256)))
    pinned_inputs(inputs)
    raw = raw_checked(inputs['reader'])
    native = dict(__name__='_sha_bound_adapter_only_reader', __file__=inputs['reader']['path'])
    exec(compile(raw, inputs['reader']['path'], 'exec'), native)
    result = qualify(inputs, native)
    raw_checked(inputs['reader']); raw_checked(dict(path=args.manifest, sha256=args.manifest_sha256))
    result['reader'] = inputs['reader']; result['input_manifest'] = dict(path=args.manifest, sha256=args.manifest_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
