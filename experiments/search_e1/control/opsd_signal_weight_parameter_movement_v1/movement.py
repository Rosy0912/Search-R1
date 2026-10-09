"""Measure saved LoRA parameters on CPU; never model weights, gradients or RNG."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
try:
    from .common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, read,
                         checked_file, same, bind_helper, execution_packet)
except ImportError:
    from common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, read,
                        checked_file, same, bind_helper, execution_packet)

SUFFIX = re.compile(r'^(.*)\.lora_([AB])\.weight$')


def publication(spec, actual_entry=None):
    manifest, mp = read(spec['source_manifest']); published, pp = read(spec['publication'])
    require(published.get('accepted') is True and published.get('branch') == 'search-e1'
            and published.get('source_commit') == published.get('remote_readback_sha')
            and isinstance(published.get('source_commit'), str) and len(published['source_commit']) == 40
            and published.get('source_manifest_sha256') == mp['sha256'], 'Missing exact source publication/readback')
    root = Path(spec['helper_root']).resolve()
    require({str(p.relative_to(root)) for p in root.rglob('*.py')} == {n for n in manifest['files'] if n.endswith('.py')},
            'Published Python inventory differs')
    for name, entry in manifest['files'].items():
        rel = Path(name); require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source path')
        checked_file(dict(path=str(root / rel), **entry))
    if actual_entry is not None:
        require(root / 'movement.py' == Path(actual_entry).resolve(), 'Actual movement entry outside published source')
    return dict(helper_root=str(root), manifest=mp, publication=pp)


def asset_files(config):
    source, sp = read(config['source_model_manifest'])
    require(source['accepted'] is True and source['source_grpo_step'] == 100 and len(source['files']) == 12,
            'Original source100 twelve-file manifest missing')
    files = [checked_file(f) for f in source['files']]
    root = Path(config['paths']['model']).resolve()
    require(len({f['path'] for f in files}) == 12 and all(Path(f['path']).parent == root for f in files)
            and {str(p.resolve()) for p in root.iterdir() if p.is_file()} == {f['path'] for f in files},
            'Source100 actual twelve-file directory differs')
    runtime, rp = read(config['runtime_manifest']); entries = runtime.get('files', runtime)
    require(rp['sha256'] == '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
            and len(entries) == 305, 'Frozen305 runtime differs')
    runtime_root = Path(config['paths']['runtime']).resolve()
    for name, entry in entries.items():
        rel = Path(name); require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe runtime path')
        checked_file(dict(path=str(runtime_root / rel), **(entry if isinstance(entry, dict) else dict(sha256=entry))))
    return dict(source_manifest=sp, source_files=files, runtime_manifest=rp, runtime_fresh_files=305)


def initial_metadata(config, ref):
    require(same(ref, config['common_initial_adapter_manifest']), 'Common initial reference differs from config')
    value, proof = read(ref)
    require(value['schema'] == 'opsd-signal-common-initial-v1' and value['accepted'] is True
            and value['tensors'] == 504 and value['zero_B_tensors'] == 252 and value['dtype'] == 'float32'
            and value['fresh_adam_required'] is True
            and value['source_model_identity'] == config['source_model_manifest']['sha256']
            and value['adapter_tensor_sha256'] == config['common_initial_adapter_tensor_sha256'],
            'Original zero-effect common initial identity differs')
    payload = [checked_file(f) for f in value['payload_files']]
    adapter = checked_file(value['adapter_model']); root = Path(adapter['path']).parent
    require(len(payload) == 2 and {Path(f['path']).name for f in payload} == {'adapter_model.safetensors', 'adapter_config.json'}
            and all(Path(f['path']).parent == root for f in payload)
            and any(same(f, adapter) for f in payload), 'Common initial payload inventory differs')
    return value, dict(manifest=proof, adapter_model=adapter, payload=payload)


def checkpoint_metadata(endpoint, config, config_ref, native):
    require(type(endpoint['optimizer_updates']) is int and endpoint['optimizer_updates'] >= 0
            and type(endpoint['outer_index']) is int and 0 <= endpoint['outer_index'] < 25,
            'Explicit true Adam/outer counters required')
    config = dict(config, _config_sha256=config_ref['sha256'])
    cp, proof = native['checkpoint'].check_checkpoint(endpoint['checkpoint'], config, config['run_id'],
                                                      endpoint['optimizer_updates'], endpoint['outer_index'])
    require(cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'], 'Raw endpoint K not fully committed')
    require(type(endpoint['cumulative_pair_exposure']) is int
            and cp['cumulative_pair_exposure'] == endpoint['cumulative_pair_exposure'], 'Pair exposure counter differs')
    model = next((f for f in proof['adapter_files'] if Path(f['path']).name == 'adapter_model.safetensors'), None)
    require(model is not None and Path(model['path']).parent == Path(cp['adapter']['path']).resolve(), 'Raw adapter payload absent')
    require(Path(cp['adapter']['path']).resolve().parent == Path(proof['manifest']['path']).parent,
            'Endpoint adapter is not physically inside its raw checkpoint; cache/best refused')
    model = checked_file(model)  # Normalize native stat-rich proof to this reader's immutable path/SHA/bytes form.
    return cp, dict(manifest=proof['manifest'], adapter_model=model,
                    rank_payloads=proof['ranks'], shared_optimizer_file=proof['shared_optimizer'],
                    optimizer_payloads_hashed_not_deserialized=True)


def validate_coverage(packet, study_ref, configs, endpoints):
    cli, execution = execution_packet(dict(packet, output=packet['actual_stdout']))
    require(cli['accepted'] is True and same(cli['output'], packet['output']), 'Coverage actual CLI output differs')
    coverage, proof = read(cli['output'])
    require(coverage['schema'] == 'opsd-signal-final-coverage-v1' and coverage['accepted'] is True
            and coverage['coverage_accepted'] is True and type(coverage['final_accepted']) is bool
            and cli['final_accepted'] == coverage['final_accepted'] and coverage['complete_groups'] == 4
            and coverage['complete_outers'] == 100 and coverage['train_trajectory_slots'] == 256000
            and coverage['formal_native512_slots'] == 4608 and coverage['torch_imported'] is False
            and same(coverage['study_manifest'], study_ref), 'Complete100/4608 coverage missing or false')
    request, rp = read(coverage['request'])
    require(same(request['study_manifest'], study_ref) and [g['variant'] for g in request['groups']] == list(VARIANTS)
            and all(same(g['config'], configs[g['variant']]) for g in request['groups']), 'Coverage config quartet differs')
    rows = coverage['outer_rows']
    require(len(rows) == 100 and {(r['variant'], r['outer_index']) for r in rows} == {(v, i) for v in VARIANTS for i in range(25)},
            'Coverage does not contain complete unique25-outer chains')
    require(set(coverage['final_checkpoints']) == set(VARIANTS), 'Coverage final checkpoint quartet absent')
    for variant in VARIANTS:
        for endpoint in endpoints[variant]:
            row = next(r for r in rows if r['variant'] == variant and r['outer_index'] == endpoint['outer_index'])
            require(same(row['checkpoint'], endpoint['checkpoint']) and row['optimizer_updates'] == endpoint['optimizer_updates']
                    and row['cumulative_pair_exposure'] == endpoint['cumulative_pair_exposure'],
                    'Movement raw endpoint counters/checkpoint differ from actual complete coverage')
        require(same(coverage['final_checkpoints'][variant], endpoints[variant][1]['checkpoint']), 'Raw final checkpoint differs')
    require(len(coverage['evaluations']) == 9, 'Coverage lacks nine actual native endpoints')
    expected = {('source100', 0)} | {(v, t) for v in VARIANTS for t in (15, 25)}
    seen = set()
    for entry in coverage['evaluations']:
        value, _ = read(entry['child']); key = (value['version'], value['completed_outer'])
        require(key in expected and key not in seen and value['accepted'] is True and value['primary_slots'] == 512
                and value['actual_child_exit_and_release'] is True, 'Raw formal evaluation endpoint missing/duplicated')
        variant = 'G_POS' if key[1] == 0 else key[0]
        require(same(value['config'], configs[variant]), 'Native endpoint config identity differs')
        if key[1]:
            endpoint = endpoints[variant][0 if key[1] == 15 else 1]
            cp, _ = read(endpoint['checkpoint'])
            require(value['raw_checkpoint']['kind'] == 'raw_end_of_outer'
                    and same(value['raw_checkpoint']['manifest'], endpoint['checkpoint'])
                    and value['input_identity']['expected_adapter_tensor_sha256'] == cp['all_rank_adapter_sha256'][0],
                    'Formal evaluation did not use this same raw adapter')
        else:
            config, _ = read(configs['G_POS'])
            require(value['raw_checkpoint']['kind'] == 'common_zero_adapter'
                    and same(value['raw_checkpoint']['manifest'], config['common_initial_adapter_manifest'])
                    and value['input_identity']['expected_adapter_tensor_sha256'] == config['common_initial_adapter_tensor_sha256'],
                    'Formal common-zero evaluation used another initial adapter')
        seen.add(key)
    require(seen == expected, 'Formal nine endpoints differ')
    return dict(report=proof, actual_execution=execution, request=rp,
                coverage_accepted=True, final_accepted=coverage['final_accepted'],
                final_accepted_not_required_for_parameter_analysis=True)


def preflight(request, actual_entry):
    require(request['schema'] == 'opsd-signal-parameter-movement-request-v1' and request['mode'] in ('final', 'prefix'),
            'Wrong parameter movement mode/schema')
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Actual production audit requires HKUST4 UID1001, empty CUDA, fresh CPU process')
    published = publication(request['audit_source'], actual_entry)
    study, sr = read(request['study_manifest'])
    require(str(Path(study['study_base']).resolve()) == STUDY_ROOT and [g['variant'] for g in study['groups']] == list(VARIANTS),
            'Wrong source study quartet')
    final = request['mode'] == 'final'; keys = set(request['configs'])
    require(keys == set(VARIANTS) if final else len(keys) == 1 and keys <= set(VARIANTS), 'Wrong config arm scope')
    require(set(request['endpoints']) == keys, 'Explicit endpoint/config keys differ')
    configs = {}; modules = {}; checked = {}; identities = None; common = None
    common_keys = ('source_model_manifest', 'runtime_manifest', 'common_initial_adapter_manifest', 'common_initial_adapter_tensor_sha256',
                   'train_identity', 'validation_identity', 'common_cohort_manifest')
    for variant in VARIANTS:
        if variant not in keys: continue
        cfg, cf = read(request['configs'][variant]); configs[variant] = cfg
        group = next(g for g in study['groups'] if g['variant'] == variant)
        require(same(group['config'], cf) and cfg['variant'] == variant and cfg['study_id'] == study['study_id']
                and cfg['helper_source_manifest']['sha256'] == ORIGINAL_SHA, 'Original config/source/study identity differs')
        if common is not None: require(all(cfg[k] == common[k] for k in common_keys), 'Common input identity differs between arms')
        else: common = cfg
        native = bind_helper(cfg['paths']['helper_root'], cfg['helper_source_manifest'], ('checkpoint', 'prepare', 'targets'))
        require(native['prepare'].validate_config(cfg, launch=False) == [], 'Original25-outer four-rank scientific config is incomplete')
        modules[variant] = native
        if identities is None: identities = asset_files(cfg)
        initial, init_proof = initial_metadata(cfg, request['common_initial'])
        endpoints = request['endpoints'][variant]
        require([e['outer_index'] for e in endpoints] == [14, 24] if final else len(endpoints) == 1,
                'Final requires exact rawouter14/24; prefix allows one protected completed endpoint')
        checked[variant] = [checkpoint_metadata(e, cfg, cf, native) for e in endpoints]
        if final:
            require(endpoints[1]['optimizer_updates'] >= endpoints[0]['optimizer_updates']
                    and endpoints[1]['cumulative_pair_exposure'] >= endpoints[0]['cumulative_pair_exposure'], 'Counters decrease between endpoints')
    coverage = validate_coverage(request['initial_coverage'], sr, request['configs'], request['endpoints']) if final else None
    require(final or 'initial_coverage' not in request, 'Prefix cannot claim full coverage evidence')
    require('torch' not in sys.modules, 'Preflight unexpectedly imported Torch')
    return dict(study=sr, source=published, identities=identities, common_initial=init_proof,
                initial=initial, configs=configs, modules=modules, checked=checked, coverage=coverage)


def validate_tensors(values, expected_digest, digest, torch, zero=False):
    require(len(values) == 504, 'Exactly504 native LoRA tensors required')
    paired = {}; counts = {'A': 0, 'B': 0}
    for name, t in values.items():
        match = SUFFIX.fullmatch(name)
        require(match is not None and t.device.type == 'cpu' and t.dtype == torch.float32 and t.ndim == 2
                and torch.isfinite(t).all().item(), 'LoRA tensor must be named finite FP32 CPU matrix')
        module, kind = match.groups(); paired.setdefault(module, {})[kind] = t; counts[kind] += 1
        require(t.shape[0 if kind == 'A' else 1] == 16, 'Original LoRA rank16 dimension differs')
        if zero and kind == 'B': require(torch.count_nonzero(t).item() == 0, 'Common initial B tensor is nonzero')
    require(counts == {'A': 252, 'B': 252} and len(paired) == 252 and all(set(p) == {'A', 'B'} for p in paired.values()),
            'Missing/duplicated A/B module inventory')
    require(digest(values) == expected_digest, 'Actual504 native tensor digest differs')
    return counts


def _metrics(rows):
    source_sq = math.fsum(x['source_squared_l2'] for x in rows); dest_sq = math.fsum(x['destination_squared_l2'] for x in rows)
    delta_sq = math.fsum(x['delta_squared_l2'] for x in rows)
    source = math.sqrt(source_sq); dest = math.sqrt(dest_sq); delta = math.sqrt(delta_sq)
    ratio = delta / source if source > 0 else None
    require(all(math.isfinite(x) for x in (source_sq, dest_sq, delta_sq, source, dest, delta))
            and (ratio is None or math.isfinite(ratio)), 'Nonfinite FP64 norm/reduction')
    return dict(tensors=len(rows), parameters=sum(x['parameters'] for x in rows),
                changed_tensors=sum(x['changed'] for x in rows), source_l2=source, destination_l2=dest,
                delta_l2=delta, relative_l2=ratio, relative_denominator='source_parameter_l2',
                relative_l2_na_reason=None if ratio is not None else 'ZERO_SOURCE_PARAMETER_NORM')


def compare(left, right, torch):
    require(set(left) == set(right) and len(left) == 504, 'Endpoint504 name inventory differs')
    rows = []; layers = {}
    for name in sorted(left):
        a, b = left[name], right[name]; match = SUFFIX.fullmatch(name)
        require(match is not None and a.shape == b.shape and a.dtype == b.dtype == torch.float32
                and a.device.type == b.device.type == 'cpu' and torch.isfinite(a).all().item()
                and torch.isfinite(b).all().item(), 'Endpoint shape/dtype/device/finite mismatch')
        # FP64 subtraction before squaring prevents FP32 cancellation/overflow in the reduction.
        aa = a.to(dtype=torch.float64); bb = b.to(dtype=torch.float64); difference = bb - aa
        row = dict(name=name, kind=match.group(2), parameters=a.numel(), changed=not torch.equal(a, b),
                   source_squared_l2=torch.sum(aa * aa, dtype=torch.float64).item(),
                   destination_squared_l2=torch.sum(bb * bb, dtype=torch.float64).item(),
                   delta_squared_l2=torch.sum(difference * difference, dtype=torch.float64).item())
        require(all(math.isfinite(row[k]) for k in ('source_squared_l2', 'destination_squared_l2', 'delta_squared_l2')),
                'Nonfinite tensor norm')
        layer = re.search(r'(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)', match.group(1))
        require(layer is not None, 'Native named LoRA module lacks explicit layer index')
        rows.append(row); layers.setdefault(int(layer.group(1)), []).append(row)
    require(set(layers) == set(range(36)), 'Native504 LoRA adapter must cover exactly36 layers')
    return dict(parameter_space='saved_named_FP32_LoRA_A_B', aggregate=_metrics(rows),
                by_factor={kind: _metrics([r for r in rows if r['kind'] == kind]) for kind in ('A', 'B')},
                by_layer=[dict(layer=index, aggregate=_metrics(items),
                                by_factor={kind: _metrics([r for r in items if r['kind'] == kind]) for kind in ('A', 'B')})
                          for index, items in sorted(layers.items())],
                reduction_dtype='float64', effective_model_weight_delta_computed=False, gradient_computed=False)


def run(request_ref):
    request, rp = read(request_ref); verified = preflight(request, __file__)
    import torch
    from safetensors.torch import load_file
    require(not torch.cuda.is_initialized(), 'CPU audit found CUDA initialized')
    digest = verified['modules'][next(iter(verified['modules']))]['targets'].tensor_digest
    initial_ref = verified['common_initial']['adapter_model']; initial = load_file(initial_ref['path'], device='cpu')
    validate_tensors(initial, verified['initial']['adapter_tensor_sha256'], digest, torch, zero=True)
    require(checked_file(initial_ref) == initial_ref, 'Initial payload changed while loading')
    comparisons = []; endpoint_identity = {}
    for variant in VARIANTS:
        if variant not in verified['configs']: continue
        endpoints = request['endpoints'][variant]; previous = initial; previous_label = 'common_zero'
        loaded = []
        for endpoint, (cp, evidence) in zip(endpoints, verified['checked'][variant]):
            values = load_file(evidence['adapter_model']['path'], device='cpu')
            validate_tensors(values, cp['all_rank_adapter_sha256'][0], digest, torch)
            require(checked_file(evidence['adapter_model']) == evidence['adapter_model'], 'Raw adapter changed while loading')
            loaded.append(values)
            label = 'outer%d' % (endpoint['outer_index'] + 1)
            comparisons.append(dict(variant=variant, source=previous_label, destination=label,
                                    source_updates=0 if previous_label == 'common_zero' else endpoints[0]['optimizer_updates'],
                                    source_pair_exposure=0 if previous_label == 'common_zero' else endpoints[0]['cumulative_pair_exposure'],
                                    destination_updates=cp['optimizer_updates'], destination_pair_exposure=cp['cumulative_pair_exposure'],
                                    metrics=compare(previous, values, torch)))
            previous, previous_label = values, label
        if request['mode'] == 'final':
            comparisons.append(dict(variant=variant, source='common_zero', destination='outer25', source_updates=0,
                                    source_pair_exposure=0,
                                    destination_updates=endpoints[1]['optimizer_updates'],
                                    destination_pair_exposure=endpoints[1]['cumulative_pair_exposure'], metrics=compare(initial, loaded[1], torch)))
        endpoint_identity[variant] = [dict(outer_index=e['outer_index'], completed_outer=e['outer_index'] + 1,
            optimizer_updates=cp['optimizer_updates'], cumulative_pair_exposure=cp['cumulative_pair_exposure'],
            usable_pair_count=cp['cursor']['usable_pair_count'], next_pair_cursor=cp['cursor']['next_pair_cursor'],
            actual_native_adapter_tensor_sha256=cp['all_rank_adapter_sha256'][0], evidence=proof)
            for e, (cp, proof) in zip(endpoints, verified['checked'][variant])]
        del loaded, previous
    # Detect every bound source/config/checkpoint and tensor payload mutation across the audit.
    for variant, checkpoints in verified['checked'].items():
        checked_file(request['configs'][variant])
        for cp, evidence in checkpoints:
            checked_file(evidence['manifest']); checked_file(evidence['adapter_model'])
            for rank in evidence['rank_payloads']: checked_file(rank['file'])
            checked_file(evidence['shared_optimizer_file'])
            for ref in cp['adapter']['files']: checked_file(ref)
    checked_file(verified['common_initial']['manifest']); checked_file(initial_ref)
    for ref in verified['common_initial']['payload']: checked_file(ref)
    require(asset_files(verified['configs'][next(iter(verified['configs']))]) == verified['identities'],
            'Source100/runtime changed across audit')
    require(not torch.cuda.is_initialized() and not any(n == 'transformers' or n.startswith('transformers.')
            or n == 'peft' or n.startswith('peft.') for n in sys.modules), 'CPU audit initialized CUDA or imported a model package')
    return dict(schema='opsd-signal-parameter-movement-v1', accepted=True,
                full_study_complete=request['mode'] == 'final', mode=request['mode'], request=rp,
                study=verified['study'], audit_source=verified['source'], common_initial=verified['common_initial'],
                common_initial_actual_native_tensor_sha256=verified['initial']['adapter_tensor_sha256'],
                source_and_runtime=verified['identities'], configs=request['configs'], endpoints=endpoint_identity,
                initial_coverage=verified['coverage'], comparisons=comparisons, cuda_initialized=False,
                torch_version=torch.__version__, tensor_read_device='cpu', stdout_only=True,
                optimizer_or_rng_restored=False, optimizer_numeric_audit=False, new_training_generation_or_signals=0,
                limitations=['LoRA A/B coordinates have factorization-dependent scale; these are parameter-space displacements.',
                            'No effective model-weight BA delta, gradient, optimizer update-vector or causal performance effect is measured.',
                            'Saved zero/15/25 endpoints cannot locate exact onset; prefix is only a protected-CP interface audit.',
                            'full_study_complete describes this parameter endpoint scope, not overall goal completion.'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    try:
        result = run(dict(path=args.request, sha256=args.request_sha256)); code = 0
    except Exception as error:
        result = dict(schema='opsd-signal-parameter-movement-v1', accepted=False, full_study_complete=False,
                      error_type=type(error).__name__, error=str(error), cuda_initialized=None,
                      torch_imported='torch' in sys.modules); code = 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)); return code


if __name__ == '__main__':
    raise SystemExit(main())
