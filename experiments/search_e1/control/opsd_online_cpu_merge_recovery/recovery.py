"""External CPU process boundary for the unchanged original BF16 PEFT merge."""
import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

WAIT_SHA = '55ee0b95e6330530ebd3da0c0f32b7699e33fae17e4aa9e049e387387176a214'
WAIT_MANIFEST_SHA = 'baa072189d0213309d72dd9d04c67bdaf03361e7e7b452b137040d4bfee9a67f'
WAIT_COMMIT = '2cbcc1b4754eff51a52b213240579f5db0e10f87'
PREFIX = 'experiments/search_e1/control/opsd_online_cpu_merge_recovery/'


def require(ok, message):
    if not ok: raise ValueError(message)


def load_wait(root, publication_ref):
    root = Path(root).resolve(); path = root / 'recovery.py'
    import hashlib
    require(hashlib.sha256(path.read_bytes()).hexdigest() == WAIT_SHA, 'Frozen wait runner SHA differs')
    spec = importlib.util.spec_from_file_location('opsd_cpu_merge_frozen_wait', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    manifest_ref = dict(path=str(root / 'source-manifest.json'), sha256=WAIT_MANIFEST_SHA)
    binding = module.verify_external_binding(manifest_ref, publication_ref, root)
    require(binding['source_commit'] == WAIT_COMMIT, 'Expected the published 2cb wait recovery')
    return module, binding


def verify_publication(wait, manifest_ref, publication_ref, root=None):
    root = Path(root or Path(__file__).parent).resolve()
    tree = wait.verify_tree(root, manifest_ref); value, proof = wait.checked_json(publication_ref)
    require(Path(tree['manifest']['path']).parent == root, 'CPU boundary manifest outside its own root')
    commit = value.get('source_commit', '')
    require(value.get('accepted') is True and value.get('branch') == 'search-e1'
            and re.fullmatch('[0-9a-f]{40}', commit) and value.get('remote_readback_sha') == commit,
            'CPU boundary publication/remote readback missing')
    require(value.get('source_manifest_sha256') == tree['manifest']['sha256']
            and value.get('config_sha256') == wait.CONFIG_SHA and value.get('run_id') == wait.RUN_ID,
            'CPU boundary publication binding differs')
    rows = value.get('files', [])
    require(len({v['path'] for v in rows}) == len(rows), 'Duplicate remote readback paths')
    indexed = {v['path']: v for v in rows}
    for item in tree['files'] + [dict(relative_path='source-manifest.json', **tree['manifest'])]:
        row = indexed.get(PREFIX + item['relative_path'])
        require(row is not None and row.get('equal') is True and row.get('sha256') == item['sha256']
                and row.get('bytes') == item['bytes'], 'Missing/different published CPU boundary file')
    return dict(source=tree, publication=proof, source_commit=commit)


def cuda_empty_and_uninitialized(torch):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU merge needs CUDA_VISIBLE_DEVICES empty before imports')
    require(not torch.cuda.is_initialized() and torch.cuda.device_count() == 0,
            'CPU merge unexpectedly has an initialized/visible CUDA device')
    return dict(cuda_initialized=False, visible_cuda_devices=0, cuda_visible_devices='')


def all_cpu(model):
    devices = [parameter.device.type for parameter in model.parameters()]
    require(devices and set(devices) == {'cpu'}, 'Actual PEFT/merged parameters are not entirely CPU')
    return dict(parameter_tensors=len(devices), device_types=['cpu'])


def original_merge_on_cpu(config, checkpoint, output, checkpoint_ref, sampling, torch, peft_type):
    """Call original function once, with identical args/methods; only inspect devices."""
    before = cuda_empty_and_uninitialized(torch); peft_proofs = []; merged_proofs = []
    descriptor = inspect.getattr_static(peft_type, 'from_pretrained')
    original = peft_type.from_pretrained
    def checked_from_pretrained(*args, **kwargs):
        model = original(*args, **kwargs)  # Same original PEFT call and same return object.
        peft_proofs.append(all_cpu(model)); cuda_empty_and_uninitialized(torch)
        merge = model.merge_and_unload
        def checked_merge(*margs, **mkwargs):
            result = merge(*margs, **mkwargs)
            merged_proofs.append(all_cpu(result)); cuda_empty_and_uninitialized(torch)
            return result
        model.merge_and_unload = checked_merge
        return model
    peft_type.from_pretrained = staticmethod(checked_from_pretrained)
    try:
        result = sampling.merge_latest(config, checkpoint, output, checkpoint_ref)
    finally:
        peft_type.from_pretrained = descriptor
    after = cuda_empty_and_uninitialized(torch)
    require(len(peft_proofs) == len(merged_proofs) == 1, 'Original PEFT construction/merge was not observed exactly once')
    return result, dict(before=before, after=after, peft_parameters=peft_proofs,
                        merged_parameters=merged_proofs, original_function_called_once=True,
                        original_bf16_method_and_arguments_unchanged=True)


def validate_result(wait, config, checkpoint, checkpoint_ref, output, value):
    require(value['kind'] == 'original-PEFT-CPU-BF16-merge-latest'
            and value['source_model_identity'] == config['source_model_manifest']['sha256']
            and value['checkpoint'] == checkpoint_ref
            and value['optimizer_updates'] == value['checkpoint_optimizer_updates'] == checkpoint['optimizer_updates']
            and value['adapter_tensor_sha256'] == checkpoint['all_rank_adapter_sha256'][0]
            and value['adapter'] == checkpoint['adapter'], 'CPU merge returned wrong latest checkpoint/source/adapter')
    require(Path(value['path']).resolve() == Path(output).resolve(), 'CPU merge output path differs')
    files = value['files']; root = Path(output).resolve()
    require(files and all(Path(f['path']).resolve().is_relative_to(root) for f in files), 'Merged files outside requested cache')
    names = {str(Path(f['path']).resolve().relative_to(root)) for f in files}
    require(names == {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}, 'Merged cache inventory differs')
    for f in files:
        actual = wait.fileproof(f['path'])
        require(actual['sha256'] == f['sha256'] and actual['bytes'] == f['bytes'], 'Merged cache bytes differ from child proof')
    # Original files include stat identity; preserve the original object hash exactly.
    require(value['sha256'] == original_object_sha(files), 'Original merged model identity hash differs')
    return True


def original_object_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def child(request_ref):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU child visibility must be empty before any model imports')
    require('torch' not in sys.modules, 'CPU child imported Torch before the boundary')
    request_data = json.loads(Path(request_ref['path']).read_text())
    wait, wait_binding = load_wait(request_data['wait_root'], request_data['wait_publication'])
    request, request_proof = wait.checked_json(request_ref)
    publication = verify_publication(wait, request['source_manifest'], request['publication'])
    require(request['config']['sha256'] == wait.CONFIG_SHA and request['run_id'] == wait.RUN_ID, 'CPU child must retain immutable config/run')
    config, config_proof = wait.checked_json(request['config'])
    frozen, original = wait.load_original(config)
    config['_config_sha256'] = wait.CONFIG_SHA; config['_run_id'] = wait.RUN_ID
    checkpoint, checkpoint_proof = frozen.check_checkpoint(request['checkpoint'], config, wait.RUN_ID)
    require(checkpoint == request['checkpoint_value'], 'CPU child checkpoint value differs from actual committed manifest')
    require(not Path(request['output']).exists(), 'CPU cache already exists; preserve rather than overwrite')
    import torch
    from peft import PeftModel
    import sampling
    result, device_proof = original_merge_on_cpu(config, checkpoint, request['output'], request['checkpoint'], sampling, torch, PeftModel)
    validate_result(wait, config, checkpoint, request['checkpoint'], request['output'], result)
    verify_publication(wait, request['source_manifest'], request['publication'])
    wait.verify_tree(config['paths']['helper_root'], config['helper_source_manifest'])
    proof = dict(schema='opsd-online-cpu-merge-child-v1', accepted=True, run_id=wait.RUN_ID,
        config_sha256=wait.CONFIG_SHA, owner=frozen.process_identity(os.getpid()), request=request_proof,
        config=config_proof, checkpoint_proof=checkpoint_proof, wait_binding=wait_binding,
        source_binding=publication, original_helper=original, device_proof=device_proof, result=result,
        original_function='sampling.merge_latest', original_source_model_and_lora_bytes_unchanged=True)
    wait.exclusive_json(request['child_receipt'], proof)


def parent_cpu_only():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Controller itself must remain CPU-only')
    require('torch' not in sys.modules, 'Controller must not import Torch/model libraries')
    return dict(cuda_visible_devices='', torch_imported=False)


def cpu_boundary(controller, frozen, wait, bindings, checkpoint, output, checkpoint_ref, sleeper=time.sleep, reverify=lambda: None):
    parent_proof = parent_cpu_only()
    require(not Path(output).exists(), 'Existing merge cache requires preservation before CPU recovery')
    controller.gate('cpu-merge', [])
    safe = lambda operation: wait.hold_evidence(operation, sleeper)
    directory = Path(controller.config['paths']['base']) / 'receipts' / 'cpu-merge-recovery' / uuid.uuid4().hex
    safe(lambda: directory.mkdir(parents=True, exist_ok=True))
    journal = safe(lambda: wait.Journal(directory))
    record = lambda event, **fields: safe(lambda: journal.record(event, **fields))
    record('CPU_MERGE_BEGIN', parent=controller.owner, parent_cpu=parent_proof, checkpoint=checkpoint_ref)
    request_value = dict(schema='opsd-online-cpu-merge-request-v1', config=controller.config_ref,
        run_id=controller.run_id, checkpoint=checkpoint_ref, checkpoint_value=checkpoint,
        output=str(Path(output).resolve()), child_receipt=str(directory / 'child-complete.json'), **bindings)
    request = safe(lambda: wait.durable_json(directory / 'request.json', request_value))
    command = [controller.config['paths']['python'], '-B', str(Path(__file__).resolve()), 'child',
               '--request', request['path'], '--request-sha256', request['sha256']]
    process, handle, identity = controller.spawn(command, directory / 'child.log', 'cpu-merge',
                                               dict(CUDA_VISIBLE_DEVICES=''))
    record('CPU_CHILD_SPAWNED', identity=identity)
    while True:
        try: rc = process.poll()
        except Exception as exc:
            record('CPU_CHILD_POLL_UNKNOWN_HOLD', error=str(exc)); sleeper(5); continue
        if rc is not None: break
        try:
            parent_cpu_only(); controller.gate('cpu-merge', [])
        except Exception as exc: record('CPU_RUNNING_ADMISSION_UNCERTAIN', error=str(exc))
        sleeper(5)
    while True:
        try:
            rc = process.wait(); require(type(rc) is int, 'Unknown CPU child wait returncode'); break
        except Exception as exc:
            record('CPU_CHILD_WAIT_UNKNOWN_HOLD', error=str(exc)); sleeper(5)
    record('ACTUAL_CPU_CHILD_RETURNED', identity=identity, actual_wait_returncode=rc)
    safe(lambda: None if handle.closed else handle.close())
    terminal = safe(lambda: wait.durable_json(directory / 'actual-terminal.json', dict(
        schema='opsd-online-cpu-merge-actual-terminal-v1', actual_wait_returncode=rc, identity=identity,
        run_id=controller.run_id, config_sha256=controller.config_ref['sha256'], signals_sent=False)))
    log = safe(lambda: wait.fileproof(handle.name))
    while True:
        try:
            parent_cpu_only()
            reverify()
            gpu = frozen.require_gpu_release(controller.config, [identity])
            controller.gate('cpu-merge', [])
            audit = getattr(controller.admit, 'require_owned_release', None)
            require(callable(audit), 'Original full-owned release callback missing')
            owned = audit(controller.config, controller.run_id, controller.owner, known_children=controller.known)
            require(owned.get('accepted') is True, 'CPU child/unknown descendant release rejected')
            controller.gate('cpu-merge', [])
            release = dict(gpu=gpu, all_owned_release=owned)
            record('CPU_RELEASE_ACCEPTED', proof=release, actual_terminal=terminal); break
        except Exception as exc:
            record('CPU_RELEASE_UNKNOWN_HOLD', error=str(exc), actual_terminal=terminal); sleeper(5)
    require(rc == 0, 'Actual CPU merge child nonzero; preserved terminal/release, no automatic restart')
    require((directory / 'child-complete.json').is_file(), 'CPU child rc0 without its completion receipt; reject after release')
    child_ref = safe(lambda: wait.fileproof(directory / 'child-complete.json'))
    value, child_proof = wait.checked_json(child_ref)
    require(value.get('accepted') is True and value.get('run_id') == controller.run_id
            and value.get('config_sha256') == controller.config_ref['sha256']
            and value.get('request', {}).get('sha256') == request['sha256'], 'Missing/wrong CPU child completion identity')
    require(all(value['owner'].get(k) == identity[k] for k in ('pid', 'uid', 'starttick')), 'CPU completion belongs to another process')
    require(value['device_proof']['before'] == value['device_proof']['after'] == dict(
        cuda_initialized=False, visible_cuda_devices=0, cuda_visible_devices=''), 'CPU CUDA-state acceptance absent')
    device = value['device_proof']
    require(device.get('original_function_called_once') is True
            and device.get('original_bf16_method_and_arguments_unchanged') is True,
            'Original CPU merge method/call proof missing')
    for name in ('peft_parameters', 'merged_parameters'):
        rows = device.get(name)
        require(isinstance(rows, list) and len(rows) == 1 and rows[0].get('device_types') == ['cpu']
                and type(rows[0].get('parameter_tensors')) is int and rows[0]['parameter_tensors'] > 0,
                'Actual PEFT/merged CPU parameter acceptance absent')
    validate_result(wait, controller.config, checkpoint, checkpoint_ref, output, value['result'])
    record('CPU_MERGE_ACCEPTED', child_completion=child_proof)
    safe(lambda: wait.durable_json(directory / 'parent-complete.json', dict(accepted=True, run_id=controller.run_id,
        config_sha256=controller.config_ref['sha256'], actual_terminal=terminal, child=child_proof,
        child_log=log, release=release, parent_cpu=parent_cpu_only(), events=wait.fileproof(journal.path))))
    return value['result']  # Exactly the original merge identity; no added scientific fields.


def validate_previous(wait, ref, proc_root='/proc'):
    value, proof = wait.checked_json(ref)
    expected = dict(pid=660173, uid=1005, starttick=924349236)
    require(value.get('schema') == 'opsd-online-cpu-merge-parent-terminal-v1' and value.get('accepted') is True
            and value.get('run_id') == wait.RUN_ID and value.get('config_sha256') == wait.CONFIG_SHA
            and value.get('old_owner') == expected and value.get('old_source_commit') == WAIT_COMMIT,
            'Previous CUDA-owning controller terminal identity differs')
    rows = value['old_owned_processes']
    require(rows and len({r['pid'] for r in rows}) == len(rows) and any(all(r.get(k) == v for k, v in expected.items()) for r in rows),
            'Previous owner absent from complete old process inventory')
    for row in rows:
        require(row['uid'] == 1005 and row['starttick'] > 0 and not (Path(proc_root) / str(row['pid'])).exists(),
                'Previous owned PID present; historical terminal proof is insufficient')
    preservation, preserved_proof = wait.checked_json(value['merge_preservation'])
    require(preservation.get('accepted') is True and preservation.get('run_id') == wait.RUN_ID
            and preservation.get('config_sha256') == wait.CONFIG_SHA, 'Old CPU merge artifacts not preserved')
    return value, dict(receipt=proof, preservation=preserved_proof, fresh_all_old_pids_absent=True)


def run_parent(args):
    require(args.resume, 'Same-run --resume required; do not reset Adam')
    # The new process's only default CUDA change is physical visibility. Original
    # sampling/trainer spawn still supplies its explicit rank/eight-GPU overrides.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parent_cpu_only()
    wait_pub = dict(path=args.wait_publication, sha256=args.wait_publication_sha256)
    wait, wait_binding = load_wait(args.wait_root, wait_pub)
    require(args.config_sha256 == wait.CONFIG_SHA and args.run_id == wait.RUN_ID, 'Immutable original config/run required')
    config_ref = dict(path=args.config, sha256=args.config_sha256); config, _ = wait.checked_json(config_ref)
    source_ref = dict(path=args.source_manifest, sha256=args.source_manifest_sha256)
    pub_ref = dict(path=args.publication, sha256=args.publication_sha256)
    publication = verify_publication(wait, source_ref, pub_ref)
    old_ref = dict(path=args.original_terminal, sha256=args.original_terminal_sha256)
    wait.validate_old_terminal(old_ref)
    previous_ref = dict(path=args.previous_terminal, sha256=args.previous_terminal_sha256)
    previous, previous_proof = validate_previous(wait, previous_ref)
    frozen, original = wait.load_original(config)
    config['_config_sha256'] = wait.CONFIG_SHA; config['_run_id'] = wait.RUN_ID
    cp, _ = frozen.check_checkpoint(previous['resume_checkpoint'], config, wait.RUN_ID, expected_updates=11, expected_outer=0)
    require(cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'], 'CP11 outer completion cursor missing')
    outer = Path(config['paths']['output_root']) / 'outers' / 'outer-00001'
    require(not (outer / 'merged-model.json').exists() and not (outer / 'merged-model-cache').exists(),
            'Old unaccepted CPU merge paths must be preserved outside original locations')
    bindings = dict(wait_root=str(Path(args.wait_root).resolve()), wait_publication=wait_pub,
                    source_manifest=source_ref, publication=pub_ref)
    def reverify():
        verify_publication(wait, source_ref, pub_ref)
        load_wait(args.wait_root, wait_pub)
        wait.checked_json(config_ref); wait.verify_tree(config['paths']['helper_root'], config['helper_source_manifest'])
    cls = wait.recovery_controller(frozen, dict(cpu_merge=publication, wait=wait_binding,
        original_helper=original, previous_terminal=previous_proof), reverify=reverify)
    controller = None; original_boundary = frozen.merge_latest
    try:
        controller = cls(config_ref, args.run_id, resume=True)
        validate_previous(wait, previous_ref); wait.validate_old_terminal(old_ref)
        controller.gate('cpu-merge', [])
        # Patch only this freshly imported module's merge call boundary; no disk
        # edits, no changes inside sampling.merge_latest or any scientific method.
        frozen.merge_latest = lambda cfg, ckpt, output, ref: cpu_boundary(controller, frozen, wait, bindings, ckpt, output, ref, reverify=reverify)
        controller.run()
    except Exception as exc:
        if controller is not None and controller.owns_output: controller.contain_failure(exc)
        raise
    finally:
        frozen.merge_latest = original_boundary


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='mode', required=True)
    child_parser = sub.add_parser('child')
    child_parser.add_argument('--request', required=True); child_parser.add_argument('--request-sha256', required=True)
    parent_parser = sub.add_parser('parent')
    for name in ('config', 'config-sha256', 'run-id', 'wait-root', 'wait-publication', 'wait-publication-sha256',
                 'source-manifest', 'source-manifest-sha256', 'publication', 'publication-sha256',
                 'original-terminal', 'original-terminal-sha256', 'previous-terminal', 'previous-terminal-sha256'):
        parent_parser.add_argument('--' + name, required=True)
    parent_parser.add_argument('--resume', action='store_true'); args = p.parse_args()
    if args.mode == 'child': child(dict(path=args.request, sha256=args.request_sha256))
    else: run_parent(args)


if __name__ == '__main__': main()
