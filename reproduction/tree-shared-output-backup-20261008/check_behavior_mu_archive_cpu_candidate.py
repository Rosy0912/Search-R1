#!/usr/bin/env python3
"""UNAPPROVED LOCAL PREPARATION: future own3341 archive-aware CPU checker.

Default prints a plan, without opening tensor assets or creating runtime files.
Never deployed or executed. Frozen training/checker files remain unchanged.
"""
import argparse
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

REPLACEMENT_ENTRY_SHA = 'f641d463d65f2f6a7b1e9a8334cb89b9a6a175342c88616467e0e5669ec239eb'
PACKAGE_MANIFEST_SHA = '9774e8854889ace8e618335d5f09c8cfbe229bd265c9b6f454d46ebc8cdfbc88'
PINS = {
    'production_contract.py':'c57001c3c790d7218973587e534e19f657cdee5b7d9d97f82298842e7886c476',
    'calibration_contract.py':'c8aef5445f1283f37dfbf23993b43ea7c2ce8cb7ca9d896406324241da1965f0',
    'attempt_accounting.py':'8426d39486e92773022dd0d40ef5e4cb0c742ee6317550d20244ed4696913c29',
    'build_package_draft.py':'4c8d26a20cb0099845f7a7cdf73e7845bfd2c975ce8b50186172eb6e5f54fb92',
    'check_behavior_mu_cpu.py':'c56e3c79959c501d6256ba51f0cd455b2de0a7540ea0c1ff9231fae066afe5bc',
}


def replacement_module():
    path = Path(__file__).parent / 'trajectory_actorinput_replacement_candidate.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPLACEMENT_ENTRY_SHA:
        raise RuntimeError('Frozen replacement helper changed')
    module = types.ModuleType('own3341_replacement'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def read_package_raw(a, fd, name, expected):
    leaf = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(leaf, 'rb') as stream:
        before = a.signature_fd(stream.fileno())
        a.require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode) and before['bytes'] < 1024*1024, 'Bounded fixed package code/JSON required')
        raw = stream.read(before['bytes']+1)
        a.require(len(raw) == before['bytes'] and a.signature_fd(stream.fileno()) == before
                  and a.digest(raw) == expected, 'Frozen package source SHA differs: ' + name)
        return raw


def load_frozen_checker(a, paths):
    package_path = a.ROOT / 'instruct300/candidates/current200-retained18-speed-v14-ebef786'
    fd = paths._walk(paths.root_fd, package_path.relative_to(a.ROOT).parts)
    try:
        package_raw = read_package_raw(a, fd, 'manifest.json', PACKAGE_MANIFEST_SHA)
        package = json.loads(package_raw)
        a.require(package['code_commit'] == a.CODE and package['common_data_identity'] == a.DATA
                  and package['model_identity']['sha256'] == a.MODEL, 'Fixed original checker package identity differs')
        raw_codes = {name:read_package_raw(a, fd, name, sha) for name,sha in PINS.items()}
    finally:
        os.close(fd)
    # Compile the bytes already SHA-bound on held package FDs. No second
    # untrusted path import of these five original modules is permitted.
    modules = {}
    for filename, raw in raw_codes.items():
        name = filename[:-3]
        a.require(name not in sys.modules, 'Use a fresh independent CPU process, not an existing training interpreter')
        module = types.ModuleType(name); module.__file__ = str(package_path / filename)
        sys.modules[name] = module
        executable = raw
        if filename == 'attempt_accounting.py':
            # Only filesystem enumeration changes; every original accounting,
            # identity, opportunity and SHA check below it remains byte-exact.
            old = b"files=sorted((run/'checkpoints/training_state/attempt-results').glob('attempt-*.json'))"
            a.require(raw.count(old) == 1, 'Frozen attempt enumeration shape changed')
            executable = raw.replace(old, b'files=_archive_adapter_attempt_files(run)')
            module._archive_adapter_attempt_files = lambda run: bounded_attempt_files(a, run)
            module._archive_adapter_executable_sha256 = a.digest(executable)
        exec(compile(executable, module.__file__, 'exec'), module.__dict__)
        modules[name] = module
    for name in ('check_behavior_mu_cpu', 'production_contract', 'attempt_accounting'):
        modules[name].read_bound = lambda path, expected=None: bounded_read_bound(a, path, expected)
    modules['check_behavior_mu_cpu'].sha = lambda path: bounded_sha(a, path)
    return modules['check_behavior_mu_cpu'], package, modules['attempt_accounting']._archive_adapter_executable_sha256


def physical_run_path(a, path):
    path = Path(path); alias = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    if path.is_relative_to(alias):
        path = a.ROOT / path.relative_to(alias)
    a.require(path.is_absolute() and path.is_relative_to(a.RUN) and '..' not in path.parts, 'Metadata must stay on own3341 RUN')
    return path


def bounded_attempt_files(a, run):
    a.require(Path(run) == a.RUN, 'Frozen accounting must receive exact own3341 run')
    with a.OwnPaths() as paths:
        fd = paths._walk(paths.run_fd, ('checkpoints','training_state','attempt-results'))
        try:
            with os.scandir(fd) as entries:
                names = sorted(e.name for e in entries if fnmatch.fnmatchcase(e.name, 'attempt-*.json'))
            return [a.RUN / 'checkpoints/training_state/attempt-results' / name for name in names]
        finally:
            os.close(fd)


def bounded_read_bound(a, path, expected=None):
    path = physical_run_path(a, path)
    with a.OwnPaths() as paths:
        fd = paths.open(path, os.O_RDONLY)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            a.require(before.st_size < 1024*1024, 'Bounded own metadata required')
            raw = stream.read(before.st_size+1); after = os.fstat(stream.fileno())
            fields = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            a.require(len(raw) == before.st_size and fields(before) == fields(after), 'Held evidence changed')
    sha = a.digest(raw)
    a.require(expected is None or sha == expected, 'Original pinned evidence bytes changed')
    return json.loads(raw), sha, raw


def bounded_sha(a, path):
    path = physical_run_path(a, path)
    with a.OwnPaths() as paths:
        return a.hash_file(path, paths)[0]


def physical_own_path(a, path):
    path = Path(path); alias = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    if path.is_relative_to(alias):
        path = a.ROOT / path.relative_to(alias)
    a.require(path.parent.parent == a.TRAJECTORIES and path.name == 'actor-inputs.pt'
              and re.fullmatch(r'outer-\d{6}-attempt-\d{6}', path.parent.name), 'Only exact own3341 actor-inputs path allowed')
    return path


def tensor_bridge(helper, a, original_held_tensor, request, records):
    def load(path, expected):
        source = physical_own_path(a, path)
        a.require(type(expected) is str and re.fullmatch('[a-f0-9]{64}', expected), 'Original manifest payload SHA required')
        subrequest = request + '-item-%03d' % (len(records)+1)
        with a.OwnPaths() as paths:
            try:
                a.read_json(a.ARCHIVES/source.parent.name/'.complete.json', paths)
                has_archive = True
            except FileNotFoundError:
                has_archive = False
        if has_archive:
            # Exercise the archive while the original is still present, so
            # actual original-logic CPU acceptance can precede any source unlink.
            with helper.verified_archive_input(source.parent.name, subrequest, expected) as handle:
                value = original_held_tensor(handle.read_path, expected)
                records.append(dict(directory=source.parent.name, actor_input_sha256=expected, archive_used=True,
                                    archive_sha256=handle.archive_sha256, archive_receipt_sha256=handle.archive_receipt_sha256))
                return value
        with a.OwnPaths() as paths:
            try:
                fd = paths.open(source, os.O_RDONLY)
            except FileNotFoundError:
                fd = None
            if fd is not None:
                with os.fdopen(fd, 'rb') as held:
                    # Unchanged original hash->torch.load CPU->stable fstat->hash
                    # logic reads this same pinned file via its kernel FD path.
                    value = original_held_tensor(Path('/proc/self/fd/%d' % held.fileno()), expected)
                records.append(dict(directory=source.parent.name, actor_input_sha256=expected, archive_used=False))
                return value
        raise RuntimeError('Neither original source nor complete verified archive exists')
    return load


def verify(outer, request, approval):
    helper = replacement_module(); a = helper.archive_module()
    a.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Future checker must be explicitly CPU-only')
    with a.own_lock() as paths:
        checker, package, attempt_executable_sha = load_frozen_checker(a, paths)
        # This narrow adapter permits only the currently observed all-credited
        # chain. It refuses allskip/partial categories rather than silently
        # dropping them or letting the frozen helper read partial native PT.
        attempts = paths._walk(paths.run_fd, ('checkpoints','training_state','attempt-results'))
        try:
            with os.scandir(attempts) as entries:
                names = sorted(e.name for e in entries if re.fullmatch(r'attempt-\d{6}\.json', e.name))
            for name in names:
                event, _ = a.read_json(a.RUN / 'checkpoints/training_state/attempt-results' / name, paths)
                a.require(event['status'] == 'COMPLETED_OPPORTUNITIES_OUTER_CREDITED', 'Partial/allskip history requires separately scoped original audit; no archive-adapter acceptance')
                if event['accepted_outer'] == outer:
                    break
        finally:
            os.close(attempts)
        helper.lowspace_module().guard(paths,helper.current_latest(a,paths),0,'cpu-before-intent')
        audit_root = a.ARCHIVES / 'cpu-audits'; paths.mkdir(audit_root, exist_ok=True)
        lease = paths.open(audit_root / '.single-audit.lock', os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = audit_root / request; paths.mkdir(target)
            paths.write_json_new(target / 'intent.json', dict(status='CPU_AUDIT_INTENT_ONLY', request_id=request,
                                 approval_reference=approval, pid=os.getpid(), pid_startticks=helper.pid_ticks(),
                                 original_checker_sha256=PINS['check_behavior_mu_cpu.py'], source_commit=a.CODE,
                                 entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), outer=outer))
        except BaseException:
            os.close(lease); raise
    # Release the archive writer lock; each archived tensor context acquires it
    # separately. The independent CPU audit lease remains held for this run.
    try:
        affinity = os.sched_getaffinity(0); os.sched_setaffinity(0, {min(affinity)})
        records = []; original_held_tensor = checker.held_tensor
        checker.held_tensor = tensor_bridge(helper, a, original_held_tensor, request, records)
        with a.OwnPaths() as paths, paths.parent(target / 'original-checker-result.pending.json') as (parent, leaf):
            output = Path('/proc/self/fd/%d/%s' % (parent, leaf))
            result = checker.verify(str(a.RUN), package, a.CONFIG, outer, str(output))
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, 'rb') as held:
                os.fsync(held.fileno()); original_raw = held.read(32*1024**2+1)
                a.require(len(original_raw) <= 32*1024**2, 'Bounded original CPU result required')
            a.require(json.loads(original_raw) == result and result['cuda_initialized'] is False, 'Original checker output or CPU-only identity differs')
            result['archive_adapter'] = dict(status='INDEPENDENT_ARCHIVE_AWARE_CPU_ACCESS_COMPLETED', request_id=request,
                approval_reference=approval, original_checker_sha256=PINS['check_behavior_mu_cpu.py'], original_checker_output_sha256=a.digest(original_raw),
                package_manifest_sha256=PACKAGE_MANIFEST_SHA, frozen_helper_sha256=PINS, replacement_entry_sha256=REPLACEMENT_ENTRY_SHA,
                archive_entry_sha256=helper.ARCHIVE_ENTRY_SHA, lowspace_entry_sha256=helper.LOWSPACE_ENTRY_SHA, new_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                accounting_io_adapted_executable_sha256=attempt_executable_sha,
                io_adaptations='Pinned own JSON/hash/enumeration and actor-input FD routing; original validation/accounting checks retained',
                actor_inputs=records, one_file_temporary_at_a_time=True, cpu_affinity_cores=1,
                original_verification_logic_unchanged=True, training_checker_source_unchanged=True, model_weights_loaded=False,
                tokenizer_loading='Original checker package-bound tokenizer metadata/CPU loader retained', optimizer_writes=0, model_forwards=0)
            helper.lowspace_module().guard(paths,helper.current_latest(a,paths),0,'cpu-final-metadata')
            paths.write_json_new(target / '.complete.json', result)
            os.fsync(parent)
        return result
    finally:
        os.close(lease)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['plan','verify'], default='plan'); p.add_argument('--outer', type=int, required=True)
    p.add_argument('--request-id'); p.add_argument('--approval-reference'); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.action == 'plan':
        print(json.dumps(dict(status='LOCAL_UNAPPROVED_PLAN_ONLY_NO_TENSOR_OR_RUNTIME_IO', outer=args.outer,
                              original_checker_sha256=PINS['check_behavior_mu_cpu.py'], package_manifest_sha256=PACKAGE_MANIFEST_SHA,
                              actor_input_mode='Single pinned original FD or verified archive temporary FD per file', frozen_training_unchanged=True), indent=2))
        return
    if not (args.execute and args.approval_reference and args.request_id and re.fullmatch('[a-z0-9-]{8,40}', args.request_id)
            and 1 <= args.outer <= 200):
        raise RuntimeError('Future explicit human approval, unique safe request, execute and outer1..200 required')
    print(json.dumps(verify(args.outer,args.request_id,args.approval_reference),sort_keys=True))


if __name__ == '__main__':
    main()
