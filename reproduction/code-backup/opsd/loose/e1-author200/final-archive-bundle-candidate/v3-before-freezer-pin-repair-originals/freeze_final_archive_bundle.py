"""Local preparation/freeze only. Never access remote or execute the collector."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import types

W = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
RUNTIME = 'd83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
BASE = '90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
PEER = '62826381f10160bc6452fe8b26a9338b586f0f3a9142b43b4767a05a58b5aed4'
HISTORY = 'a855ab5b63e607218d40db8de87647438c424868e83af654a231936fbbad2349'
READY = '24d876c05496678648fa1d19228fe87c19685bd35903c2f0d1f641ef73af4892'
RUNTIME_NAME = 'collect_actual3261_final_originals.py'
FORMAL = ROOT / 'tools/formal-bb5f4872142f'
SNAP = W / 'actual-snapshots/3261-20261005T232501896059Z/original-files'


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode() + b'\n'


def bootstrap_runtime():
    path = W / 'final-archive-v2-candidate' / RUNTIME_NAME
    need(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid(),
         'Physical owned reviewed runtime required')
    before = path.stat()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_uid', 'st_mtime_ns', 'st_ctime_ns')
        need(tuple(getattr(before, k) for k in fields) == tuple(getattr(opened, k) for k in fields)
             == tuple(getattr(after, k) for k in fields) == tuple(getattr(path.stat(), k) for k in fields),
             'Reviewed runtime changed while loading')
        need(digest(raw) == RUNTIME, 'Only exact d83 collector may supply IO helpers')
    finally:
        os.close(fd)
    module = types.ModuleType('reviewed_final_archive_collector_io')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module, raw


def read(io, path, pin):
    fd = io.open_physical(path, uid=os.getuid())
    try:
        before = io.metadata(os.fstat(fd))
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            raw = handle.read()
        need(io.metadata(os.fstat(fd)) == before == io.physical_metadata(path, os.getuid())
             and len(raw) == before['bytes'] and digest(raw) == pin,
             'Exact stable input SHA differs: ' + str(path))
        return raw
    finally:
        os.close(fd)


def history_contract(io, raw):
    value = io.unique_json(raw)
    need(value['errors'] == [] and value['scope'] ==
         'READ_ONLY_INFLIGHT_RAY_STDERR_SLURM_BINDING_NOT_CLOSED_LIFETIME_OR_RELEASE',
         'Historical reader scope/errors differ')
    need(set(value['processes']) == {str(p) for p, _, _ in io.RAY}, 'Actual nine historical PIDs differ')
    result = {}
    for binding in io.expected_bindings():
        pid = binding['pid']
        process = value['processes'][str(pid)]
        for key in ('pid', 'start_ticks', 'cgroup', 'stdout_fd_target', 'stderr_fd_target'):
            need(type(process[key]) is type(binding[key]) and process[key] == binding[key],
                 'Historical original PID/start/FD identity differs')
        need(process['uids'] == [0, 0, 0, 0] and all(type(v) is int for v in process['uids']),
             'Historical four UID fields differ')
        for stream, record in [('stdout', process['stdout']),
                               ('stderr', value['files'][process['stderr_key']])]:
            need(record['physical_path'] == binding[stream + '_fd_target'], 'Historical physical path differs')
            need(record['before'] == record['after'], 'Historical pathname samples changed')
            old = record['before']
            need(all(type(old[k]) is int for k in ('device', 'inode', 'uid', 'mode', 'bytes'))
                 and old['uid'] == 0 and old['bytes'] >= 0, 'Typed historical metadata required')
            result[f'ray-pid-{pid}-{stream}'] = {
                'physical_path': record['physical_path'], 'device': old['device'], 'inode': old['inode'],
                'uid': 0, 'permission_mode': old['mode'], 'minimum_final_bytes': old['bytes'],
                'historical_prefix_sha256': record.get('sha256'),
                'prefix_scope': 'ACTUAL_RECORDED_BYTES' if 'sha256' in record else
                                'NO_PREFIX_HASH_IN_EXISTING_READER_NOT_GUESSED'}
    for role in ('slurm-formal-3261.out', 'slurm-formal-3261.err'):
        record = value['files'][role]
        need(record['physical_path'] == str(ROOT / 'logs' / role.removeprefix('slurm-')),
             'Historical Slurm path differs')
        need(record['before'] == record['after'] and record['source_offset'] == 0,
             'Historical Slurm prefix origin differs')
        old = record['before']
        need(all(type(old[k]) is int for k in ('device', 'inode', 'uid', 'mode', 'bytes')) and old['uid'] == 0,
             'Typed historical Slurm identity required')
        result[role] = {'physical_path': record['physical_path'], 'device': old['device'], 'inode': old['inode'],
            'uid': 0, 'permission_mode': old['mode'], 'minimum_final_bytes': old['bytes'],
            'historical_prefix_sha256': record['sha256'], 'prefix_scope': 'ACTUAL_RECORDED_BYTES'}
    need(len(result) == 20 and sum(v['historical_prefix_sha256'] is not None for v in result.values()) == 11,
         'Exactly20 original inodes and11 original prefix hashes required')
    return result


def review_inputs():
    return {
        'FINAL_ARCHIVE_COLLECTOR_SCOPE.md': ('final-archive-v2-candidate/FINAL_ARCHIVE_COLLECTOR_SCOPE.md', 'bb92376ea95f64761cdaf0f920f80b4ad273005e40adbcf145bc0f1dea70e37e'),
        'COLLECTOR_SOFTWARE_MANIFEST_ORIGINAL.json': ('final-archive-v2-candidate/manifest-software-v2.json', '566bb572f6bac6362685beab648624ab7171039f9c3e4e8a3cf3cb79f81da251'),
        'COLLECTOR_CPU51_STDOUT_ORIGINAL': ('final-archive-v2-candidate/cpu-tests-v2-original.stdout', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
        'COLLECTOR_CPU51_STDERR_ORIGINAL': ('final-archive-v2-candidate/cpu-tests-v2-original.stderr', '552a591e539137a55d7204584a43d1a86c43315d3f5c3245a8bb090bf3ad0a90'),
        'COLLECTOR_CPU51_TEST_SOURCE.py': ('final-archive-v2-candidate/test_collect_actual3261_final_originals.py', '0b292bd74293ab23140971a16644922343ad43fd6908f25dcd3b26ba7ccf97b8'),
        'FINAL_ARCHIVE_COLLECTOR_V2_INDEPENDENT_SOFTWARE_REVIEW.json': ('FINAL_ARCHIVE_COLLECTOR_V2_INDEPENDENT_SOFTWARE_REVIEW.json', PEER),
        'INDEPENDENT_COLLECTOR_CPU51_STDOUT_ORIGINAL': ('independent-final-archive-v2-cpu-original.stdout', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
        'INDEPENDENT_COLLECTOR_CPU51_STDERR_ORIGINAL': ('independent-final-archive-v2-cpu-original.stderr', 'a8d01652e5568aead86c7e0fb208e7473110e2e0c1b855355ad27c5d6be4e41a'),
        'INDEPENDENT_COLLECTOR_BOUNDARY6_TEST_SOURCE.py': ('independent_final_archive_boundary_tests.py', '926a8002705c02de414d039e19757569cd693dad6e62524dc6bcd4352785f451'),
        'INDEPENDENT_COLLECTOR_BOUNDARY6_STDERR_ORIGINAL': ('independent-final-archive-boundary-original.stderr', 'b3e836f87a17dbc8b7669b9065f8f70d4b1bdd08b6079a447c6a0d12c3be249a'),
        'AUTHOR200_FINAL_ACCEPTANCE_EVIDENCE_PLAN.md': ('AUTHOR200_FINAL_ACCEPTANCE_EVIDENCE_PLAN.md', '82d1b2ef5fdb54ab4bfb29fabaa6679073aee481f2b496dbd575125e2cedf4a6'),
        'AUTHOR200_FINAL_ACCEPTANCE_INDEPENDENT_DESIGN_REVIEW.json': ('AUTHOR200_FINAL_ACCEPTANCE_INDEPENDENT_DESIGN_REVIEW.json', '8ab24df20d4ca2d824b4c33a8461b68305c9271d396b8c637c2be9bdee3b9c2f'),
        'AUTHOR200_FINAL_LOGS_TERMINAL_READ_ONLY_REVIEW.md': ('AUTHOR200_FINAL_LOGS_TERMINAL_READ_ONLY_REVIEW.md', '479a32781c4fe231c361499fd6baf7e615a80b2371b05731d8d7dfc9321f1b25'),
        'ACTUAL3261_INFLIGHT_STDERR_SLURM_PREFIX_INDEPENDENT_REVIEW.json': ('ACTUAL3261_INFLIGHT_STDERR_SLURM_PREFIX_INDEPENDENT_REVIEW.json', 'e47ebd3b3a9e73619b5bdd0ec700760ed6aa0ea045266854baff76cab73b83c6'),
        'SOURCE_FORMAL_LOG_FRAMING_TERMINAL_IMPACT_REVIEW_V2.json': ('SOURCE_FORMAL_LOG_FRAMING_TERMINAL_IMPACT_REVIEW_V2.json', 'd16d79903b41ab43f95315d6eceb3e84d5da5dacb4e42350631e140a5b52233e'),
        'AUTHOR200_FINAL_EVIDENCE_PLAN_INDEPENDENT_REVIEW.json': ('AUTHOR200_FINAL_EVIDENCE_PLAN_INDEPENDENT_REVIEW.json', '2082a72bb9177c665ef1e9ea64397edd16c483017009e81b5ab9e93e3b63b964'),
        'ACTUAL_FD_BINDINGS_SUMMARY_ORIGINAL.json': ('ray-stderr-slurm-bindings-inflight-20261005T233707660781Z/bindings-summary.json', HISTORY),
    }


def construct(io, runtime_raw, design_path):
    base = io.unique_json(read(io, W / 'final-acceptance-90c2695a7f46/manifest.json', BASE))
    need(base['source_commit'] == SOURCE and len(base['dependencies']) == 278, 'Exact existing90 reference differs')
    source_raw = read(io, SNAP / 'bindings/source.json', base['dependencies'][str(RUN / 'bindings/source.json')])
    source = io.unique_json(source_raw)
    need(source['source_commit'] == SOURCE and source['repo'] == str(REPO) and source['clean'] is True
         and len(source['source_files']) == 16, 'Actual source188 identity differs')
    dependencies = {str(REPO / n): pin for n, pin in source['source_files'].items()}
    selected = [RUN / n for n in ('bindings/source.json', 'bindings/allocation.json', 'bindings/manifest.json',
        'bindings/import_closure.json', 'bindings/public.json', 'full_state/global_step_5/resolved-config.json',
        'full_state/global_step_5/manifest.json', 'acceptance-5-receipt.json')]
    selected += [FORMAL / n for n in ('manifest.json', 'formal_supervisor.py', 'formal_job.sh',
                                     'owned_runtime.py', 'resource_sample.py')]
    for path in selected:
        dependencies[str(path)] = base['dependencies'][str(path)]
    dependencies[str(ROOT / 'tools/final-acceptance-90c2695a7f46/manifest.json')] = BASE
    dependencies[str(RUN / 'approval-ready.json')] = READY
    need(len(dependencies) == 31, 'Exact collector-only31 existing dependency paths required')
    localrepo = W.parent / 'opsd-e1-300'
    head = subprocess.check_output(['git', '-C', str(localrepo), 'rev-parse', 'HEAD']).decode().strip()
    clean = subprocess.check_output(['git', '-C', str(localrepo), 'status', '--porcelain'])
    need(head == SOURCE and not clean, 'Local source188 must remain clean')
    for name, pin in source['source_files'].items():
        read(io, localrepo / name, pin)
    small = {}
    for path in selected:
        if RUN in path.parents:
            local = SNAP / path.relative_to(RUN)
        else:
            local = W / 'formal-bb5f4872142f' / path.name
        small[str(path)] = read(io, local, dependencies[str(path)])
    ready_raw = read(io, SNAP / 'approval-ready.json', READY)
    m = io.unique_json(small[str(RUN / 'full_state/global_step_5/manifest.json')])
    receipt = io.unique_json(small[str(RUN / 'acceptance-5-receipt.json')])
    need(m['driver_pid'] == 1533312 and type(m['driver_pid']) is int and m['source_commit'] == SOURCE
         and m['job_id'] == '3261' and m['config_sha256'] == CONFIG and m['run_root'] == str(RUN)
         and m['method'] == 'pure_opsd' and m['global_step'] == 5 and m['horizon'] == 200,
         'Actual accepted5 manifest identity differs')
    need(receipt['payload']['checkpoint_manifest_sha256'] == digest(small[str(RUN / 'full_state/global_step_5/manifest.json')])
         and receipt['payload']['driver_pid'] == 1533312, 'Exact accepted5 signed envelope binding differs')
    ready = io.unique_json(ready_raw)
    need(ready['job_id'] == '3261' and ready['source_commit'] == SOURCE and ready['run_root'] == str(RUN),
         'Actual ready/source identity differs')
    reviews = {name: read(io, W / relative, pin) for name, (relative, pin) in review_inputs().items()}
    peer = io.unique_json(reviews['FINAL_ARCHIVE_COLLECTOR_V2_INDEPENDENT_SOFTWARE_REVIEW.json'])
    need(peer['status'] == 'PASS_INDEPENDENT_LOCAL_FINAL_ORIGINAL_ARCHIVE_COLLECTOR_SOFTWARE_ONLY'
         and peer['runtime']['sha256'] == RUNTIME and peer['independent_cpu_replay']['tests'] == 51
         and peer['independent_cpu_replay']['actual_exit_code'] == 0
         and peer['additional_independent_closure_tests']['tests'] == 6
         and peer['additional_independent_closure_tests']['actual_exit_code'] == 0,
         'Exact51+6 independent collector software review differs')
    io.validate_bindings(io.unique_json(reviews['ACTUAL3261_INFLIGHT_STDERR_SLURM_PREFIX_INDEPENDENT_REVIEW.json']))
    historical = history_contract(io, reviews['ACTUAL_FD_BINDINGS_SUMMARY_ORIGINAL.json'])
    # The local design is a new reviewed candidate artifact, never a future terminal.
    design_raw = read(io, design_path, digest(Path(design_path).read_bytes()))
    reviews['FINAL_ARCHIVE_BUNDLE_PREPARATION_DESIGN.md'] = design_raw
    for name in ('freeze_final_archive_bundle.py', 'publish_final_archive_bundle.py', 'test_final_archive_bundle.py'):
        path = Path(__file__).parent / name
        reviews['PREPARATION_ONLY_' + name] = read(io, path, digest(path.read_bytes()))
    manifest = {'version': 1, 'kind': 'AUTHOR200_FINAL_ORIGINAL_ARCHIVE_PREPARATION_TOOLS',
        'scope': 'IMMUTABLE_SOFTWARE_AND_EXISTING_IDENTITY_NOT_COLLECTOR_EXECUTION_OR_FINAL_ACCEPTANCE',
        'source_commit': SOURCE, 'tools': {RUNTIME_NAME: RUNTIME},
        'review_records': {n: digest(raw) for n, raw in reviews.items()}, 'dependencies': dependencies,
        'contract': {'training_job_id': '3261', 'run_root': str(RUN), 'source_root': str(REPO),
            'config_sha256': CONFIG, 'original_driver_pid': 1533312, 'original_driver_start_ticks': '1829085316',
            'world_size': 8, 'horizon': 200, 'collector_runtime_count': 1,
            'runtime_execution_authorized_by_freeze': False, 'requires_live_training_driver': False,
            'deployment_requires_original_driver_alive': True,
            'deployment_requires_typed_running_allocation': True,
            'future_final_review_requires_historical_inode_compare': True,
            'historical_binding_summary_sha256': HISTORY, 'historical_inode_members': 20,
            'historical_prefix_members': 11, 'fresh_existing_dependency_count': 31},
        'historical_file_bindings': historical,
        'limits': [
            'Only exact d83 is runtime; test sources and publisher/freezer candidates are non-runtime review artifacts',
            '31 collector-specific existing source/identity/formal/reference dependencies are distinct from native210 or checkpoint278 closure',
            'Import-closure JSON full SHA is bound as historical context; its210 module files are not reverified by this bundle',
            'No future terminal, resources final hash, final checkpoint200 or final51713 result is included or guessed',
            'Deployment separately verifies actual original live driver/RUNNING typed8 exclusive noRestart; collector future terminal mode needs no live driver',
            'Deploying this bundle does not execute collector, assume loopback state, signal or acquire locks',
            'Final independent review must compare20 opened inode/device/UID identities and11 actual original prefix hashes',
            'Nine stdout historical records contain no prefix hash; missing historical bytes are not reconstructed or claimed',
            'Closed current bytes, scientific effect, checkpoint recovery, training child exit, wrapper/Slurm exit and resource release remain separate',
        ]}
    members = {RUNTIME_NAME: runtime_raw, **reviews}
    return manifest, members


def freeze(io, parent, manifest, members):
    raw = canonical(manifest)
    pin = digest(raw)
    directory = Path(parent) / ('final-archive-' + pin[:12])
    need(directory.parent.resolve(strict=True) == directory.parent
         and directory.parent.stat().st_uid == os.getuid(), 'Physical owned freeze parent required')
    need(set(members) == set(manifest['tools']) | set(manifest['review_records']), 'Member union differs')
    need(all(Path(n).name == n and n not in ('manifest.json', '.', '..') for n in members), 'Unsafe bundle member')
    need(all(digest(value) == {**manifest['tools'], **manifest['review_records']}[name]
             for name, value in members.items()), 'Full member SHA differs')
    os.mkdir(directory, 0o700)
    published = []
    for name, value in list(members.items()) + [('manifest.json', raw)]:
        io.write_exclusive(directory / name, value)
        (directory / name).chmod(0o444)
        fd = io.open_physical(directory / name, uid=os.getuid())
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        need(read(io, directory / name, digest(value)) == value, 'New frozen member readback differs')
        published.append(name)
    need(published[-1] == 'manifest.json', 'Manifest must be last')
    need(set(p.name for p in directory.iterdir()) == set(members) | {'manifest.json'}, 'Frozen member set differs')
    directory.chmod(0o555)
    for path in (directory, directory.parent):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {'directory': str(directory), 'remote_directory': str(ROOT / 'tools' / directory.name),
            'manifest_sha256': pin, 'manifest_last': True, 'runtime_count': 1,
            'existing_dependency_count': len(manifest['dependencies']),
            'review_member_count': len(manifest['review_records']), 'published_members': published,
            'directory_mode': oct(directory.stat().st_mode & 0o777),
            'scope': 'LOCAL_IMMUTABLE_PREPARATION_NOT_DEPLOYMENT_EXECUTION_OR_FINAL_PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Explicitly write a new local immutable bundle')
    args = parser.parse_args()
    io, runtime_raw = bootstrap_runtime()
    manifest, members = construct(io, runtime_raw, Path(__file__).parent / 'FINAL_ARCHIVE_BUNDLE_PREPARATION_DESIGN.md')
    if not args.freeze:
        print(json.dumps({'scope': 'LOCAL_VERIFIED_PLAN_ONLY_NO_BUNDLE_WRITE',
            'manifest_sha256': digest(canonical(manifest)), 'runtime_count': 1,
            'existing_dependency_count': 31, 'review_count': len(manifest['review_records'])}, sort_keys=True))
        return
    result = freeze(io, W, manifest, members)
    receipt = W / ('final-archive-freeze-' + result['manifest_sha256'][:12] + '.json')
    io.write_exclusive(receipt, canonical(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
