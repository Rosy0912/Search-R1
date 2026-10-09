"""Explicit accepted R/W/P/J protected backups: exact XZ, then duplicate-only reclaim."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
PREFIX = 'experiments/search_e1/control/opsd_ablation_protected_compact_v1/'
ANALYSIS = Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008')
NODE = '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008'
GOAL_SHA = 'c65bea46dbb4ef6451ed56b2c4e26adafde5fd247e6b67fb5bea593b8689c94f'
RUNS = dict(R='779e853e37b540e8847336c142129377', W='dc8a4e85822547548c0f79be4c87e13e',
            P='c0a425974d2142b1955429c85a4fb9f3', J='f40bcfd2693a4c36a154de5fd81a475d')
CONFIGS = dict(R='8cc81ae28882f0fb32a924d7a4d96fc6f7f0702b210ef62847e409c404d36ab9',
               W='c7f48e13328835c84c4c09a9b7bd51be74c99e902e68d1f4ddb21ef989dd1494',
               P='dcc0a03750dec59651550609889a8b82ba0d967a728da01626216af6d7714412',
               J='da6f7396970b77d3bbf2572f437ed2e05f0c3469bb2bcbe5a63dbbe12306b01f')
ENGINES = dict(
    p=('opsd_backup_duplicate_reclaim_v1/reclaim.py', '4c4d2518c9fc44f8ef75ef0ef31c3fe51d7cb00f2de0242932587c26f2bd3e6c'),
    c=('opsd_local_archive_recompress_v1/recompress.py', '9bfb249e78957e7f34c38b1d8beacf0d75cadae68b15106793b91dce212ac921'),
    t=('opsd_local_single_tar_reclaim_v1/reclaim.py', '70ea28a46393db506f3992010e7558c94781adb565dbfe14556ecc0b7ca5aa7e'))


def engines():
    result = {}
    for name, (relative, sha) in ENGINES.items():
        path = HERE.parent/relative
        for part in [*reversed(path.parents), path]:
            if part.is_symlink(): raise ValueError('Engine symlink refused')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha: raise ValueError('Pinned engine source differs')
        module = types.ModuleType('_protected_compact_'+name); module.__file__ = str(path)
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        result[name] = module
    return result


def read_ref(p, ref):
    value, proof = p.checked_json(ref['path'], ref['sha256'])
    p.require('bytes' not in ref or ref['bytes'] == proof['bytes'], 'Reference bytes differ')
    return value, proof


def published(p, root, manifest_ref, publication_ref, commit, prefix, run):
    manifest, mp = read_ref(p, manifest_ref); pub, pp = read_ref(p, publication_ref)
    p.require({x.name for x in root.iterdir()} == set(manifest['files'])|{'source-manifest.json'}, 'Source inventory differs')
    p.require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
              and re.fullmatch('[0-9a-f]{40}', commit or '')
              and pub.get('source_commit') == pub.get('remote_readback_sha') == commit
              and pub.get('source_manifest_sha256') == mp['sha256']
              and (pub.get('run_id') == run or run in pub.get('run_ids', {}).values()), 'Published run/source binding absent')
    files = []
    for name, old in manifest['files'].items():
        p.require(Path(name).name == name, 'Source path escape')
        fp = p.fileproof(root/name)
        p.require(all(fp[k] == old[k] for k in ('bytes', 'sha256')), 'Source SHA/bytes differ')
        files.append(fp)
    for fp in files+[mp]:
        rows = [r for r in pub['files'] if r['path'] == prefix+Path(fp['path']).name]
        p.require(len(rows) == 1 and rows[0].get('equal') is True
                  and all(rows[0][k] == fp[k] for k in ('bytes', 'sha256')), 'Remote byte readback differs')
    return dict(manifest=mp, publication=pp, source_commit=commit, files=files)


def bundle(p, input_ref):
    value, ip = read_ref(p, input_ref)
    p.require(value.get('schema') == 'opsd-ablation-protected-compact-input-v1'
              and value.get('variant') in RUNS, 'Explicit R/W/P/J input required')
    receipt, rp = read_ref(p, value['backup']); root = Path(rp['path']).parent
    p.checked_path(root, directory=True)
    p.require(ANALYSIS in root.parents and rp['path'] == str(root/'complete-sha-verified.json'), 'Only analysis protected package')
    identity = receipt['identity']; variant = value['variant']; run = RUNS[variant]
    p.require(receipt.get('schema') == 'opsd-ablation-protected-local-backup-v1'
              and receipt.get('accepted') is True and receipt.get('actual_transfer_returncode') == 0
              and receipt.get('file_count') == 42 and receipt.get('core_file_count') == 39
              and receipt.get('closure_file_count') == 3
              and receipt.get('independent_full_local_disk_readback') is True
              and receipt.get('all_source_full_SHA_before') is True
              and receipt.get('all_source_metadata_stable_after') is True
              and receipt.get('node_originals_retained') is True, 'Complete 42-member acceptance absent')
    p.require(identity['variant'] == variant and identity['run_id'] == run
              and identity['output_root'] == NODE+'/group-'+variant+'/runs'
              and identity['config']['path'] == NODE+'/group-'+variant+'/receipts/authorized-launch-20261009/config.json'
              and identity['config']['sha256'] == CONFIGS[variant]
              and identity['goal'] == dict(path=NODE+'/receipts/outer-backup-v2-554c52c-20261009/goal.json', sha256=GOAL_SHA), 'Original group identity differs')
    inv, ivp = read_ref(p, receipt['inventory']); scope, sp = read_ref(p, receipt['scope'])
    after, ap = read_ref(p, receipt['source_after']); refs = [ip, rp, ivp, sp, ap]
    p.require(inv.get('schema') == 'opsd-ablation-protected-inventory-v1' and inv.get('accepted') is True
              and inv['identity'] == scope['identity'] == after['identity'] == identity
              and scope.get('accepted') is True and after.get('accepted') is True
              and inv['scope_sha256'] == after['scope_sha256'] == sp['sha256']
              and after['inventory_sha256'] == ivp['sha256']
              and after['all_source_metadata_stable'] is True and after['all_core_file_sets_unchanged'] is True
              and after['files'] == 42 and inv['protected'] == scope['protected'] == receipt['protected'] == after['protected']
              and [r['optimizer_updates'] for r in scope['protected']] == [0, 1, 5], 'Original inventory/scope/after differs')
    for key in ('pin_execution', 'inventory_execution', 'source_after_execution'):
        execution, fp = read_ref(p, receipt[key]); refs.append(fp)
        p.require(execution.get('actual_returncode') == 0, 'Original read-only operation did not exit0')
    transfer, fp = p.checked_json(root/'transfer-terminal.json', p.fileproof(root/'transfer-terminal.json')['sha256']); refs.append(fp)
    p.require(transfer.get('actual_returncode') == 0 and transfer.get('status_not_inferred_from_absence') is True, 'Original stream actual wait0 absent')
    rows = {p.safe_name(r['relative']): r for r in inv['files']}
    p.require(len(rows) == len(inv['files']) == len(receipt['files']) == 42
              and set(rows) == set(receipt['files']) and sum(r['bytes'] for r in rows.values()) == inv['total_bytes'], 'Exact original member set differs')
    for name, row in rows.items():
        old = receipt['files'][name]
        p.require(old['path'] == str(root/'verified-files'/name)
                  and all(old[k] == row[k] for k in ('bytes', 'sha256')), 'Original loose/member SHA mapping differs')
    archive = receipt['archive']; p.require(archive['path'] == str(root/'protected-checkpoints.tar'), 'Only named original tar')
    source = receipt['source_publication']; original_root = Path(source['manifest']['path']).parent
    oldpub = published(p, original_root, source['manifest'], source['publication'], source['source_commit'],
                       'experiments/search_e1/control/opsd_ablation_protected_backup_v1/', run)
    refs += oldpub['files']+[oldpub['manifest'], oldpub['publication']]
    return dict(input=ip, identity=identity, root=str(root), archive=archive, rows=rows, preserve=refs)


def source_proof(p, args, run):
    proof = published(p, HERE, dict(path=str(HERE/'source-manifest.json'), sha256=args.source_manifest_sha256),
                      dict(path=args.publication, sha256=args.publication_sha256), args.source_commit, PREFIX, run)
    proof['engines'] = [p.fileproof(HERE.parent/path) for path, _ in ENGINES.values()]
    p.require([r['sha256'] for r in proof['engines']] == [sha for _,sha in ENGINES.values()], 'Engine bytes changed after import')
    return proof


def compress(p, c, args, package, source):
    out = Path(args.output); p.checked_path(out.parent, directory=True)
    p.require(out.is_absolute() and not out.exists() and not out.is_symlink()
              and Path(package['root']) not in out.parents and out not in Path(package['root']).parents, 'New independent output required')
    original = p.metadata(package['archive']['path']); p.require(original['bytes'] == package['archive']['bytes'], 'Original tar size changed')
    ceiling = original['bytes']+original['bytes']//100+(1<<20)
    p.require(shutil.disk_usage(out.parent).free >= ceiling+(5<<30), 'Storage BACKLOG, never a GPU hold')
    p.no_open_files([original['path']]); out.mkdir(mode=0o700); p.fsync_dir(out.parent)
    start = p.write_json(out/'start.json', dict(actual_pid=os.getpid(), uid=os.getuid(), argv=[sys.executable, *sys.argv],
        source=source, package=package, output=str(out), original=original, no_deletion=True))
    written = c.compress_exact(original['path'], out/'archive.tar.xz.partial', package['archive']['sha256'], original['bytes'])
    recovery = c.verify_disk_xz(out/'archive.tar.xz.partial', package['archive']['sha256'], original['bytes'])
    for ref in package['preserve']+source['files']+[source['manifest'], source['publication'], *source['engines']]: p.stable(ref)
    p.stable(original); p.stable(recovery['compressed'])
    final = out/'archive.tar.xz'; os.rename(out/'archive.tar.xz.partial', final); p.fsync_dir(out)
    compressed = dict(**p.metadata(final), sha256=recovery['compressed']['sha256'])
    return p.write_json(out/'complete-xz-verified.json', dict(schema='opsd-protected-exact-ustar-xz-v1', accepted=True,
        actual_pid=os.getpid(), uid=os.getuid(), package=package, source=source, start=start, compression=written,
        recovery=recovery, compressed=compressed, original=dict(**original, sha256=package['archive']['sha256']),
        historical_42_member_SHA_preserved_via_exact_decoded_tar_bytes=True, fresh_member_hashes_repeated=False,
        no_deletion=True, original_tar_retained=True, no_training_dependency=True))


def accepted_compression(p, args, package, source):
    value, cp = p.checked_json(args.completion, args.completion_sha256)
    parent, pp = p.checked_json(args.parent_terminal, args.parent_terminal_sha256)
    start, sp = read_ref(p, value['start']); wait, wr = read_ref(p, parent['actual_wait_evidence'])
    p.require(value.get('schema') == 'opsd-protected-exact-ustar-xz-v1' and value.get('accepted') is True
              and value['package'] == package and value['source'] == source and value.get('no_deletion') is True,
              'Exact original compression/package/source absent')
    p.require(parent.get('schema') == 'opsd-protected-compact-actual-parent-terminal-v1'
              and parent.get('actual_pid') == value['actual_pid'] == start['actual_pid']
              and parent.get('uid') == value['uid'] == start['uid'] == 501
              and parent.get('actual_returncode') == 0 and parent.get('status_not_inferred_from_absence') is True
              and parent['completion']['path'] == cp['path'] and parent['completion']['sha256'] == cp['sha256']
              and parent['start']['path'] == sp['path'] and parent['start']['sha256'] == sp['sha256']
              and parent['argv'] == start['argv'] and wait.get('actual_returncode') == 0
              and wait.get('actual_pid') == value['actual_pid'], 'Actual compression parent wait0 absent/differs')
    rec = value['recovery']; original = value['original']; compressed = value['compressed']
    p.require(start['source'] == source and start['package'] == package and start['original'] == value['compression']['input']
              and {k: original[k] for k in p.META} == value['compression']['input']
              and rec['decoded_sha256'] == original['sha256'] == package['archive']['sha256']
              and rec['decoded_bytes'] == original['bytes'] == package['archive']['bytes']
              and rec.get('check') == 'CRC64' and rec.get('single_stream') is True
              and rec.get('independent_disk_read') is True and rec.get('no_extracted_payload_written') is True
              and value['compression']['original_sha256'] == original['sha256']
              and value['compression']['original_bytes'] == original['bytes']
              and compressed['path'] == str(Path(cp['path']).parent/'archive.tar.xz')
              and rec['compressed']['path'] == str(Path(cp['path']).parent/'archive.tar.xz.partial')
              and all(compressed[k] == rec['compressed'][k] for k in ('sha256','bytes','device','inode','mtime_ns','mode','uid','nlink')),
              'Exact independent decoded whole tar/final inode differs')
    p.stable(compressed)
    return value, [cp, pp, sp, wr, compressed]


def prepare(p, args, package, source):
    value, refs = accepted_compression(p, args, package, source)
    root = Path(package['root']); out = Path(args.output); p.checked_path(out.parent, directory=True)
    p.require(out.is_absolute() and not out.exists() and root not in out.parents
              and out not in root.parents and Path(args.completion).parent not in out.parents, 'New independent plan required')
    expected = {str(root/'verified-files'/n) for n in package['rows']}
    p.require(p.extraction_set(root/'verified-files') == expected, 'Unexpected/missing loose file')
    targets = []
    for name, old in sorted(package['rows'].items()):
        fp = p.fileproof(root/'verified-files'/name)
        p.require(all(fp[k] == old[k] for k in ('sha256', 'bytes')), 'Fresh loose SHA differs from accepted member')
        targets.append(dict(relative=name, **fp))
    p.stable(value['original']); targets.append(dict(relative=None, **value['original']))
    p.require(len({(r['device'],r['inode']) for r in targets}) == 43, 'Duplicate target inode')
    preserve = package['preserve']+refs+source['files']+[source['manifest'],source['publication'],*source['engines']]
    p.no_open_files([r['path'] for r in targets+preserve])
    for row in targets+preserve: p.stable(row)
    out.mkdir(mode=0o700); p.fsync_dir(out.parent)
    return p.write_json(out/'plan.json', dict(schema='opsd-protected-compact-reclaim-plan-v1', accepted=True,
        package=package, source=source, compression=refs[0], parent_terminal=refs[1], targets=targets, preserve=preserve,
        logical_duplicate_bytes=sum(r['bytes'] for r in targets), fresh_loose_SHA_checked=42,
        historical_member_SHA_via_exact_decoded_tar_bytes=True, tar_is_last_target=True))


def reclaim_targets(p, t, out, pref, targets, preserve):
    """Original one-file mutation primitive; durable per-file restart evidence."""
    journal = out/'journal'; intent = out/'execution-intent.json'
    intended = dict(plan=pref, targets=targets)
    if intent.exists():
        old, _ = p.checked_json(intent, p.fileproof(intent)['sha256']); p.require(old == intended, 'Execution intent differs')
        p.checked_path(journal, directory=True)
    else:
        for row in targets: p.stable(row)
        p.no_open_files([r['path'] for r in targets+preserve])
        journal.mkdir(mode=0o700); p.fsync_dir(out); p.write_json(intent, intended)
    # Validate every remaining target before any new unlink.
    for index, row in enumerate(targets):
        pending = journal/('%03d-pending.json'%index); done = journal/('%03d-done.json'%index)
        wanted = dict(plan=pref, index=index, file=row)
        if pending.exists():
            old, _ = p.checked_json(pending, p.fileproof(pending)['sha256']); p.require(old == wanted, 'Pending differs')
        if done.exists():
            old, _ = p.checked_json(done, p.fileproof(done)['sha256'])
            p.require(old['intent'] == wanted and pending.exists() and not Path(row['path']).exists()
                      and not Path(row['path']).is_symlink(), 'Done target reappeared/differs')
        elif Path(row['path']).exists() or Path(row['path']).is_symlink(): p.stable(row)
        else: p.require(pending.exists(), 'Missing target without exact durable pending')
    for index, row in enumerate(targets):
        pending = journal/('%03d-pending.json'%index); done = journal/('%03d-done.json'%index)
        wanted = dict(plan=pref, index=index, file=row)
        if not done.exists():
            outcome = t.unlink_one(p, row, pending, wanted, preserve)
            p.write_json(done, dict(intent=wanted, outcome=outcome, observed_unix=time.time()))
    for row in targets: p.require(not Path(row['path']).exists() and not Path(row['path']).is_symlink(), 'Target remains')
    for row in preserve+[pref]: p.stable(row)


def execute(p, t, args, source):
    plan, pref = p.checked_json(args.plan, args.plan_sha256); package = bundle(p, plan['package']['input'])
    p.require(plan.get('schema') == 'opsd-protected-compact-reclaim-plan-v1' and plan.get('accepted') is True
              and plan['package'] == package and plan['source'] == source and plan['tar_is_last_target'] is True, 'Plan/source/package differs')
    args.completion = plan['compression']['path']; args.completion_sha256 = plan['compression']['sha256']
    args.parent_terminal = plan['parent_terminal']['path']; args.parent_terminal_sha256 = plan['parent_terminal']['sha256']
    value, refs = accepted_compression(p, args, package, source)
    preserve = package['preserve']+refs+source['files']+[source['manifest'],source['publication'],*source['engines']]
    targets = plan['targets']; names = sorted(package['rows'])
    p.require(plan['preserve'] == preserve and len(targets) == 43 and [r['relative'] for r in targets] == names+[None], 'Exact 42 loose plus tar plan differs')
    for row, name in zip(targets[:-1], names):
        p.require(row['path'] == str(Path(package['root'])/'verified-files'/name)
                  and all(row[k] == package['rows'][name][k] for k in ('sha256','bytes')), 'Target escape/member mismatch')
    p.require(targets[-1] == dict(relative=None, **value['original'])
              and plan['logical_duplicate_bytes'] == sum(r['bytes'] for r in targets)
              and plan['fresh_loose_SHA_checked'] == 42
              and plan['historical_member_SHA_via_exact_decoded_tar_bytes'] is True, 'Original tar/total differs')
    extract = Path(package['root'])/'verified-files'
    p.require(p.extraction_set(extract) == {r['path'] for r in targets[:-1] if Path(r['path']).exists()}, 'Unexpected loose file before reclaim')
    reclaim_targets(p, t, Path(args.plan).parent, pref, targets, preserve)
    p.require(not p.extraction_set(extract), 'Unexpected remaining loose file')
    result = Path(args.plan).parent/'reclaim-complete.json'
    receipt = dict(schema='opsd-protected-compact-reclaim-complete-v1', accepted=True, plan=pref,
        original_identity=package['identity'], compressed=value['compressed'], logical_duplicate_bytes=plan['logical_duplicate_bytes'],
        preserved=preserve, target_files=43, directories_receipts_failed_partials_node_preserved=True,
        actual_apfs_free_gain_not_measured=True, no_model_numerical_acceptance=True, no_training_dependency=True)
    if result.exists():
        old, ref = p.checked_json(result, p.fileproof(result)['sha256']); p.require(old == receipt, 'Completion differs'); return ref
    return p.write_json(result, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['compress','prepare','execute'])
    for name in ('source-manifest-sha256','publication','publication-sha256','source-commit'):
        parser.add_argument('--'+name, required=True)
    for name in ('input','input-sha256','output','completion','completion-sha256','parent-terminal','parent-terminal-sha256','plan','plan-sha256'):
        parser.add_argument('--'+name)
    args = parser.parse_args(); e = engines(); p,c,t = (e[n] for n in ('p','c','t'))
    p.require(sys.platform == 'darwin' and os.getuid() == os.geteuid() == 501, 'Only actual Mac UID501')
    if args.mode == 'execute':
        p.require(args.plan and args.plan_sha256 and not args.input and not args.output, 'Execute only exact immutable plan')
        plan, _ = p.checked_json(args.plan, args.plan_sha256); package = bundle(p, plan['package']['input'])
    else:
        p.require(args.input and args.input_sha256 and args.output and not args.plan, 'Explicit input/SHA/new output required')
        package = bundle(p, dict(path=args.input, sha256=args.input_sha256))
    source = source_proof(p, args, package['identity']['run_id'])
    lock = Path(package['root'])/'.protected-compact.lock'
    fd = os.open(lock, os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd); p.require(stat.S_ISREG(st.st_mode) and st.st_uid == 501 and st.st_nlink == 1, 'Local lease differs')
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode == 'compress': result = compress(p,c,args,package,source)
        elif args.mode == 'prepare': result = prepare(p,args,package,source)
        else: result = execute(p,t,args,source)
        print(json.dumps(dict(accepted=True,mode=args.mode,result=result),sort_keys=True))
    finally: os.close(fd)


if __name__ == '__main__': main()
