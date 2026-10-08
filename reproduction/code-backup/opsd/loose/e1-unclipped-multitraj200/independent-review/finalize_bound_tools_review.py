"""Independent exact local binding verification; never invokes runtime entries."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from datetime import datetime, timezone

H = Path(__file__).resolve().parents[1]
R = H.parent / 'opsd-unclipped-multitraj200'
SOURCE = '667c68e0f91b4bc439324d0e095ce13df10da559'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))


def raw(path):
    assert path.is_absolute() and path.resolve(strict=True) == path and path.stat().st_uid == os.getuid()
    return path.read_bytes()


def record(path):
    value = raw(path)
    return dict(path=str(path), sha256=sha(value), bytes=len(value))


def main():
    bp = H / 'runtime-tools-bound-667c68e0f91b.json'
    binding_raw = raw(bp)
    binding = unique(binding_raw)
    software = unique(raw(H / 'INDEPENDENT_SOFTWARE_REVIEW.json'))
    assert software['status'] == 'PASS_INDEPENDENT_SOURCE_SOFTWARE_ONLY'
    assert binding['source_commit'] == software['source_commit'] == SOURCE
    assert binding['source_files'] == software['reviewed_source_files']
    assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip() == SOURCE
    assert subprocess.check_output(['git','-C',str(R),'status','--porcelain'],text=True) == ''
    for name, pin in binding['source_files'].items():
        assert sha(raw(R / name)) == pin
    review = unique(raw(H / 'independent-review/SOURCE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.json'))
    replacements = {'BLOCKED_UNBOUND_SOURCE_COMMIT': SOURCE}
    candidate = {'formal': H / 'formal-tools-candidate', 'native': H / 'first5-native-candidate',
                 'first5': H / 'first5-receipt-candidate'}
    assert binding['scope'] == 'LOCAL_BOUND_TOOL_PREPARATION_NO_ACTUAL_JOB_ID_NO_GPU_NO_SIGNATURE'
    assert binding['remote_repo'] == binding['remote_root'] + '/repo-' + SOURCE[:12]
    bundles = {}
    inodes = set()
    for role in ('native', 'formal', 'first5'):
        if role == 'first5':
            replacements.update({'BLOCKED_NATIVE_TOOLS_PATH': binding['native']['remote_directory'],
                                 'BLOCKED_NATIVE_INSPECTOR_SHA': binding['native']['files']['inspect_payloads.py'],
                                 'BLOCKED_NATIVE_MANIFEST_SHA': binding['native']['manifest_sha256'],
                                 'BLOCKED_GATE_SOURCE_SHA': binding['source_files']['search_r1/baselines/checkpoint_gate.py']})
        row = binding[role]
        directory = Path(row['local_directory'])
        assert directory.parent == H and directory.resolve(strict=True) == directory
        assert stat.S_IMODE(directory.stat().st_mode) == 0o555
        assert row['remote_directory'] == binding['remote_root'] + '/tools/' + directory.name
        assert set(p.name for p in directory.iterdir()) == set(row['files']) | {row['manifest_name']}
        mp = directory / row['manifest_name']
        mraw = raw(mp)
        assert sha(mraw) == row['manifest_sha256']
        manifest = unique(mraw)
        assert type(manifest['version']) is int and manifest['version'] == 1 and manifest['source_commit'] == SOURCE
        assert manifest['tools'] == row['files']
        if role in ('formal', 'first5'):
            assert manifest['source_files'] == binding['source_files']
        if role == 'native':
            assert manifest['profile'] == 'multitraj200'
            assert type(manifest['horizon']) is int and manifest['horizon'] == 200
            assert type(manifest['warmup_steps']) is int and manifest['warmup_steps'] == 57
            assert type(manifest['saved_adam_steps']) is int and manifest['saved_adam_steps'] == 50
            assert manifest['saved_lr'] == 1e-6 * (5 / 57)
        if role == 'formal':
            assert manifest['common_data_manifest_sha256'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
            assert manifest['official_model_manifest_sha256'] == '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
            assert manifest['retrieval_asset_manifest_sha256'] == '87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
            assert manifest['native_import_closure_sha256'] == manifest['public_key_sha256'] == 'EXTERNAL_ENV_REQUIRED_NOT_GUESSED'
        if role == 'first5':
            assert manifest['native_payload_inspector_sha256'] == binding['native']['files']['inspect_payloads.py']
            assert manifest['native_payload_tools_manifest_sha256'] == binding['native']['manifest_sha256']
        member_records = {}
        for name in (*row['files'], row['manifest_name']):
            path = directory / name
            value = raw(path)
            assert stat.S_IMODE(path.stat().st_mode) == 0o444
            inode = (path.stat().st_dev, path.stat().st_ino)
            assert inode not in inodes
            inodes.add(inode)
            member_records[name] = record(path)
            if name == row['manifest_name']:
                continue
            assert sha(value) == row['files'][name]
            original = raw(candidate[role] / name)
            expected_pin = review['reviewed_candidate_tool_files'].get(str((candidate[role]/name).relative_to(H)))
            if expected_pin is not None:
                assert sha(original) == expected_pin
            else:
                assert role == 'native' and name == 'test_inspector.py'
                assert sha(original) == '9632f0a3b6fede523f3966512c15e8ff2995e00295c293597b095438fb3571e3'
            for token, target in replacements.items():
                original = original.replace(token.encode(), target.encode())
            assert value == original, (role, name, 'Unexpected non-binding logic change')
            assert b'BLOCKED_' not in value
            if name.endswith('.py'):
                compile(value, str(path), 'exec', dont_inherit=True)
        bundles[role] = {'manifest':record(mp),'member_records':member_records,'manifest_scope':manifest['scope'],
                         'exact_derivation_from_reviewed_candidate_with_only_binding_replacements':True,
                         'immutable_modes_0444_0555_inventory_and_distinct_inodes_validated':True}
    cp = H / 'control-tools-bound-667c68e0f91b.json'
    controls = unique(raw(cp))
    assert controls['source_commit'] == SOURCE and controls['binding_sha256'] == sha(binding_raw)
    ctl_replacements = {
        'BLOCKED_UNBOUND_SOURCE_COMMIT':SOURCE, 'BLOCKED_BOUND_SPEC_FILENAME':bp.name,
        'BLOCKED_NATIVE_INVENTORY_FILENAME':'native-source-inventory-'+SOURCE[:12]+'-ACTUAL.json',
        'BLOCKED_FORMAL_MANIFEST_SHA':binding['formal']['manifest_sha256'],
        'BLOCKED_RECEIPT_MANIFEST_SHA':binding['first5']['manifest_sha256'],
        'BLOCKED_GATE_SOURCE_SHA':binding['source_files']['search_r1/baselines/checkpoint_gate.py'],
        'BLOCKED_LOCAL_RECEIPT_MANIFEST':Path(binding['first5']['local_directory']).name+'/'+binding['first5']['manifest_name'],
        'BLOCKED_NATIVE_MANIFEST_RELATIVE':Path(binding['native']['local_directory']).name+'/'+binding['native']['manifest_name'],
        'BLOCKED_FORMAL_MANIFEST_RELATIVE':'tools/'+Path(binding['formal']['local_directory']).name+'/'+binding['formal']['manifest_name'],
    }
    control_records = {}
    for name, rec in controls['files'].items():
        path = Path(rec['path'])
        assert path == H / (Path(name).stem+'-bound-'+SOURCE[:12]+'.py')
        value = raw(path)
        assert sha(value) == rec['sha256']
        template = raw(H / name)
        for token, target in ctl_replacements.items():
            template = template.replace(token.encode(), target.encode())
        assert value == template and b'BLOCKED_' not in value
        compile(value, str(path), 'exec', dont_inherit=True)
        control_records[name] = dict(record(path), template=record(H / name), same_logic_only_binding_replacements=True)
    assert set(control_records) == {'deploy_native_cpu.py','deploy_tools.py','submit_reviewed.py','publish_gpu_init.py','publish_first5_release.py'}
    cpu_dir = H / 'independent-review/bound-tools-independent-cpu'
    cpu = unique(raw(cpu_dir / 'execution.original.json'))
    assert cpu['returncode'] == 0 and cpu['all_bundle_bytes_stable'] is True
    assert cpu['runtime_binding_sha256'] == sha(binding_raw)
    assert b'139 passed' in raw(cpu_dir / 'stdout.original')
    for name, pin in cpu['after_sha256'].items():
        assert sha(raw(Path(name))) == pin
    assert raw(bp) == binding_raw
    report = {'version':1,'status':'PASS_INDEPENDENT_SOURCE_BOUND_SOFTWARE_ONLY',
              'utc':datetime.now(timezone.utc).isoformat(),'source_commit':SOURCE,
              'runtime_binding_sha256':sha(binding_raw),'runtime_binding':record(bp),
              'source_software_review':record(H/'INDEPENDENT_SOFTWARE_REVIEW.json'),
              'reviewed_source_files':binding['source_files'],'reviewed_bundles':bundles,
              'control_binding':record(cp),'reviewed_control_files':control_records,
              'control_static_review':record(H/'CONTROL_TOOLS_STATIC_REVIEW_V2.json'),
              'independent_bound_modules_cpu_replay':{'test_count':139,'returncode':0,'byte_stability':True,
                  'scope':cpu['scope'],'originals':[record(cpu_dir/name) for name in ('execution.original.json','stdout.original','stderr.original')]},
              'remaining_proven_bound_software_blockers':[],
              'scope':'Current exact immutable formal/native/first5 source-bound tools and rendered local control scripts. Binding verification plus CPU fixture execution against exact bound modules only; no tool deployment, remote/native execution, real Slurm job, GPU, signing, actual first5, effects or200/final acceptance.',
              'required_actual_next':['Native CPU import closure from this committed source and actual native2.4 environment; external closure/public-key SHA supplied and verified at actual prepare',
                  'Actual owned single8H100 allocation/source/data/model/tool seven-role bindings and typed pre-GPU independent ready review before root-only signed initialization',
                  'Actual native8rank numerical/capacity probe, then realfresh5 saved-state/HF/raw-archive/resource/data/signed independent gate before fetch6',
                  'Root-only local private-key signing; neither software PASS nor fixtures permit signatures for guessed future job/PID/nonce/checkpoint'],
              'scope_limits':['The native scope reads stored checkpoint content; it does not prove new live training-process restore',
                  'No actual GPU capacity/equivalence, teacher diagnostic outputs or runtime closures have been accepted here',
                  'The shell submits only one exclusive h100-10 typed8H100 job through durable intent; unknown submit outcome retains originals and must not be automatically retried',
                  'Formal manifest external-required native/public placeholders are intentionally supplied through verified actual environment and copied into actual seven-role binding; they are not guessed native evidence']}
    target = H/'INDEPENDENT_BOUND_TOOLS_REVIEW.json'
    value = (json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with target.open('xb') as f:f.write(value)
    md = H/'INDEPENDENT_BOUND_TOOLS_REVIEW.md'
    with md.open('x') as f:
        f.write('PASS：当前实际源码绑定工具的软件独审通过。\n\n'+SOURCE+' 的formal/native/first5三份不可变manifest、全部16成员、19sourcepins、native50Adam/57warmup/200budget及五份rendered control文件均精确核对，实际bound模块139项CPU重放通过，文件前后未变。\n\n没有GPU、正式job、签名、真实首5或效果资格；实际原生closure和单节点8H100资源/数值/容量/首5原件仍须逐阶段核验。私钥只允许根代理在真实独审ready后本地签名。\n')
    print(json.dumps({'report':record(target),'markdown':record(md),'bundle_count':len(bundles),'members':len(inodes),'independent_bound_cpu_tests':139}))


if __name__ == '__main__':
    main()
