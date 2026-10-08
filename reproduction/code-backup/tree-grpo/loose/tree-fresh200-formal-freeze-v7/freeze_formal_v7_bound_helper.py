"""Local-only strict code freeze of an existing actual Formal V7 builder draft.

Never runs builder, Hydra, supervisor, remote tools, assets or scheduling.
Four actual byte-bound observation snapshots are required before any output.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import types

LOCAL_ROOT=Path(__file__).resolve().parents[2]
RUNTIME=LOCAL_ROOT/'work/tree-fresh200-train-runtime-v7'
HELPERS=LOCAL_ROOT/'work/tree-fresh200-formal-dispatch-v7'
COMMIT='53082e1667a6ab131a036ca2ef66e004989ed1a9'
RUNTIME_INVENTORY_SHA='42d225fb38510e9726107617e72e06b9deead8539acfa858998ce7830d5ca55e'
HELPER_COMMON_SHA='0f9cfc571eef4213bca06be071ad9955384624b3795b22afa5652a6cfed3a2c9'
HELPER_ENTRY_SHA='324ae15b76755c4f70b508a9c9e96f3645b1a5640030757713bba3a313ce9b79'
HELPER_REVIEW_SHA='e21ce57729ea6a2d71c07e4fa99af49c800eb5cfedbe6e0246fa3da2b499d113'  # Actual root fork:none V7 source review pin is still missing.
INPUT_NAMES=('actual_calibration_ready','actual_calibration_acceptance','threshold_acceptance','source_context_free_review')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary_bytes(path,expected=None):
    path=Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), 'Ordinary no-symlink input required: '+str(path))
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read();after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Input changed while read')
    require(expected is None or sha(raw)==expected, 'Input SHA differs: '+str(path))
    return raw


def pinned_module(path,expected,name):
    raw=ordinary_bytes(path,expected)
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def require_four_nonnull(values):
    require(type(values) is dict and set(values)==set(INPUT_NAMES), 'Four actual observation descriptors required')
    for name in INPUT_NAMES:
        value=values[name]
        require(type(value) is dict and set(value)=={'path','sha256'}, 'Actual non-null descriptor required: '+name)


def regular_inventory(directory):
    require(directory.is_absolute() and stat.S_ISDIR(directory.lstat().st_mode), 'Ordinary absolute draft directory required')
    result=set()
    for current,dirs,files in os.walk(directory,followlinks=False):
        for name in dirs:
            require(stat.S_ISDIR((Path(current)/name).lstat().st_mode), 'Draft directory symlink/special file refused')
        for name in files:
            path=Path(current)/name
            require(stat.S_ISREG(path.lstat().st_mode), 'Draft symlink/special file refused')
            result.add(str(path.relative_to(directory)))
    return result


def filtered_manifest(original,common):
    """Reuse accepted strict runtime path predicate; preserve all identity fields."""
    manifest=copy.deepcopy(original)
    before=manifest['source_selection']
    selected={};removed={};files={}
    required_non_source=set(common.WRAPPER_FILES)|{'observed-inputs/source-interface-manifest.json'}|{
        'observed-inputs/'+name+'.json' for name in INPUT_NAMES}
    for name,digest in original['files'].items():
        if name.startswith('source/repo/'):
            relative=name[len('source/repo/'):]
            try:
                common.relative_file(name)
            except ValueError:
                removed[relative]=digest
                continue
            selected[relative]=digest
        else:
            require(name in required_non_source, 'Unexpected non-source draft payload: '+name)
            common.relative_file(name)
        files[name]=digest
    require(required_non_source.issubset(files), 'Full33 wrappers and five actual metadata snapshots required')
    require({name[len('source/repo/'):]:value for name,value in original['files'].items()
             if name.startswith('source/repo/')}==before['selected_files'], 'Builder selected provenance differs from actual inventory')
    excluded=dict(before['excluded_files'])
    require(not set(excluded).intersection(before['selected_files']), 'Builder selected/excluded provenance overlaps')
    excluded.update(removed)
    before.update(protocol='runtime_review_code_metadata_strict_common_scope_v2',
        allowed_source_root_files=sorted(common.SOURCE_ROOT_FILES),allowed_source_prefixes=['search_r1/','verl/'],
        selected_files=selected,selected_count=len(selected),excluded_files=excluded,excluded_count=len(excluded),
        not_a_full_repository_asset_copy=True)
    manifest.update(status='FROZEN_FORMAL_V7_CODE_ONLY_PENDING_EXACT_PACKET_REVIEW',
        scope='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW',files=files,
        actual_remote_executed=False,actual_gpu_executed=False,actual_slurm_submitted=False,
        actual_cpu_prepared=False,actual_ready=False,training_release=False)
    common.validate_manifest(manifest)
    return manifest,removed


def write_json(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n'
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return sha(raw)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('draft','draft-manifest-sha','output','runtime-archive','runtime-target'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-uncompressed-bytes',type=int,default=32*1024**2)
    args=parser.parse_args()
    common=pinned_module(HELPERS/'formal_cpu_common.py',HELPER_COMMON_SHA,'formal_cpu_common')
    common.valid_sha(HELPER_REVIEW_SHA)
    common.valid_sha(args.draft_manifest_sha)
    inventory=json.loads(ordinary_bytes(RUNTIME/'FILE_HASHES_V7.json',RUNTIME_INVENTORY_SHA))
    require(inventory['wrapper_files']==common.WRAPPER_HASHES, 'Current formal33 wrapper inventory differs')
    production=pinned_module(RUNTIME/'production_contract.py',common.WRAPPER_HASHES['production_contract.py'],'production_contract')
    draft=Path(args.draft);output=Path(args.output)
    require(output.is_absolute() and str(output).startswith('/Users/') and not output.exists()
            and not output.is_symlink(), 'New local no-clobber freeze output required')
    actual_files=regular_inventory(draft)
    manifest_raw=ordinary_bytes(draft/'manifest.json',args.draft_manifest_sha);original=json.loads(manifest_raw)
    require(original['status']=='LOCAL_TRAIN200_PACKAGE_MATERIALIZED_NOT_RELEASED'
            and original['scope']=='LOCAL_DRAFT_NOT_RELEASED' and original['code_commit']==COMMIT
            and original['stage']==common.STAGE, 'Only actual current-source Formal V7 builder draft may freeze')
    require_four_nonnull(original['production_inputs'])
    common.require_observation_descriptors(original['production_inputs'])
    require(actual_files==set(original['files'])|{'manifest.json'}, 'Draft inventory has missing/unlisted files')
    require(0<args.max_uncompressed_bytes<=256*1024**2, 'Explicit bounded code archive size required')
    runtime_archive=common.local_path(args.runtime_archive);runtime_target=common.local_path(args.runtime_target)
    require(runtime_archive.name=='CODE_PACKAGE.tar.gz'
            and runtime_archive.parent.parent.resolve()==(common.ROOT/'instruct300').resolve(), 'Explicit direct own preparation archive path required')
    require(runtime_target.parent.resolve()==(common.ROOT/'instruct300/candidates').resolve(), 'New own direct candidate target required')
    # Read only local actual snapshots; remote descriptor paths remain unchanged.
    values={name:production.read_bound(draft/'observed-inputs'/(name+'.json'),original['production_inputs'][name]['sha256'])
            for name in INPUT_NAMES}
    thresholds=production.observed_production_inputs(original,values)
    require(thresholds==original['bootstrap_thresholds']
            and {name:value[1] for name,value in values.items()}==original['production_input_sha256'], 'Actual observation/threshold bytes differ')
    spec=json.loads(ordinary_bytes(draft/'CONFIGURATION_SPEC.json',common.WRAPPER_HASHES['CONFIGURATION_SPEC.json']))
    require(production.configuration_overrides(spec,thresholds)==original['configuration_overrides'], 'Actual formal overrides differ')
    manifest,removed=filtered_manifest(original,common)
    payload={name:ordinary_bytes(draft/name,digest) for name,digest in manifest['files'].items()}
    helper_review_raw=ordinary_bytes(HELPERS/'INDEPENDENT_FORMAL_V7_HELPER_STATIC_REVIEW.json',HELPER_REVIEW_SHA)
    helper_review=json.loads(helper_review_raw)
    require(helper_review['status']=='FORMAL_V6_CPU_PREPARE_HELPERS_SOURCE_ONLY_ACCEPTED'
            and helper_review['fork_turns']=='none' and helper_review['training_release'] is False,
            'Accepted immutable formal helper source review required')
    helpers={'formal_cpu_common.py':ordinary_bytes(HELPERS/'formal_cpu_common.py',HELPER_COMMON_SHA),
        'prepare_formal_tree_fresh200_package.py':ordinary_bytes(HELPERS/'prepare_formal_tree_fresh200_package.py',HELPER_ENTRY_SHA),
        'HELPER_SOURCE_REVIEW.json':helper_review_raw}
    require(helper_review['helper_files']=={name:sha(raw) for name,raw in helpers.items() if name.endswith('.py')}, 'Helper review bytes differ')
    manifest_bytes=json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n'
    total=sum(len(raw) for raw in payload.values())+len(manifest_bytes)
    require(total<=args.max_uncompressed_bytes, 'Code-only uncompressed size exceeds reviewed bound')
    # All actual observation checks precede creating output. Partial later failure is retained.
    output.mkdir(parents=True)
    package=output/'package-code-only';package.mkdir()
    with (output/'DRAFT_MANIFEST_BEFORE_FREEZE.json').open('xb') as stream:
        stream.write(manifest_raw)
    for name,raw in payload.items():
        destination=package/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw)
    manifest_sha=write_json(package/'manifest.json',manifest)
    require(ordinary_bytes(package/'manifest.json',manifest_sha)==manifest_bytes, 'Written manifest bytes differ')
    archive_path=output/'CODE_PACKAGE.tar.gz'
    with archive_path.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w',format=tarfile.PAX_FORMAT) as bundle:
                for name,raw in sorted({**payload,'manifest.json':manifest_bytes}.items()):
                    member=tarfile.TarInfo(name);member.size=len(raw);member.mode=0o600;member.mtime=0
                    bundle.addfile(member,io.BytesIO(raw))
    archive_sha=sha(ordinary_bytes(archive_path))
    with tarfile.open(archive_path,'r:gz') as bundle:
        members=bundle.getmembers()
        require(all(member.isfile() and not member.issym() and not member.islnk() for member in members)
                and len(members)==len(payload)+1 and {member.name for member in members}==set(payload)|{'manifest.json'}, 'Final archive inventory differs')
    inputs=dict(status='FORMAL_V7_EXACT_CPU_PREPARATION_INPUTS_PENDING_PACKET_REVIEW',stage=common.STAGE,
        code_commit=COMMIT,manifest_sha256=manifest_sha,archive_sha256=archive_sha,
        target=str(runtime_target),archive=str(runtime_archive),max_uncompressed_bytes=args.max_uncompressed_bytes,
        production_inputs=manifest['production_inputs'],actual_cpu_prepared=False,training_release=False)
    inputs_sha=write_json(output/'PREPARATION_INPUTS.json',inputs)
    for name,raw in helpers.items():
        with (output/name).open('xb') as stream:
            stream.write(raw)
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1',
        str(common.ROOT/'env/bin/python'),'-B',str(runtime_archive.parent/'prepare_formal_tree_fresh200_package.py'),
        '--archive',str(runtime_archive),'--archive-sha',archive_sha,'--target',str(runtime_target),
        '--manifest-sha',manifest_sha,'--inputs',str(runtime_archive.parent/'PREPARATION_INPUTS.json'),
        '--inputs-sha',inputs_sha,'--helper-review',str(runtime_archive.parent/'HELPER_SOURCE_REVIEW.json'),
        '--helper-review-sha',HELPER_REVIEW_SHA]
    write_json(output/'CPU_PREPARATION_ARGV_PENDING_REVIEW.json',dict(argv=argv,executed=False,
        exact_packet_review_required=True,actual_remote_executed=False,training_release=False))
    write_json(output/'FREEZE_RECEIPT.json',dict(status='LOCAL_FORMAL_V7_CODE_ONLY_FROZEN_PENDING_EXACT_PACKET_REVIEW',
        freezer_source_sha256=sha(ordinary_bytes(Path(__file__))),draft=str(draft),draft_manifest_sha256=args.draft_manifest_sha,
        code_commit=COMMIT,manifest_sha256=manifest_sha,archive_sha256=archive_sha,preparation_inputs_sha256=inputs_sha,
        source_selected_count=manifest['source_selection']['selected_count'],removed_source_files=removed,
        uncompressed_code_bytes=total,wrapper_count=33,source_interface_count=20,
        production_input_sha256=manifest['production_input_sha256'],runtime_archive=str(runtime_archive),runtime_target=str(runtime_target),
        actual_local_code_freeze_executed=True,actual_builder_executed=False,actual_hydra_executed=False,
        actual_cpu_prepared=False,actual_remote_executed=False,actual_gpu_executed=False,actual_slurm_submitted=False,
        actual_official_assets_read=False,training_release=False,dispatch_release_created=False))
    print(json.dumps(dict(output=str(output),manifest_sha256=manifest_sha,archive_sha256=archive_sha,
        preparation_inputs_sha256=inputs_sha,actual_cpu_prepared=False,training_release=False)),flush=True)


if __name__=='__main__':
    main()
