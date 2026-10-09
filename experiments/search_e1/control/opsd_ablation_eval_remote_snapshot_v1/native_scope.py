"""Closed native evaluator scope; V3 native backup contract plus common zero.

Only scope naming/zero role and its FP16 cache exclusion are extended.
No model/scoring/process/resource execution is introduced.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

sys.dont_write_bytecode=True
N=Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
STUDY_ID='reverse-seven-a1dacc9d8c77451c8b04140e8e355777'
EVAL_ROOT='eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851'
DRIVER_ROOT='eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851'
VARIANTS=('zero','C0','C1','M','R','W','P','J')
GROUPS=('C0','C1','M','R','W','P','J')
DEPENDENCY_SHA='d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
NODE_DEPENDENCY=N/'source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py'


def require(value,message):
    if not value:raise ValueError(message)


def primitives(path):
    path=Path(path);require(path.is_absolute() and '..' not in path.parts,'Absolute dependency required')
    p=Path(path.anchor)
    for part in path.parts[1:]:
        p/=part;require(not p.is_symlink(),'Dependency symlink refused')
    fields=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before=fields(path.stat());raw=path.read_bytes()
    require(fields(path.stat())==before and hashlib.sha256(raw).hexdigest()==DEPENDENCY_SHA,'Published V2 dependency differs')
    ns=dict(__name__='_closed_native_primitives',__file__=str(path))
    exec(compile(raw,str(path),'exec'),ns);return ns


def versions(variant):
    require(variant in VARIANTS,'Group outside this study')
    return ['bf16_zero','fp16_zero'] if variant=='zero' else ['raw10_bf16','raw10_fp16'] if variant in ('P','J') else ['raw10_bf16']


def roots(variant):
    return N/EVAL_ROOT/variant/'runs',N/DRIVER_ROOT/'runs'/('common-zero' if variant=='zero' else '%02d-%s'%(GROUPS.index(variant),variant))/'evaluation'


def same_ref(ref,proof):
    require(ref['path']==proof['path'] and ref['sha256']==proof['sha256'],'Immutable reference differs')
    if 'bytes' in ref:require(ref['bytes']==proof['bytes'],'Immutable byte count differs')


def scope_shape(scope):
    require(scope.get('schema')=='opsd-ablation-closed-native-scope-v1' and scope.get('accepted') is True
            and scope['study_id']==STUDY_ID and scope['variant'] in VARIANTS
            and re.fullmatch('[0-9a-f]{32}',scope['run_id']),'Explicit closed study/group/run scope required')
    variant=scope['variant'];S,T=roots(variant)
    expected={'config':T.parent/'evaluation-config.json','summary':S/'summary.json',
              'controller':S/'controller-complete.json','accepted':T/'result-accepted.json',
              'parent_exit':T/'actual-child-exit.json','parent_release':T/'actual-child-release.json'}
    require(set(scope['pins'])==set(expected),'Exact config/controller/summary/parent closure required')
    for name,path in expected.items():
        ref=scope['pins'][name]
        require(ref['path']==str(path) and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'Pinned path/SHA outside closed roots')
    require(scope['versions']==versions(variant) and set(scope['phases'])==set(scope['versions']),'Exact one/two native phases required')
    for version in scope['versions']:
        require(set(scope['phases'][version])=={'phase','terminal'},'Both phase and actual wait terminal required')
        for key,name in [('phase','phase-complete.json'),('terminal','actual-terminal.json')]:
            ref=scope['phases'][version][key]
            require(ref['path']==str(S/version/name) and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'Phase pin differs')
    owner=scope['parent_identity']
    require(type(owner['pid']) is int and owner['pid']>0 and owner['uid']==1005
            and type(owner['starttick']) is int and owner['starttick']>0,'Explicit actual parent identity required')
    return scope


def select_ref(refs,path):
    matches=[r for r in refs if r['path']==str(path)]
    require(len(matches)==1,'Original summary must bind exactly one artifact')
    return matches[0]


def closure(b,scope):
    scope_shape(scope);S,T=roots(scope['variant']);run=scope['run_id'];value={};proofs={}
    for name,ref in scope['pins'].items():
        value[name],proofs[name]=b['checked_json'](ref['path'],ref['sha256']);same_ref(ref,proofs[name])
    cfg=value['config'];summary=value['summary'];controller=value['controller'];accepted=value['accepted']
    require(cfg['schema']=='opsd-ablation-native512-eval-config-v1' and cfg['run_id']==run
            and cfg['study_id']==STUDY_ID and cfg['variant']==('C0' if scope['variant']=='zero' else scope['variant']) and cfg['evaluation_role']==('common_zero' if scope['variant']=='zero' else 'group')
            and cfg['paths']['output_root']==str(S) and cfg['paths']['base']==str(S.parent)
            and cfg['paths']['helper_root']==str(N/'source/eval-stats-v2-72e1e12/opsd_ablation_eval_v1')
            and cfg['scientific']['versions']==scope['versions']
            and cfg['scientific']['validation_questions']==512
            and cfg['scientific']['total_primary_slots']==512*len(scope['versions']), 'Actual native config identity differs')
    require(controller['accepted'] is True and controller['run_id']==run and controller['release']['accepted'] is True
            and controller['primary_slots']==512*len(scope['versions']), 'Native controller not closed')
    same_ref(controller['config'],proofs['config']);same_ref(controller['summary'],proofs['summary'])
    require(accepted['accepted'] is True,'Original driver did not accept this native result')
    same_ref(accepted['config'],proofs['config']);same_ref(accepted['controller_completion'],proofs['controller'])
    same_ref(accepted['summary'],proofs['summary'])
    require(summary['schema']=='opsd-ablation-native512-summary-v1' and summary['accepted'] is True
            and summary['run_id']==run and summary['study_id']==STUDY_ID and summary['variant']==cfg['variant']
            and summary['evaluation_role']==cfg['evaluation_role'] and summary['primary_slots']==512*len(scope['versions'])
            and set(summary['models'])==set(scope['versions']), 'Closed summary scope differs')
    parent=value['parent_exit'];owner=parent['identity']
    require(parent['run_id']==run and parent['actual_wait_returncode']==0
            and parent['signal_termination_not_exit0'] is False
            and all(owner[k]==scope['parent_identity'][k] for k in ('pid','uid','starttick'))
            and owner['run_id']==run,'Actual parent wait/identity differs')
    command=[cfg['paths']['python'],'-B',str(Path(cfg['paths']['helper_root'])/'controller.py'),
             '--config',proofs['config']['path'],'--config-sha256',proofs['config']['sha256']]
    require(parent['command']==command and owner['argv']==command,'Actual native parent entry/config differs')
    require(value['parent_release']['accepted'] is True,'Original parent release not accepted')
    phase_proofs={};worker_proofs=[];by_version={};common_rows=None
    for version in scope['versions']:
        pins=scope['phases'][version]
        phase,pp=b['checked_json'](pins['phase']['path'],pins['phase']['sha256'])
        terminal,tp=b['checked_json'](pins['terminal']['path'],pins['terminal']['sha256'])
        phase_proofs[version]=dict(phase=pp,terminal=tp)
        require(phase['accepted'] is True and phase['release']['accepted'] is True
                and phase['run_id']==terminal['run_id']==run and phase['version']==terminal['version']==version
                and phase['actual_worker_count']==phase['expected_worker_count']==len(phase['child_exit_evidence'])==8
                and phase['actual_exit_codes_never_inferred'] is True and terminal['all_actual_wait_zero'] is True
                and terminal['child_exit_evidence']==phase['child_exit_evidence'],'Native phase/actual waits not closed')
        ids=[x['identity'] for x in phase['child_exit_evidence']]
        require(len({i['pid'] for i in ids})==8 and len({(i['pid'],i['starttick']) for i in ids})==8
                and all(x['actual_wait_returncode']==0 and x['identity']['uid']==1005
                        and x['identity']['run_id']==run and type(x['identity']['starttick']) is int
                        and x['identity']['ppid']==owner['pid'] for x in phase['child_exit_evidence']), 'Eight unique actual owned rc0 workers required')
        same_ref(select_ref(summary['inputs'],S/version/'phase-complete.json'),pp)
        model=summary['models'][version]
        require(model['overall']['questions']==512 and all(model['datasets'][d]['questions']==256 for d in ('nq','hotpotqa')),
                'Original512/256+256 summary counts differ')
        records={};hits={'nq':0,'hotpotqa':0};datasets={'nq':0,'hotpotqa':0}
        for rank in range(8):
            stage=S/version/('worker-%d'%rank)
            ref=select_ref(summary['inputs'],stage/'complete.json')
            worker,wp=b['checked_json'](ref['path'],ref['sha256']);same_ref(ref,wp)
            identity=worker['process'];expected=ids[rank]
            require(all(identity[k]==expected[k] for k in ('pid','uid','starttick','ppid','run_id')),
                    'Complete worker differs from actual waited identity')
            require(worker['accepted'] is True and worker['status']=='COMPLETE' and worker['primary_rows']==64
                    and worker['rank']==rank and worker['physical_gpu']==rank and worker['rank_seed']==42
                    and worker['version']==version and worker['run_id']==run and worker['study_id']==STUDY_ID
                    and worker['variant']==cfg['variant'] and worker['privileged_reference'] is False
                    and worker['no_training'] is True, 'Complete64 original worker identity differs')
            require(worker['gpu_uuid']==cfg['owner']['gpu_uuids'][rank], 'Recorded physical GPU identity differs')
            same_ref(worker['config'],proofs['config'])
            require(worker['source']['manifest']['sha256']==summary['identities']['source_model_identity']
                    and worker['science']['manifest']['sha256']==summary['identities']['runtime_identity'], 'Worker source/runtime differs')
            resultref=select_ref(summary['inputs'],stage/'results.jsonl');same_ref(worker['results'],resultref)
            raw,rp=b['fileproof'](resultref['path'],resultref,retain=True)
            rows=[json.loads(line) for line in raw.splitlines() if line]
            require(len(rows)==64,'Original worker result row count differs')
            for slot,row in enumerate(rows):
                index=rank+slot*8
                require(row['status']=='complete' and row['primary'] is True and row['run_id']==run
                        and row['version']==version and row['rank']==rank and row['slot']==slot
                        and row['validation_index']==index and index not in records
                        and row['dataset'] in datasets and row['protocol_em'] in (0,1), 'Original primary slot identity differs')
                records[index]=(row['question_id'],row['row_semantic_sha256'],row['dataset'])
                datasets[row['dataset']]+=1;hits[row['dataset']]+=int(row['protocol_em'])
            worker_proofs.append(dict(version=version,rank=rank,complete=wp,results=rp,actual_wait_identity=expected))
        require(set(records)==set(range(512)) and len({r[0] for r in records.values()})==512,
                'Duplicate/missing original512 question identities')
        require(datasets=={'nq':256,'hotpotqa':256} and sum(hits.values())==model['overall']['strict_hits']
                and all(hits[d]==model['datasets'][d]['strict_hits'] for d in hits),'Original recorded row/summary counts differ')
        if common_rows is None:common_rows=records
        else:require(records==common_rows,'Two phases did not use the same original512 cohort')
        by_version[version]=dict(primary_slots=512,strict_hits=sum(hits.values()),dataset_hits=hits,actual_worker_wait_codes=[0]*8)
    return dict(config=proofs['config'],source_receipts=proofs,phases=phase_proofs,workers=worker_proofs,
                actual_native_parent_wait_returncode=0,primary_slots=512*len(scope['versions']),models=by_version,
                scope='Original closed native assets/recorded identity only; no new inference/scoring/current-resource or whole-study acceptance')


def enumerate_files(b,scope):
    S,T=roots(scope['variant']);paths=[Path(scope['pins']['config']['path'])]
    excluded={S/version/'cpu-merge/model' for version in scope['versions'] if version in ('raw10_fp16','fp16_zero')}
    for tree in (S,T):
        b['checked_path'](tree,directory=True)
        for parent,dirs,names in os.walk(tree,topdown=True,followlinks=False):
            parent=Path(parent);b['checked_path'](parent,directory=True)
            for name in list(dirs):
                path=parent/name;b['checked_path'](path,directory=True)
                if path in excluded:dirs.remove(name)
                else:require(name not in ('model','models','cache','checkpoints','merged-model-cache'),'Unrecognized model/cache subtree HOLD')
            for name in names:paths.append(b['checked_path'](parent/name))
    paths=sorted(paths,key=lambda p:p.relative_to(N).as_posix())
    require(paths and len(paths)==len(set(paths)),'Duplicate closed source file')
    for path in paths:
        name=b['safe_name'](path.relative_to(N).as_posix());tarfile.TarInfo(name).tobuf(format=tarfile.USTAR_FORMAT)
    return paths,[str(p.relative_to(N)) for p in sorted(excluded)]


def inventory(b,scope,sha):
    closures=closure(b,scope);paths,excluded=enumerate_files(b,scope)
    files=[dict(relative=p.relative_to(N).as_posix(),**b['fileproof'](p)) for p in paths]
    for row in files:b['stable'](row)
    require(closures==closure(b,scope) and (paths,excluded)==enumerate_files(b,scope),'Closed source changed during inventory')
    return dict(schema='opsd-ablation-closed-native-inventory-v1',accepted=True,source=str(N),scope_sha256=sha,
                run_id=scope['run_id'],variant=scope['variant'],versions=scope['versions'],config=closures['config'],closures=closures,
                files=files,expected_file_count=len(files),bytes=sum(r['bytes'] for r in files),excluded_subtrees=excluded,
                primary_slots=512*len(scope['versions']),node_originals_retained=True,backup_is_training_dependency=False,
                observed_unix=time.time())


def validate_inventory(b,scope,sha,value):
    scope_shape(scope);S,T=roots(scope['variant']);cfg=Path(scope['pins']['config']['path'])
    require(value.get('schema')=='opsd-ablation-closed-native-inventory-v1' and value.get('accepted') is True
            and value['scope_sha256']==sha and value['source']==str(N) and value['run_id']==scope['run_id']
            and value['variant']==scope['variant'] and value['versions']==scope['versions']
            and value['primary_slots']==512*len(scope['versions']),'Pinned native inventory differs')
    same_ref(value['config'],scope['pins']['config'])
    files=value['files'];require(len(files)==value['expected_file_count'] and files
        and len({r['relative'] for r in files})==len(files) and len({r['path'] for r in files})==len(files)
        and len({(r['device'],r['inode']) for r in files})==len(files),'Inventory aliases')
    for row in files:
        path=Path(row['path']);name=b['safe_name'](row['relative'])
        require(path==N/name and (path==cfg or path.is_relative_to(S) or path.is_relative_to(T))
                and not any(path.is_relative_to(S/v/'cpu-merge/model') for v in ('raw10_fp16','fp16_zero'))
                and all(k in row for k in (*b['META'],'sha256')) and b['SHA'].fullmatch(row['sha256'])
                and type(row['bytes']) is int and row['bytes']>=0,'Inventory outside closed native scope')
    require(sum(r['bytes'] for r in files)==value['bytes'],'Inventory byte total differs')
    mapped={r['path']:r for r in files}
    pins=list(scope['pins'].values())+[r for phase in scope['phases'].values() for r in phase.values()]
    for ref in pins:
        require(ref['path'] in mapped,'Pinned closure omitted');same_ref(ref,mapped[ref['path']])
    return {r['relative']:r for r in files}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('inventory','stream','after'))
    p.add_argument('--scope-base64',required=True);p.add_argument('--scope-sha256',required=True);a=p.parse_args()
    require(os.getuid()==os.geteuid()==1005 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'Read-only UID1005/CVD empty/no Torch required')
    raw_scope=base64.b64decode(a.scope_base64,validate=True)
    require(hashlib.sha256(raw_scope).hexdigest()==a.scope_sha256,'Inline original scope SHA differs')
    scope=json.loads(raw_scope);b=primitives(NODE_DEPENDENCY)
    if a.mode=='inventory':value=inventory(b,scope,a.scope_sha256)
    else:
        raw=sys.stdin.buffer.read();value=json.loads(raw);validate_inventory(b,scope,a.scope_sha256,value)
        fresh=closure(b,scope);paths,excluded=enumerate_files(b,scope)
        require(value['closures']==fresh and [r['path'] for r in value['files']]==[str(p) for p in paths]
                and value['excluded_subtrees']==excluded,'Closed inventory/source after differs')
        for row in value['files']:b['stable'](row)
        if a.mode=='stream':b['stream_archive'](value,sys.stdout.buffer);return
        value=dict(schema='opsd-ablation-closed-native-source-after-v1',accepted=True,source=str(N),
            scope_sha256=a.scope_sha256,inventory_sha256=hashlib.sha256(raw).hexdigest(),run_id=scope['run_id'],
            versions=scope['versions'],config=fresh['config'],closures=fresh,files=len(paths),
            all_source_metadata_stable=True,all_closed_file_set_unchanged=True,observed_unix=time.time())
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
