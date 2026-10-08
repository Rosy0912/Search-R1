"""Prepare local NOT_READY assets/package only. No real production inputs inferred."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from deployment_contract_v1 import *
HERE=Path(__file__).resolve().parent
LOCAL=HERE.parents[2]
OUT=LOCAL/'outputs/gigpo-e1-policy-safety'
RUNTIME=HERE.with_name('formal-policy-safety-v4')
CONTROL_NAMES={'deployment_contract_v1.py','freeze_ready_v1.py','install_package_v1.py','prepare_candidate_v1.py','submit_once_v1.py','verify_physical_v1.py','formal_v1.sbatch'}
FORMAL_PROOFS=('FORMAL-PROTECTION-DRAFT-IMPLEMENTATION-v4.json','FORMAL-PROTECTION-DRAFT-INDEPENDENT-REVIEW-v4.json','FORMAL-PROTECTION-DRAFT-DIFF-v4.patch','FORMAL-PROTECTION-DRAFT-CPU-v4.json','FORMAL-PROTECTION-DRAFT-CPU-v4.log','FORMAL-PROTECTION-INTERFACE-v4.json','FORMAL-PROTECTION-DRAFT-BINDING-NOT-READY-v4.json')


def existing(name,expected):
    path=OUT/name;require(sha(path)==expected,'Existing original changed: '+name)
    return json.loads(path.read_text())


def remote_record(local):
    item=record(local);item['path']=str(ROOT/'logs'/Path(local).name);return item


def preview_config(candidate,directory):
    """Resolved schema only; null production inputs remain null and cannot start."""
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    q=candidate['quality_selection_control']
    env=dict(GIGPO_E1_ROOT=str(ROOT),GIGPO_E1_RUN=str(ROOT/'runs/__ACTUAL_FORMAL_RUN__'),AGENT_ROOT=str(ROOT),
        SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root',SEARCH_R1_BASELINE_OUTPUT=str(ROOT/'runs/__ACTUAL_FORMAL_RUN__'),
        GIGPO_QUALITY_GATE='30',GIGPO_QUALITY_CHECKPOINT=candidate['restore_checkpoint'],GIGPO_QUALITY_MANIFEST_SHA256=candidate['restore_manifest_sha256'],
        GIGPO_QUALITY_SELECTION=q['selection_path'],GIGPO_QUALITY_SELECTION_SHA256=q['selection_sha256'],
        GIGPO_QUALITY_REVIEW=q['selection_review_path'],GIGPO_QUALITY_REVIEW_SHA256=q['selection_review_sha256'])
    previous={k:os.environ.get(k) for k in env};os.environ.update(env)
    try:
        with initialize_config_dir(config_dir=str(directory),version_base=None):config=compose(config_name='ppo_gigpo_instruct300_quality10_safe')
        config.actor_rollout_ref.actor.policy_safety=OmegaConf.create(candidate['policy_safety'])
        OmegaConf.resolve(config)
        return OmegaConf.to_container(config,resolve=True)
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def draft():
    parent=existing('CALIBRATION-CANDIDATE-52BEFE9-v9.json',PARENT_CANDIDATE_SHA)
    ready=existing('policy-safety-calibration-52befe9-reviewed-binding-v9.json',PARENT_READY_SHA)
    manifest=existing('FORMAL-PROTECTION-DRAFT-IMPLEMENTATION-v4.json',RUNTIME_MANIFEST_SHA)
    existing('FORMAL-PROTECTION-DRAFT-INDEPENDENT-REVIEW-v4.json',RUNTIME_REVIEW_SHA)
    local_inputs={name:OUT/name for name in FORMAL_PROOFS}
    for row in manifest['files']:
        original=Path(row['path']);require(record(original)==row,'Frozen formalv4 byte changed')
    runtime=[dict(row,path=str(HELPERS/Path(row['path']).name)) for row in manifest['files']]
    controls=[dict(record(HERE/name),path=str(CONTROLS/name)) for name in sorted(CONTROL_NAMES)]
    removed={r['path'] for r in parent['runtime_helpers']}
    removed|={r['path'] for r in parent['files'] if Path(r['path']).parent==Path(parent['controls'])}
    inherited=[r for r in parent['files'] if r['path'] not in removed];require(len(inherited)==42327,'Base closure changed')
    evidence=list(parent['evidence'])
    def add(item):
        old=[r for r in evidence if r['path']==item['path']]
        require(not old or old==[item],'Conflicting existing proof SHA/bytes')
        if not old:evidence.append(item)
    for path in local_inputs.values():add(remote_record(path))
    for name in ('CALIBRATION-CANDIDATE-52BEFE9-v9.json','policy-safety-calibration-52befe9-reviewed-binding-v9.json'):
        add(remote_record(OUT/name))
    # Native acceptance already exists. Its original paths are preserved, not
    # normalized/replayed; extract only known byte inventory from downloadedraw.
    native=dict(ready['source_native_acceptance'],native_helpers=str(ROOT/'tools/policy-safety-native-52befe9-v2'))
    raw=OUT/'policy-safety-native3267-terminal-raw-v4.json.gz'
    require(sha(raw)==native['physical_payload_sha256'],'Existing native physical original changed')
    physical=json.loads(gzip.decompress(raw.read_bytes()))
    by_path={r['path']:{k:r[k] for k in ('path','bytes','sha256')} for r in physical['originals']}
    for key in ('result','native_binding','terminal'):
        item=by_path[native[key+'_path']];require(item['sha256']==native[key+'_sha256'],'Native inventory SHA mismatch');add(item)
    add(dict(remote_record(raw),path=native['physical_payload_path']))
    native_review=OUT/'NATIVE3267-INDEPENDENT-REVIEW-v2.json';require(sha(native_review)==native['review_sha256'],'Existing native review changed')
    add(dict(remote_record(native_review),path=native['review_path']))
    c=json.loads((OUT/'FORMAL-PROTECTION-DRAFT-BINDING-NOT-READY-v4.json').read_text())
    c.update(status='CANDIDATE_NOT_READY',scope=SCOPE,code_commit=SOURCE,repo=str(REPO),helpers=str(HELPERS),controls=str(CONTROLS),
        node='h100-8',files=inherited+runtime+controls,evidence=evidence,runtime_helpers=runtime,control_helpers=controls,inherited_file_count=len(inherited),
        inherited_file_records_sha256=hashlib.sha256(json.dumps(inherited,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        runtime_manifest=remote_record(OUT/FORMAL_PROOFS[0]),runtime_static_review=remote_record(OUT/FORMAL_PROOFS[1]),
        inherited_candidate=remote_record(OUT/'CALIBRATION-CANDIDATE-52BEFE9-v9.json'),
        inherited_launch_READY_not_actual_calibration_acceptance=remote_record(OUT/'policy-safety-calibration-52befe9-reviewed-binding-v9.json'),
        inherited_physical=dict(parent['inherited_physical'],bytes=(OUT/'independent-policy-safety-native-physical-v2.json').stat().st_size),guard_identity=native['guard_identity'],native_acceptance=native,native_status='PASS',
        fresh_model_revision=parent['fresh_model_revision'],common_manifest_sha256=parent['common_manifest_sha256'],
        quality_selection_control=parent['quality_selection_control'],restore_checkpoint=parent['restore_checkpoint'],restore_manifest_sha256=parent['restore_manifest_sha256'],restore_step=25,
        formal_budget=BUDGET,expected_config_preview=None,training_submitted=False,extra_AB_diagnostic_Adam_required=0,
        first_formal_hook_acceptance_stage='SAME_PROCESS26_TO30_BEFORE31',runtime_full_SHA_before_GPU_required=True,
        preparation_only=True,actual_source52_native_acceptance_preserved=True,actual_v9_calibration_acceptance=None,
        inspected_actual_tokenizer_identity_not_probability_acceptance=ready['actual_tokenizer_identity'])
    # These fields are intentionally unresolved. Do not turn observed PAD/EOS or
    # a static helper review into independently accepted sampler probabilities.
    require(sha(OUT/'independent-policy-safety-native-physical-v2.json')==c['inherited_physical']['sha256'],'Inherited native physical original changed')
    add(c['inherited_physical'])
    require(c['production_artifact'] is None and c['policy_safety']['target_old_policy_kl'] is None
        and c['policy_safety']['negative_tail_limit'] is None and c['probability_identity']['tolerances'] is None,
        'Draft was populated with guessed production evidence')
    return c,local_inputs


def archive(path,sources):
    with Path(path).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz:
        with tarfile.open(fileobj=gz,mode='w') as tar:
            for name,local in sorted(sources.items()):
                payload=Path(local).read_bytes();info=tarfile.TarInfo(name)
                info.size,info.mtime,info.uid,info.gid,info.mode=len(payload),0,0,0,0o644
                tar.addfile(info,io.BytesIO(payload))


if __name__=='__main__':
    # Current preparation never freezes production READY; all real inputs null.
    c,local=draft();target=OUT/CANDIDATE.name;write_once(target,c)
    sources={'runtime/'+Path(r['path']).name:Path(r['path']) for r in json.loads((OUT/FORMAL_PROOFS[0]).read_text())['files']}
    sources.update({'control/'+name:HERE/name for name in CONTROL_NAMES})
    sources.update({'proof/'+name:path for name,path in local.items()});sources['candidate/'+CANDIDATE.name]=target
    package=HERE/'policy-safety-formal-52befe9-v1-draft-package.tar.gz';archive(package,sources)
    # The install plan lists exact runtime/control/proof bytes; actual install
    # still requires a separate exact installer review. No target is accessed.
    planned=[]
    by_file={r['path']:r for r in c['files']};by_evidence={r['path']:r for r in c['evidence']}
    for name,path in sorted(sources.items()):
        category,filename=name.split('/',1)
        destination=(HELPERS/filename if category=='runtime' else CONTROLS/filename if category=='control' else ROOT/'logs'/filename)
        item=record(path);item.update(path=str(destination),member=name,reuse_same_bytes_allowed=category in ('proof','candidate'))
        if category in ('runtime','control'):require(by_file[str(destination)]=={k:item[k] for k in ('path','bytes','sha256')},'Planned new byte closure differs')
        elif category=='proof':require(by_evidence[str(destination)]=={k:item[k] for k in ('path','bytes','sha256')},'Planned proof closure differs')
        planned.append(item)
    plan=dict(status='DRAFT_IMMUTABLE_INSTALL_PLAN_NOT_PRODUCTION_READY',scope=SCOPE,source_commit=SOURCE,package=record(package),candidate=record(target),candidate_target=str(CANDIDATE),
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA,runtime_static_review_sha256=RUNTIME_REVIEW_SHA,
        members=planned,required_existing_evidence=[r for r in c['evidence'] if r['path'] not in {x['path'] for x in planned}],
        actual_v9_calibration_acceptance=None,probability_tolerances=None,production_thresholds=None,
        ready=False,submitted=False,target_accessed=False)
    plan_path=OUT/'FORMAL-PROTECTION-INSTALL-PLAN-v1.json';write_once(plan_path,plan)
    print(json.dumps(dict(candidate=record(target),package=record(package),plan=record(plan_path),ready=False,submitted=False)))
