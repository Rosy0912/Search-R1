"""Bind an entirely fresh run to the exact canonical shared data and official model."""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from reproduction.common8.verify_data import digest, probe_question_id, verify
from reproduction.common8.formal_contract import CONTRACT_KIND, TARGET_UPDATES, assert_scientific_run

REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
OFFICIAL_MANIFEST_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'


def derive_calibration(base):
    import numpy as np
    import pyarrow.parquet as pq
    proof=verify(base/'data')
    table=pq.read_table(base/'data/train.parquet')
    sources=table['data_source'].to_pylist()
    ids=[probe_question_id(x) for x in table['prompt'].to_pylist()]
    rng=np.random.RandomState(42)
    chosen=[]; seen=set()
    for source in ('nq','hotpotqa'):
        candidates=[i for i,s in enumerate(sources) if s==source]
        count=0
        for i in rng.permutation(candidates):
            i=int(i)
            if ids[i] in seen:continue
            chosen.append(i);seen.add(ids[i]);count+=1
            if count==150:break
        assert count==150
    chosen.sort()
    path=base/'data/calibration-300.parquet'
    assert not path.exists(), 'Existing derivative must not be silently replaced'
    pq.write_table(table.take(chosen),path)
    audit=dict(calibration_count=300,calibration_question_ids=[ids[i] for i in chosen],
        original_train_row_indices=chosen, original_train_sha256=proof['products']['train']['sha256'],
        semantics='train-derived probe fitting subset; every canonical training row retained')
    with (base/'data/audit.json').open('x') as f:json.dump(audit,f,indent=2)
    assert verify(base/'data')==proof, 'Canonical shared files changed during derivative creation'
    return audit


def verify_model(model, official_path):
    assert digest(official_path)==OFFICIAL_MANIFEST_SHA
    official=json.loads(official_path.read_text())
    assert official['repo']=='Qwen/Qwen2.5-3B-Instruct'
    assert official['revision']==REVISION
    index=json.loads((model/'model.safetensors.index.json').read_text())
    names=set(index['weight_map'].values())|{'model.safetensors.index.json','config.json',
        'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json'}
    assert len(names)==9
    records={r['file']:r for r in official['files']}
    hashes={}
    for name in sorted(names):
        record=records[name];path=model/name
        assert path.stat().st_size==record['bytes']
        actual=digest(path)
        assert actual==record['sha256']
        if record.get('expected_lfs_sha256'):
            assert actual==record['expected_lfs_sha256']
        else:
            content=path.read_bytes()
            git=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            assert git==record['git_blob_sha1']
        hashes[name]=actual
    config=json.loads((model/'config.json').read_text())
    assert config['model_type']=='qwen2' and config['hidden_size']==2048
    return hashes


def prepare(base,model,official_path,commit,*,execution_role='production',measurement_parent=None,measurement_method=None):
    base,model,official_path=map(Path,(base,model,official_path))
    from reproduction.common8.warmup_measurement_plan import ROLE,PARENT_PATH,METHOD_PATH,verify_parent,verify_method
    assert execution_role in ('production',ROLE), 'Unsupported scientific execution role'
    assert (execution_role==ROLE)==(measurement_parent is not None and measurement_method is not None), 'Measurement requires actual parent and prospective method'
    if execution_role=='production':
        assert measurement_parent is None and measurement_method is None
    target=base/'logs/run-manifest.json'
    assert not target.exists(), 'Inspect the existing run rather than create a duplicate'
    data_proof=verify(base/'data')
    model_hashes=verify_model(model,official_path)
    audit=derive_calibration(base)
    repo=base/'repo'
    from reproduction.common8.source_files import runtime_files
    source={str(p.relative_to(repo)):digest(p) for p in runtime_files(repo)}
    budget=json.loads((repo/'reproduction/common8/budget.json').read_text())
    assert budget['model_revision']==REVISION and budget['fresh_actor']
    assert budget['common_manifest_sha256']==data_proof['manifest_sha256']
    inputs={name:digest(base/'data'/name) for name in
        ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet',
         'exclusions-audit.json','calibration-300.parquet','audit.json')}
    result=dict(policy_safety_required='actor_rollout_ref.actor.policy_safety.enabled=true' in json.loads((repo/'reproduction/common8/recipe.json').read_text()),profile='common8',run_id=uuid.uuid4().hex,started_unix=time.time(),
        completed_updates=0,initial_model=str(model.resolve()),reference_model=str(model.resolve()),
        initial_model_sha256=model_hashes,model_revision=REVISION, budget=budget,
        source_commit=commit,source_sha256=source,
        source_snapshot_semantics='disk fingerprint before launch; not proof of loaded modules',
        input_sha256=inputs,calibration_question_ids=audit['calibration_question_ids'],
        contract_kind=CONTRACT_KIND,target_updates=TARGET_UPDATES,
        save_interval=50,acceptance_update=5,initial_probe_version=None,
        checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
        train_log=str(base/'logs/search-g1-common8-training.log'),
        optimizer_semantics='fresh Adam, scheduler, RNG and sampler; official fresh actor; no prior checkpoint or diagnostic state',
        retrieval_placement='pending actual resource and retrieval-equivalence admission',
        probe_visible_device=None)
    if execution_role==ROLE:
        parent_path,method_path=Path(measurement_parent),Path(measurement_method)
        assert parent_path.is_file() and method_path.is_file() and not parent_path.is_symlink() and not method_path.is_symlink()
        parent=json.loads(parent_path.read_text());method=json.loads(method_path.read_text())
        result.update(policy_calibration_role=ROLE,measurement_outer_count=2,
                      measurement_parent_run_id=parent['run_id'],measurement_parent_manifest_sha256=digest(parent_path),
                      measurement_method_sha256=digest(method_path))
        verify_parent(result,parent);verify_method(method,result)
        for relative,source in ((PARENT_PATH,parent_path),(METHOD_PATH,method_path)):
            destination=base/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as f:
                f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
    assert_scientific_run(result,expected_role=execution_role)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--model',required=True);p.add_argument('--official-manifest',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--execution-role',default='production',choices=('production','warmup57_measurement'))
    p.add_argument('--measurement-parent');p.add_argument('--measurement-method');a=p.parse_args()
    print(json.dumps(prepare(a.base,a.model,a.official_manifest,a.source_commit,execution_role=a.execution_role,
                             measurement_parent=a.measurement_parent,measurement_method=a.measurement_method)))
