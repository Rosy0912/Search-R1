"""CPU-only preflight; creates one immutable plan after all model bytes verify."""
import json
import sys
import time
import uuid
from pathlib import Path
from .contract import ROOT, OLD, PROFILE, models, model_path, verify_model, verify_data, make_config, validate_plan
from reproduction.seven_dataset.contracts import sha, digest, DATA_SHA, IDS_SHA


def prepare(kind, repeat):
    assert kind in ('original','treegrpo','base','searchr1')
    assert repeat == 0 if kind in ('original','treegrpo') else repeat in (1,2,3)
    label = kind if kind in ('base','searchr1') else 'base'
    root = ROOT/'runs'/kind if repeat == 0 else ROOT/'evaluations'/kind/('r'+str(repeat))
    planfile = ROOT/'plans'/(kind+'-'+str(repeat)+'.json')
    assert not root.exists() and not planfile.exists(), 'Existing namespace: inspect, never replay'
    files = verify_model(label)
    data, manifest = verify_data()
    source = Path(__file__).resolve().parents[2]
    deployment = json.loads((ROOT/'deployment.json').read_text())
    assert source == Path(deployment['source'])
    source_manifest = Path(deployment['manifest'])
    assert sha(source_manifest) == deployment['manifest_sha256']
    retriever = OLD/'receipts/gpu-retriever-ready.json'
    rr = json.loads(retriever.read_text())
    from reproduction.seven_dataset.health import wait_retriever_health
    wait_retriever_health(rr['url'], rr['job'])
    m = models()[label]
    plan = dict(profile=PROFILE,base=str(ROOT),kind=kind,repeat_index=repeat,repeat_count=3,
        model_label=label,model=str(model_path(label)),model_revision=m['revision'],model_files=files,
        checkpoint_identity=digest(dict(profile=PROFILE,model=m['repo'],revision=m['revision'],files=files)),
        root=str(root),output=str(root/'output'),job='zhangjunming-7b-'+kind+'-'+str(repeat),
        source=str(source),source_manifest=str(source_manifest),source_manifest_sha256=sha(source_manifest),
        evaluation_code_commit=deployment['commit'],evaluation_run_id=uuid.uuid4().hex,
        data_root=str(data),data_manifest_sha256=DATA_SHA,ordered_ids_sha256=IDS_SHA,
        final_content_identity=manifest['splits']['final']['content_identity'],
        retriever_receipt=str(retriever),retriever_receipt_sha256=sha(retriever),retriever=rr['url'],
        backend='slurm_lumia',group='7B-'+kind,outer=200 if kind=='original' else 0,
        protocol='sevenb_four_way_v1',no_auto_retry=True)
    if kind == 'treegrpo':
        from reproduction.official_treegrpo.contract import recipe, JOB
        previous = json.loads((OLD/'official-treegrpo/receipts/plan-v2.json').read_text())
        src = Path(previous['upstream_source'])
        for n,h in previous['source_files'].items():
            assert sha(src/n)==h,n
        plan.update(upstream_source=str(src),upstream_commit=previous['upstream_commit'],
            upstream_files=previous['source_files'],required_versions=previous['required_versions'],
            overrides=[v.replace('='+JOB,'='+plan['job']) for v in recipe(
                (src/'train_multihopqa_tree_search.sh').read_text(),model_path('base'),data,root,rr['url'])],
            expected_iterations=179,node='node11')
    else:
        from omegaconf import OmegaConf
        host='gpu-ada6000-1' if kind=='original' else 'gpu-l40s-2'
        cfg = make_config(root, model_path(label), host, evaluation=kind!='original')
        if kind == 'searchr1':
            verify_model('base')
            from reproduction.seven_dataset.model_view import TOKENIZER_FILES, create_view, validate_view
            from reproduction.seven_dataset.searchr1_contracts import validate_tokenizer_equivalence
            base_files = {f['file']:f['sha256'] for f in models()['base']['files']}
            plan.update(canonical_tokenizer=str(model_path('base')),
                canonical_tokenizer_files={k:v for k,v in base_files.items() if k in TOKENIZER_FILES},
                inference_model=str(ROOT/'model-views'/'searchr1'))
            validate_tokenizer_equivalence(plan)
            view=Path(plan['inference_model'])
            if not view.exists():create_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
            else:validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
            cfg.actor_rollout_ref.model.path=cfg.actor_rollout_ref.ref.model_path=str(view)
        from search_r1.baselines.controller import validate_config
        validate_config(cfg)
        plan['config_text']=OmegaConf.to_yaml(cfg,resolve=True)
        plan['node']='node8' if kind=='original' else None
    validate_plan(plan)
    root.mkdir(parents=True,exist_ok=False)
    (ROOT/'plans').mkdir(exist_ok=True)
    with planfile.open('x') as f:json.dump(plan,f,indent=2)
    (root/'preflight.json').write_text(json.dumps(dict(at=time.time(),status='FULL_SHA_DATA_CONFIG_VERIFIED',plan_sha256=sha(planfile)),indent=2))
    print(json.dumps(dict(plan=str(planfile),root=str(root),sha256=sha(planfile))))


if __name__=='__main__':prepare(sys.argv[1],int(sys.argv[2]))
