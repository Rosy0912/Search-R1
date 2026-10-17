"""Prepare a reviewable immutable-model plan, without allocating or submitting."""
import argparse
import json
from pathlib import Path
import uuid
from .contracts import DATA_SHA,IDS_SHA,digest,sha,validate_milestone,validate_manifest


def build(base,group,step,commit,backend):
    validate_milestone(group,step)
    base=Path(base).resolve(strict=True);run=base/'runs'/group
    receipt_path=run/'receipts/configuration.json'
    receipt=json.loads(receipt_path.read_text())
    config=Path(receipt['config'])
    manifest_path=run/'checkpoints/training_state'/('global_step_'+str(step))/'manifest.json'
    manifest=json.loads(manifest_path.read_text());identity=manifest['identity']
    # Continued groups can have a different immutable configuration receipt.
    if receipt['config_sha256']!=identity['config_sha256']:
        entry=json.loads((run/'receipts/current-entry.json').read_text())
        receipt_path=Path(entry['configuration_receipt']);receipt=json.loads(receipt_path.read_text())
        config=Path(receipt['config'])
    from .archived_assets import resolve_assets
    assets,asset_binding=resolve_assets(base,group,backend,receipt)
    data=assets['data']
    data_manifest=json.loads((data/'manifest.json').read_text())
    model=run/'checkpoints/actor'/('global_step_'+str(step))
    files=manifest['hf_identity']['files']
    cid=digest(dict(group=group,outer=step,run=receipt['run_id'],
        training_config=identity['config_sha256'],model_files_sha256=digest(files),data=DATA_SHA))
    retriever=Path(receipt['retriever_receipt'])
    result=dict(protocol='tree_seven_milestone_v1',base=str(base),group=group,outer=step,
        checkpoint_manifest=str(manifest_path),checkpoint_manifest_sha256=sha(manifest_path),
        training_config=str(config),training_config_sha256=identity['config_sha256'],
        configuration_receipt=str(receipt_path),configuration_receipt_sha256=sha(receipt_path),
        training_run_id=receipt['run_id'],training_job=identity['job_id'],training_code_commit=identity['code_commit'],
        model=str(model),model_files=files,checkpoint_identity=cid,data_root=str(data),
        data_manifest_sha256=sha(data/'manifest.json'),ordered_ids_sha256=IDS_SHA,
        final_content_identity=data_manifest['splits']['final']['content_identity'],
        retriever_receipt=str(retriever),retriever_receipt_sha256=sha(retriever),
        evaluation_code_commit=commit,evaluation_run_id=uuid.uuid4().hex,backend=backend,
        job='rt16-seven-'+group.lower().replace('generation','depth')+'-'+str(step)+'-'+cid[:8],
        output=str(base/'seven-evaluations'/cid/'output'))
    from .model_view import TOKENIZER_FILES
    canonical=assets['model'].resolve(strict=True)
    if asset_binding:result['archived_asset_resolution']=asset_binding
    result.update(canonical_tokenizer=str(canonical),
        canonical_tokenizer_files={n:sha(canonical/n) for n in TOKENIZER_FILES if (canonical/n).is_file()},
        inference_model=str(base/'seven-model-views'/cid/commit[:12]))
    validate_manifest(result,manifest)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--group',required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--backend',choices=['pj_pt','slurm_lumia'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=build(a.base,a.group,a.step,a.commit,a.backend)
    from .model_view import create_view,validate_view
    from omegaconf import OmegaConf
    expected=OmegaConf.load(value['training_config']).baselines.tree.token_preservation.tokenizer_sha256
    if Path(value['inference_model']).exists():validate_view(value,expected)
    else:create_view(value,expected)
    with Path(a.output).open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(status='PLAN_ONLY_NOT_SUBMITTED',path=a.output,job=value['job'],identity=value['checkpoint_identity'])))

if __name__=='__main__':main()
