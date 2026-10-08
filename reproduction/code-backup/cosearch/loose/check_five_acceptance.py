"""Strict five-update and HF weight reload check; no exact-resume claim."""
import argparse, hashlib, json, math, re, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');run=Path(args.run).resolve()
assert run.is_relative_to((root/'runs').resolve())
assert os.environ.get('SLURM_JOB_NODELIST')=='h100-16'
torch.set_num_threads(8)
from metric_log_source import validated_metric_log

text,metric_source=validated_metric_log(run,root)
records=[]
for line in text.splitlines():
    marker='COSEARCH_OUTER_METRICS_JSON='
    if marker in line:
        raw=line.split(marker,1)[1]
        value,end=json.JSONDecoder().raw_decode(raw)
        records.append(value)
assert [r['step'] for r in records]==[1,2,3,4,5], 'Require exactly five logged actual updates'
for i,r in enumerate(records,1):
    m=r['metrics']
    assert m['training/actual_outer_updates']==i and m['actor/optimizer_steps']>0
    assert all(math.isfinite(v) for v in m.values())
previous_ranker_steps=0
ranker_steps_advanced=set()
for record in records:
    current_ranker_steps=record['metrics'].get('ranker/optimizer_steps_cumulative',0)
    assert current_ranker_steps>=previous_ranker_steps
    if current_ranker_steps>previous_ranker_steps: ranker_steps_advanced.add(record['step'])
    previous_ranker_steps=current_ranker_steps
rl=[r for r in records if r['step'] in ranker_steps_advanced and r['metrics'].get('ranker/informative_groups_with_distinct_rankings',0)>0
    and r['metrics'].get('ranker/gradient_norm_max',0)>1e-5
    and r['metrics'].get('ranker/parameter_probe_delta_l1',0)>0
    and r['metrics'].get('ranker/optimizer_steps_cumulative',0)>0]
assert len(rl)>=2, 'Require two genuine joint-update milestones; decay alone is insufficient'
assert records[-1]['metrics'].get('val/qa_count/all')==64, 'Require all 64 validation questions'
assert 'COSEARCH_VALIDATION_FINAL_COST_JSON=' in text, 'Missing validation-64 costs'
assert (run/'driver-exit.json').is_file()
assert json.loads((run/'driver-exit.json').read_text())['returncode']==0
launch=json.loads((run/'launch.json').read_text())
assert launch['local_node_only'] and launch['endpoints']==['127.0.0.1:18086','127.0.0.1:16386']
report=dict(status='reload_in_progress',five_updates=records,ranker_rl_steps=[r['step'] for r in rl],
    source_commit=launch['commit'],metric_source=metric_source,local_node='h100-16',exact_joint_resume_supported=False)
output=run/'acceptance.json'
def save(): output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def probes(model):
    result={}
    for name,param in model.named_parameters():
        assert torch.isfinite(param).all(), f'Nonfinite saved parameter {name}'
        flat=param.detach().reshape(-1);count=min(8,flat.numel())
        idx=torch.arange(count)*(flat.numel()-1)//max(count-1,1)
        result[name]=flat[idx].float().cpu()
    return result
resolved=OmegaConf.load(run/'resolved.yaml')
actor_model_dtype=OmegaConf.select(resolved,'actor_rollout_ref.actor.fsdp_config.model_dtype')
assert actor_model_dtype is None or str(actor_model_dtype).lower() in ('fp32','float32','32'), 'This checker requires the actual FP32 actor master initialization'
report['actor_initial_master_dtype']='float32'
base=root/'assets-root/assets/qwen2.5-3b'
source=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
base_probe=probes(source);del source
base_tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
experiment=next(arg.split('=',1)[1] for arg in launch['command'] if arg.startswith('trainer.experiment_name='))
assert '/' not in experiment and experiment.startswith(('cosearch-local-five-', 'cosearch-local-gpu-five-'))
checkpoint=run/'output/checkpoints'/experiment
models={'actor':checkpoint/'actor/global_step_5', 'ranker':checkpoint/'algorithm/global_step_5/ranker'}
warm=launch.get('warm_start_manifest')
if warm:
    warm_path=Path(warm);assert warm_path==run/'warm-start/manifest.json'
    warm_manifest=json.loads(warm_path.read_text())
    assert warm_manifest['completed'] and warm_manifest['acceptance_updates']==0
    assert warm_manifest['roles']['actor']['supervised_optimizer_steps']==32
    assert warm_manifest['roles']['ranker']['supervised_optimizer_steps']==16
    initialization=[]
    for line in text.splitlines():
        if 'COSEARCH_RANKER_INITIALIZATION_JSON=' in line:
            initialization.append(json.JSONDecoder().raw_decode(line.split('COSEARCH_RANKER_INITIALIZATION_JSON=',1)[1])[0])
    assert len(initialization)==1 and initialization[0]['optimizer_states']==0
    assert initialization[0]['rl_optimizer_steps']==0
    assert initialization[0]['model_path']==str(run/'warm-start/ranker')
    def verify_hash(path,expected):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==expected, 'Warm-start initialization artifact changed: '+str(path)
    verify_hash(run/'warm-start/training-only-corpus.json',warm_manifest['corpus_sha256'])
    verify_hash(Path(warm_manifest['retained_train_path']),warm_manifest['train_sha256'])
    for role in ('actor','ranker'):
        for name,digest in warm_manifest['roles'][role]['files_sha256'].items():
            verify_hash(run/'warm-start'/role/name,digest)
    audit_config=OmegaConf.load(run/'resolved.yaml')
    from search_r1.baselines.holdout import prepare_common_holdout
    actual_split=prepare_common_holdout(audit_config)
    assert actual_split['content_identity']['sha256']==warm_manifest['split_content_identity']
    assert warm_manifest['stage_periodic_and_final_questions_excluded']
    actor_initialization=[]
    for line in (run/'training.log').read_text(errors='replace').splitlines():
        if 'COSEARCH_ACTOR_INITIALIZATION_JSON=' in line:
            actor_initialization.append(json.JSONDecoder().raw_decode(line.split('COSEARCH_ACTOR_INITIALIZATION_JSON=',1)[1])[0])
    assert len(actor_initialization)==8 and {r['rank'] for r in actor_initialization}==set(range(8))
    assert all(r['optimizer_states']==0 and r['rl_optimizer_steps']==0 and str(r['job'])==launch['job_id']
        and r['model_path']==str(run/'warm-start/actor') for r in actor_initialization)
    report['warm_start_manifest']=str(warm_path)
    report['supervised_steps_excluded_from_rl']=True
for role,path in models.items():
    role_base_probe=base_probe
    if warm:
        initial_path=run/'warm-start'/role
        assert launch['warm_'+role+'_initialization']==str(initial_path)
        initial=AutoModelForCausalLM.from_pretrained(initial_path,torch_dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).eval()
        role_base_probe=probes(initial);del initial
    assert (path/'config.json').is_file() and (path/'tokenizer_config.json').is_file()
    model,loading=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys']
    model.eval();tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    assert tokenizer.get_vocab()==base_tokenizer.get_vocab()
    assert tokenizer.bos_token_id==base_tokenizer.bos_token_id and tokenizer.eos_token_id==base_tokenizer.eos_token_id
    probe=probes(model);assert probe.keys()==role_base_probe.keys()
    delta=sum(float((probe[key]-role_base_probe[key]).abs().sum()) for key in probe)
    assert delta>0, f'{role} saved weights did not change at sampled coordinates'
    tokens=tokenizer('Question: What is the capital of France?',return_tensors='pt',return_token_type_ids=False)
    with torch.no_grad():
        logits=model(**tokens,num_logits_to_keep=1).logits
        assert torch.isfinite(logits).all()
        decode=model.generate(**tokens,max_new_tokens=8,do_sample=False)
    def file_hash(file):
        digest=hashlib.sha256()
        with file.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''): digest.update(block)
        return digest.hexdigest()
    hashes={f.name:file_hash(f) for f in path.iterdir() if f.is_file() and f.suffix in ('.json','.safetensors','.bin','.pt')}
    report[role]=dict(hf_path=str(path),loading_complete=True,probe_delta_l1_from_rl_initialization=delta,
        finite_reload_logits=True,short_reload_generation=tokenizer.decode(decode[0],skip_special_tokens=True),checkpoint_files_sha256=hashes)
    if role=='ranker':
        state=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
        optimizer.load_state_dict(state)
        assert optimizer.state
        assert any(float(s['step'])>0 for s in optimizer.state.values())
        assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
        report[role]['optimizer_load_finite']=True
        del optimizer,state
    save();del model
report['status']='five_update_validation_save_reload_passed'
save();print('COSEARCH_FIVE_ACCEPTANCE_JSON='+json.dumps(report,allow_nan=False),flush=True)
