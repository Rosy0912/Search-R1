"""CPU inputs for posthoc replay of the actual new batched heldout responses.

No extra generate/retrieval/update. New batched HF outcomes are never relabelled as
old native-HF outcomes. All large canonical traces/rows stay on HKUST4.
"""
import argparse, json, os, sys
from pathlib import Path
try:
    from .common import (require, read_json, read_rows, check_file, file_ref,
        exclusive_json, remote_output, bind_training, semantic_sha, cpu_only,
        VARIANTS, ENDPOINTS)
except ImportError:
    from common import (require, read_json, read_rows, check_file, file_ref,
        exclusive_json, remote_output, bind_training, semantic_sha, cpu_only,
        VARIANTS, ENDPOINTS)

PROTOCOL='hf-sdpa-greedy-rep1p05-unmerged-fp32-lora-batch32-fixed512-v1'

def same_ref(a,b):
    return a['path']==b['path'] and a['sha256']==b['sha256']

def canonical_rows(data, records, reader, scorer):
    require(len(data)==len(records)==512 and len({q['question_id'] for q in data})==512,
        'Full fixed512, including no-reference/wrong questions, is required')
    byindex={r['validation_index']:r for r in records}
    require(len(byindex)==512 and set(byindex)==set(range(512)), 'Missing/duplicate endpoint slot')
    out=[]
    for i,q in enumerate(data):
        row=byindex[i];trace=reader(row['trace']);result=trace['native_result']
        require(row['primary'] is True and (row['version'],row['completed_outer']) in ENDPOINTS
            and row['question_id']==trace['question_id']==q['question_id']
            and trace['validation_index']==i and trace['question']==q['question']
            and trace['answers']==q['answers'] and row['row_semantic_sha256']==semantic_sha(q)
            and row['native_result_sha256']==semantic_sha(result), 'Saved response/question changed')
        decoded=scorer(result,q['answers'])
        require(decoded==trace['protocol'] and decoded['protocol_em']==row['protocol_em']==result['reward'],
            'Saved EM differs from unchanged authored-final scorer')
        require(type(result['search_calls']) is int and 0<=result['search_calls']<=4, 'Search count invalid')
        value=dict(row,dataset=q['data_source'],
            protocol_valid_final_action=decoded['protocol_valid_final_action'],
            failure_class=decoded['failure_class'],termination=result['termination'],
            search_calls=result['search_calls'])
        out.append(value)
    require(len({(r['version'],r['completed_outer']) for r in out})==1, 'Mixed endpoint models')
    return out

def prepare(request_ref,output):
    cpu_only();request,rp=read_json(request_ref)
    require(request['schema']=='opsd-signal-batched-replay-preparation-v1', 'Wrong preparation request')
    cfgs={};refs={}
    for ref in request['configs']:
        cfg,cp=read_json(ref);cfg['_config_sha256']=cp['sha256']
        require(cfg['variant'] not in cfgs,'Duplicate config');cfgs[cfg['variant']]=cfg;refs[cfg['variant']]=cp
    require(set(cfgs)==set(VARIANTS),'All four immutable configs required')
    cfg=cfgs['G_POS'];mods=bind_training(cfg)
    try:
        from .replay import verify_four_group_terminal, verify_analysis_publication
    except ImportError:
        from replay import verify_four_group_terminal, verify_analysis_publication
    terminal,tp=read_json(request['four_group_terminal']);verify_four_group_terminal(terminal,mods)
    done,dp=read_json(request['evaluation_complete'])
    require(done['accepted'] is True and done['protocol']==PROTOCOL
        and done['formal_endpoint_tests']==8 and done['common_baseline_evaluations']==1
        and len(done['summaries'])==len(done['posthoc_replay_inputs'])==9, 'All nine real batched endpoints required')
    verify_analysis_publication(Path(done['evaluation_source_manifest']['path']).parent,done['evaluation_source_manifest'],done['evaluation_publication'])
    summaries={s['endpoint']:s for s in done['summaries']}
    require(len(summaries)==9,'Duplicate endpoint summary')
    bank,bp=read_json(done['fixed_reference_bank']);data,dataref=read_rows(cfg['validation_identity'])
    require(bank['accepted'] is True and bank['protocol']==PROTOCOL
        and bank['fixed_before_eight_new_endpoint_outputs'] is True
        and bank['frozen_before_all_group_15_25_outputs'] is True
        and set(bank['configs'])==set(refs) and all(same_ref(bank['configs'][v],refs[v]) for v in VARIANTS) and bank['validation_identity']['sha256']==dataref['sha256']
        and bank['reference_generation_added']==0 and len(bank['entries'])==512, 'Frozen new baseline bank changed')
    root=remote_output(output);root.mkdir(parents=True,exist_ok=False)
    # This adds only namespace lineage to an already frozen bank; entries unchanged.
    adapted=dict(bank,study_id=cfg['study_id'],configs=refs,original_frozen_batched_bank=bp,
        canonical_metadata_only=True,reference_entries_semantic_sha256=semantic_sha(bank['entries']))
    bankref=exclusive_json(root/'bank.json',adapted)
    import importlib
    mods['prepare'].verify_runtime(cfg)
    core=importlib.import_module('search_e1.core');qa=importlib.import_module('search_r1.qa_protocol')
    scorer=lambda result,answers:mods['training_metrics'].protocol(core,qa,result,answers)
    endpoints=[];seen=set()
    for ref in done['posthoc_replay_inputs']:
        ep,epp=read_json(ref);key=(ep['version'],ep['completed_outer'])
        require(ep['accepted'] is True and ep['schema']=='opsd-signal-batched-replay-input-v1'
            and key in ENDPOINTS and key not in seen and ep['generation_protocol']==PROTOCOL,
            'Duplicate/foreign endpoint input');seen.add(key)
        variant='G_POS' if key[1]==0 else key[0];ecfg=cfgs[variant]
        require(same_ref(ep['config'],refs[variant]) and ep['validation_identity']['sha256']==dataref['sha256'],
            'Endpoint raw config/data changed')
        summ,sp=read_json(ep['endpoint_summary'])
        require(summ==summaries[ep['endpoint']] and summ['actual_worker_wait_returncodes']==[0]*8
            and summ['raw_checkpoint']==ep['raw_checkpoint'], 'Actual endpoint exit/raw checkpoint differs')
        directory=Path(epp['path']).parent
        for wave in range(2):
            wait,_=read_json(file_ref(directory/('wave-%d-exit.json'%wave)))
            require(wait['actual_worker_wait_returncodes']==[0]*4,'Actual wave exit0 missing')
            release,_=read_json(file_ref(directory/('wave-%d-release.json'%wave)))
            require(release['accepted'] is True and release['actual_worker_wait_returncodes']==[0]*4
                and len(release['actual_admission']['devices'])==4,'Actual wave GPU release missing')
        records,_=read_json(ep['records'])
        rows=canonical_rows(data,records,lambda ref:read_json(ref)[0],scorer)
        native=[];loaded=[]
        for rank in range(8):
            worker,_=read_json(file_ref(directory/('worker-%d-complete.json'%rank)))
            require(worker['accepted'] is True and worker['rank']==rank
                and worker['request']['checkpoint']==ep['raw_checkpoint']
                and worker['protocol']==PROTOCOL and worker['rank_seed']==42
                and worker['formal_weights']=='raw_unmerged_FP32_LoRA','Wrong actual worker model')
            loaded.append(worker['actual_loaded_adapter'])
            shard,_=read_rows(worker['results']);require(len(shard)==64,'Incomplete shard');native.extend(shard)
        require(len({r['question_id'] for r in native})==512
            and {semantic_sha(r) for r in native}=={r['native_result_sha256'] for r in rows},
            'Canonical traces differ from actual eight worker outputs')
        identity,_=mods['evaluation'].input_identity(ecfg,key[1],ep['raw_checkpoint'])
        require(all(p['tensor_sha256']==identity['expected_adapter_tensor_sha256']
            and p['tensors']==504 and p['dtype']=='torch.float32' and p['all_finite'] for p in loaded),
            'Actual HF loaded adapter differs from bound raw FP32 checkpoint')
        dest=root/ep['endpoint'];dest.mkdir()
        path=dest/'records.jsonl'
        with path.open('x') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        endpoints.append(exclusive_json(dest/'endpoint.json',dict(
            schema='opsd-signal-batched-canonical-endpoint-v1',accepted=True,primary_slots=512,
            version=key[0],completed_outer=key[1],run_id=ecfg['run_id'],study_id=ecfg['study_id'],
            config=refs[variant],raw_checkpoint=identity['raw_checkpoint'],
            input_identity=identity,results=file_ref(path),batched_input=epp,batched_summary=sp,
            actual_worker_wait_returncodes=[0]*8,generation_protocol=PROTOCOL,
            probability_scope='posthoc HF BF16 base/FP32 raw LoRA, retokenized/unprocessed; not HF processed scores',
            single_question_HF_endpoint_not_claimed=True,new_samples_or_training=0)))
    require(seen==ENDPOINTS,'Missing endpoint')
    plan=dict(schema='opsd-signal-batched-heldout-replay-plan-v1',
        analysis_run_id=request['analysis_run_id'],helper_root=request['helper_root'],
        source_manifest=request['source_manifest'],publication=request['publication'],
        configs=list(refs.values()),bank=bankref,endpoints=endpoints,four_group_terminal=tp,
        evaluation_complete=dp,preparation=rp,new_samples_or_training=0)
    return exclusive_json(root/'plan.json',plan)

def main():
    p=argparse.ArgumentParser()
    for n in ('request','request-sha256','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(prepare(dict(path=a.request,sha256=a.request_sha256),a.output)))
if __name__=='__main__':main()
