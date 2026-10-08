"""Exact held-out512 coverage, strict scoring, paired question-unit bootstrap."""
import argparse
from collections import Counter
from pathlib import Path
import random
try:
    from .utils import *
    from . import worker,prepare
except ImportError:
    from utils import *
    import worker,prepare


def counts(items):
    require(items,'Empty evaluation denominator')
    return dict(questions=len(items),strict_hits=sum(x['protocol_em'] for x in items),em=sum(x['protocol_em'] for x in items)/len(items),
        final_invalid=sum(x['failure_class']=='final_invalid' for x in items),valid_gold_mismatch=sum(x['failure_class']=='valid_gold_mismatch' for x in items),
        no_executed_search=sum(x['search_calls']==0 for x in items),termination=dict(Counter(x['termination'] for x in items)),search_calls=dict(Counter(x['search_calls'] for x in items)))


def bootstrap(differences,replicates=10000,seed=42):
    require(differences and all(x in (-1,0,1) for x in differences),'Paired binary question differences required')
    require(type(replicates) is int and replicates>=1000,'Prespecified bootstrap count too small')
    rng=random.Random(seed);n=len(differences)
    draws=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(replicates))
    def quantile(q):
        position=q*(len(draws)-1);index=int(position);fraction=position-index
        return draws[index]*(1-fraction)+draws[min(index+1,len(draws)-1)]*fraction
    return dict(method='paired nonparametric percentile bootstrap; resample question slots, never siblings',question_units=n,
        replicates=replicates,seed=seed,confidence=.95,em_difference=sum(differences)/n,lower=quantile(.025),upper=quantile(.975))


def paired(a,b,replicates=10000):
    require(len(a)==len(b)==512 and [r['validation_index'] for r in a]==[r['validation_index'] for r in b]==list(range(512)), 'Paired512 ordering differs')
    require(all(x['row_semantic_sha256']==y['row_semantic_sha256'] and x['question_id']==y['question_id'] for x,y in zip(a,b)),'Different heldout question identities')
    c=Counter((x['protocol_em'],y['protocol_em']) for x,y in zip(a,b))
    lost=[y for x,y in zip(a,b) if x['protocol_em']==1 and y['protocol_em']==0]
    gained=[y for x,y in zip(a,b) if x['protocol_em']==0 and y['protocol_em']==1]
    return dict(questions=512,new_hits=c[0,1],lost_hits=c[1,0],preserved_hits=c[1,1],still_wrong=c[0,0],net_hits=c[0,1]-c[1,0],
        net_percentage_points=100*(c[0,1]-c[1,0])/512,
        lost_final_invalid=sum(x['failure_class']=='final_invalid' for x in lost),lost_valid_gold_mismatch=sum(x['failure_class']=='valid_gold_mismatch' for x in lost),
        gained_no_search=sum(x['search_calls']==0 for x in gained),lost_no_search=sum(x['search_calls']==0 for x in lost),
        bootstrap=bootstrap([y['protocol_em']-x['protocol_em'] for x,y in zip(a,b)],replicates))


def aggregate(records,data,config,verify_traces=True):
    versions=config['scientific']['versions'];require(len(records)==512*len(versions),'Incomplete primary model/question slots')
    index={};core=qa=None
    if verify_traces:core,qa,_=science(config)
    for row in records:
        key=(row['version'],row['validation_index']);require(key not in index and key[0] in versions and key[1] in range(512),'Duplicate/unexpected primary slot')
        source=data[key[1]]
        require(row['primary'] is True and row['status']=='complete' and row['run_id']==config['run_id']
            and row['rank']==key[1]%8 and row['slot']==key[1]//8 and row['question_id']==source['question_id']
            and row['dataset']==source['data_source'] and row['row_semantic_sha256']==object_sha(source),'Native rank64/row identity differs')
        require(type(row['protocol_em']) is int and row['protocol_em'] in (0,1) and type(row['protocol_valid_final_action']) is bool,'Wrong strict scoring schema')
        require(row['failure_class']==(None if row['protocol_em'] else 'final_invalid' if not row['protocol_valid_final_action'] else 'valid_gold_mismatch')
            and (not row['protocol_em'] or row['protocol_valid_final_action']),'Failure labels overlap or correct answer has invalid final action')
        require(type(row['search_calls']) is int and 0<=row['search_calls']<=4,'Executed search count differs')
        if row['slot']==0:
            require(row['hook_control']['accepted'] is True and row['hook_control']['excluded_from_primary_denominator'] is True,'Actual hook null-effect control absent')
            if key[0]=='bf16_zero':require(row['zero_effect_control']['accepted'] is True,'Actual BF16 zero/raw-base control absent')
        if verify_traces:
            trace,_=checked_json(row['trace']);result=trace['native_result'];decoded=worker.protocol(core,qa,result,source['answers'])
            require(trace['version']==key[0] and trace['validation_index']==key[1] and trace['question_id']==source['question_id']
                and trace['question']==source['question'] and trace['answers']==source['answers'] and trace['privileged_reference'] is False
                and trace['hooks_restored'] is True and not trace['logging_errors'] and object_sha(result)==row['native_result_sha256']
                and decoded==trace['protocol'] and all(decoded[k]==row[k] for k in ('protocol_em','protocol_valid_final_action','failure_class'))
                and result['search_calls']==row['search_calls'] and result['termination']==row['termination'],'Native result/transparent trace/strict QA re-score differs')
        index[key]=row
    require(set(index)=={(v,i) for v in versions for i in range(512)},'Omitted fixed model/question')
    models={v:dict(overall=counts([index[v,i] for i in range(512)]),datasets={d:counts([index[v,i] for i in range(512) if data[i]['data_source']==d]) for d in ('nq','hotpotqa')}) for v in versions}
    comparisons={}
    if versions==['bf16_zero','fp16_zero']:
        comparisons['BF16zero->FP16zero']=paired([index['bf16_zero',i] for i in range(512)],[index['fp16_zero',i] for i in range(512)])
    elif versions==['raw10_bf16','raw10_fp16']:
        comparisons['same_raw10_BF16_unmerged->FP16_deployed']=paired([index['raw10_bf16',i] for i in range(512)],[index['raw10_fp16',i] for i in range(512)])
    return dict(schema='opsd-ablation-native512-summary-v1',accepted=True,study_id=config['study_id'],variant=config['variant'],run_id=config['run_id'],
        evaluation_role=config['evaluation_role'],primary_slots=512*len(versions),models=models,paired=comparisons,
        excluded_extra_actual_generations=dict(hook_null_effect_controls=8*len(versions),bf16_zero_disabled_adapter_controls=8 if 'bf16_zero' in versions else 0),
        scope='Fixed ordinary heldout512, NQ256+Hotpot256. Strict final-action EM; invalid vs valid answer mismatch, no semantic-only correctness inference. Raw tenth outer, no best fallback; no algorithm-causal conclusion from these counts.',
        source239_diagnostic_only=True,not_a_match_hits_gate=True,no_training=True)


def collect(config):
    data,validation=cohort(config);records=[];inputs=[];model_weights={};config_ref=fileproof(config['_config_path']);train,_=original_config(config)
    for version in config['scientific']['versions']:
        stage=Path(config['paths']['output_root'])/version
        phase,pp=checked_json(fileproof(stage/'phase-complete.json'))
        require(phase['accepted'] is True and phase['version']==version and len(phase['child_exit_evidence'])==8
            and all(x['actual_wait_returncode']==0 for x in phase['child_exit_evidence']),'Actual model phase8 exits not complete rc0')
        inputs.append(pp);expected=prepare.adapter(config,version,full=False)
        model_weights[version]=dict(adapter=expected,model_kind='BF16_base_FP32_unmerged_LoRA' if version in ('bf16_zero','raw10_bf16') else 'CPU_FP32_merged_once_cast_FP16',
            deployment=None)
        for rank in range(8):
            value,cp=checked_json(fileproof(stage/f'worker-{rank}/complete.json'))
            require(value['accepted'] is True and value['status']=='COMPLETE' and value['primary_rows']==64 and value['rank']==rank
                and value['version']==version and value['run_id']==config['run_id'] and value['variant']==config['variant'] and value['study_id']==config['study_id']
                and value['config']['sha256']==config_ref['sha256'] and value['adapter']['manifest']['sha256']==expected['manifest']['sha256']
                and value['source']['manifest']['sha256']==train['source_model_manifest']['sha256'] and value['science']['manifest']['sha256']==train['runtime_manifest']['sha256']
                and value['actual_adapter']['tensor_sha256']==expected['tensor_sha256'] and value['actual_adapter']['all_finite'] is True
                and value['actual_adapter']['tensors']==504 and value['actual_adapter']['dtype']=='torch.float32'
                and value['physical_gpu']==rank and value['gpu_uuid']==config['owner']['gpu_uuids'][rank] and value['rank_seed']==42
                and value['privileged_reference'] is False and value['no_training'] is True,'Actual worker/model/scorer/run identity differs')
            if version in ('bf16_zero','raw10_bf16'):
                require(value['model_kind']=='BF16_base_FP32_unmerged_LoRA' and value['actual_loaded_adapter']['tensor_sha256']==expected['tensor_sha256'],'Primary512 actual unmerged adapter differs')
            else:
                require(value['model_kind']=='CPU_FP32_merged_once_cast_FP16' and value['deployment'] is not None and value['actual_loaded_adapter'] is None,'Deployment precision/model labeling differs')
                if model_weights[version]['deployment'] is None:model_weights[version]['deployment']=value['deployment']
                require(model_weights[version]['deployment']==value['deployment'],'Eight ranks loaded different deployed models')
            rs,rp=rows(value['results']);require(len(rs)==64,'Worker64 actual rows incomplete');records+=rs;inputs.extend([cp,rp])
    result=aggregate(records,data,config);result['inputs']=inputs;result['model_weights']=model_weights
    private=Path(config['paths']['output_root'])/'per-question.private.jsonl'
    with private.open('x') as sink:
        for row in sorted(records,key=lambda r:(r['version'],r['validation_index'])):append_json(sink,row)
    result['private_primary_records']=fileproof(private)
    result['identities']=dict(source_model_identity=train['source_model_manifest']['sha256'],runtime_identity=train['runtime_manifest']['sha256'],
        common_initial_adapter_tensor_sha256=train['common_initial_adapter_tensor_sha256'],validation_sha256=validation['file']['sha256'],
        greedy_contract={k:config['scientific'][k] for k in ('temperature','rank_seed','max_context','max_new_tokens','turns','max_search_calls','topk','observation_tokens','repetition_penalty','eos_token_id','attention_backend')})
    return result


def compare(input_manifest):
    manifest,mp=checked_json(input_manifest);require(manifest['schema']=='opsd-ablation-eleven-fixed-model-inputs-v1','Wrong cross-group input manifest')
    expected={'bf16_zero','fp16_zero','C0','C1','M','R','W','P','J','P_fp16','J_fp16'}
    require(set(manifest['models'])==expected,'Exactly eleven fixed models /5632 primary predictions required')
    models={};records={};model_weights={};identity=None;study=None;inputs=[]
    for label,item in manifest['models'].items():
        value,fp=checked_json(item['summary']);version=item['version']
        require(value['accepted'] is True and version in value['models'],'Input fixed model summary not accepted')
        if identity is None:identity=value['identities'];study=value['study_id']
        require(value['identities']==identity and value['study_id']==study,'Different data/source/initial/science across group evaluation')
        require(version==('bf16_zero' if label=='bf16_zero' else 'fp16_zero' if label=='fp16_zero' else 'raw10_fp16' if label.endswith('_fp16') else 'raw10_bf16')
            and (label in ('bf16_zero','fp16_zero') or value['variant']==label.split('_')[0]),'Cross-group version/variant mislabeled')
        actual,rp=rows(value['private_primary_records']);selected=sorted([r for r in actual if r['version']==version],key=lambda r:r['validation_index'])
        require(len(selected)==512 and [r['validation_index'] for r in selected]==list(range(512)),'Cross-group private512 slots omitted/duplicate')
        require(counts(selected)==value['models'][version]['overall'],'Cross-group private strict scores differ from summary')
        records[label]=selected;models[label]=value['models'][version];model_weights[label]=value['model_weights'][version]
        inputs.append(dict(label=label,summary=fp,private_records=rp))
    pairs=[('bf16_zero',v) for v in ('C0','C1','M','R','W','P','J')]+[('fp16_zero',v) for v in ('P_fp16','J_fp16')]+[
        ('C0','C1'),('C1','M'),('C1','R'),('R','W'),('C1','P'),('C1','J'),('P','P_fp16'),('J','J_fp16')]
    return dict(schema='opsd-ablation-eleven-fixed-model-summary-v1',accepted=True,study_id=study,primary_slots=5632,
        models=models,model_weights=model_weights,paired={a+'->'+b:paired(records[a],records[b]) for a,b in pairs},identities=identity,inputs=inputs,input_manifest=mp,
        scope='All eleven native512 fixed models. FP16 deployment compared with actual FP16zero; changed K/Adam budgets and sampling remain report covariates. Paired bootstrap uses512 question units; no extra controls or teacher diagnostics in denominator. No automatic algorithm-causal attribution.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--compare-manifest',required=True);p.add_argument('--compare-manifest-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=compare(dict(path=a.compare_manifest,sha256=a.compare_manifest_sha256));fp=atomic_json(a.output,result,exclusive=True)
    print(json.dumps(dict(accepted=True,primary_slots=5632,summary=fp)))


if __name__=='__main__':main()
