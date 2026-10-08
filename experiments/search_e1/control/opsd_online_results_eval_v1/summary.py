"""Private per-question input; compact strict native512 and paired summaries."""
from collections import Counter
import argparse
from pathlib import Path
try:
    from .utils import *
    from . import worker,prepare
except ImportError:
    from utils import *
    import worker,prepare

def aggregate(records,data,config,verify_traces=True):
    require(len(records)==1536,'Incomplete actual1536 primary slots')
    index={}
    core=qa=None
    if verify_traces:core,qa,_=science(config)
    for row in records:
        key=(row['version'],row['validation_index']);require(key not in index and key[0] in VERSIONS and key[1] in range(512),'Duplicate/unexpected primary slot')
        source=data[key[1]]
        require(row['primary'] is True and row['status']=='complete' and row['run_id']==config['run_id']
            and row['rank']==key[1]%8 and row['slot']==key[1]//8 and row['question_id']==source['question_id']
            and row['dataset']==source['data_source'] and row['row_semantic_sha256']==object_sha(source),'Primary validation slice/identity differs')
        require(row['protocol_em'] in (0,1) and type(row['protocol_em']) is int and type(row['protocol_valid_final_action']) is bool,'Invalid strict scoring schema')
        require(row['failure_class']==(None if row['protocol_em'] else 'final_invalid' if not row['protocol_valid_final_action'] else 'valid_gold_mismatch'),'Nonexclusive failure class')
        require(not row['protocol_em'] or row['protocol_valid_final_action'],'Hit without protocol-valid final')
        require(type(row['search_calls']) is int and 0<=row['search_calls']<=4,'Invalid executed search count')
        if row['slot']==0:
            require(row['hook_control']['accepted'] is True and row['hook_control']['excluded_from_primary_denominator'] is True,'Missing real hook null-effect control')
            if row['version']=='source100':require(row['zero_effect_control']['accepted'] is True,'Missing CP0 raw-base zero-effect control')
        if verify_traces:
            trace,_=checked_json(row['trace']);result=trace['native_result'];decoded=worker.protocol(core,qa,result,source['answers'])
            require(trace['version']==key[0] and trace['validation_index']==key[1] and trace['question_id']==source['question_id']
                and trace['question']==source['question'] and trace['answers']==source['answers'] and trace['privileged_reference'] is False
                and trace['hooks_restored'] is True and not trace['logging_errors'] and object_sha(result)==row['native_result_sha256']
                and decoded==trace['protocol'] and all(decoded[k]==row[k] for k in ('protocol_em','protocol_valid_final_action','failure_class'))
                and result['search_calls']==row['search_calls'] and result['termination']==row['termination'],'Actual native trace/result/strict reward differs')
        index[key]=row
    require(set(index)=={(v,i) for v in VERSIONS for i in range(512)},'Omitted model/question')
    def counts(items):
        return dict(questions=len(items),strict_hits=sum(x['protocol_em'] for x in items),em=sum(x['protocol_em'] for x in items)/len(items),
            final_invalid=sum(x['failure_class']=='final_invalid' for x in items),valid_gold_mismatch=sum(x['failure_class']=='valid_gold_mismatch' for x in items),
            no_executed_search=sum(x['search_calls']==0 for x in items),termination=dict(Counter(x['termination'] for x in items)),search_calls=dict(Counter(x['search_calls'] for x in items)))
    models={v:dict(overall=counts([index[v,i] for i in range(512)]),datasets={d:counts([index[v,i] for i in range(512) if data[i]['data_source']==d]) for d in ('nq','hotpotqa')}) for v in VERSIONS}
    paired={}
    for a,b in (('source100','cp89'),('source100','final199'),('cp89','final199')):
        c=Counter((index[a,i]['protocol_em'],index[b,i]['protocol_em']) for i in range(512))
        lost=[index[b,i] for i in range(512) if index[a,i]['protocol_em']==1 and index[b,i]['protocol_em']==0]
        paired[a+'->'+b]=dict(new_hits=c[0,1],lost_hits=c[1,0],preserved_hits=c[1,1],still_wrong=c[0,0],net_hits=c[0,1]-c[1,0],questions=512,
            lost_final_invalid=sum(x['failure_class']=='final_invalid' for x in lost),lost_valid_gold_mismatch=sum(x['failure_class']=='valid_gold_mismatch' for x in lost),
            lost_termination=dict(Counter(x['termination'] for x in lost)),lost_search_calls=dict(Counter(x['search_calls'] for x in lost)))
    actual_source_hits=models['source100']['overall']['strict_hits']
    return dict(schema='opsd-online-results-native512-summary-v1',accepted=True,run_id=config['run_id'],primary_slots=1536,
        models=models,paired=paired,hook_controls_excluded=24,cp0_zero_effect_controls_excluded=8,
        scope='Fixed held-out512,256 NQ+256 Hotpot each. Strict final-action EM; no word-semantic correctness inference; no training population EM; no algorithm-causal attribution',
        source239_is_not_a_gate=True,source100_previous239_condition_check=dict(historical_hits=239,current_hits=actual_source_hits,
            decision='MATCHED_HITS_CONDITIONS_STILL_BOUND_TO_THIS_RUN' if actual_source_hits==239 else 'NEEDS_CONDITION_DIAGNOSIS',
            no_generation_parameter_adjustment=True,not_a_training_failure=True),final_raw_checkpoint_not_best=True,formal_checkpoint_control_count=3)

def collect(config):
    data,_=cohort(config);records=[];proofs=[];config_proof=fileproof(config['_config_path'])
    train,_=original_config(config)
    for version in VERSIONS:
        stage=Path(config['paths']['output_root'])/version
        phase,pp=checked_json(fileproof(stage/'phase-complete.json'));require(phase['accepted'] is True and phase['version']==version,'Stage exits incomplete')
        require(len(phase['child_exit_evidence'])==8 and all(x['actual_wait_returncode']==0 for x in phase['child_exit_evidence']),'Stage true child exit is not zero')
        proofs.append(pp)
        for rank in range(8):
            complete,cp=checked_json(fileproof(stage/f'worker-{rank}/complete.json'))
            require(complete['accepted'] is True and complete['status']=='COMPLETE' and complete['primary_rows']==64
                and complete['rank']==rank and complete['version']==version and complete['run_id']==config['run_id'],'Worker64 not complete')
            cp_value,cp_identity=prepare.checkpoint(config,version,full=False)
            require(complete['config']['sha256']==config_proof['sha256'] and complete['checkpoint']['checkpoint']['sha256']==cp_identity['checkpoint']['sha256']
                and complete['source']['manifest']['sha256']==train['source_model_manifest']['sha256']
                and complete['science']['manifest']['sha256']==train['runtime_manifest']['sha256']
                and complete['actual_loaded_adapter']['tensor_sha256']==complete['actual_adapter']['tensor_sha256']==cp_value['all_rank_adapter_sha256'][0]
                and complete['actual_loaded_adapter']['all_finite'] is True and complete['actual_loaded_adapter']['dtype']=='torch.float32'
                and complete['actual_loaded_adapter']['tensors']==504 and complete['physical_gpu']==rank
                and complete['gpu_uuid']==config['owner']['gpu_uuids'][rank] and complete['rank_seed']==42
                and complete['privileged_reference'] is False and complete['no_training'] is True,'Worker actual model/config/scorer/device identity differs')
            rs,rp=rows(complete['results']);require(len(rs)==64,'Worker rowcount differs');records+=rs;proofs.extend([cp,rp])
    result=aggregate(records,data,config);result['inputs']=proofs;return result

def main():
    p=argparse.ArgumentParser()
    for n in ('config','config-sha256','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();cfg,_=checked_json(dict(path=a.config,sha256=a.config_sha256));cfg['_config_path']=a.config
    result=collect(cfg);out=atomic_json(a.output,result,exclusive=True)
    print(json.dumps(dict(accepted=True,output=out,models=result['models'],paired=result['paired'])))

if __name__=='__main__':main()
