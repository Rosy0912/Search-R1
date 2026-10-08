"""Independent CPU-only actual-raw review; never produces execution approval."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from safetensors.torch import load_file,save


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise RuntimeError(message)
def load(path):return json.loads(Path(path).read_text())
def thash(value):return hashlib.sha256(save(value)).hexdigest()


def review(directory):
    result=load(directory/'result.json');manifest=load(directory/'teacher-ipc/manifest.json')
    need(result['status']=='HARDWARE_AFUNIX_PARITY_RELEASE_CORESIDENCY_PASS_NO_TRAINING','Raw hardware status not complete PASS')
    need(result['NO_TRAINING'] is True,'Unexpected training scope')
    keys=('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')
    rows=[list(map(json.loads,(directory/f'teacher-ipc/scorer-{rank}.jsonl').read_text().splitlines())) for rank in range(8)]
    need(all(len(x)==3 for x in rows),'Require all24 worker rows')
    oracle_ready=load(directory/'original-gpu-oracle/ready.json')
    need(manifest['code_commit']=='438af3529b40663aae7b8293080e09fda57f6a87' and manifest['source_sha256']=='101eb46fdd5cc493d8be3be11cb9d63313b3e94819bd9d2034b37a1f94e3a9a6','Actual source freeze differs')
    for name in ('probe.py','original_gpu_oracle.py'):
        need(result['script_sha256'][name]==sha(Path(__file__).parent/'afunix-scorer-probe'/name),'Actual deployed producer script differs')
    need(oracle_ready['cuda_initialized'] is False and oracle_ready['status']=='CPU_READY_NO_CUDA_CONTEXT' and oracle_ready['offline'] is True,'Oracle startup is not offline CPU-ready')
    hashes=set();phases=[]
    for step in (1,2,3):
        tensors=load_file(str(directory/f'parity-round{step}.safetensors'),device='cpu')
        p,r,a,g,o=(tensors[k] for k in ('prompts','responses','attention','gpu_log_probs','original_gpu_log_probs'))
        need(p.dtype==r.dtype==a.dtype==torch.int64 and bool(((a==0)|(a==1)).all()) and g.dtype==o.dtype==torch.float32 and g.shape==r.shape==o.shape==a.shape,'Raw tensor shape/dtype/mask differs')
        need(bool(torch.isfinite(g).all() and torch.isfinite(o).all()) and bool((g[a==0]==0).all() and (o[a==0]==0).all()),'Finite/mask failure')
        need(bool(torch.allclose(g,o,atol=1e-4,rtol=1e-4)),'Actual original GPU parity failed')
        expected_p=[256]*16 if step<3 else [256]*15+[2048]
        expected_r=[32 if step==1 else 500]*16 if step<3 else [32 if i%2==0 else 500 for i in range(15)]+[4096]
        pad=manifest['student_tokenizer']['pad_token_id'];token_counts=[]
        need((p!=pad).sum(-1).tolist()==expected_p and a.sum(-1).tolist()==expected_r,'Actual longest/control rows differ')
        aggregate=load(directory/f'receipts-round{step}.json')
        for rank in range(8):
            rec=rows[rank][step-1];owner=manifest['services'][rank];region=slice(rank*2,rank*2+2);indices=torch.arange(rank*2,rank*2+2)
            need(rec==aggregate[rank] and all(rec[k]==manifest[k] for k in keys),'Actual receipt identity/aggregate differs')
            need((rec['rank'],rec['pid'],rec['uuid'],rec['cgroup'],rec['outer_step'])==(rank,owner['pid'],owner['uuid'],manifest['cgroup'],step),'Actual owner/rank/step differs')
            need(rec['input_sha256']==thash(dict(prompts=p[region].contiguous(),responses=r[region].contiguous(),attention=a[region].contiguous(),row_indices=indices)),'Input wire differs')
            need(rec['output_sha256']==thash(dict(teacher_log_probs=g[region].contiguous(),row_indices=indices)) and rec['row_indices_sha256']==thash(dict(row_indices=indices)),'Output/order wire differs')
            need(rec['status']=='RELEASED' and rec['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0),'Actual worker release not zero')
            expected=int((p[region]!=pad).sum()+a[region].sum());need(rec['scored_tokens_delta']==expected,'Actual worker token count differs');token_counts.append(expected)
            for state in (rec['loaded_cpu'],rec['released_cpu'],rec['gpu_before'],rec['gpu_after']):
                if state is None:continue
                hashes.add(state['parameter_sha256']);need(state['dtypes']==['torch.float32'] and not state['training'] and state['training_modules_count']==state['requires_grad_parameters']==state['grad_parameters']==0,'Frozen FP32 state differs')
        odir=directory/'original-gpu-oracle';receipt=load(odir/f'receipt-round{step}.json');request=load(odir/f'request-round{step}.json');incoming=load_file(str(odir/f'input-round{step}.safetensors'),device='cpu');out=load_file(str(odir/f'reference-round{step}.safetensors'),device='cpu')
        need(set(out)=={'original_gpu_log_probs'} and torch.equal(out['original_gpu_log_probs'],o),'Original GPU reference differs')
        need(torch.equal(incoming['prompts'],p) and torch.equal(incoming['responses'],r) and torch.equal(incoming['attention'],a) and torch.equal(incoming['row_indices'],torch.arange(16)),'Oracle full16 exact inputs differ')
        need(receipt['pid']==oracle_ready['pid'] and receipt['uuid']==manifest['services'][4]['uuid'] and all(receipt[k]==manifest[k] for k in keys),'Oracle owner/identity differs')
        need(receipt['status']=='RELEASED' and receipt['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0) and not receipt.get('error') and not receipt.get('release_error'),'Oracle release not zero')
        need(request['candidate_receipts_sha256']==receipt['candidate_receipts_sha256']==sha(directory/f'receipts-round{step}.json'),'Oracle candidate artifact binding differs')
        need(request['input_sha256']==receipt['input_sha256']==sha(odir/f'input-round{step}.safetensors') and receipt['output_sha256']==sha(odir/f'reference-round{step}.safetensors'),'Oracle bytes binding differs')
        need(receipt['row_indices_sha256']==thash(dict(row_indices=incoming['row_indices'])) and receipt['attention_sha256']==thash(dict(attention=a)),'Oracle row/mask SHA differs')
        expected_tokens=[int((p[i]!=pad).sum()+a[i].sum()) for i in range(16)]
        need(receipt['scored_tokens_by_row']==expected_tokens and receipt['scored_tokens_delta']==sum(expected_tokens),'Oracle actual tokens differ')
        flags=receipt['actual_flags'];need(flags['matmul_allow_tf32'] is False and flags['cudnn_allow_tf32'] is False and flags['matmul_precision']=='highest' and not flags['autocast_enabled'],'Oracle actual FP32 flags differ')
        for state in (receipt['loaded_cpu'],receipt['gpu_before'],receipt['gpu_after'],receipt['released_cpu']):
            hashes.add(state['parameter_sha256'])
            need(state['dtypes']==['torch.float32'] and not state['training'] and state['training_modules_count']==state['requires_grad_parameters']==state['grad_parameters']==0,'Oracle frozen FP32 boundary differs')
        comparison=result['ipc_rounds'][step-1]
        times=[comparison['scoring_started_epoch'],comparison['scoring_completed_epoch'],comparison['scorers_released_epoch'],receipt['cuda_started_epoch'],receipt['scoring_started_epoch'],receipt['scoring_completed_epoch'],receipt['completed_epoch'],comparison['original_oracle_completed_epoch']]
        need(all(math.isfinite(t) for t in times) and times==sorted(times),'Scorer/oracle phase overlap or timestamp disorder')
        cpu=None
        if step<3:
            c=tensors['cpu_log_probs'];need(bool(torch.isfinite(c).all()),'Retained CPU diagnostic is nonfinite')
            d=(g-c).abs();bad=d>(1e-4+1e-4*c.abs());claimed=comparison['cpu_cross_backend_diagnostic']
            need(claimed['scope']=='NON_BLOCKING_CROSS_BACKEND_DIAGNOSTIC' and claimed['atol']==claimed['rtol']==1e-4 and claimed['numeric_close']==bool(torch.allclose(g,c,atol=1e-4,rtol=1e-4)) and claimed['max_absolute_delta']==float(d.max()) and claimed['mean_absolute_delta']==float(d.mean()),'CPU diagnostic scope/numbers hidden')
            counts=bad.sum(-1).tolist();need(claimed['violation_count']==int(bad.sum()) and claimed['per_row_violation_counts']==counts,'CPU diagnostic violations hidden')
            cpu=dict(allclose=bool(torch.allclose(g,c,atol=1e-4,rtol=1e-4)),max_abs=float(d.max()),mean_abs=float(d.mean()),violations=int(bad.sum()),per_row_violations=counts)
        else:need(comparison['cpu_cross_backend_diagnostic'] is None,'Long CPU diagnostic fabricated')
        phases.append(dict(step=step,gpu_max_abs=float((g-o).abs().max()),gpu_mean_abs=float((g-o).abs().mean()),all16_rows_gpu_close=True,active_response_tokens=int(a.sum()),oracle_scored_tokens=sum(expected_tokens),perrank_scored_tokens=token_counts,cpu_cross_backend_diagnostic=cpu,ordered_timestamps=times))
    need(len(hashes)==1,'Frozen full parameter hashes differ')
    need(rows[7][2]['scored_tokens_delta']==6432,'Rank7 maximum context missing')
    samples=list(map(json.loads,(directory/'gpu-memory-timeseries.jsonl').read_text().splitlines()))
    mapping=result['worker_gpu_mapping'];vpids=result['owned_gpu_model_pids']['vllm'];retriever=result['owned_gpu_model_pids']['retriever'];coexist=[]
    for sample in samples:
        actual={(line.split(',')[0].strip(),int(line.split(',')[1])):float(line.split(',')[2]) for line in sample['processes']}
        if all(actual.get((s['uuid'],s['pid']),0)>=11000 for s in manifest['services']) and all(actual.get((m['uuid'],pid),0)>=10000 for m,pid in zip(mapping,vpids)) and all((s['uuid'],retriever) in actual for s in manifest['services']):coexist.append(sample['time'])
    need(coexist,'No actual single NVML coexistence sample')
    owners=result['owned_gpu_model_pids'];allpids=set(owners['scorers']+owners['vllm']+[owners['retriever'],owners['original_oracle']]);need(len(allpids)==18,'Not18 distinct owners')
    final=samples[-1];need(not {int(line.split(',')[1]) for line in final['processes']} & allpids,'Actual18 GPU owners did not disappear')
    initial_identities={(line.split(',')[0].strip(),int(line.split(',')[1])) for line in result['original_baseline']['processes']}
    final_identities={(line.split(',')[0].strip(),int(line.split(',')[1])) for line in final['processes']}
    need(initial_identities<=final_identities,'Actual foreign baseline PID/UUID disappeared')
    cleanup=result['own_cleanup_record'];need(cleanup['actual_cgroup_empty'] is True and not cleanup['remaining'] and result['own_children_remaining']==0 and result['own_gpu_model_pids_remaining']==[] and result['baseline_preserved'] is True and result['monitor_shutdown']['stopped'] is True,'Raw cleanup/baseline/monitor failed')
    need(result['original_gpu_oracle_exit_code']==0 and result['parent_driver_cuda_initialized'] is False,'Oracle exit/parent device differs')
    return dict(status='INDEPENDENT_ACTUAL_RAW_CHECKS_PASS_HARDWARE_ONLY_NOT_EXECUTION_APPROVAL',scope='NO_TRAINING_NO_REAL_SELECTED_TEACHER_NO_RL_EFFECT',phases=phases,frozen_parameter_sha256=list(hashes)[0],actual_single_sample_coexistence_times=coexist,actual_owned_gpu_model_processes=18,cleanup_actual_empty=True,
                raw_result_sha256=sha(directory/'result.json'),raw_manifest_sha256=sha(directory/'teacher-ipc/manifest.json'),reader_sha256=sha(__file__),reader_runtime_torch=torch.__version__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(review(a.directory),indent=2)+'\n')
