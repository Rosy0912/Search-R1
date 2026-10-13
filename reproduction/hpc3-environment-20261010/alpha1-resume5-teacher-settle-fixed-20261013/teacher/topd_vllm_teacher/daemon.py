"""Owned local TP4/BF16 frozen teacher; one batched full-rollout request/outer."""
import os,sys
sys.dont_write_bytecode=True
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import signal
def terminate(signum,frame):raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate)
import argparse,json,pathlib,secrets,socket,time,hashlib,traceback,subprocess
from .protocol import peer,send,receive,write
from .alignment import pack,unpack

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--directory',required=True);p.add_argument('--preflight',required=True);p.add_argument('--max-model-len',type=int,default=8192);p.add_argument('--gpu-memory-utilization',type=float,default=.25);p.add_argument('--start-outer',type=int,choices=range(1,201),default=1);a=p.parse_args()
    root=pathlib.Path('/data/user/jxu957/experiments/topd-20261008/runtime');teacher=pathlib.Path(a.teacher).resolve();directory=pathlib.Path(a.directory).resolve()
    if root not in teacher.parents or root/'logs' not in directory.parents:raise ValueError('Only owned local paths')
    if not (teacher/'config.json').is_file():raise ValueError('Actual GRPO100 full HF is required')
    import importlib.util
    spec=importlib.util.spec_from_file_location('actual_teacher_lease',a.preflight);pre=importlib.util.module_from_spec(spec);spec.loader.exec_module(pre)
    from .lease_wait import wait_for_lease
    lease_idle_retries=wait_for_lease(pre,4)
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);uuids=[r['uuid'] for r in leases]
    if len(set(uuids))!=4:raise ValueError('Eight actual UUID leases required')
    gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={line.split(',')[1].strip():line.split(',')[0].strip() for line in gpu_inventory.splitlines()}
    if not set(uuids)<=set(mapping):raise ValueError('Leased GPU UUID mapping missing')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(mapping[u] for u in uuids)
    os.umask(0o077);directory.mkdir(mode=0o700,exist_ok=False)
    import torch
    from vllm import LLM,SamplingParams
    from .executor import TeacherExecutor
    from safetensors.torch import load,save
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(teacher,local_files_only=True)
    token_sha=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest()
    llm=LLM(model=str(teacher),tokenizer=str(teacher),dtype='bfloat16',tensor_parallel_size=4,distributed_executor_backend=TeacherExecutor,enforce_eager=True,quantization=None,trust_remote_code=False,max_model_len=a.max_model_len,max_num_batched_tokens=2048,max_num_seqs=16,gpu_memory_utilization=a.gpu_memory_utilization,disable_log_stats=True,enable_prefix_caching=False,enable_chunked_prefill=True)
    executor=llm.llm_engine.model_executor
    executor._run_workers('topd_freeze')
    def release():
        executor.stop_remote_worker_execution_loop()
        records=executor._run_workers('topd_release_kv')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        actual_pairs={(int(line.split(',')[1]),line.split(',')[0].strip()) for line in apps.splitlines()}
        for record in records:
            record['uuid']=uuids[record['rank']]
            if (record['pid'],record['uuid']) not in actual_pairs:raise RuntimeError('Actual teacher PID/leased UUID not observed')
        if len(records)!=4 or {r['rank'] for r in records}!=set(range(4)) or any(not r['cache_engine_absent'] or not r['gpu_cache_absent'] or r['requires_grad_parameters'] or r['model_training'] or r['org_vocab_size']!=152064 or r['model_parameter_dtype']!='torch.bfloat16' for r in records):raise RuntimeError('Actual TP4 frozen/KV release ACK incomplete')
        return records
    ready=release();cap=secrets.token_hex(32);sockpath=directory/'teacher.sock'
    server=socket.socket(socket.AF_UNIX);server.bind(str(sockpath));os.chmod(sockpath,0o600);server.listen(2)
    # capability exists only in private client config, never raw receipts/logs.
    write(directory/'client-config.json',dict(socket=str(sockpath),capability=cap,job=os.environ['SLURM_JOB_ID'],server_pid=os.getpid(),server_starttime=pathlib.Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19],teacher_path=str(teacher),token_vocab_sha256=token_sha,leased_uuids=uuids,max_model_len=a.max_model_len))
    write(directory/'ready.json',dict(status='FROZEN_BF16_TP4_WEIGHTS_RESIDENT_KV_RELEASED',first_expected_outer=a.start_outer,lease_idle_retries=lease_idle_retries,pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],teacher_path=str(teacher),dtype='bfloat16',backend='vllm0.6.3_tp4_prompt_logprobs',quantization=None,full_vocab=152064,no_gold=True,not_fp32_equivalence=True,worker_ack=ready))
    step=a.start_outer;sampler=SamplingParams(temperature=1.,top_p=1.,top_k=-1,repetition_penalty=1.,frequency_penalty=0.,presence_penalty=0.,max_tokens=1,prompt_logprobs=1,detokenize=False)
    try:
        while True:
            with server.accept()[0] as connection:
                connection.settimeout(14400);client=peer(connection,os.environ['SLURM_JOB_ID']);meta,payload=receive(connection)
                if meta.get('operation')=='SHUTDOWN' and meta.get('capability')==cap:
                    break
                if step>200:raise ValueError('Student300 scoring budget exhausted')
                if meta.get('capability')!=cap or meta.get('outer_step')!=step or meta.get('teacher_path')!=str(teacher) or meta.get('temperature')!=1. or meta.get('targets') is not None:raise ValueError('Teacher request authority/order/math')
                tensors=load(payload)
                if set(tensors)!={'prompts','responses','attention'}:raise ValueError('No gold/unknown tensor field allowed')
                prompts,responses,attention=[tensors[k] for k in ('prompts','responses','attention')]
                if any(t.dtype!=torch.long or t.device.type!='cpu' or t.ndim!=2 for t in (prompts,responses,attention)) or responses.shape!=attention.shape or len(prompts)!=len(responses) or not 0<len(prompts)<=2560:raise ValueError('Malformed teacher batch')
                if not bool(((attention==0)|(attention==1)).all()):raise ValueError('Invalid attention mask')
                rows,positions=pack(prompts.tolist(),responses.tolist(),attention.tolist(),int(meta['student_pad_token_id']),a.max_model_len)
                start=time.time();rebuilt=executor._run_workers('topd_rebuild_kv')
                outputs=llm.generate([{'prompt_token_ids':row} for row in rows],sampling_params=sampler,use_tqdm=False)
                # One generated completion is required by vLLM API, discarded entirely.
                values=torch.tensor(unpack(outputs,rows,positions,responses.tolist(),attention.tolist()),dtype=torch.float32)
                ack=release();end=time.time();body=save({'teacher_log_probs':values})
                receipt=dict(status='ACTUAL_TP4_KV_RELEASE_ACK_BEFORE_STUDENT_UPDATE',outer_step=step,job=os.environ['SLURM_JOB_ID'],peer=client,teacher_path=str(teacher),dtype='bfloat16',inference_backend='vllm0.6.3',not_fp32_equivalence=True,no_gold=True,temperature=1.,full_vocab_normalization=152064,rows=len(rows),scored_tokens=sum(map(len,rows)),response_token_count=int(attention.sum()),generated_tokens_used_for_supervision=0,request_sha256=hashlib.sha256(payload).hexdigest(),output_sha256=hashlib.sha256(body).hexdigest(),started_epoch=start,completed_epoch=end,score_and_release_wall_seconds=end-start,weights_resident=True,rebuilt_workers=rebuilt,released_workers=ack,leased_uuids=uuids)
                write(directory/('round-%04d.json'%step),receipt);send(connection,receipt,body)
                final,_=receive(connection)
                if final!={'status':'CLIENT_ACCEPTED_RELEASED_RESULT','outer_step':step,'output_sha256':receipt['output_sha256']}:raise ValueError('Unknown ACK; actor update forbidden')
                step+=1
    except BaseException as error:
        write(directory/'first-error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),epoch=time.time(),outer_step=step))
        raise
    finally:
        server.close()
        executor.shutdown()
if __name__=='__main__':main()
