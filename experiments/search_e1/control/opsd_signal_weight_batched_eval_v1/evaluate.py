"""Deferred batched512 evaluation. GPU run is forbidden before all100 outers.

Uses frozen native vLLM executor, original CPU PEFT/BF16 merge and strict scorer.
The common zero baseline is freshly evaluated with this same engine; old HF
239 is historical only. No training, signals, selected-best or mutable-latest.
"""
import argparse,fcntl,hashlib,importlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
from metrics import aggregate,paired,require
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
PROTOCOL='vllm-greedy-rep1p05-bf16-merged-fixed512-v1'

def read(ref):
    b=Path(ref['path']).read_bytes();require(hashlib.sha256(b).hexdigest()==ref['sha256'],'Pinned input changed')
    return json.loads(b)

def proof(path):
    b=Path(path).read_bytes();return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return proof(path)

def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]

def modules(config):
    helper=config['paths']['helper_root'];sys.path.insert(0,helper)
    sys.path.insert(0,config['paths']['runtime']);sys.path.insert(0,str(Path(config['paths']['runtime'])/'upstream/Search-R1'))
    return importlib.import_module('gates'),importlib.import_module('sampling')

def admission(config,allowed_run=None,start=True):
    import gates
    devices,apps=gates.inventory();own=[d for d in devices if d['index'] in (4,5,6,7)]
    require(len(own)==4 and [d['uuid'] for d in own]==config['owner']['gpu_uuids'],'Four authorized GPU UUIDs differ')
    oracle=gates.retriever_admission(config,read(config['gates']['retriever_admission']))
    oracle_pid=oracle['owner']['pid']
    for a in apps:
        if a['gpu_uuid'] not in {d['uuid'] for d in own}:continue
        if a['pid']==oracle_pid:require(a['used_mib']<=1024 and a['gpu_uuid']==own[-1]['uuid'],'Oracle sharing differs');continue
        actual=gates.process(a['pid']);require(not start and actual['uid']==1001 and actual['run_id']==allowed_run,'Foreign/old GPU process; never signal it')
    require(min(d['actual_free_mib'] for d in own)>=(20480 if start else 6144),'Direct GPU memory.free below floor')
    return dict(devices=own,oracle=oracle,checked_unix=time.time())

def merge(request,directory):
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only merge required')
    config=read(request['config']);gates,sampling=modules(config);config['_config_sha256']=request['config']['sha256']
    cp=None
    if request['checkpoint']:
        from checkpoint import check_checkpoint
        cp,_=check_checkpoint(request['checkpoint'],config,config['run_id'],expected_outer=request['completed_outer']-1)
    result=sampling.merge_latest(config,cp,str(directory/'model'),request['checkpoint'])
    write(directory/'merged.json',dict(accepted=True,result=result,protocol=PROTOCOL))

def worker(request,directory,rank):
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')==str(4+rank%4)
        and os.environ.get('E1_RUN_ID')==request['evaluation_run_id'],'Worker UID/GPU/run differs')
    config=read(request['config']);gates,sampling=modules(config)
    require(proof(request['data']['path'])['sha256']==request['data']['sha256'],'Worker data SHA differs')
    from search_e1 import vllm_rollout as native,core
    from search_r1 import qa_protocol
    from training_metrics import protocol
    model=read(proof(directory/'merged.json'))['result'];questions=rows(request['data']['path'])[rank::8]
    require(len(questions)==64,'Logical rank must keep64 questions')
    # CPU validated source/data/model once in parent. No per-question hashing or gates.
    old_llm,old_params,old_requests=native.LLM,native.SamplingParams,native.requests
    generated=(directory/('generation-%d.jsonl'%rank)).open('x');http=(directory/('http-%d.jsonl'%rank)).open('x')
    class LoggedLLM:
        def __init__(self,*a,**k):self.inner=old_llm(*a,**k)
        def __getattr__(self,n):return getattr(self.inner,n)
        def generate(self,prompts,params,**kwargs):
            out=self.inner.generate(prompts,params,**kwargs)
            for text,result in zip(prompts,out):generated.write(json.dumps(dict(input=text,generated_token_ids=result.outputs[0].token_ids,
                output=result.outputs[0].text,temperature=params.temperature,repetition_penalty=params.repetition_penalty,
                stop_token_ids=params.stop_token_ids),ensure_ascii=False)+'\n')
            generated.flush();return out
    generation=json.loads((Path(config['paths']['model'])/'generation_config.json').read_text())
    require(float(generation.get('repetition_penalty',1.))==1.05,'Original greedy repetition penalty differs')
    eos=generation['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    def params(**kwargs):return old_params(**kwargs,repetition_penalty=1.05,stop_token_ids=eos)
    proxy=sampling.RequestsTraceProxy(old_requests,http,rank,model['sha256'])
    native.LLM,native.SamplingParams,native.requests=LoggedLLM,params,proxy
    resultpath=directory/('results-%d.jsonl'%rank)
    try:
        with resultpath.open('x') as output:
            for r in native.batched_rollouts(model['path'],questions,1,config['paths']['retriever_url'],0.,42+rank,64):
                r['reward']=core.reward(r['response'],r['answers'],r['policy_spans'])
                r['protocol']=protocol(core,qa_protocol,r,r['answers'])
                output.write(json.dumps(r,ensure_ascii=False)+'\n');output.flush()
    finally:
        native.LLM,native.SamplingParams,native.requests=old_llm,old_params,old_requests;generated.close();http.close()
    require(len(rows(resultpath))==64,'Incomplete64-row shard')
    write(directory/('worker-%d-complete.json'%rank),dict(accepted=True,rank=rank,results=proof(resultpath),request=request,
        generation=proof(directory/('generation-%d.jsonl'%rank)),http=proof(directory/('http-%d.jsonl'%rank)),protocol=PROTOCOL))

def run(studyref,publication):
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU HKUST4 parent required')
    pub=read(publication);manifest_path=Path(__file__).parent/'source-manifest.json'
    require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha']
        and proof(manifest_path)['sha256']==pub['source_manifest_sha256'],'Published evaluator identity differs')
    for name,item in json.loads(manifest_path.read_text())['files'].items():
        require(proof(Path(__file__).parent/name)['sha256']==item['sha256'],'Evaluator source changed')
    study=read(studyref);require(study['study_base']==str(S),'Authorized study root differs')
    status=json.loads((S/'study-status.json').read_text());require(status['phase']=='FOUR_GROUPS_TRAIN_COMPLETE_TESTS_PENDING','All100 training outers must finish first')
    terminal=subprocess.check_output(['tmux','display-message','-p','-t','e1-signal-training-priority-ed7f976-20261010','#{pane_pid}|#{pane_dead}|#{pane_dead_status}'],text=True).strip()
    require(terminal=='1861244|1|0','Actual training supervisor exit0 required')
    configs=[read(g['config']) for g in study['groups']];config=configs[0];gates,_=modules(config)
    require([c['variant'] for c in configs]==['G_POS','G_NEG','H_POS','H_NEG'],'Four-arm order differs')
    from prepare import validate_config
    for c in configs:
        validate_config(c,launch=True)
        require(c['source_model_manifest']==config['source_model_manifest'] and c['common_initial_adapter_manifest']==config['common_initial_adapter_manifest']
            and c['validation_identity']==config['validation_identity'],'Common source/zero/data differ')
    locks=[]
    names={str(S/'study-sequence.lock')}
    for c in configs:
        names.add(str(Path(c['paths']['base'])/'online-experiment.lock'))
        names.update(c['resources']['shared_gpu_lease_paths']+[c['resources']['study_lease_path']])
    for name in sorted(names):
        f=open(name,'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    admission(config)
    from prepare import verify_runtime,verify_common
    verify_runtime(config);verify_common(config)
    data=config['validation_identity']
    # Fixed-data SHA already verified above; exactly the same512 and strict QA.
    require(proof(data['path'])['sha256']==data['sha256'],'Fixed512 SHA differs')
    questions=rows(data['path']);require(len(questions)==512,'Fixed validation512 required')
    requests=[]
    common=dict(config=study['groups'][0]['config'],checkpoint=None,completed_outer=0,data=data)
    requests.append(('common-zero',common))
    for group,c in zip(study['groups'],configs):
        for completed in (15,25):
            completion=Path(c['paths']['output_root'])/'outers'/('outer-%05d'%(completed-1))/'training/outer-complete.json'
            value=json.loads(completion.read_text());require(value['accepted'] and value['outer_index']==completed-1,'Saved raw endpoint incomplete')
            requests.append((group['variant']+'-%02d'%completed,dict(config=group['config'],checkpoint=value['checkpoint'],completed_outer=completed,data=data)))
    marker='signal-batched-tests-'+uuid.uuid4().hex;root=S/'batched-endpoint-evaluations'/marker;root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',dict(protocol=PROTOCOL,study=studyref,requests=requests,run_id=marker,old_HF_baseline_is_not_reused=True))
    baseline=None;summaries=[]
    env=dict(os.environ,E1_RUN_ID=marker,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HOME=str(S/'cache/hf'),TMPDIR=str(S/'tmp'),VLLM_ATTENTION_BACKEND='XFORMERS',E1_INFERENCE_UTIL='.85')
    for name,req in requests:
        directory=root/name;directory.mkdir();req=dict(req,evaluation_run_id=marker);requestref=write(directory/'request.json',req)
        command=[config['paths']['python'],str(Path(__file__).resolve()),'--request',requestref['path'],'--request-sha256',requestref['sha256'],'--directory',str(directory)]
        with (directory/'merge.log').open('x') as f:
            cp=subprocess.run(command+['--mode','merge'],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT)
        write(directory/'merge-exit.json',dict(actual_wait_returncode=cp.returncode));require(cp.returncode==0,'CPU merge failed; retained, no GPU start')
        codes=[]
        for wave in range(2):
            admission(config);jobs=[]
            for rank in range(wave*4,wave*4+4):
                f=(directory/('worker-%d.log'%rank)).open('x');p=subprocess.Popen(command+['--mode','worker','--rank',str(rank)],
                    env=dict(env,CUDA_VISIBLE_DEVICES=str(4+rank%4)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((p,f))
            while any(p.poll() is None for p,f in jobs):
                try:admission(config,marker,start=False)
                except Exception as e:print('RUNNING_ADMISSION_OBSERVATION',type(e).__name__,str(e),flush=True)
                time.sleep(10)
            actual=[p.wait() for p,f in jobs]
            for p,f in jobs:f.close()
            codes+=actual;write(directory/('wave-%d-exit.json'%wave),dict(actual_worker_wait_returncodes=actual))
            require(actual==[0]*4,'Actual worker failure retained; no assumed completion')
            for retry in range(30):
                try:admission(config);break
                except (OSError,gates.Hold,ValueError):time.sleep(2)
            else:raise RuntimeError('GPU release unconfirmed; no nextwave launched')
        results=[]
        for rank in range(8):
            complete=read(proof(directory/('worker-%d-complete.json'%rank)));require(complete['accepted'],'Complete shard missing');results+=rows(complete['results']['path'])
        metrics=aggregate(results,questions)
        if baseline is None:baseline=results
        comparison=paired(results,baseline)
        summary=dict(accepted=True,endpoint=name,protocol=PROTOCOL,metrics=metrics,paired=comparison,
            raw_checkpoint=req['checkpoint'],actual_worker_wait_returncodes=codes)
        write(directory/'summary.json',summary);summaries.append(summary)
    write(root/'complete.json',dict(accepted=True,protocol=PROTOCOL,formal_endpoint_tests=8,common_baseline_evaluations=1,summaries=summaries))
    print(json.dumps(dict(accepted=True,output=str(root),formal_tests=8)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['run','merge','worker'],required=True)
    p.add_argument('--study-manifest');p.add_argument('--study-manifest-sha256');p.add_argument('--publication');p.add_argument('--publication-sha256');p.add_argument('--request');p.add_argument('--request-sha256');p.add_argument('--directory');p.add_argument('--rank',type=int)
    a=p.parse_args()
    if a.mode=='run':run(dict(path=a.study_manifest,sha256=a.study_manifest_sha256),dict(path=a.publication,sha256=a.publication_sha256))
    elif a.mode=='merge':merge(read(dict(path=a.request,sha256=a.request_sha256)),Path(a.directory))
    else:worker(read(dict(path=a.request,sha256=a.request_sha256)),Path(a.directory),a.rank)
if __name__=='__main__':main()
