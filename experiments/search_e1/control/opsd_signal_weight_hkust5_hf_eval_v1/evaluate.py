"""Deferred batched512 evaluation. GPU run is forbidden before all100 outers.

Keeps frozen HF executor semantics and BF16 base/unmerged FP32 LoRA.
The common zero baseline is freshly evaluated with this same engine; old HF
239 is historical only. No training, signals, selected-best or mutable-latest.
"""
import argparse,fcntl,hashlib,importlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
from metrics import aggregate,paired,require
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
PROTOCOL='hf-sdpa-greedy-rep1p05-unmerged-fp32-lora-batch32-fixed512-v1'

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

def prepare_endpoint(request,directory):
    # Original identity reader: raw15/25 and common zero, never cache/best/merge.
    config=read(request['config']);config['_config_sha256']=request['config']['sha256'];modules(config)
    import evaluation
    identity,data=evaluation.input_identity(config,request['completed_outer'],request['checkpoint'])
    require(identity['validation']['sha256']==request['data']['sha256'],'Fixed512 identity differs')
    write(directory/'model-identity.json',dict(accepted=True,identity=identity,protocol=PROTOCOL,
        precision='BF16_base_unmerged_FP32_LoRA',CPU_only=True,no_weight_merge=True))

def worker(request,directory,rank):
    from worker import run_worker
    return run_worker(request,directory,rank)

def completed_training_terminal(status, receipt, pane_terminal):
    """Bind the real replacement supervisor; never reuse failed ed7f976 exit."""
    require(status['phase']=='FOUR_GROUPS_TRAIN_COMPLETE_TESTS_PENDING',
        'All100 training outers must finish first')
    require(receipt['accepted'] and receipt['completed_groups']==4
        and receipt['completed_outers']==100 and receipt['full_training_em_rows']==100,
        'Final native training ledger is not all100')
    owner=receipt['actual_cpu_supervisor']
    require(owner['pid']==2297236 and owner['uid']==1001
        and owner['starttick']==693190113
        and owner['run_id']=='signal-study-port-resume-84064393f92b476ea78f7b0b13b52889',
        'Final training terminal belongs to a different supervisor')
    require('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/port-resume-0146350/opsd_signal_weight_port_resume_v1/entry.py'
        in owner['argv'] and '10d94f6a413924c4f869846989e8fdb7cd34274d9ca8f0a6661111949781f828'
        in owner['argv'],'Actual replacement source/input identity differs')
    require(pane_terminal=='2297233|1|0', 'Actual replacement foreground exit0 required')
    return owner


def run(studyref,publication):
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU HKUST4 parent required')
    pub=read(publication);manifest_path=Path(__file__).parent/'source-manifest.json'
    require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha']
        and proof(manifest_path)['sha256']==pub['source_manifest_sha256'],'Published evaluator identity differs')
    for name,item in json.loads(manifest_path.read_text())['files'].items():
        require(proof(Path(__file__).parent/name)['sha256']==item['sha256'],'Evaluator source changed')
    from dependencies import verify_core
    dependency=verify_core()
    study=read(studyref);require(study['study_base']==str(S),'Authorized study root differs')
    status=json.loads((S/'study-status.json').read_text());require(status['phase']=='FOUR_GROUPS_TRAIN_COMPLETE_TESTS_PENDING','All100 training outers must finish first')
    terminal=subprocess.check_output(['tmux','display-message','-p','-t','e1-signal-port-resume-0146350-20261010','#{pane_pid}|#{pane_dead}|#{pane_dead_status}'],text=True).strip()
    completed_training_terminal(status, read(status['receipt']), terminal)
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
    marker='signal-hf-batch-tests-'+uuid.uuid4().hex;root=S/'hf-batched-endpoint-evaluations'/marker;root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',dict(protocol=PROTOCOL,study=studyref,requests=requests,run_id=marker,old_HF_baseline_is_not_reused=True,evaluation_publication=publication,evaluation_source_manifest=proof(manifest_path),batch_core=dependency))
    baseline=None;summaries=[];replay_inputs=[];bank_ref=None
    env=dict(os.environ,E1_RUN_ID=marker,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HOME=str(S/'cache/hf'),TMPDIR=str(S/'tmp'),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_ATTENTION_BACKEND='XFORMERS',E1_INFERENCE_UTIL='.85')
    for name,req in requests:
        directory=root/name;directory.mkdir();req=dict(req,evaluation_run_id=marker,batch_core=dependency,analysis_contract=dict(configs={g['variant']:g['config'] for g in study['groups']},source_model_manifest=config['source_model_manifest'],runtime_manifest=config['runtime_manifest'],common_initial_adapter=config['common_initial_adapter_manifest']));requestref=write(directory/'request.json',req)
        command=[config['paths']['python'],str(Path(__file__).resolve()),'--request',requestref['path'],'--request-sha256',requestref['sha256'],'--directory',str(directory)]
        with (directory/'identity-preparation.log').open('x') as f:
            cp=subprocess.run(command+['--mode','prepare'],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT)
        write(directory/'identity-preparation-exit.json',dict(actual_wait_returncode=cp.returncode));require(cp.returncode==0,'Raw model identity preparation failed; no GPU start')
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
                try:
                    released=admission(config);write(directory/('wave-%d-release.json'%wave),dict(accepted=True,actual_admission=released,actual_worker_wait_returncodes=actual));break
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
        summary_ref=write(directory/'summary.json',summary);summaries.append(summary)
        from bank_adapter import export_endpoint
        from training_metrics import protocol as strict_protocol
        from search_e1 import core
        from search_r1 import qa_protocol
        replay_ref,new_bank=export_endpoint(root,directory,name,questions,results,summary_ref,req,
            lambda result,answers:strict_protocol(core,qa_protocol,result,answers),write,proof)
        replay_inputs.append(replay_ref)
        if new_bank is not None:bank_ref=new_bank
    write(root/'complete.json',dict(accepted=True,protocol=PROTOCOL,formal_endpoint_tests=8,common_baseline_evaluations=1,evaluation_publication=publication,evaluation_source_manifest=proof(manifest_path),summaries=summaries,fixed_reference_bank=bank_ref,posthoc_replay_inputs=replay_inputs))
    print(json.dumps(dict(accepted=True,output=str(root),formal_tests=8)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['run','prepare','worker'],required=True)
    p.add_argument('--study-manifest');p.add_argument('--study-manifest-sha256');p.add_argument('--publication');p.add_argument('--publication-sha256');p.add_argument('--request');p.add_argument('--request-sha256');p.add_argument('--directory');p.add_argument('--rank',type=int)
    a=p.parse_args()
    if a.mode=='run':run(dict(path=a.study_manifest,sha256=a.study_manifest_sha256),dict(path=a.publication,sha256=a.publication_sha256))
    elif a.mode=='prepare':prepare_endpoint(read(dict(path=a.request,sha256=a.request_sha256)),Path(a.directory))
    else:worker(read(dict(path=a.request,sha256=a.request_sha256)),Path(a.directory),a.rank)
if __name__=='__main__':main()
