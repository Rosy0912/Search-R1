"""Independent A800 G1 pipeline: actual GPU admission, both probes, GRPO, final proof.

R1 must finish its requested update300 and stop first. No foreign Ray shutdown,
no H20 restart, and no use of physical GPUs4/7. Existing ledgers require cleanup
rather than automatic duplicate launch.
"""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from reproduction.a800.owned_processes import OwnedProcesses,publish
from reproduction.a800.prepare_run import digest

ALLOWED=(0,1,2,3,5,6)
_interrupt_requested=None


def request_interrupt(sig,frame):
    # Never raise asynchronously between Popen and durable ownership registration.
    global _interrupt_requested
    _interrupt_requested=sig


def gpu_rows(fields,kind='gpu'):
    return list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True))))


def assert_idle():
    mapping={int(r[0]):r[1].strip() for r in gpu_rows('index,uuid')}
    assert set(ALLOWED)<=set(mapping)
    uuids={mapping[i] for i in ALLOWED}
    busy=[r for r in gpu_rows('gpu_uuid,pid','compute-apps') if r[0].strip() in uuids]
    assert not busy, 'Authorized devices still occupied: '+repr(busy)


def r1_finished(logs=Path('/data/search-r1/logs')):
    state=json.loads((logs/'r1-stop300-status.json').read_text())
    assert state['run_id']=='cc437e7315c94fccba5750433ff850b1'
    if state['state'] not in ('STOPPED_AFTER_VERIFIED_CHECKPOINT','STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT'):return False
    filename='r1-update300-acceptance.json' if state['state']=='STOPPED_AFTER_VERIFIED_CHECKPOINT' else 'r1-update300-failed-snapshot.json'
    proof=json.loads((logs/filename).read_text());assert proof['run_id']==state['run_id'] and proof['target_update']==300
    assert proof['completed_updates']==list(range(101,301)), 'R1 requested actual update sequence incomplete'
    # Both terminal paths retain actual requested checkpoint files and their hashes.
    evidence=proof['checkpoint_evidence'] if 'checkpoint_evidence' in proof else proof
    assert evidence and evidence.get('file_sha256'), 'Inspect terminal R1 checkpoint evidence schema'
    for name,value in evidence['file_sha256'].items():assert digest(Path(evidence['checkpoint'])/name)==value
    return True


def bind_sources(base,run,commit):
    # No G1 process/calibration has been launched yet; never rebind a running run.
    assert not (base/'logs/process-ledger.json').exists()
    assert not list((base/'artifacts').glob('round_*'))
    assert run['initial_probe_version'] is None
    for name,value in run['input_sha256'].items():assert digest(base/'data'/name)==value
    for name,value in run['initial_model_sha256'].items():assert digest(Path(run['initial_model'])/name)==value
    current={str(p.relative_to(base/'repo')):digest(p) for p in sorted((base/'repo').rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.yaml','.yml') and '__pycache__' not in p.parts and '.git' not in p.parts}
    if current!=run['source_sha256']:
        run.setdefault('prelaunch_source_revisions',[]).append(dict(source_commit=run['source_commit'],source_sha256=run['source_sha256'],replaced_unix=time.time(),reason='authorized code preparation before any GPU launch'))
    run.update(source_commit=commit,source_sha256=current,source_bound_unix=time.time())
    publish(base/'logs/run-manifest.json',run)


class Pipeline:
    def __init__(self,base,run):
        self.base=base;self.run=run;self.repo=base/'repo';self.python=str(base/'env/bin/python')
        self.owned=OwnedProcesses(base/'logs/process-ledger.json',run['run_id'])
        self.stop_sampling=threading.Event();self.max_used={};self.sample_error=None
        self.env=dict(os.environ,SEARCH_G1_BASE=str(base),PYTHONPATH=str(self.repo),PATH=str(base/'env/bin')+':'+os.environ['PATH'],
                      HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',
                      TMPDIR=str(base/'tmp'),RAY_TMPDIR=str(base/'ray'),HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
                      TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
                      C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include',
                      NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost')
        for key in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):self.env.pop(key,None)
    def stage(self,name,**extra):publish(self.base/'logs/pipeline-status.json',dict(run_id=self.run['run_id'],stage=name,unix=time.time(),**extra))
    def sample(self):
        path=self.base/'logs'/('gpu-samples-'+self.run['run_id']+'.jsonl')
        try:
            with path.open('a') as f:
                while not self.stop_sampling.is_set():
                    rows=[]
                    for r in gpu_rows('index,memory.used,memory.total,utilization.gpu'):
                        if int(r[0]) in ALLOWED:
                            item=dict(index=int(r[0]),used_mib=int(r[1]),total_mib=int(r[2]),util_percent=int(r[3]));rows.append(item)
                            self.max_used[item['index']]=max(self.max_used.get(item['index'],0),item['used_mib'])
                    f.write(json.dumps(dict(run_id=self.run['run_id'],unix=time.time(),gpus=rows))+'\n');f.flush()
                    self.stop_sampling.wait(2)
        except Exception as e:self.sample_error=repr(e)
    def start(self,name,args,gpus,extra=None):
        assert _interrupt_requested is None, 'Pipeline stop requested before component launch'
        assert set(gpus)<=set(ALLOWED)
        env=dict(self.env,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)));env.update(extra or {})
        process=self.owned.start(name,args,env,self.base/'logs'/(name+'.log'),self.repo)
        self.observe_all() # Deferred stop can now safely clean this registered child.
        return process
    def observe_all(self):
        for name,job in self.owned.jobs.items():
            if job['state']!='STOPPED':self.owned.observe(name)
        assert _interrupt_requested is None, 'Pipeline interrupted; clean up registered owned processes'
        assert self.sample_error is None, 'GPU sampling failed: '+str(self.sample_error)
    def wait(self,name,process,timeout=None):
        started=time.monotonic()
        while True:
            self.observe_all() # before poll can reap a session leader
            code=process.poll()
            if code is not None:
                self.owned.jobs[name]['exit_code']=code;self.owned.save()
                assert code==0, name+' failed; see project log'
                assert not self.owned.observe(name), name+' left owned subprocesses'
                self.owned.jobs[name]['state']='STOPPED';self.owned.save()
                return
            if timeout:assert time.monotonic()-started<timeout, name+' exceeded admission timeout'
            time.sleep(1)
    def endpoint(self,name,process,path,method='get',payload=None):
        import requests
        session=requests.Session();session.trust_env=False;started=time.monotonic()
        while True:
            self.observe_all();assert process.poll() is None, name+' exited during endpoint readiness'
            try:
                response=session.request(method,'http://127.0.0.1:'+path,json=payload,timeout=10);response.raise_for_status();return response.json()
            except requests.RequestException:
                assert time.monotonic()-started<1800, name+' endpoint readiness failed'
                time.sleep(2)
    def execute(self):
        sampler=threading.Thread(target=self.sample,daemon=True);sampler.start()
        try:
            self.stage('FOUR_GPU_ADMISSION')
            p=self.start('gpu-preflight',[self.python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','reproduction/a800/gpu_preflight.py'],[0,1,2,3]);self.wait('gpu-preflight',p,600)
            for rank in range(4):
                proof=json.loads((self.base/'logs'/f'gpu-preflight-rank{rank}.json').read_text());assert proof['run_id']==self.run['run_id']
            self.stage('STARTING_LOCAL_RETRIEVER')
            p=self.start('retriever',[self.python,'reproduction/a800/retriever_local.py','--index_path','/data/search-r1/assets/wiki/e5_Flat.index','--corpus_path','/data/search-r1/assets/wiki/wiki-18.jsonl','--topk','3','--retriever_name','e5','--retriever_model','/data/search-r1/assets/e5-base-v2','--faiss_gpu'],[5,6],{'PYTHONPATH':'/data/search-r1/retriever-gpu-lib:'+str(self.repo)})
            response=self.endpoint('retriever',p,'18010/retrieve','post',{'queries':['capital of France'],'topk':3,'return_scores':True});assert len(response['result'][0])==3
            publish(self.base/'logs/retrieval-http-admission.json',dict(run_id=self.run['run_id'],result=response))
            self.stage('TRAINING_BOTH_INITIAL_PROBES')
            c=self.start('initial-calibration',[self.python,'reproduction/a800/refit.py','--step','0','--model',self.run['initial_model']],[6]);self.wait('initial-calibration',c)
            self.stage('STARTING_FROZEN_PROBE_ENDPOINT')
            e=self.start('probe-server',[self.python,'probe_server.py','--manifest',str(self.base/'artifacts/live/manifest.json'),'--port','18012'],[6])
            health=self.endpoint('probe-server',e,'18012/health');assert health['ok']
            released=self.endpoint('probe-server',e,'18012/suspend','post',{'version':health['version']})
            assert released['suspended'] is True and released['allocated_bytes_after_release']<256*1024*1024
            resumed=self.endpoint('probe-server',e,'18012/resume','post',{'version':health['version']})
            assert resumed['resumed'] is True and resumed['version']==health['version']
            publish(self.base/'logs/evaluator-release-admission.json',dict(run_id=self.run['run_id'],released=released,resumed=resumed))
            a=self.start('probe-admission',[self.python,'reproduction/a800/probe_admission.py'],[6]);self.wait('probe-admission',a,300)
            assert self.max_used, 'Missing actual GPU sampling'
            totals={int(r[0]):int(r[1]) for r in gpu_rows('index,memory.total')}
            assert all(v/totals[i]<=.98 for i,v in self.max_used.items()), 'Measured initial pipeline peak exceeds GPU admission limit'
            publish(self.base/'logs/initial-resource-admission.json',dict(run_id=self.run['run_id'],sampled_peak_mib=self.max_used,sample_interval_seconds=2,scope='initial calibration and probe/retrieval sampled occupancy; not proof of all instantaneous peaks. Periodic refit requires confirmed old-encoder release.',accepted=True))
            self.stage('GRPO_1_TO_300')
            log=Path(self.run['train_log']);assert not log.exists();log.write_text('SEARCH_G1_RUN '+self.run['run_id']+'\n')
            t=self.start('trainer',['bash','reproduction/a800/train.sh'],[0,1,2,3],{'SEARCH_G1_TRAIN_LOG':str(log)});self.wait('trainer',t)
            self.stage('FINAL300_ACCEPTANCE')
            v=self.start('final-acceptance',[self.python,'reproduction/a800/verify_full.py'],[]);self.wait('final-acceptance',v,1800)
            self.stage('FULL300_ACCEPTED_PENDING_OWNED_CLEANUP')
        except Exception as e:
            self.stage('FAILED',error=repr(e));raise
        finally:
            try:self.owned.stop_all()
            except Exception as cleanup_error:
                self.stage('OWNED_CLEANUP_FAILED',error=repr(cleanup_error));raise
            finally:self.stop_sampling.set();sampler.join(5)
        try:
            recorded={r['pid'] for job in self.owned.jobs.values() for r in job['ledger'].values()}
            remaining=[r for r in gpu_rows('gpu_uuid,pid','compute-apps') if int(r[1]) in recorded]
            assert not remaining, 'Recorded project subprocess still occupies GPU'
            assert_idle() # Report unfamiliar occupants; never clean them up.
        except Exception as error:
            self.stage('CLEANUP_VERIFICATION_FAILED',error=repr(error));raise
        self.stage('FINISHED_FULL300_AND_OWNED_CLEANUP')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit');parser.add_argument('--cleanup-owned',action='store_true');parser.add_argument('--wait-for-r1',action='store_true');args=parser.parse_args()
    base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'))
    for name in ('logs','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts'):(base/name).mkdir(parents=True,exist_ok=True)
    lock=(base/'gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.cleanup_owned:
        run=json.loads((base/'logs/run-manifest.json').read_text())
        owned=OwnedProcesses(base/'logs/process-ledger.json',run['run_id']);assert owned.recovery_only
        owned.stop_all()
        recorded={r['pid'] for job in owned.jobs.values() for r in job['ledger'].values()}
        assert not [r for r in gpu_rows('gpu_uuid,pid','compute-apps') if int(r[1]) in recorded]
        publish(base/'logs/cleanup-status.json',dict(run_id=run['run_id'],state='OWNED_CLEANUP_FINISHED',unix=time.time()))
        return
    assert args.source_commit, 'Launch requires committed source identity'
    signal.signal(signal.SIGTERM,request_interrupt)
    signal.signal(signal.SIGINT,request_interrupt)
    if args.wait_for_r1:
        from reproduction.a800.owned_processes import identity
        while not r1_finished():
            assert _interrupt_requested is None, 'Stop requested while waiting for R1'
            handles=[]
            for pid,ticks in ((939751,507870420),(1360378,511195985),(1476488,512002058)):
                try:
                    record=identity(pid)
                    if record['start_ticks']==ticks and record['state']!='Z':handles.append(record)
                except (FileNotFoundError,ProcessLookupError):pass
            assert handles, 'R1 not terminal and no bound runtime/stop monitor remains; inspect before launch'
            publish(base/'logs/pipeline-status.json',dict(stage='WAITING_R1_SAVED_AND_STOPPED_UPDATE300',unix=time.time(),verified_live_r1_handles=handles))
            time.sleep(5)
    else:
        assert r1_finished(), 'R1 requested checkpoint300 has not finished and stopped'
    assert_idle()
    for port in (18010,18012):
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    run=json.loads((base/'logs/run-manifest.json').read_text());bind_sources(base,run,args.source_commit)
    Pipeline(base,run).execute()


if __name__=='__main__':main()
