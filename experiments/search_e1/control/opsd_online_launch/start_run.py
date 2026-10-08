"""Start one admitted full run and independent observer; never signals a process."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--helper-root',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--controller-session',required=True);p.add_argument('--observer-session',required=True)
    a=p.parse_args();sys.path.insert(0,a.helper_root)
    from checkpoint import checked_json,fileproof,exclusive_json,require
    from prepare import validate_config
    from acceptance import process,require_launch
    c,ref=checked_json(dict(path=a.config,sha256=a.config_sha256));validate_config(c,launch=True)
    require(c['run_id']==a.run_id and os.environ.get('E1_RUN_ID')==a.run_id,'Bound new-run environment required')
    require(Path(c['paths']['helper_root']).resolve()==Path(a.helper_root).resolve(),'Wrong helper root')
    require(not Path(c['paths']['output_root']).exists(),'Fresh start refuses existing outputs; preserve and use explicit resume')
    owner=process(os.getpid());require_launch(c,'cpu-merge',a.run_id,owner,[])
    pci=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True)
    rows=[[v.strip() for v in line.split(',')] for line in pci.splitlines()]
    require([int(x[0]) for x in rows]==list(range(8)) and [x[1] for x in rows]==c['owner']['gpu_uuids'],'Physical GPU inventory changed')
    buses=[int(x[2].replace(':','').replace('.',''),16) for x in rows]
    require(buses==sorted(buses),'PCI_BUS_ID ordering does not match physical index inventory')
    base=Path(c['paths']['base']);receipts=Path(c['paths']['receipts']);logs=base/'logs';logs.mkdir(exist_ok=True)
    for name in ('tmp','hf-cache','hf-datasets','cache'):(base/name).mkdir(exist_ok=True)
    for session in (a.controller_session,a.observer_session):
        require(subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,'Existing tmux session; no duplicate launch')
    observer_id='online-memory-'+a.run_id
    shared=dict(c['bootstrap_launch_environment'],PYTHONDONTWRITEBYTECODE='1')
    require(shared['CUDA_DEVICE_ORDER']=='PCI_BUS_ID' and shared['NCCL_P2P_DISABLE']=='1'
        and shared['NCCL_IB_DISABLE']=='1' and shared['VLLM_ATTENTION_BACKEND']=='XFORMERS'
        and float(shared['E1_INFERENCE_UTIL'])==.85,'Changed physical environment')
    python=c['paths']['python'];root=Path(a.helper_root)
    binding=receipts/'memory-binding.json';live=receipts/'memory-live.json'
    def launch(session,cmd,env,log):
        require(not log.exists(),'Refuse overwrite launch log')
        shell=shlex.join(['env']+[key+'='+value for key,value in env.items()]+cmd)+' > '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
        subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
        return dict(session=session,command=cmd,environment=env,log=str(log),shell_sha256=__import__('hashlib').sha256(shell.encode()).hexdigest())
    observer=launch(a.observer_session,[python,'-B',str(root/'memory_observer.py'),'--config',a.config,
        '--output',str(live),'--binding-output',str(binding),'--interval','2'],dict(shared,E1_RUN_ID=observer_id,CUDA_VISIBLE_DEVICES=''),logs/'memory-observer.log')
    deadline=time.monotonic()+30
    while not (binding.exists() and live.exists()):
        require(time.monotonic()<deadline,'Observer did not publish startup; no controller started')
        time.sleep(.5)
    observed,_=checked_json(fileproof(binding));identity=process(observed['observer_identity']['pid'])
    require(all(identity[k]==observed['observer_identity'][k] for k in ('pid','uid','starttick','run_id')),'Observer identity changed')
    sample=json.loads(live.read_text());require(time.time()-sample['updated_unix']<10 and sample['samples_below_limit']==0,'Observer not fresh/healthy before controller')
    require_launch(c,'cpu-merge',a.run_id,owner,[])
    controller=launch(a.controller_session,[python,'-B',str(root/'controller.py'),'--config',a.config,
        '--config-sha256',a.config_sha256,'--run-id',a.run_id],dict(shared,E1_RUN_ID=a.run_id,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),logs/'controller.log')
    result=exclusive_json(receipts/'launch.json',dict(schema='opsd-online-tmux-launch-v1',accepted=True,run_id=a.run_id,
        config=ref,launcher_owner=owner,launcher_source=fileproof(__file__),controller=controller,observer=observer,
        observer_binding=fileproof(binding),physical_index_uuid_pci=rows,uuid_scope='Numeric visible mapping with PCI_BUS_ID, not direct CUDA properties UUID',
        checked_unix=time.time(),five_real_updates_accepted=False,no_signals_sent=True))
    print(json.dumps(result))


if __name__=='__main__':main()
