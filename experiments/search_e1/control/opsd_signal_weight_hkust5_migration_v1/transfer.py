"""Explicit HKUST4 -> HKUST5 pipe: no large local file or payload capture."""
import subprocess,shlex,json,time,threading,hashlib
from pathlib import Path
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ProxyCommand=none','-o','ProxyJump=none','-o','ControlMaster=no','-o','ControlPath=none']
S='weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
O='weka_3/shaoshitong/search-e1-config-study-20261004'
DEST='/home/shaoshitong/search-e1-migration-20261011'
MEMBERS=[S+'/'+x for x in ['assets','runtime','receipts','source','runs','group-H_POS','group-H_NEG','preserved']]+[O+'/runs/archive/round-1-grpo/actor/global_step_100',O+'/data/train.jsonl',O+'/data/validation.jsonl','weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004/receipts/gpu-oracle-acceptance.json']
def run(proof):
    assert all(x.startswith('weka_3/shaoshitong/') and '..' not in x.split('/') for x in MEMBERS)
    source=SSH+['hkust4',shlex.join(['tar','-cf','-','-C','/']+MEMBERS)]
    target=SSH+['hkust5',shlex.join(['tar','-xf','-','-C',DEST])]
    started=time.time();a=subprocess.Popen(source,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    errors=[]
    def drain():errors.append(a.stderr.read().decode(errors='replace'))
    thread=threading.Thread(target=drain);thread.start()
    b=subprocess.Popen(target,stdin=a.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);a.stdout.close()
    out,err=b.communicate();ac=a.wait();thread.join()
    value=dict(source='HKUST4',target='HKUST5',destination=DEST,members=MEMBERS,source_wait=ac,target_wait=b.returncode,source_stderr=errors,target_stderr=err.decode(errors='replace'),started=started,ended=time.time(),large_local_files_created=0,accepted=ac==b.returncode==0)
    Path(proof).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));assert value['accepted']
if __name__=='__main__':
    import sys;run(sys.argv[1])
