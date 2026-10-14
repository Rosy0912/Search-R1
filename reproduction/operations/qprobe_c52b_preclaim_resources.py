"""One read-only resource/terminal audit; cannot launch or signal any process."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
RUNTIME = ROOT/'qprobe-teacher-warmup-a800-snapshot-repair'
HANDLE = ROOT/'preparation/qprobe-teacher-warmup-a800-snapshot-repair-deployment.json'
COMMIT = '3bd1c173664c581c0eb6fd9e6a872a718bf0d5b3'
REMOTE = r'''
import ast,csv,hashlib,io,json,os,socket,subprocess,time
from pathlib import Path
payload=json.load(__import__('sys').stdin);intent=payload['intent'];base=Path(intent['base'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-c52b44701a0544bd9e219122e53d3c77'
def read(path):
 assert path.resolve()==path and path.stat().st_uid==os.getuid() and path.stat().st_size<=1024**2
 return json.loads(path.read_text())
run=read(base/'logs/run-manifest.json')
assert read(base/'preparation/qprobe-intent.json')==intent
assert (run['run_id'],run['execution_id'],run['source_commit'])==('afa30a5dab184b32a6c89c155e550235','79fba41ac07a414aaab7f85b87d55c7f','3bd1c173664c581c0eb6fd9e6a872a718bf0d5b3')
assert run['execution_host']==socket.gethostname() and run['source_sha256']==intent['source_sha256']
names=('pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json','pipeline-status.json','native-allocation-admission.json','actual-updates.jsonl')
absent={name:not (base/'logs'/name).exists() for name in names}
receipt=read(base/'logs/deployment-launch-receipt.json')
proc=Path('/proc')/str(receipt['pid']);alive=False
try:
 raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 alive=int(fields[19])==receipt['start_ticks'] and int(fields[3])==receipt['sid'] and proc.stat().st_uid==os.getuid() and fields[0] not in ('Z','X')
except (FileNotFoundError,ProcessLookupError):pass
log=base/'logs/deployment-pipeline.log';assert log.resolve()==log and log.stat().st_uid==os.getuid() and log.stat().st_size<=1024**2
raw=log.read_bytes();prefix='AssertionError: Authorized local GPUs are occupied: '
lines=[line[len(prefix):] for line in raw.decode().splitlines() if line.startswith(prefix)]
assert len(lines)==1;original_occupied=ast.literal_eval(lines[0])
assert all(isinstance(row,list) and len(row)==2 and row[0].startswith('GPU-') and row[1].isdigit() for row in original_occupied)
def query(kind,fields):
 raw=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=30)
 assert len(raw.encode())<=65536
 return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(raw)) if row]
gpus=query('gpu','index,uuid,memory.total,memory.used,utilization.gpu');mapping={row[1]:int(row[0]) for row in gpus}
apps=query('compute-apps','gpu_uuid,pid,used_gpu_memory');assigned={0,1,2,3,5,6};current=[]
for uuid,pid,memory in apps:
 if mapping[uuid] not in assigned:continue
 row=dict(physical_index=mapping[uuid],gpu_uuid=uuid,pid=int(pid),used_memory_mib=memory,belongs_to_audited_namespace=False)
 try:
  p=Path('/proc')/pid;s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();cwd=(p/'cwd').resolve()
  row.update(start_ticks=int(f[19]),sid=int(f[3]),uid=p.stat().st_uid,state=f[0],belongs_to_audited_namespace=cwd==base or base in cwd.parents)
 except (FileNotFoundError,ProcessLookupError,PermissionError):row['process_identity_unknown']=True
 current.append(row)
output=dict(role='qprobe_preclaim_native_resource_observation',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],base=str(base),observed_unix=time.time(),original_pipeline_alive=alive,absence_checks=absent,original_occupied=[dict(physical_index=mapping[uuid],gpu_uuid=uuid,pid=int(pid)) for uuid,pid in original_occupied],current_assigned_compute_apps=current,current_gpu_inventory=gpus,original_log_sha256=hashlib.sha256(raw).hexdigest(),no_signals_or_mutation=True,metadata_only=True)
print('RESOURCE_JSON:'+json.dumps(output),flush=True)
'''


def main():
    own = Path(__file__).resolve()
    ops = own.parents[2]
    head = subprocess.check_output(['git','-C',str(ops),'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(ops),'show',head+':'+str(own.relative_to(ops))])==own.read_bytes()
    intent=json.loads(HANDLE.read_text());assert intent['source_commit']==COMMIT
    path=RUNTIME/'reproduction/a800_common/deploy.py';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==intent['source_sha256']['reproduction/a800_common/deploy.py']
    assert subprocess.check_output(['git','-C',str(RUNTIME),'show',COMMIT+':reproduction/a800_common/deploy.py'])==raw
    spec=importlib.util.spec_from_file_location('read_only_preclaim_route',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ast.parse(REMOTE);route=module.ssh_command(REMOTE);assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    result=subprocess.run(route,input=json.dumps(dict(intent=intent)).encode(),capture_output=True,timeout=90)
    assert len(result.stdout)<=131072
    rows=[json.loads(line.split(':',1)[1]) for line in result.stdout.decode().splitlines() if line.startswith('RESOURCE_JSON:')]
    record=dict(exit_code=result.returncode,observations=rows,stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),operator_commit=head,source_code_sha256=hashlib.sha256(own.read_bytes()).hexdigest(),checked_unix=time.time())
    output=ROOT/'preparation'/('qprobe-c52b-preclaim-resources-'+str(time.time_ns())+'.json')
    with output.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(receipt=str(output),**record)))
    assert result.returncode==0 and len(rows)==1


if __name__=='__main__':main()
