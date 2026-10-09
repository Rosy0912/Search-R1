"""Generate isolated zero-update retry tools; never rewrites old remote artifacts."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
OLD=ROOT/'tools/deployment';NEW=OLD/'formal-rayip-v2';NEWHEAD=ROOT/'scheduler-head/formal-rayip-v2'
OUT=HERE/'formal-rayip-v2';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
rawdir=HERE.parents[2]/'evidence/aligned-em200-formal3348-failure-full'
manifest=json.loads((rawdir/'file-manifest.json').read_text())
for row in manifest:
 p=rawdir/Path(row['path']).relative_to(ROOT)
 if sha(p.read_bytes())!=row['sha256']:raise ValueError('original failure evidence changed: '+str(p))
previous={r['path']:r['sha256'] for r in manifest}
change=[]
files=['formal.sbatch','formal_worker.sh','formal_inner.sh','formal_driver.py','ray_cli_compat.py','formal_service_proof.py','bind_retrieval_proof.py','submit_formal.py']
for name in files:
 raw=(rawdir/'tools/deployment'/name).read_bytes();s=raw.decode()
 if name!='bind_retrieval_proof.py':
  s=s.replace('tools/deployment','tools/deployment/formal-rayip-v2')
 if name=='submit_formal.py':
  s=s.replace("head=ROOT/'scheduler-head'","head=ROOT/'scheduler-head/formal-rayip-v2'")
  s=s.replace("lock=head/'submission-global.lock'","lock=ROOT/'scheduler-head/submission-global.lock'")
  s=s.replace("value=dict(source=SOURCE", "value=dict(deployment_directory=str(ROOT/'tools/deployment/formal-rayip-v2'),submission_namespace='formal-rayip-v2',previous_failed_job='3348',source=SOURCE")
 if name=='formal_service_proof.py':
  s=s.replace('import argparse, json, os, sys, time','import argparse, json, os, sys, time, subprocess, ipaddress')
  start=s.index('\ndef verify_ray_connection():')
  helpers='''\ndef active_local_ipv4():
    result=subprocess.run(['ip','-j','-4','address','show','up'],capture_output=True,text=True,timeout=10,check=True)
    rows=json.loads(result.stdout)
    require(isinstance(rows,list) and rows,'actual local interface inventory absent')
    addresses=set()
    for interface in rows:
        if 'UP' not in interface.get('flags',[]) or (interface.get('operstate')!='UP' and 'LOOPBACK' not in interface.get('flags',[])):continue
        for address in interface.get('addr_info',[]):
            if address.get('family')=='inet':
                ip=ipaddress.IPv4Address(address['local'])
                require(not ip.is_unspecified and not ip.is_multicast,'invalid local interface address')
                addresses.add(str(ip))
    require(addresses,'no active local IPv4')
    return dict(addresses=sorted(addresses),interfaces=rows)


def validate_local_ray_node(nodes, local):
    require(len(nodes)==1 and nodes[0].get('Resources',{}).get('GPU')==8,'connected Ray is not one eight-GPU node')
    ip=ipaddress.IPv4Address(nodes[0].get('NodeManagerAddress',''))
    require(str(ip) in local['addresses'],'Ray advertises a nonlocal/inactive interface address')
    require(type(nodes[0].get('NodeManagerPort')) is int and 0<nodes[0]['NodeManagerPort']<65536,'invalid own raylet port')
    return str(ip)

'''
  s=s[:start]+helpers+s[start:]
  s=s.replace("require(len(nodes)==1 and nodes[0]['NodeManagerAddress']=='127.0.0.1' and nodes[0]['Resources'].get('GPU')==8, 'connected Ray is not the own eight-GPU head')","local=active_local_ipv4()\n        advertised_ip=validate_local_ray_node(nodes,local)")
  s=s.replace("validate_services()\n        freeze('own-formal-ray-connection.json',dict(node=nodes[0],raylet=raylet,session_dir=str(session),address='127.0.0.1:16383'))","require(port_owner(int(nodes[0]['NodeManagerPort']))==raylet,'own raylet changed during validation')\n        require(advertised_ip in active_local_ipv4()['addresses'],'local Ray interface disappeared')\n        validate_services()\n        freeze('own-formal-ray-connection.json',dict(node=nodes[0],raylet=raylet,session_dir=str(session),address='127.0.0.1:16383',advertised_ip=advertised_ip,local_interface_evidence=local))")
 (OUT/name).write_text(s);change.append({'file':name,'original_sha256':sha(raw),'retry_sha256':sha(s.encode())})
admission=(rawdir/'tools/deployment/formal_admission.py').read_text()
# Preserve all original hardware references/UUID/input/source gates; only formal tools and receipt are namespaced.
admission=admission.replace("ROOT / 'tools/deployment' / name","ROOT / 'tools/deployment/formal-rayip-v2' / name")
admission=admission.replace("'tools/deployment', 'repo'","'tools/deployment/formal-rayip-v2', 'repo'")
admission=admission.replace("ROOT / 'scheduler-head/formal-submit-receipt.json'","ROOT / 'scheduler-head/formal-rayip-v2/formal-submit-receipt.json'")
lines=admission.splitlines();pins={n:sha((OUT/n).read_bytes()) for n in files if n!='submit_formal.py'}
lines=[('FORMAL_FILES = '+repr(pins)) if line.startswith('FORMAL_FILES =') else line for line in lines];admission='\n'.join(lines)+'\n'
pre='''\ndef validate_previous_failure():
    for original,expected in PREVIOUS_FAILURE_FILES.items():
        require(sha(original)==expected,'immutable original3348 failure evidence changed')
    old=ROOT/'runs/formal-aligned-em200-66cb18840a404586ab3333f8c393b8b7-3348'
    require(raw_file(old/'original-exit-code.txt').strip()==b'1' and raw_file(old/'cleanup-exit-code.txt').strip()==b'0','old own startup failure/cleanup differ')
    cleanup=json.loads(raw_file(old/'own-process-cleanup.json'))
    require(cleanup.get('own_marker')==str(old) and cleanup.get('remaining')==[],'old task cleanup incomplete')
    text=raw_file(old/'train.log').decode()
    require('FRESH200_ATTEMPT_JSON=' not in text and 'BAPO_TRAIN_METRICS_JSON=' not in text and not (old/'checkpoints').exists(),'previous attempt has updates/state: zero-update retry forbidden')
    trace=raw_file(old/'driver-time.log').decode()
    require('connected Ray is not the own eight-GPU head' in trace,'different startup failure')
    queue=subprocess.run(['squeue','-h','-j','3348','-o','%i'],capture_output=True,text=True,timeout=15)
    require(not queue.stdout.strip() and (queue.returncode==0 or 'Invalid job id' in queue.stderr),'old3348 allocation remains')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:env=(proc/'environ').read_bytes().split(b'\\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        require(('BAPO_RUN_ID='+str(old)).encode() not in env,'original3348 process remains; no duplicate retry')

'''
admission=admission.replace('\ndef validate_admission():','\nPREVIOUS_FAILURE_FILES = '+repr(previous)+'\n'+pre+'\ndef validate_admission():')
admission=admission.replace('hardware = validate_hardware()','hardware = validate_hardware()\n    validate_previous_failure()')
admission=admission.replace("    require(receipt.get('scope') == 'ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION', 'wrong submission scope')", "    require(receipt.get('deployment_directory') == str(ROOT/'tools/deployment/formal-rayip-v2') and receipt.get('submission_namespace') == 'formal-rayip-v2' and receipt.get('previous_failed_job') == '3348', 'foreign retry namespace')\n    require(receipt.get('scope') == 'ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION', 'wrong submission scope')")
(OUT/'formal_admission.py').write_text(admission)
# Isolated evidence consumers: algorithm/CPU load/hash logic is byte-for-byte unchanged except fixed new receipt/tool directory paths.
e=OUT/'evidence-tools';e.mkdir()
for name in ['aligned_first5_cpu_evidence.py','aligned_first5_math.py','aligned-first5-math-provenance.json','publish_first5_release.py','aligned_final_evidence.py','aligned_final_phase2.py','collect_hardware_evidence.py']:
 s=(HERE/name).read_text()
 s=s.replace('scheduler-head/formal-submit-receipt.json','scheduler-head/formal-rayip-v2/formal-submit-receipt.json')
 s=s.replace("'tools/deployment/first5-cpu-v1'","'tools/deployment/formal-rayip-v2/evidence-tools'")
 s=s.replace("'tools/deployment/formal.sbatch'","'tools/deployment/formal-rayip-v2/formal.sbatch'")
 s=s.replace("'tools/deployment' / name","'tools/deployment/formal-rayip-v2' / name").replace("'tools/deployment'/name","'tools/deployment/formal-rayip-v2'/name")
 if name=='aligned_first5_cpu_evidence.py':
  s=s.replace("job not in ('3315','3345')", "job not in ('3315','3345','3348')")
  s=s.replace("    job,token=receipt.get('job'),receipt.get('token')", "    require(receipt.get('deployment_directory')==str(root/'tools/deployment/formal-rayip-v2') and receipt.get('submission_namespace')=='formal-rayip-v2' and receipt.get('previous_failed_job')=='3348','foreign retry receipt namespace')\n    job,token=receipt.get('job'),receipt.get('token')")
 (e/name).write_text(s)
(OUT/'isolation-manifest.json').write_text(json.dumps({'source':'080ebbc46b89d61567cd33e1a5e2f1b40afd8b84','root':str(ROOT),'deployment':str(NEW),'scheduler':str(NEWHEAD),'shared_global_submission_lock':str(ROOT/'scheduler-head/submission-global.lock'),'shared_global_execution_lock':str(ROOT/'scheduler-head/execution-global.lock'),'changes':change,'original_failure_files':previous,'scope':'ISOLATED_ZERO_UPDATE_RETRY_CANDIDATE_NOT_EXECUTED_OR_ACCEPTED'},indent=2)+'\n')
print(OUT)
