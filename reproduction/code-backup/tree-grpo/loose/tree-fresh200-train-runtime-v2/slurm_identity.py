"""Own local scheduler identity; runtime calls are read-only, no dispatch/cancel."""
import base64
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from production_contract import require, read_bound, ROOT

CLUSTER='wuhan-h100'
NODE='h100-9'
EXCLUDED={'SLURM_CLUSTERS','SLURM_CONF','SLURM_CONF_SERVER','SLURM_CLUSTER_NAME','SLURM_FEDERATION','SLURM_FEDERATION_CLUSTER'}
PREFIXES=('SBATCH_','SQUEUE_','SACCT_','SCONTROL_','SCANCEL_')

def local_environment(environment):
    return {k:v for k,v in environment.items() if k not in EXCLUDED and not k.startswith(PREFIXES)}

def install_local_environment(environment):
    removed=sorted(k for k in environment if k in EXCLUDED or k.startswith(PREFIXES))
    clean=local_environment(environment)
    environment.clear(); environment.update(clean)
    return removed

def runtime_job(environment,hostname):
    require(hostname.split('.')[0]==NODE,'Only the own h100-9 node is permitted')
    job=environment.get('SLURM_JOB_ID')
    require(type(job) is str and re.fullmatch('[1-9][0-9]*',job),'Exact numeric own allocation ID required')
    require(environment.get('SLURM_JOB_NODELIST')==NODE,'Exact one-node runtime allocation required')
    value=environment.get('SLURM_CLUSTER_NAME')
    require(value is None or value==CLUSTER,'Runtime declares a foreign cluster')
    return job

def query(command,run,environment):
    result=run(command,env=local_environment(environment),text=True,capture_output=True,timeout=30)
    require(result.returncode==0 and result.stderr=='','Local scheduler identity query failed or ambiguous')
    return result.stdout

def cluster_identity(run=subprocess.run,environment=None):
    environment=dict(os.environ if environment is None else environment)
    config=query(['scontrol','--local','show','config'],run,environment)
    clusters=re.findall(r'^ClusterName\s*=\s*([A-Za-z0-9_.-]+)\s*$',config,re.MULTILINE)
    require(clusters==[CLUSTER],'Default local controller cluster differs or is ambiguous')
    federation=query(['scontrol','--local','show','federation'],run,environment)
    require(federation.strip()=='','Federated lineage is not permitted for this own task')
    return dict(cluster_name=CLUSTER,configuration_stdout=config,federation_stdout=federation,local_queries=True)

def allocation_identity(text,job,cluster):
    require(cluster==CLUSTER and type(job) is str and re.fullmatch('[1-9][0-9]*',job),'Untrusted allocation tuple')
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_\-/]*)=([^\s]+)',text)
    tokens={}
    for key,value in pairs:
        require(key not in tokens,'Duplicate/ambiguous allocation field: '+key)
        tokens[key]=value
    expected={'JobId':job,'JobState':'RUNNING','NodeList':NODE,'BatchHost':NODE,'NumNodes':'1',
              'OverSubscribe':'NO','TresPerNode':'gres:gpu:8','CPUs/Task':'32',
              'Requeue':'0','Restarts':'0','TimeLimit':'3-00:00:00','MinMemoryNode':'256G'}
    require(all(tokens.get(k)==v for k,v in expected.items()),'Exact own allocation/resources/72h/no-requeue identity differs')
    require(tokens.get('Cluster',CLUSTER)==CLUSTER,'Allocation declares foreign cluster')
    require(re.fullmatch('[1-9][0-9]*',tokens.get('NumCPUs','')) is not None and int(tokens['NumCPUs'])>=32,'Actual reserved CPUs missing')
    return dict(cluster_name=CLUSTER,job_id=int(job),node_list=NODE,requested_cpus=32,
                allocated_cpus=int(tokens['NumCPUs']),gpus=8,memory_gib=256,wall_hours=72,
                no_requeue=True,restarts=0,allocation_stdout=text)

def verify_runtime(environment=None,hostname=None,run=subprocess.run):
    environment=dict(os.environ if environment is None else environment)
    job=runtime_job(environment,socket.gethostname() if hostname is None else hostname)
    cluster=cluster_identity(run,environment)
    raw=query(['scontrol','--local','show','job',job],run,environment)
    return dict(cluster,**{k:v for k,v in allocation_identity(raw,job,cluster['cluster_name']).items() if k!='cluster_name'})

def fsync_directory(path):
    fd=os.open(Path(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def create_journal(path,value):
    """Durable initial intent only; existence always blocks retry."""
    path=Path(path)
    require(value['status']=='SUBMITTING' and value['cluster_name']==CLUSTER,'Explicit own initial journal required')
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    fsync_directory(path.parent)

def finalize_journal(path,value):
    """Keep original SUBMITTING if final replace fails; never remove/retry."""
    path=Path(path);initial,_,_=read_bound(path)
    require(initial['status']=='SUBMITTING' and initial['cluster_name']==value['cluster_name']==CLUSTER,'Own original pending journal required')
    require(all(initial[k]==value[k] for k in ('manifest_sha256','review_sha256','dispatch_intent_path')),'Journal finalization changed dispatch bytes')
    pending=path.with_name(path.name+'.final-'+str(os.getpid()))
    with pending.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path);fsync_directory(path.parent)

def timeout_output(value):
    if isinstance(value,bytes):return {'encoding':'base64','bytes':base64.b64encode(value).decode('ascii')}
    require(value is None or type(value) is str,'Unrecognized timeout evidence type')
    return value

def submitted_lineage(value,job,manifest_sha,review_sha,intent_path):
    require(value['status']=='SUBMITTED' and value.get('cluster_name')==CLUSTER and value.get('cluster') in (None,CLUSTER),'Foreign/legacy/unknown submitted lineage')
    require(type(value.get('job_id')) is int and value['job_id']==int(job),'Same-number aliases/non-numeric IDs forbidden')
    require(value.get('manifest_sha256')==manifest_sha and value.get('review_sha256')==review_sha
            and value.get('dispatch_intent_path')==str(intent_path),'Once journal byte/path binding differs')
    require(value.get('node_list')==NODE and value.get('run_id')=='tree-fresh200-train-'+str(job),'Once journal own node/run binding missing')
    output=value.get('parsable_stdout');require(type(output) is str and output in (str(job),str(job)+';'+CLUSTER),'Foreign/ambiguous parsable suffix')
    require(value.get('stdout') in (output,output+'\n') and value.get('stderr')=='' and value.get('returncode')==0,'Exact successful submit bytes missing')
    proof=value.get('local_cluster_identity')
    require(type(proof) is dict and proof.get('cluster_name')==CLUSTER and proof.get('local_queries') is True,'Original local controller proof missing')
    require(re.findall(r'^ClusterName\s*=\s*([A-Za-z0-9_.-]+)\s*$',proof.get('configuration_stdout',''),re.MULTILINE)==[CLUSTER]
            and proof.get('federation_stdout')=='','Original cluster/federation proof differs')
    return value
