"""Job3231-only guards; no receipt signing and no remote execution."""
import hashlib,json,os,re,stat
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
SOURCE='39cdcdd1404213b897bb0de99cc52d72707682f0'
JOB=3231
RUN=ROOT/'runs'/f'bapo-instruct300-formal300-{JOB}'
EXTRACTOR_SHA='a27674a26f4958f9514c19be6f9f5ba5fab9bfd640c586b5bf293266da29ee74'
def require(value,message):
    if not value:raise ValueError(message)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def owned_file(path,base=RUN):
    path=Path(path)
    require(path.is_absolute() and path==path.resolve() and base in path.parents,'Exact owned absolute path required')
    require(stat.S_ISREG(path.lstat().st_mode),'Regular nonsymlink file required')
    return path
def read(path):return json.loads(Path(path).read_text())
def events(text,marker):return [json.loads(line.split(marker,1)[1])for line in text.splitlines()if marker in line]
def evidence_identity(e):
    require(e.get('extraction_complete')is True and e.get('issues')==[] and e.get('actual_training_accepted')is False,'Complete unsigned extraction required')
    require(type(e.get('job_id'))is int and e['job_id']==JOB and e.get('run_root')==str(RUN) and e.get('source_expected')==SOURCE,'Extraction job/run/source mismatch')
    require(e.get('tool',{}).get('sha256')==EXTRACTOR_SHA,'Reviewed extractor hash mismatch')
    identity=e.get('release_identity',{})
    require(identity.get('run_root')==str(RUN)and identity.get('source_commit')==SOURCE,'Release identity mismatch')
    for key,value in [('global_step',5),('actual_outer_updates',5),('horizon',300),('warmup_steps',114)]:
        require(type(identity.get(key))is int and identity[key]==value,'Wrong actual gate '+key)
    pid=e.get('expected_driver_pid');require(type(pid)is int and pid>0,'Actual driver PID required')
    require(e.get('driver_state',{}).get('driver_pid')==pid,'Driver state PID mismatch')
    processes=e.get('actual_processes_after_cpu',[])
    require(len(processes)==9 and len({p.get('pid')for p in processes})==9,'Actual nine process proofs required')
    require(type(e.get('driver_state',{}).get('driver_pid'))is int,'Driver PID must be integer')
    driver=next((p for p in processes if p.get('pid')==pid),None)
    require(driver is not None and type(driver.get('start_ticks'))is int and driver['start_ticks']>0,'Driver lifecycle proof missing')
    for p in processes:
        require(type(p.get('pid'))is int and p['pid']>0 and type(p.get('start_ticks'))is int and p['start_ticks']>0 and p.get('state')not in ('Z','X'),'Bad process proof')
        require(p.get('selected_env',{}).get('SLURM_JOB_ID')==str(JOB)and p['selected_env'].get('BAPO_RUN_ID')==str(RUN),'Process job/run mismatch')
    return pid,processes

def live_pause(e):
    pid,processes=evidence_identity(e)
    prefix=e['train_log_prefix'];snapshot=owned_file(prefix['snapshot_path'])
    require(prefix['live_path']==str(RUN/'train.log')and type(prefix['bytes'])is int and prefix['bytes']>0,'Wrong prefix identity')
    require(snapshot.stat().st_size==prefix['bytes']and digest(snapshot)==prefix['sha256'],'Snapshot changed')
    with owned_file(RUN/'train.log').open('rb')as f:require(hashlib.sha256(f.read(prefix['bytes'])).hexdigest()==prefix['sha256'],'Actual live prefix changed')
    manifest=owned_file(RUN/'checkpoints/full_state/global_step_5/manifest.json')
    require(digest(manifest)==e['checkpoint_manifest']['sha256'],'Actual manifest changed')
    text=(RUN/'train.log').read_text();paused=events(text,'E1_STEP5_PAUSED_JSON=')
    require(len(paused)==1 and type(paused[0].get('pid'))is int and paused[0]['pid']==pid and paused[0]['manifest_path']==str(RUN/'checkpoints/full_state/global_step_5/manifest.json') and paused[0]['manifest_sha256']==e['checkpoint_manifest']['sha256'],'Actual pause identity mismatch')
    require(len(events(text,'BAPO_TRAIN_METRICS_JSON='))==5 and not events(text,'E1_STEP5_RELEASED_JSON='),'Run advanced/released')
    require(not os.path.lexists(RUN/'logs/step5-independent-release.json'),'Release already present')
    return pid,processes

def proc_proof(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    env=dict(x.split('=',1)for x in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0')if '='in x)
    return dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],selected_env={k:env.get(k)for k in ('SLURM_JOB_ID','BAPO_RUN_ID')})
def check_processes(processes):
    for previous in processes:
        now=proc_proof(previous['pid'])
        require(now['start_ticks']==previous['start_ticks']and now['state']not in ('Z','X')and now['selected_env']=={k:previous['selected_env'][k]for k in ('SLURM_JOB_ID','BAPO_RUN_ID')},'Actual process lifecycle changed')
def validate_slurm(text):
    require(re.search(r'(?:^|\s)JobId=3231(?:\s|$)',text)and re.search(r'(?:^|\s)JobState=RUNNING(?:\s|$)',text)and re.search(r'(?:^|\s)Restarts=0(?:\s|$)',text)and re.search(r'(?:^|\s)NodeList=h100-5(?:\s|$)',text),'Actual Slurm job/state/node/restarts mismatch')
def signed_binding(receipt,evidence_path,evidence_sha):
    require(receipt.get('status')=='PASS_INDEPENDENT_E1_STEP5_REVIEW','New independent PASS receipt required')
    bindings={x['path']:x['sha256']for x in receipt.get('evidence',[])}
    require(bindings.get(str(evidence_path))==evidence_sha,'Receipt must bind actual reviewed extraction evidence')

def publish_link(receipt,release,expected_sha):
    receipt=owned_file(receipt)
    require(release==RUN/'logs/step5-independent-release.json'and not os.path.lexists(release),'Exact fresh release path required')
    require(digest(receipt)==expected_sha,'Receipt changed before atomic publish')
    os.link(receipt,release,follow_symlinks=False)
