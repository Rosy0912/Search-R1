"""User-authorized pending-job update; retain original submission and runtime."""
import datetime,json,os,re,subprocess
from pathlib import Path
BASE=Path('/mnt/data/shaoshitong/search-g1-speed-resume5-653712b-20261012')
AUTHORIZATION='允许核实并改为Slurm共用，保留视频和8卡训练预算'

def query(job):
    p=subprocess.run(['scontrol','show','job',str(job),'-o'],capture_output=True,text=True)
    return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def fields(text):return dict(re.findall(r'(?:^|\s)(\w+)=(\S+)',text))
def write_once(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())

def main():
    intent=BASE/'preparation/slurm-shared-update-intent-v1.json'
    receipt=BASE/'preparation/slurm-shared-update-receipt-v1.json'
    assert not intent.exists() and not receipt.exists(),'Inspect existing intent/result and actual job; do not replay'
    submitted=json.loads((BASE/'preparation/native-submit-receipt.json').read_text())
    assert submitted['slurm_job_id']=='3377' and submitted['source_commit']=='653712ba5cb3f7f96dcdf570625cc789b77e3024'
    before=query(3377);assert before['returncode']==0;job=fields(before['stdout'])
    assert job['JobState']=='PENDING' and job['JobName']=='search-g1-mixed200'
    assert job['ReqNodeList']=='h100-8' and job['TresPerNode']=='gres:gpu:h100:8'
    assert job['Command']==str(BASE/'preparation/launch-mixed-native.sh')
    assert job['OverSubscribe']=='NO' and job['NumCPUs']=='96' and job['MinMemoryNode']=='512G'
    command=['scontrol','update','JobId=3377','OverSubscribe=YES']
    write_once(intent,dict(authorization=AUTHORIZATION,authorization_source='direct user question reply',
        authorization_recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='only pending own3377; preserve video3371 and8GPU budget',
        before=before,video_before=query(3371),command=command,training_runtime_unchanged=True))
    p=subprocess.run(command,capture_output=True,text=True,timeout=30)
    after=query(3377)
    result=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,after=after,video_after=query(3371),
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_once(receipt,result)
    print(json.dumps(result),flush=True)
    assert p.returncode==0 and after['returncode']==0
    current=fields(after['stdout']);assert current['OverSubscribe']!='NO'
    assert current['TresPerNode']==job['TresPerNode'] and current['MinMemoryNode']==job['MinMemoryNode']
    assert current['Command']==job['Command'] and current['NumCPUs']==job['NumCPUs']
if __name__=='__main__':main()
