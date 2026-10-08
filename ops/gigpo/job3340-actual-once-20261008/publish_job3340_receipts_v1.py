import json, pathlib, hashlib, datetime
from publish_cudasynchronous_v19_once import BASE, OUT, REPO, run, sha

def main():
    target='ops/gigpo/job3340-actual-once-20261008'
    inputs=[pathlib.Path(__file__).resolve(), BASE/'work/record_xformers_v20_submission_actual_v1.py',
        BASE/'work/read_gigpo_job3340_retained18_actual_template.sh', BASE/'work/read_gigpo_job3332_actual.py', BASE/'work/record_job3340_actual_observation_v1.py',
        pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/FAILURES-AND-FIXES-20261008.md')]
    inputs += [OUT/n for n in ['FRESH200-XFORMERS-V20-PUBLICATION-v1.json',
        'FRESH200-XFORMERS-V20-GATE-TEMPLATE-v1.json', 'FRESH200-XFORMERS-V20-ACTUAL-GATE-v1.json',
        'FRESH200-XFORMERS-V20-REPAIR3338-SUBMIT-LOCAL-INTENT-v1.json',
        'FRESH200-XFORMERS-V20-REPAIR3338-SUBMIT-TRANSPORT-v1.json',
        'JOB3340-XFORMERS-ACTUAL-SUBMISSION-v1.json', 'RETAINED18-JOB3340-ACTUAL-STATUS-v1-actual-allocation.json',
        'RETAINED18-JOB3340-ACTUAL-STATUS-v2-actual-startup.json']]
    assert not run(['git','status','--porcelain']).strip()
    rows=[]
    for p in inputs:
        raw=p.read_bytes();rel=target+'/'+p.name;dst=REPO/rel
        dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();dst.write_bytes(raw)
        rows.append(dict(path=rel,bytes=len(raw),sha256=sha(raw),original_source_path=str(p)))
    run(['git','add','--',target]);run(['git','commit','-m','Preserve consumed XFormers job3340 submission and actual startup observers'])
    commit=run(['git','rev-parse','HEAD']).decode().strip();branch=run(['git','branch','--show-current']).decode().strip()
    run(['git','push','origin','HEAD:refs/heads/'+branch],timeout=180)
    remote=run(['git','ls-remote','origin','refs/heads/'+branch],timeout=60).decode().split()[0];assert remote==commit
    bare=BASE/'work/xformers-v20-independent-git-readback.git'
    run(['git','fetch','--depth=1','--filter=blob:none','origin',commit],cwd=bare,timeout=180)
    tree=run(['git','ls-tree','-r',commit,'--',target],cwd=bare).decode().splitlines()
    blobs={line.split('\t',1)[1]:line.split()[2] for line in tree};assert set(blobs)=={r['path'] for r in rows}
    ids=list(dict.fromkeys(blobs.values()));run(['git','fetch','--no-tags','origin']+ids,cwd=bare,timeout=180)
    data=run(['git','cat-file','--batch'],cwd=bare,input=('\n'.join(ids)+'\n').encode(),timeout=120)
    offset=0;actual={}
    for oid in ids:
        nl=data.index(b'\n',offset);h=data[offset:nl].decode().split();assert h[:2]==[oid,'blob']
        length=int(h[2]);actual[oid]=data[nl+1:nl+1+length];offset=nl+2+length
    assert offset==len(data)
    for row in rows:
        raw=actual[blobs[row['path']]];assert sha(raw)==row['sha256'] and len(raw)==row['bytes'];row['downloaded_bytes_match']=True
    record=dict(status='PUSHED_AND_INDEPENDENT_GIT_DOWNLOAD_VERIFIED',new_commit=commit,remote_sha=remote,branch=branch,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest=rows,execution=False,
        verification=dict(all_file_sha256_and_bytes_match=True,independent_bare_repository=str(bare)))
    p=OUT/'JOB3340-ACTUAL-ONCE-CONTROLS-PUBLICATION-v1.json'
    with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(dict(commit=commit,files=len(rows),sha256=sha(p.read_bytes()),receipt=str(p))))

if __name__=='__main__':main()
