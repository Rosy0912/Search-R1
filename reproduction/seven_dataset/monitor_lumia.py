"""Bounded evaluation monitor honoring audited in-place scheduler renames."""
import re

def validated_rename_names(receipt):
    result = {}
    for row in receipt['rows']:
        job = str(row['job'])
        old = re.search(r'(?:^| )JobName=(\S+)', row['before']).group(1)
        new = re.search(r'(?:^| )JobName=(\S+)', row['after']).group(1)
        for evidence in (row['before'], row['after']):
            assert re.search(r'(?:^| )JobId=(\S+)', evidence).group(1) == job
            assert 'UserId=jmzhang(' in evidence
        assert new == row['name'] and new.startswith('zhangjunming-ttt-')
        assert job not in result
        result[job] = dict(old=old, new=new)
    return result

def main():
    """Bounded live Slurm evaluation monitoring and incremental token/GPU accounting."""
    from pathlib import Path
    import subprocess,json,textwrap
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True)
    P=Path(parser.parse_args().study_dir).resolve()
    renames=validated_rename_names(json.loads((P/'rename-lumia-zhangjunming-ttt-20261017.json').read_text()))
    submissions=[json.loads(f.read_text()) for f in sorted(P.glob('seven-*-lumia-submission.json'))]
    assert len({(s['checkpoint_identity'],s['job_id']) for s in submissions})==len(submissions)
    cp=P/'seven-evaluation-token-cursors.json';cache=json.loads(cp.read_text()) if cp.exists() else {}
    code='RENAMES='+repr(renames)+'\nSUBMISSIONS='+repr(submissions)+'\nCACHE='+repr(cache)+'\n'+textwrap.dedent(r'''
    import pathlib,json,subprocess,time,os,re
    b=pathlib.Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1');results=[]
    for sub in SUBMISSIONS:
     root=pathlib.Path(sub['output']).parent;receipt=json.loads((root/'submission.json').read_text())
     assert all(receipt[k]==sub[k] for k in ['job','job_id','checkpoint_identity','plan_sha256','source'])
     text=subprocess.check_output(['sacct','-X','-n','-P','-j',sub['job_id'],'--format=JobIDRaw,User,JobName%80,State,ElapsedRaw,AllocTRES%200,NodeList,Start,End'],text=True)
     rows=[r.split('|') for r in text.splitlines() if r.startswith(sub['job_id']+'|')];assert len(rows)==1
     r=rows[0];expected=sub['job']
     if sub['job_id'] in RENAMES:
      rename=RENAMES[sub['job_id']];assert rename['old']==sub['job'];expected=rename['new']
     assert r[1]=='jmzhang' and r[2]==expected, (sub['job_id'],r[1],r[2],expected)
     seconds=int(r[4]);match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',r[5]);gpus=int(match.group(1)) if match else 0
     assert gpus in (0,8);assert seconds==0 or gpus==8
     log=b/'logs'/('seven-'+sub['checkpoint_identity'][:12]+'-'+sub['job_id']+'.log');out=pathlib.Path(sub['output'])
     item=dict(job_id=sub['job_id'],job=sub['job'],scheduler_job=r[2],group=sub['group'],outer=sub['outer'],state=r[3],seconds=seconds,gpus=gpus,gpu_hours=gpus*seconds/3600,node=r[6],start=r[7],end=r[8],output=str(out))
     for name in ['progress.json','summary.json','entry-exit.json']:
      p=out/name;item[name]=json.loads(p.read_text()) if p.exists() else None
     item['exit_code']=(root/'exit.code').read_text().strip() if (root/'exit.code').exists() else None
     c=CACHE.get(sub['job_id'],dict(offset=0,raw_emitted_tokens=0,events=0))
     if log.exists():
      stat=log.stat();assert stat.st_size>=c['offset']
      if 'inode' in c:assert stat.st_ino==c['inode']
      with log.open('rb') as f:
       f.seek(c['offset']);chunk=f.read(4*1024**2);end=chunk.rfind(b'\n')+1
       for line in chunk[:end].decode(errors='replace').splitlines():
        if 'SEVEN_EVAL_GPU=' not in line:continue
        event=json.JSONDecoder().raw_decode(line.split('SEVEN_EVAL_GPU=',1)[1])[0]
        assert event['checkpoint_identity']==sub['checkpoint_identity'];count=event['raw_emitted_tokens'];assert type(count)is int and count>=0
        c['raw_emitted_tokens']+=count;c['events']+=1
       c.update(offset=c['offset']+end,inode=stat.st_ino,observed_size=stat.st_size)
       f.seek(max(0,stat.st_size-32768));item['log_tail']=f.read().decode(errors='replace')[-12000:]
      item.update(log_bytes=stat.st_size,log_mtime=stat.st_mtime)
     pred=out/'predictions.jsonl';item['prediction_bytes']=pred.stat().st_size if pred.exists() else 0
     item['recorded_raw_emitted_tokens_lower_bound']=c['raw_emitted_tokens'];item['token_events']=c['events'];CACHE[sub['job_id']]=c;results.append(item)
    s=os.statvfs(b);print(json.dumps(dict(at=time.time(),entries=results,cache=CACHE,free_bytes=s.f_bavail*s.f_frsize,gpu_hours=sum(x['gpu_hours'] for x in results),raw_emitted_tokens_lower_bound=sum(x['recorded_raw_emitted_tokens_lower_bound'] for x in results))))
    ''')
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lumia','python3 -'],input=code,text=True,capture_output=True,timeout=45);assert r.returncode==0, r.stderr;v=json.loads(r.stdout);cp.write_text(json.dumps(v.pop('cache'),indent=2));(P/'seven-evaluation-status.latest.json').write_text(json.dumps(v,indent=2));print(json.dumps(dict(at=v['at'],free_bytes=v['free_bytes'],gpu_hours=v['gpu_hours'],tokens=v['raw_emitted_tokens_lower_bound'],jobs=[{k:e.get(k) for k in ['job_id','state','gpus','prediction_bytes','progress.json','exit_code']} for e in v['entries']])))

if __name__ == '__main__':
    main()
