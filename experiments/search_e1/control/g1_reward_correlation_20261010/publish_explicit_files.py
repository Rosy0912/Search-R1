"""Publish explicit own files without changing the shared checkout or real index."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, tempfile

def run(args, **kw):
    r=subprocess.run(args, capture_output=True, **kw)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors="replace")[-3000:])
    return r.stdout

def publish(repo, mapping, message, proof):
    repo=pathlib.Path(repo)
    items=json.loads(pathlib.Path(mapping).read_text())
    records=[]
    for item in items:
        rel=item['target']; assert not rel.startswith('/') and '..' not in pathlib.PurePosixPath(rel).parts
        raw=pathlib.Path(item['source']).read_bytes()
        records.append((rel,raw,hashlib.sha256(raw).hexdigest()))
    base=['git','-C',str(repo)]
    for attempt in range(6):
        run(base+['fetch','origin','search-e1'])
        parent=run(base+['rev-parse','FETCH_HEAD']).decode().strip()
        with tempfile.TemporaryDirectory(prefix='g1-own-index-') as d:
            env=os.environ.copy(); env['GIT_INDEX_FILE']=str(pathlib.Path(d)/'index')
            run(base+['read-tree',parent],env=env)
            for rel,raw,digest in records:
                blob=run(base+['hash-object','-w','--stdin'],input=raw).decode().strip()
                run(base+['update-index','--add','--cacheinfo','100644',blob,rel],env=env)
            tree=run(base+['write-tree'],env=env).decode().strip()
            commit=run(base+['commit-tree',tree,'-p',parent],input=(message+'\n').encode()).decode().strip()
        push=subprocess.run(base+['push','origin',commit+':refs/heads/search-e1'],capture_output=True)
        if push.returncode:
            if b'non-fast-forward' in push.stderr or b'fetch first' in push.stderr or b'failed to update ref' in push.stderr: continue
            raise RuntimeError(push.stderr.decode(errors='replace')[-3000:])
        run(base+['fetch','origin','search-e1'])
        remote=run(base+['rev-parse','FETCH_HEAD']).decode().strip()
        for rel,raw,digest in records:
            assert run(base+['show',remote+':'+rel])==raw, 'remote file differs: '+rel
        run(base+['update-ref','refs/heads/codex/g1-reward-correlation-20261010',commit])
        result=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=commit,
            remote_readback=remote,files=[dict(path=rel,sha256=digest,bytes=len(raw)) for rel,raw,digest in records],
            checkout_preserved=True,no_force_push=True)
        pathlib.Path(proof).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result)); return
    raise RuntimeError('publication retries exhausted')

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--repo',required=True);a.add_argument('--mapping',required=True);a.add_argument('--message',required=True);a.add_argument('--proof',required=True);x=a.parse_args()
    publish(x.repo,x.mapping,x.message,x.proof)
