"""Record deployed source hashes after independently verifying its publication commit."""
import hashlib,json,os,time
from pathlib import Path

def write(root,publication_commit):
    root=Path(root)
    if len(publication_commit)!=40 or any(c not in '0123456789abcdef' for c in publication_commit):raise ValueError('Exact verified publication SHA required')
    files=[root/'author-cli.json']
    for folder in ('tools','repo'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json','.sh','.sbatch') and '__pycache__' not in p.parts)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    record={'epoch':time.time(),'publication_commit':publication_commit,'published_remote_sha_verified':True,'sha256':hashes,'training_started':False}
    destination=root/'deployment-manifest.json';temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,destination)
    return record
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('publication_commit');a=p.parse_args()
    result=write(a.root,a.publication_commit);print(json.dumps({'files':len(result['sha256']),'publication_commit':a.publication_commit}))
