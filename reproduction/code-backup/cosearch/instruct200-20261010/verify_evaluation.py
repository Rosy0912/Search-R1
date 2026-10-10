"""Check complete native evaluation output counts and finite scores; keep full output remote."""
import hashlib,json,math,time
from pathlib import Path

def verify(path,rows,step):
    path=Path(path);hasher=hashlib.sha256();count=0
    with path.open('rb') as f:
        for line in f:
            hasher.update(line)
            if not line.strip():raise RuntimeError('Blank evaluation row')
            item=json.loads(line)
            if item.get('step')!=step:raise RuntimeError('Wrong evaluation step')
            if not all(k in item for k in ('input','output','gts','score')):raise RuntimeError('Missing native evaluation fields')
            if not isinstance(item['score'],(float,int)) or not math.isfinite(item['score']):raise RuntimeError('Nonfinite evaluation score')
            count+=1
    if count!=rows:raise RuntimeError('Incomplete evaluation: '+str(count)+' vs '+str(rows))
    return {'status':'NATIVE_EVALUATION_ROW_COUNT_AND_SHA_PASS','epoch':time.time(),'path':str(path),'rows':count,'step':step,'sha256':hasher.hexdigest()}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('rows',type=int);p.add_argument('step',type=int);a=p.parse_args()
    print(json.dumps(verify(a.path,a.rows,a.step)))
