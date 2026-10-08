#!/usr/bin/env python3
"""Two original head fits from original vs staged CPU fixture dumps; not calibration."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,sys,time,types
EXPECTED_TRAIN='f12881dfbd2df07f51632949d06fa9dca5a41ad08ad9da1d7de5b9bd1e9117a0'
EXPECTED_PRODUCER='0b4e2575a774d22a93039630b2dc65f277d26e8cfe8c96bc595af299e1ea68c2'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--producer',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args()
    producer=pathlib.Path(a.producer).resolve();directory=pathlib.Path(a.directory).resolve();os.umask(0o077)
    manifest=producer/'files.json'
    if sha(manifest)!=EXPECTED_PRODUCER:raise ValueError('frozen producer manifest mismatch')
    for name,r in json.loads(manifest.read_text())['files'].items():
        p=producer/name
        if p.is_symlink() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:raise ValueError('producer source drift '+name)
    train=producer/'original/ts_probe_train.py'
    if sha(train)!=EXPECTED_TRAIN:raise ValueError('original train source drift')
    directory.mkdir(parents=True,exist_ok=False)
    receipt={'scope':'CPU_FIXTURE_ORIGINAL_HEAD_FIDELITY_NOT_REAL_CALIBRATION_NO_TRAINING','status':'STARTED','real_32b_loaded':False,'real_data_or_labels':False,'manifest_sha256':EXPECTED_PRODUCER,'train_source_sha256':EXPECTED_TRAIN,'elapsed_seconds':None}
    started=time.monotonic()
    try:
        if importlib.util.find_spec('sklearn') is None or importlib.util.find_spec('joblib') is None:
            raise ModuleNotFoundError('original headfit requires sklearn and joblib; no installation performed')
        import numpy as np,pandas as pd,torch,joblib
        if torch.cuda.is_initialized():raise RuntimeError('CPU fixture parent CUDA context')
        sys.path.insert(0,str(producer));import test_producer as fixture
        from original_blocks import execute,trace,restore
        def namespace():
            ns=fixture.original_namespace()
            rows=pd.DataFrame([{'prompt':[{'content':f'Question: q{i}'}],'reward_model':{'ground_truth':{'target':['gold']}}} for i in range(32)])
            ns['pd']=types.SimpleNamespace(read_parquet=lambda path:rows)
            return ns
        original=directory/'original-dump';staged=directory/'staged-dump';original.mkdir();staged.mkdir()
        sys.argv=['dump','--model','fixture','--data','fixture','--tag','nq','--cap','32','--G','5','--out',str(original)]
        namespace()['main']()
        args=types.SimpleNamespace(model='fixture',data='fixture',tag='nq',cap=32,offset=0,G=5,layers='21,24,27',max_turns=4,gpu_mem=.45,out=str(staged))
        A=namespace();A.update(args=args,layers=[21,24,27],llm=fixture.Engine(),t0=time.time())
        for block in ('prepare','generation'):execute(producer/'original/ts_probe_dump.py',block,A)
        persisted=json.dumps(trace(A),sort_keys=True);(directory/'generation-trace.json').write_text(persisted)
        B=namespace();restore(json.loads(persisted),args,B);B.update(tok=fixture.Tokenizer(),hf=fixture.Model().eval(),device=torch.device('cpu'),t0=time.time())
        for block in ('hidden','assembly'):execute(producer/'original/ts_probe_dump.py',block,B)
        with np.load(original/'nq_0.npz',allow_pickle=False) as one,np.load(staged/'nq_0.npz',allow_pickle=False) as two:
            assert set(one.files)==set(two.files)
            for key in one.files:np.testing.assert_array_equal(one[key],two[key]);assert one[key].dtype==two[key].dtype
        assert (original/'nq_0.jsonl').read_bytes()==(staged/'nq_0.jsonl').read_bytes()
        receipt.update(dump_fields=15,rows=160,questions=32,npz_original_sha256=sha(original/'nq_0.npz'),npz_staged_sha256=sha(staged/'nq_0.npz'),jsonl_original_sha256=sha(original/'nq_0.jsonl'),jsonl_staged_sha256=sha(staged/'nq_0.jsonl'))
        head_dirs=[]
        for label,dump in (('original',original),('staged',staged)):
            heads=directory/(label+'-heads');head_dirs.append(heads)
            command=[sys.executable,str(train),'--dumps',str(dump),'--layers','21,24,27','--commit-position','Xa','--commit-label','flip','--save',str(heads)]
            with open(directory/(label+'-fit.log'),'wb') as log:
                result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if result.returncode:raise RuntimeError('original headfit failed '+label)
        raw=[]
        with np.load(original/'nq_0.npz',allow_pickle=False) as z:
            for name in ('commit_probe.pkl','boundary_probe.pkl'):
                one,two=[joblib.load(p/name) for p in head_dirs]
                for field in ('position','layer','label','model','auc'):
                    if field in one or field in two:assert one.get(field)==two.get(field)
                key={'Xa':'Xa','Xc':'Xc','Xp':'Xp'}[one['position']]+'_'+str(one['layer']);inputs=z[key].astype(np.float32)
                p,q=one['clf'].predict_proba(inputs),two['clf'].predict_proba(inputs)
                assert np.isfinite(p).all() and np.isfinite(q).all()
                np.testing.assert_allclose(p,q,atol=1e-7,rtol=1e-7)
                raw.append({'name':name,'original_sha256':sha(head_dirs[0]/name),'staged_sha256':sha(head_dirs[1]/name),'max_absolute_delta':float(np.max(np.abs(p-q))),'metadata':{k:one[k] for k in ('position','layer','label','model','auc') if k in one},'prediction_rows':len(inputs)})
        receipt.update(status='PASS_CPU_FIXTURE_ONLY',head_comparisons=raw,parent_cuda_initialized=torch.cuda.is_initialized())
    except BaseException as exc:
        receipt.update(status='UNAVAILABLE_DEPENDENCIES' if isinstance(exc,ModuleNotFoundError) else 'FAILED',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started;(directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
