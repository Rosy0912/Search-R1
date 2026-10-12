"""Append a RUN-only closure to the original successful3356 binding template.

No driver probe, model loading, large native hashing, target authorization, or
changes to the original stage/template are performed.
"""
from pathlib import Path
import hashlib,json,os,re,sys
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009')
RUN=ROOT/'runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356'
SOURCE='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def binding(p):
    p=Path(p);require(p.is_file()and not p.is_symlink()and p.resolve()==p,'ordinary canonical metadata required')
    require(p.suffix!='.pt'and p.stat().st_size<=64*1024*1024,'small metadata only')
    raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def main():
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'optimized request forbidden')
    require(sys.platform=='linux'and os.uname().nodename=='h100-11','actual node11 required')
    original=ROOT/'evidence/partial-resume-bindings3356.json';out=ROOT/'evidence/partial-resume-bindings3356-v2.json'
    require(not out.exists()and not out.is_symlink(),'unknown or existing v2 outcome: no overwrite')
    before=binding(original);value=json.loads(original.read_bytes())
    require(value['old_run_root']==str(RUN)and value['old_source_commit']==SOURCE and not any(k.startswith('target_')for k in value),'original3356 unbound template required')
    require(value['driver_cpu_proof']['status']=='ACTUAL_SMALL_DRIVER_CPU_BINDING_VERIFIED_NOT_NATIVE_RELOAD','successful original driver probe required')
    stage=value['stage_metrics'];require(binding(stage['path'])==stage,'original stage changed')
    require(len(value['ranks'])==8 and {r['rank']for r in value['ranks']}==set(range(8)),'all original ranks required')
    closure=[]
    for b in value['original_closure']:
        if RUN in Path(b['path']).parents and Path(b['path']).suffix!='.pt':
            require(binding(b['path'])==b,'original small proof changed');closure.append(b)
    native=[r['state']for r in value['ranks']]+[value['driver']]
    for b in native:
        p=Path(b['path']);require(RUN in p.parents and p.is_file()and not p.is_symlink()and p.resolve()==p and p.stat().st_size==b['bytes']and re.fullmatch('[0-9a-f]{64}',b['sha256']),'original native producer binding/stat differs')
        side=next((r['report']for r in value['ranks']if r['state']==b),value['driver_report'])
        require(binding(side['path'])==side,'original producer sidecar changed')
        report=json.loads(Path(side['path']).read_bytes())
        require(report['path']==str(p)and report['sha256']==b['sha256']and report['status']=='PARTIAL_CURRENT_STATE_SAVED','producer native SHA differs')
        closure.append(b)
    require(len({b['path']for b in closure})==len(closure),'duplicate normalized closure')
    require(value['original_config']in closure and value['driver_report']in closure,'original small closure incomplete')
    value['original_closure']=closure
    value['normalization']=dict(schema='original3356-run-only-closure-v2',original_template=before,stage_unchanged=stage,no_driver_probe_repeated=True,no_large_native_read=True,no_target_identity=True)
    require(binding(original)==before and binding(stage['path'])==stage,'original inputs changed during normalization')
    with out.open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(json.dumps(dict(status='ORIGINAL3356_TEMPLATE_CLOSURE_NORMALIZED_NOT_RESTORED',original_template=before,output=binding(out),stage_unchanged=stage)))
if __name__=='__main__':main()
