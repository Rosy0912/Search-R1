"""Bind control scripts locally to real immutable manifests; never deploy or sign."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parent/'opsd-unclipped-multitraj200-fullteacher'
NAMES=('deploy_tools.py','deploy_native_cpu.py','submit_reviewed.py',
       'publish_gpu_init.py','publish_first5_release.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--revision',default='')
    a=p.parse_args();binding=a.binding.resolve(strict=True)
    if a.revision and re.fullmatch('[a-z0-9-]+',a.revision) is None:raise ValueError('Simple control revision required')
    if binding.parent!=HERE:raise ValueError('Local actual runtime binding required')
    spec=json.loads(binding.read_bytes());source=spec['source_commit']
    if re.fullmatch('[0-9a-f]{40}',source) is None:raise ValueError('Actual source required')
    if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=source or subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Actual committed source changed')
    for role in ('formal','native','first5'):
        row=spec[role];d=Path(row['local_directory'])
        if d.parent!=HERE:raise ValueError('Unexpected local tool directory')
        pins=dict(row['files']);pins[row['manifest_name']]=row['manifest_sha256']
        if any(hashlib.sha256((d/name).read_bytes()).hexdigest()!=pin for name,pin in pins.items()):
            raise ValueError('Frozen tool bytes changed')
    first5=Path(spec['first5']['local_directory']).name+'/'+spec['first5']['manifest_name']
    native=Path(spec['native']['local_directory']).name+'/'+spec['native']['manifest_name']
    formal='tools/'+Path(spec['formal']['local_directory']).name+'/'+spec['formal']['manifest_name']
    replacements={
        'BLOCKED_UNBOUND_SOURCE_COMMIT':source,
        'BLOCKED_BOUND_SPEC_FILENAME':binding.name,
        'BLOCKED_BOUND_REVIEW_JSON':'INDEPENDENT_BOUND_TOOLS_REVIEW'+('_'+a.revision.upper() if a.revision else '')+'.json',
        'BLOCKED_BOUND_REVIEW_MD':'INDEPENDENT_BOUND_TOOLS_REVIEW'+('_'+a.revision.upper() if a.revision else '')+'.md',
        'BLOCKED_NATIVE_INVENTORY_FILENAME':'native-source-inventory-'+source[:12]+'-ACTUAL.json',
        'BLOCKED_FORMAL_MANIFEST_SHA':spec['formal']['manifest_sha256'],
        'BLOCKED_RECEIPT_MANIFEST_SHA':spec['first5']['manifest_sha256'],
        'BLOCKED_GATE_SOURCE_SHA':spec['source_files']['search_r1/baselines/checkpoint_gate.py'],
        'BLOCKED_LOCAL_RECEIPT_MANIFEST':first5,
        'BLOCKED_NATIVE_MANIFEST_RELATIVE':native,
        'BLOCKED_FORMAL_MANIFEST_RELATIVE':formal,
    }
    records={}
    for name in NAMES:
        raw=(HERE/name).read_bytes()
        for before,after in replacements.items():raw=raw.replace(before.encode(),after.encode())
        if b'BLOCKED_' in raw:raise ValueError('Unbound control script '+name)
        compile(raw,name,'exec',dont_inherit=True)
        target=HERE/(Path(name).stem+'-bound-'+source[:12]+('-'+a.revision if a.revision else '')+'.py')
        if target.exists():
            if target.read_bytes()!=raw:raise ValueError('Immutable control script exists with other bytes')
        else:
            with target.open('xb') as f:f.write(raw)
        records[name]={'path':str(target),'sha256':hashlib.sha256(raw).hexdigest()}
    record={'status':'LOCAL_SOURCE_BOUND_CONTROL_SOFTWARE_ONLY_NOT_DEPLOYED_OR_SIGNED',
            'source_commit':source,'binding_sha256':hashlib.sha256(binding.read_bytes()).hexdigest(),
            'files':records}
    raw=(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()
    target=HERE/('control-tools-bound-'+source[:12]+('-'+a.revision if a.revision else '')+'.json')
    if target.exists():
        if target.read_bytes()!=raw:raise ValueError('Control manifest immutable collision')
    else:
        with target.open('xb') as f:f.write(raw)
    print(raw.decode(),end='')

if __name__=='__main__':main()
