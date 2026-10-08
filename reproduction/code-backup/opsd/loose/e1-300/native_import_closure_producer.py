#!/usr/bin/env python3
"""Prepare externally reviewable native source pins, without importing GPU modules."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

CHECKER='reproduction.baselines.e1_import_bootstrap'
LIBRARIES=('hydra.main','torch.utils._contextlib','contextlib','dataclasses')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def physical(path):
    p=Path(path)
    if not p.is_absolute() or p.resolve(strict=True)!=p or not p.is_file():
        raise ValueError('Must be an existing physical absolute file')
    return p

def produce(repo,checker_sha):
    repo=Path(repo)
    if not repo.is_absolute() or repo.resolve(strict=True)!=repo:raise ValueError('Repository must be physical absolute path')
    if not re.fullmatch('[0-9a-f]{64}',checker_sha):raise ValueError('Externally reviewed checker SHA required')
    checker=physical(repo/'reproduction/baselines/e1_import_bootstrap.py');raw=checker.read_bytes()
    if sha(raw)!=checker_sha:raise ValueError('Checker source differs from reviewed pin')
    # Exact schema consumed by frozen runtime verifier. No acceptance flags.
    modules={}
    for namespace in ('search_r1','verl'):
        root=repo/namespace
        if not root.is_dir() or root.is_symlink():raise ValueError('Missing physical namespace')
        for p in sorted(root.rglob('*.py')):
            physical(p)
            rel=p.relative_to(repo)
            parts=list(rel.with_suffix('').parts)
            if parts[-1]=='__init__':parts.pop()
            if not parts or not all(x.isidentifier() for x in parts):raise ValueError('Unsupported importable source name: '+str(rel))
            name='.'.join(parts)
            if name in modules:raise ValueError('Ambiguous package/module source: '+name)
            source=p.read_bytes();compile(source,str(p),'exec',dont_inherit=True,optimize=sys.flags.optimize)
            modules[name]={'path':str(p),'sha256':sha(source)}
    modules[CHECKER]={'path':str(checker),'sha256':checker_sha}
    # Native environment library paths are required. find_spec may import the
    # library's parent packages, but never imports a project GPU module.
    for name in LIBRARIES:
        spec=importlib.util.find_spec(name)
        if spec is None or not spec.origin:raise ValueError('Required native library source missing: '+name)
        p=physical(Path(spec.origin).resolve());raw=p.read_bytes()
        compile(raw,str(p),'exec',dont_inherit=True,optimize=sys.flags.optimize)
        modules[name]={'path':str(p),'sha256':sha(raw)}
    if sha(checker.read_bytes())!=checker_sha:raise ValueError('Checker changed during inventory')
    for row in modules.values():
        if sha(Path(row['path']).read_bytes())!=row['sha256']:raise ValueError('Source changed during inventory')
    return {'modules':dict(sorted(modules.items()))}

def publish(value,output):
    p=Path(output)
    if not p.is_absolute() or p.parent.resolve(strict=True)!=p.parent or p.is_symlink():raise ValueError('Output must have physical absolute parent')
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    # Never overwrite a previously reviewed inventory.
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return sha(raw)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--checker-sha256',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();result=produce(args.repo,args.checker_sha256);digest=publish(result,args.output)
    print(json.dumps({'scope':'source inventory only; not GPU/import/acceptance proof','output':args.output,'sha256':digest,'module_count':len(result['modules'])},sort_keys=True))
if __name__=='__main__':main()
