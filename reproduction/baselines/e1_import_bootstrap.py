#!/usr/bin/env python3
"""Actual import/source/code checks before the E1 launch; inventory is not acceptance."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import struct
import types

REQUIRED = tuple('search_r1.baselines.'+n for n in
    ('main','holdout','sharedholdout','e1_profile','preflight','controller','actor','distillation','full_state','checkpoint_gate')) + (
    'verl.workers.fsdp_workers','verl.trainer.ppo.ray_trainer')
PREFIXES=('search_r1','verl','reproduction.baselines')
LIBRARY_MODULES=('hydra.main','torch.utils._contextlib','contextlib','dataclasses')
CHECKER_NAME='reproduction.baselines.e1_import_bootstrap'
PREFLIGHT='search_r1.baselines.preflight'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def unique_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate JSON key: '+k)
            result[k]=v
        return result
    return json.loads(text,object_pairs_hook=pairs)
def pinned(path,sha):
    import re
    if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha) or digest(path)!=sha:
        raise ValueError('External import/binding pin differs')
    return unique_json(Path(path).read_bytes())

def repo_first(repo):
    repo=Path(repo).resolve(strict=True)
    if not (repo/'search_r1/baselines/main.py').is_file():raise ValueError('Actual E1 repository missing')
    sys.path[:]=[str(repo)]+[p for p in sys.path if p and Path(p).resolve()!=repo]
    return repo

def source_path(repo,name):
    if name in LIBRARY_MODULES:
        spec=importlib.util.find_spec(name)
        if spec is None or not spec.origin:raise ValueError('Known decorator library origin missing')
        return Path(spec.origin).resolve(strict=True)
    stem=repo/Path(*name.split('.'))
    file=stem.with_suffix('.py')
    if not file.is_file():file=stem/'__init__.py'
    path=file.resolve(strict=True)
    if repo not in path.parents:raise ValueError('Module source escapes E1 repository')
    return path

def actual_path(repo,name,module):
    path=source_path(repo,name)
    if not getattr(module,'__file__',None) or Path(module.__file__).resolve()!=path:
        raise ValueError('Actual imported module differs: '+name)
    return path

def code_fingerprint(code):
    # Structural fields avoid marshal string-intern/reference encoding artifacts.
    def constant(v):
        if isinstance(v,types.CodeType):return ('code',code_fingerprint(v))
        if isinstance(v,frozenset):return ('frozenset',tuple(sorted((constant(x) for x in v),key=repr)))
        if isinstance(v,tuple):return ('tuple',tuple(constant(x) for x in v))
        if v is None:return ('NoneType',None)
        if v is Ellipsis:return ('Ellipsis',)
        if type(v) in (str,bytes,int,bool):return (type(v).__name__,v)
        if type(v) is float:return ('float64',struct.pack('!d',v))
        if type(v) is complex:return ('complex128',struct.pack('!dd',v.real,v.imag))
        raise ValueError('Unsupported executable code constant: '+type(v).__name__)
    consts=tuple(constant(v) for v in code.co_consts)
    names=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags',
           'co_code','co_names','co_varnames','co_filename','co_name','co_firstlineno','co_freevars','co_cellvars',
           'co_lnotab','co_linetable','co_exceptiontable')
    return tuple((name,getattr(code,name,None)) for name in names)+(('consts',consts),)

def compiled_codes(path):
    root=compile(Path(path).read_bytes(),str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
    result=set()
    def visit(code):
        result.add(code_fingerprint(code))
        for v in code.co_consts:
            if isinstance(v,types.CodeType):visit(v)
    visit(root);return result

def check_live_code(module,allowed):
    seen=set();count=0
    declarations=set();dataclass_classes=set();dataclass_names={'dataclass'}
    source_ast=ast.parse(Path(module.__file__).read_bytes())
    for node in source_ast.body:
        if isinstance(node,ast.ImportFrom) and node.module=='dataclasses':
            dataclass_names.update(alias.asname or alias.name for alias in node.names if alias.name=='dataclass')
    def declared(body,owner=''):
        for node in body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                declarations.add(owner+node.name)
            elif isinstance(node,ast.ClassDef):
                declarations.add(owner+node.name)
                for deco in node.decorator_list:
                    deco=deco.func if isinstance(deco,ast.Call) else deco
                    if (isinstance(deco,ast.Name) and deco.id in dataclass_names) or (isinstance(deco,ast.Attribute) and isinstance(deco.value,ast.Name) and deco.value.id=='dataclasses' and deco.attr=='dataclass'):
                        dataclass_classes.add(owner+node.name)
                declared(node.body,owner+node.name+'.')
    declared(source_ast.body)
    def visit(obj,generated=False,slot=''):
        nonlocal count
        if id(obj) in seen:return
        seen.add(id(obj))
        if isinstance(obj,(staticmethod,classmethod)):obj=obj.__func__
        if isinstance(obj,property):
            for accessor in (obj.fget,obj.fset,obj.fdel):
                if accessor is not None:visit(accessor,slot=slot)
            return
        if inspect.isfunction(obj):
            # Check every project wrapper layer as well as its wrapped function.
            chain=set()
            while inspect.isfunction(obj):
                if id(obj) in chain:raise ValueError('Cyclic function wrapping')
                chain.add(id(obj));path=Path(obj.__code__.co_filename)
                if path.is_absolute() and str(path.resolve()) in allowed:
                    if code_fingerprint(obj.__code__) not in allowed[str(path.resolve())]:
                        raise ValueError('Actual function code differs: '+module.__name__+'.'+obj.__qualname__)
                    count+=1
                elif any(str(getattr(obj,'__module__',''))==p or str(getattr(obj,'__module__','')).startswith(p+'.') for p in PREFIXES) or slot in declarations:
                    if not (generated and slot not in declarations and obj.__code__.co_filename=='<string>' and obj.__code__.co_name in ('__init__','__repr__','__eq__','__lt__','__le__','__gt__','__ge__','__hash__','__setattr__','__delattr__')):
                        raise ValueError('Project function source is outside pinned closure: '+module.__name__+'.'+obj.__qualname__)
                obj=getattr(obj,'__wrapped__',None)
        elif inspect.isclass(obj) and (obj.__module__==module.__name__ or slot in declarations):
            generated_owner=slot in dataclass_classes and hasattr(obj,'__dataclass_fields__') and hasattr(obj,'__dataclass_params__')
            for name,member in vars(obj).items():
                visit(member,generated_owner and name in ('__init__','__repr__','__eq__','__lt__','__le__','__gt__','__ge__','__hash__','__setattr__','__delattr__'),slot=slot+'.'+name)
    for name,obj in vars(module).items():visit(obj,slot=name)
    return count

def loaded_inventory(repo):
    records={}
    for name,module in list(sys.modules.items()):
        if any(name==p or name.startswith(p+'.') for p in PREFIXES):
            if module is None:raise ValueError('Partial project import: '+name)
            if getattr(module,'__file__',None):
                path=actual_path(repo,name,module)
                records[name]={'path':str(path),'sha256':digest(path)}
    for name in LIBRARY_MODULES:
        if name in sys.modules:
            path=actual_path(repo,name,sys.modules[name]);records[name]={'path':str(path),'sha256':digest(path)}
    return records

def verify_imports(repo,closure,required=REQUIRED):
    if set(closure)!={'modules'} or not isinstance(closure['modules'],dict):raise ValueError('Invalid import closure schema')
    records=closure['modules']
    if not set(required).union(LIBRARY_MODULES)<=set(records):raise ValueError('Required actual import closure incomplete')
    allowed={}
    for name,record in records.items():
        if not (name in LIBRARY_MODULES or any(name==p or name.startswith(p+'.') for p in PREFIXES)) or set(record)!={'path','sha256'}:
            raise ValueError('Invalid project closure record')
        path=source_path(repo,name)
        if str(path)!=record['path'] or digest(path)!=record['sha256']:
            raise ValueError('Pinned module path/SHA differs: '+name)
        allowed[str(path)]=compiled_codes(path)
    for name in tuple(required)+tuple(LIBRARY_MODULES):importlib.import_module(name)
    actual=loaded_inventory(repo)
    for name,record in actual.items():
        if records.get(name)!=record:raise ValueError('Actual imported project module lacks exact pin: '+name)
    count=sum(check_live_code(sys.modules[name],allowed) for name in actual if name not in LIBRARY_MODULES)
    if not count:raise ValueError('No actual function code verified')
    return {'modules':actual},count

def collect(repo,required=REQUIRED):
    if CHECKER_NAME is not None:importlib.import_module(CHECKER_NAME)
    for name in tuple(required)+tuple(LIBRARY_MODULES):importlib.import_module(name)
    result={'modules':loaded_inventory(repo)}
    verify_imports(repo,result,required)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['cfg','validate','run','collect','preflight'],required=True)
    parser.add_argument('--profile',choices=['e1_300','author200'],default='e1_300')
    parser.add_argument('preflight_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();repo=repo_first(Path(__file__).resolve().parents[2])
    if args.mode!='preflight' and args.preflight_args:raise ValueError('Unexpected bootstrap arguments')
    if args.mode in ('cfg','validate'):
        # Lightweight CPU profile imports also cannot come from old editable source.
        profile=importlib.import_module('search_r1.baselines.e1_profile')
        for name in ('search_r1.baselines.e1_profile','search_r1.baselines.sharedholdout','search_r1.baselines.holdout'):
            actual_path(repo,name,sys.modules[name])
        sys.argv=[sys.argv[0],'--profile',args.profile]+(['--validate-assets'] if args.mode=='validate' else [])
        profile.main();return
    if args.mode=='collect':
        print(json.dumps(collect(repo),indent=2));return  # CPU source inventory, not PASS/receipt.
    binding=pinned(os.environ['OPSD_E1_BINDING_MANIFEST'],os.environ['OPSD_E1_BINDING_MANIFEST_SHA256'])
    record=binding['files']['import_closure'];closure=pinned(record['path'],record['sha256'])
    if args.mode=='preflight':
        if not set(REQUIRED)<=set(closure['modules']):raise ValueError('Required preflight import pins incomplete')
        name=PREFLIGHT
        verify_imports(repo,closure,required=(name,))  # Includes already imported project dependencies.
        module=sys.modules[name]
        tail=args.preflight_args[1:] if args.preflight_args[:1]==['--'] else args.preflight_args
        sys.argv=[name]+tail;module.main();return
    actual,count=verify_imports(repo,closure)
    print('E1_ACTUAL_IMPORT_CLOSURE_JSON='+json.dumps({'repo':str(repo),'module_count':len(actual['modules']),
          'function_code_count':count,'external_closure_sha256':record['sha256']},sort_keys=True),flush=True)
    # Compose/asset validation and the actual imported Hydra main use one selector.
    from search_r1.baselines.e1_profile import PROFILE_CONFIGS
    sys.argv=['search_r1.baselines.main','--config-name',PROFILE_CONFIGS[args.profile]]
    # The verified module object is invoked; runpy would create a second unchecked object.
    sys.modules['search_r1.baselines.main'].main()
if __name__=='__main__':main()
