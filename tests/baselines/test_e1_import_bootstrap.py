"""Fresh temporary CPU packages test executable import origins/code/pins; no GPU imports."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

PATH=Path(__file__).resolve().parents[2]/'reproduction/baselines/e1_import_bootstrap.py'
spec=importlib.util.spec_from_file_location('e1_bootstrap_test',PATH);bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)

@pytest.fixture
def packages(tmp_path,monkeypatch):
    # Use separate prefix only for this temporary fixture; production constants immutable.
    monkeypatch.setattr(bootstrap,'PREFIXES',('e1fixture',))
    monkeypatch.setattr(bootstrap,'LIBRARY_MODULES',())
    monkeypatch.setattr(bootstrap,'CHECKER_NAME',None)
    saved=list(sys.path)
    root=tmp_path/'new';(root/'search_r1/baselines').mkdir(parents=True);(root/'search_r1/baselines/main.py').write_text('# repo anchor')
    (root/'e1fixture').mkdir();(root/'e1fixture/__init__.py').write_text('')
    (root/'e1fixture/a.py').write_text('def f(x):\n    return x + 1\n')
    old=tmp_path/'old';(old/'e1fixture').mkdir(parents=True);(old/'e1fixture/__init__.py').write_text('');(old/'e1fixture/a.py').write_text('def f(x):\n    return x - 999\n')
    yield root,old
    sys.path[:]=saved
    for name in list(sys.modules):
        if name=='e1fixture' or name.startswith('e1fixture.'):sys.modules.pop(name,None)

def test_new_repo_precedes_old_editable_and_actual_code(packages):
    root,old=packages;sys.path.insert(0,str(old));bootstrap.repo_first(root)
    result=bootstrap.collect(root,('e1fixture.a',));assert sys.modules['e1fixture.a'].f(1)==2
    actual,count=bootstrap.verify_imports(root,result,('e1fixture.a',))
    assert count==1 and actual==result

def test_cached_old_module_refused(packages):
    root,old=packages;sys.path.insert(0,str(old));importlib.import_module('e1fixture.a');bootstrap.repo_first(root)
    with pytest.raises(ValueError,match='Actual imported module'):bootstrap.collect(root,('e1fixture.a',))

def test_required_and_actual_transitive_import_pins(packages):
    root,_=packages;bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',))
    wrong={'modules':{'e1fixture.a':result['modules']['e1fixture.a']}}
    with pytest.raises(ValueError,match='lacks exact pin'):bootstrap.verify_imports(root,wrong,('e1fixture.a',))
    with pytest.raises(ValueError,match='incomplete'):bootstrap.verify_imports(root,result,('e1fixture.missing',))

def test_file_and_code_mutation_refused(packages):
    root,_=packages;bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',))
    module=sys.modules['e1fixture.a'];original=module.f
    exec(compile('def f(x):\n    return x - 10\n',str(root/'e1fixture/a.py'),'exec'),module.__dict__)
    with pytest.raises(ValueError,match='Actual function code'):bootstrap.verify_imports(root,result,('e1fixture.a',))
    module.f=original;(root/'e1fixture/a.py').write_text('def f(x):\n    return x + 17\n')
    with pytest.raises(ValueError,match='path/SHA'):bootstrap.verify_imports(root,result,('e1fixture.a',))
    # Coherent file+pin change still cannot accept the old live function object.
    result['modules']['e1fixture.a']['sha256']=bootstrap.digest(root/'e1fixture/a.py')
    with pytest.raises(ValueError,match='Actual function code'):bootstrap.verify_imports(root,result,('e1fixture.a',))

def test_external_pin_duplicate_key_and_path_drift(packages,tmp_path):
    root,old=packages;bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',))
    result['modules']['e1fixture.a']['path']=str(old/'e1fixture/a.py')
    with pytest.raises(ValueError,match='path/SHA'):bootstrap.verify_imports(root,result,('e1fixture.a',))
    p=tmp_path/'closure.json';p.write_text('{"modules":{},"modules":{}}')
    with pytest.raises(ValueError,match='Duplicate'):bootstrap.pinned(p,bootstrap.digest(p))
    with pytest.raises(ValueError,match='pin differs'):bootstrap.pinned(p,'0'*64)

def test_malicious_project_wrapper_with_unchanged_wrapped_function_refused(packages):
    root,_=packages;bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',))
    module=sys.modules['e1fixture.a'];original=module.f
    exec(compile('def f(x):\n    return x + 999\n',str(root/'e1fixture/a.py'),'exec'),module.__dict__)
    module.f.__wrapped__=original
    with pytest.raises(ValueError,match='Actual function code'):bootstrap.verify_imports(root,result,('e1fixture.a',))


def test_light_preflight_executes_new_pinned_code_and_keeps_arguments(packages,tmp_path,monkeypatch,capsys):
    root,_=packages
    (root/'e1fixture/a.py').write_text('def main():\n    import sys\n    print("CPU_PREFLIGHT_ONLY", repr(sys.argv[1:]))\n')
    original=bootstrap.repo_first
    original(root);result=bootstrap.collect(root,('e1fixture.a',))
    closure=tmp_path/'closure.json';closure.write_text(json.dumps(result))
    binding=tmp_path/'binding.json';binding.write_text(json.dumps({'files':{'import_closure':{'path':str(closure),'sha256':bootstrap.digest(closure)}}}))
    monkeypatch.setattr(bootstrap,'repo_first',lambda ignored:original(root))
    monkeypatch.setattr(bootstrap,'REQUIRED',('e1fixture.a',));monkeypatch.setattr(bootstrap,'PREFLIGHT','e1fixture.a')
    monkeypatch.setenv('OPSD_E1_BINDING_MANIFEST',str(binding));monkeypatch.setenv('OPSD_E1_BINDING_MANIFEST_SHA256',bootstrap.digest(binding))
    monkeypatch.setattr(sys,'argv',['bootstrap','--mode','preflight','--','--reuse-inherited','--','python','child'])
    bootstrap.main();assert "['--reuse-inherited', '--', 'python', 'child']" in capsys.readouterr().out
    # Jointly changed pinned file, with stale live object, cannot reach the call.
    (root/'e1fixture/a.py').write_text('def main():\n    raise RuntimeError("WRONG_CODE_EXECUTED")\n')
    result['modules']['e1fixture.a']['sha256']=bootstrap.digest(root/'e1fixture/a.py');closure.write_text(json.dumps(result))
    binding.write_text(json.dumps({'files':{'import_closure':{'path':str(closure),'sha256':bootstrap.digest(closure)}}}))
    monkeypatch.setenv('OPSD_E1_BINDING_MANIFEST_SHA256',bootstrap.digest(binding))
    monkeypatch.setattr(sys,'argv',['bootstrap','--mode','preflight','--','--','python','child'])
    with pytest.raises(ValueError,match='Actual function code'):bootstrap.main()

@pytest.mark.parametrize('filename',['<synthetic>','/foreign/unpinned.py'])
def test_owned_function_unknown_filename_cannot_hide_behind_good(packages,filename):
    root,_=packages;(root/'e1fixture/a.py').write_text('def good():\n    return 1\ndef bad():\n    return 2\n')
    bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a']
    exec(compile('def bad():\n    return 999\n',filename,'exec'),m.__dict__)
    with pytest.raises(ValueError,match='outside pinned closure'):bootstrap.verify_imports(root,result,('e1fixture.a',))

@pytest.mark.parametrize('accessor',['fget','fset','fdel'])
def test_property_accessors_unknown_code_refused(packages,accessor):
    root,_=packages;(root/'e1fixture/a.py').write_text('def good():\n    return 1\nclass C:\n    @property\n    def x(self):\n        return 2\n    @x.setter\n    def x(self,v):\n        pass\n    @x.deleter\n    def x(self):\n        pass\n')
    bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a'];old=m.C.x
    values={k:getattr(old,k) for k in ('fget','fset','fdel')};namespace={'__name__':m.__name__}
    exec(compile('def evil(*args):\n    return 999\n','<synthetic>','exec'),namespace)
    values[accessor]=namespace['evil'];m.C.x=property(**values)
    with pytest.raises(ValueError,match='outside pinned closure'):bootstrap.verify_imports(root,result,('e1fixture.a',))

def test_known_library_wrapper_pinned_and_dataclass_explicit_exception(packages,monkeypatch):
    root,_=packages
    # Real stdlib dataclass-generated functions, whose constructor has no source file.
    (root/'e1fixture/a.py').write_text('from dataclasses import dataclass\n@dataclass\nclass C:\n    value: int\ndef good():\n    return 1\n')
    monkeypatch.setattr(bootstrap,'LIBRARY_MODULES',('dataclasses',))
    bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',))
    actual,count=bootstrap.verify_imports(root,result,('e1fixture.a',));assert count>=1
    assert 'dataclasses' in actual['modules']
    bad=json.loads(json.dumps(result));bad['modules']['dataclasses']['sha256']='0'*64
    with pytest.raises(ValueError,match='path/SHA'):bootstrap.verify_imports(root,bad,('e1fixture.a',))


def test_changing_module_metadata_cannot_hide_declared_source_slot(packages):
    root,_=packages;bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a']
    namespace={'__name__':'pretend_foreign_library'}
    exec(compile('def f(x):\n    return 999\n','<synthetic>','exec'),namespace);m.f=namespace['f']
    with pytest.raises(ValueError,match='outside pinned closure'):bootstrap.verify_imports(root,result,('e1fixture.a',))


def test_stale_project_alias_unregistered_module_cannot_escape(packages):
    root,_=packages;bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a']
    namespace={'__name__':'e1fixture.other_cached_source'}
    exec(compile('def alias(x):\n    return 999\n','/old_repo/alias.py','exec'),namespace);m.old_alias=namespace['alias']
    with pytest.raises(ValueError,match='outside pinned closure'):bootstrap.verify_imports(root,result,('e1fixture.a',))

def test_frozenset_literal_order_normalized_without_semantic_drift(packages):
    root,_=packages;(root/'e1fixture/a.py').write_text('def f(x):\n    return x in {"alpha", "beta", "gamma", "delta", "epsilon"}\n')
    bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a'];original=m.f.__code__
    m.f.__code__=original.replace(co_consts=tuple(frozenset(reversed(list(c))) if isinstance(c,frozenset) else c for c in original.co_consts))
    bootstrap.verify_imports(root,result,('e1fixture.a',))
    m.f.__code__=original.replace(co_consts=tuple(frozenset({'CHANGED'}) if isinstance(c,frozenset) else c for c in original.co_consts))
    with pytest.raises(ValueError,match='Actual function code'):bootstrap.verify_imports(root,result,('e1fixture.a',))


@pytest.mark.parametrize('explicit',[False,True])
def test_spoofed_dataclass_marker_cannot_bypass_source_owned_constructor(packages,explicit):
    root,_=packages
    text='def good():\n    return 1\nclass C:\n'+('    def __init__(self):\n        pass\n' if explicit else '    pass\n')
    (root/'e1fixture/a.py').write_text(text);bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a']
    m.C.__dataclass_fields__={};m.C.__dataclass_params__=object()
    namespace={'__name__':m.__name__};exec(compile('def __init__(self):\n    return None\n','<string>','exec'),namespace);m.C.__init__=namespace['__init__']
    with pytest.raises(ValueError,match='outside pinned closure'):bootstrap.verify_imports(root,result,('e1fixture.a',))

def test_actual_dataclass_explicit_source_constructor_not_generated_exception(packages,monkeypatch):
    root,_=packages;(root/'e1fixture/a.py').write_text('from dataclasses import dataclass\n@dataclass\nclass C:\n    value: int\n    def __init__(self,value):\n        self.value=value\ndef good():\n    return 1\n')
    monkeypatch.setattr(bootstrap,'LIBRARY_MODULES',('dataclasses',));bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.a',));m=sys.modules['e1fixture.a']
    namespace={'__name__':m.__name__};exec(compile('def __init__(self,value):\n    self.value=999\n','<string>','exec'),namespace);m.C.__init__=namespace['__init__']
    with pytest.raises(ValueError,match='outside pinned closure'):bootstrap.verify_imports(root,result,('e1fixture.a',))

def test_imported_dataclass_checked_in_defining_module(packages,monkeypatch):
    root,_=packages
    (root/'e1fixture/a.py').write_text('from dataclasses import dataclass\n@dataclass\nclass C:\n    value: int\n')
    (root/'e1fixture/b.py').write_text('from .a import C\ndef good():\n    return C(1).value\n')
    monkeypatch.setattr(bootstrap,'LIBRARY_MODULES',('dataclasses',))
    bootstrap.repo_first(root);result=bootstrap.collect(root,('e1fixture.b',))
    assert 'e1fixture.a' in result['modules']
    bootstrap.verify_imports(root,result,('e1fixture.b',))
    namespace={'__name__':'e1fixture.a'}
    exec(compile('def __init__(self,value):\n    self.value=999\n','foreign.py','exec'),namespace)
    sys.modules['e1fixture.a'].C.__init__=namespace['__init__']
    with pytest.raises(ValueError,match='outside pinned closure'):
        bootstrap.verify_imports(root,result,('e1fixture.b',))
