"""Source-only loading of the exact frozen authored-final scorer and executor."""
import hashlib
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
try:
    from .checkpoint import checked_json, checked_file, require, stat_identity
except ImportError:
    from checkpoint import checked_json, checked_file, require, stat_identity

QA_RELATIVE='upstream/Search-R1/search_r1/qa_protocol.py'
INIT_RELATIVE='upstream/Search-R1/search_r1/__init__.py'
QA_SHA256='9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
EMPTY_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
CORE_SHA256='0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc'
MARKER='_opsd_ablation_source_sha256'


class SourceOnlyScience(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self,root,files):self.root=Path(root).resolve();self.files=files
    def resolve(self,name):
        package=name.replace('.','/')
        prefix='upstream/Search-R1/' if name=='search_r1' or name.startswith('search_r1.') else ''
        for relative,is_package in ((prefix+package+'/__init__.py',True),(prefix+package+'.py',False)):
            if relative in self.files:
                entry=self.files[relative];digest=entry['sha256'] if isinstance(entry,dict) else entry
                path=self.root/relative
                require(path.resolve().is_relative_to(self.root),'Scientific source escaped bound runtime')
                return path,digest,is_package
        raise ValueError('Scientific module not in exact305 manifest: '+name)
    def find_spec(self,fullname,path=None,target=None):
        if not (fullname=='search_e1' or fullname.startswith('search_e1.') or fullname=='search_r1' or fullname.startswith('search_r1.')):return None
        source,_,package=self.resolve(fullname)
        return importlib.util.spec_from_loader(fullname,self,origin=str(source),is_package=package)
    def create_module(self,spec):return None
    def exec_module(self,module):
        path,digest,package=self.resolve(module.__name__)
        proof=checked_file(dict(path=str(path),sha256=digest));raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==digest,'Actual compiled scientific bytes differ')
        module.__file__=str(path);module.__package__=module.__name__ if package else module.__name__.rpartition('.')[0]
        if package:module.__path__=[str(path.parent)]
        exec(compile(raw,str(path),'exec'),module.__dict__);setattr(module,MARKER,digest)
        require(stat_identity(path)=={k:proof[k] for k in stat_identity(path)},'Source changed during compile')


def verify_scoring(config):
    manifest,proof=checked_json(config['runtime_manifest']);files=manifest.get('files',manifest)
    require(len(files)==305,'Expected exact305 scientific source tree')
    root=Path(config['paths']['runtime']).resolve();finder=SourceOnlyScience(root,files)
    actual=[]
    for relative,expected in ((QA_RELATIVE,QA_SHA256),(INIT_RELATIVE,EMPTY_SHA256),('search_e1/core.py',CORE_SHA256)):
        entry=files[relative];digest=entry['sha256'] if isinstance(entry,dict) else entry
        require(digest==expected,'Frozen scoring/core identity differs')
        actual.append(checked_file(dict(path=str(root/relative),sha256=digest)))
    for name,module in list(sys.modules.items()):
        if name=='search_e1' or name.startswith('search_e1.') or name=='search_r1' or name.startswith('search_r1.'):
            path,digest,_=finder.resolve(name)
            require(Path(getattr(module,'__file__','')).resolve()==path and getattr(module,MARKER,None)==digest,
                    'Foreign/cached scientific module lacks source-only binding')
    sys.path[:0]=[str(root/'upstream/Search-R1'),str(root)];sys.dont_write_bytecode=True
    # Do not accumulate one finder for every completed outer.
    sys.meta_path[:]=[x for x in sys.meta_path if not isinstance(x,SourceOnlyScience)]
    sys.meta_path.insert(0,finder)
    qa=importlib.import_module('search_r1.qa_protocol');core=importlib.import_module('search_e1.core')
    require(callable(qa.extract_final_answer) and callable(qa.parse_model_action) and callable(core.reward),'Frozen scorer API missing')
    return dict(accepted=True,runtime_manifest=proof,actual_imported_scoring_files=actual,
        source_only_compile=True,scope='Unchanged authored-final strict scorer; original executor/source compiled from SHA-checked bytes, never pyc')
