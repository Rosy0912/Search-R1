"""Exact namespace-only pruning recovery. No changes to objective or state."""
import pathlib,sys,ast,hashlib,json
B=pathlib.Path('/data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010')
H=B/'source/opsd_teacher_offpolicy200_v1'
def apply():
 sys.path.insert(0,str(H))
 import checkpoint
 manifest=json.loads((H/'source-manifest.json').read_text());src=(H/'checkpoint.py').read_text();assert hashlib.sha256(src.encode()).hexdigest()==manifest['files']['checkpoint.py']['sha256']
 tree=ast.parse(src);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prune_checkpoints');text='\n'.join(src.splitlines()[fn.lineno-1:fn.end_lineno])
 old="root.parent.name.startswith('search-e1-opsd-j-minus-p-bf16-')";new="root.parent.name == 'search-e1-opsd-fixed-teacher-bf16-rwm200-20261010'";assert text.count(old)==1
 exec(compile(text.replace(old,new),str(pathlib.Path(__file__).resolve()),'exec'),checkpoint.__dict__)
 return checkpoint
