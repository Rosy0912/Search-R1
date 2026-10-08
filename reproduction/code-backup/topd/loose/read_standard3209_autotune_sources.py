import pathlib,shlex,subprocess,json
source="""import pathlib,json,hashlib
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/env/lib/python3.10/site-packages')
names=['torch/_inductor/config.py','torch/_inductor/runtime/triton_heuristics.py','torch/_inductor/runtime/runtime_utils.py','triton/testing.py']
rows={}
for name in names:
 p=base/name;raw=p.read_bytes();rows[name]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
print(json.dumps({'scope':'READONLY_INSTALLED_SOURCES_NO_IMPORT_NO_CUDA','files':rows}))
"""
command='python3 -c '+shlex.quote(source)
p=pathlib.Path('outputs/standard3209-autotune-installed-source-readonly.json')
with p.open('x') as out:
 r=subprocess.run(['python3','work/remote_topd.py'],input=command,text=True,stdout=out,stderr=subprocess.PIPE,timeout=90)
assert r.returncode==0,r.stderr
obj=json.loads(p.read_text());folder=pathlib.Path('outputs/remote-evidence/standard3209-autotune-source');folder.mkdir(exist_ok=True)
for name,row in obj['files'].items(): (folder/name.replace('/','__')).write_text(row['text'])
print(json.dumps({name:row['sha256'] for name,row in obj['files'].items()}))
