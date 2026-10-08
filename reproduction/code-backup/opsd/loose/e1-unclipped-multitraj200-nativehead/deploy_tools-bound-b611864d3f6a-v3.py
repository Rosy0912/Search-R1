"""Publish reviewed immutable tool files, each manifest last; no submission."""
import pathlib,json,hashlib,shlex
from transport import remote,publish
H=pathlib.Path(__file__).resolve().parent
spec=json.loads((H/'runtime-tools-bound-b611864d3f6a-v3.json').read_text())
report=H/'INDEPENDENT_BOUND_TOOLS_REVIEW_V3.md';assert report.is_file() and 'PASS' in report.read_text()
review=json.loads((H/'INDEPENDENT_BOUND_TOOLS_REVIEW_V3.json').read_text())
assert review['status']=='PASS_INDEPENDENT_SOURCE_BOUND_SOFTWARE_ONLY'
assert review['source_commit']==spec['source_commit']
assert review['runtime_binding_sha256']==hashlib.sha256((H/'runtime-tools-bound-b611864d3f6a-v3.json').read_bytes()).hexdigest()
records={}
for role in ('formal','native','first5'):
 row=spec[role];local=pathlib.Path(row['local_directory']);dest=row['remote_directory']
 m=local/row['manifest_name'];assert hashlib.sha256(m.read_bytes()).hexdigest()==row['manifest_sha256']
 for p,pin in row['files'].items():assert hashlib.sha256((local/p).read_bytes()).hexdigest()==pin
 remote('mkdir '+shlex.quote(dest))
 for p,pin in row['files'].items():publish(dest+'/'+p,(local/p).read_bytes())
 publish(dest+'/'+row['manifest_name'],m.read_bytes())
 pins=dict(row['files']);pins[row['manifest_name']]=row['manifest_sha256']
 code=f'import pathlib,hashlib,json;d=pathlib.Path({dest!r});pins={pins!r};assert all(hashlib.sha256((d/p).read_bytes()).hexdigest()==h for p,h in pins.items());print(json.dumps(dict(directory=str(d),verified_files=pins)))'
 q=remote('python3 -c '+shlex.quote(code));records[role]=json.loads(q.stdout)
(H/'tools-deployment-actual.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
