import sys
sys.dont_write_bytecode=True
import json,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,os,time,struct
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');rows=[]
for kind,root,manifest,expected_sha in [('source',R/'repo-search-r1-32b-grpo100','r1-files.json','50fcaec86dcc0f7266f49cdf482a17d3b414b54a439f43a10617e49573320a25'),('tools',R/'tools/search-r1-32b-grpo100-tools-v3','files.json','9516f88f8e701cc40e360575b9205d23a9878456b94aef2f535a6a54bdfea8ec')]:
 p=root/manifest;b=p.read_bytes();m=json.loads(b);expected=m['files'];actual={};aliases=[]
 for parent,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   f=pathlib.Path(parent,name)
   if f.is_symlink():aliases.append(str(f.relative_to(root)))
  for name in files:
   f=pathlib.Path(parent,name);stat=f.stat();raw=f.read_bytes();n=str(f.relative_to(root));rec={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,'inode':stat.st_ino,'mode':oct(stat.st_mode&0o777),'path':str(f)}
   if n.endswith('.pyc') and len(raw)>=16:rec['pyc_header_hex']=raw[:16].hex();rec['pyc_flags']=struct.unpack('<I',raw[4:8])[0]
   actual[n]=rec
 missing=sorted(set(expected)-set(actual));extras={n:r for n,r in actual.items() if n not in set(expected)|{manifest}};changed={n:{'expected':expected[n],'actual':actual[n]} for n in set(expected)&set(actual) if expected[n]['size']!=actual[n]['size'] or expected[n]['sha256']!=actual[n]['sha256']}
 rows.append({'kind':kind,'root':str(root),'manifest_sha256':hashlib.sha256(b).hexdigest(),'expected_manifest_sha256':expected_sha,'manifest_matches':hashlib.sha256(b).hexdigest()==expected_sha,'expected_payload_count':len(expected),'actual_file_count':len(actual),'missing':missing,'extras':extras,'changed':changed,'aliases':aliases,'actual_files':actual})
print(json.dumps({'scope':'READ_ONLY_STANDARD_R1_V3_POST3158_EXACT_TREE_DIFF_NO_MUTATION','queried_epoch':time.time(),'directories':rows},sort_keys=True))
'''
r=send(source,attempts=1,timeout=180)
pathlib.Path('outputs/standard-r1-v3-3158-tree-transport.log').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode()
out=pathlib.Path('outputs/remote-evidence/standard-r1-v3-3158-tree.json');out.write_bytes(r.stdout)
data=json.loads(r.stdout)
print(json.dumps({'saved':str(out),'directories':[{k:v for k,v in d.items() if k!='actual_files'} for d in data['directories']]},indent=2))
