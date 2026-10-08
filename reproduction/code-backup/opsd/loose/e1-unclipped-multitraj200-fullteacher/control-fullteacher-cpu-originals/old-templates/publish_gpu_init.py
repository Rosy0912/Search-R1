"""Root-only draft: actual pre-GPU approval, with local RSA key and no-clobber publication.

Preparing this file does not run it. Execution requires observed actual ready/review
pins and --publish. Never upload this script's private-key file or retry an unknown
transport outcome without inspecting the original remote artifacts first.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
SOURCE = 'BLOCKED_UNBOUND_SOURCE_COMMIT'
BOUND_SPEC = HERE/'BLOCKED_BOUND_SPEC_FILENAME'
LOCAL_KEY = HERE.parent/'e1-300/review-signing/private.pem'
LOCAL_PUBLIC = HERE.parent/'e1-300/review-signing/public.json'
PUBLIC_SHA = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
FORMAL_SHA = 'BLOCKED_FORMAL_MANIFEST_SHA'
ROLES = {'allocation','source','import_closure','data_manifest','model_manifest','tool_manifest','environment'}
REVIEW_STATUS = 'ACTUAL_PRE_GPU_INDEPENDENT_REVIEW_PASS_INITIALIZATION_ONLY'
OPTIONS = ['-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
           '-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=none']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def local_pinned(path, pin):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and path.stat().st_uid == os.getuid(), 'Physical local owned file required')
    require(re.fullmatch('[0-9a-f]{64}', pin) is not None, 'Actual SHA pin required')
    raw = path.read_bytes()
    require(sha(raw) == pin, 'Local actual review pin changed')
    return raw


def route(command):
    worker = 'env -u LD_LIBRARY_PATH ssh '+shlex.join(OPTIONS+['h100-10', command])
    head = 'env -u LD_LIBRARY_PATH ssh '+shlex.join(OPTIONS+['-o','HostName=10.33.0.16',
           '-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16', worker])
    return ['ssh']+OPTIONS+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p',
                           'mgmt-host', head]


def remote(command, data=None):
    return subprocess.run(route(command), input=data, capture_output=True, check=True, timeout=60)


def owned_key():
    require(LOCAL_KEY.resolve(strict=True) == LOCAL_KEY and LOCAL_KEY.is_file()
            and LOCAL_KEY.stat().st_uid == os.getuid(), 'Private key must remain a physical local owned file')
    require(stat.S_IMODE(LOCAL_KEY.stat().st_mode) == 0o600
            and stat.S_IMODE(LOCAL_KEY.parent.stat().st_mode) == 0o700,
            'Local private key permissions differ')
    public = unique(local_pinned(LOCAL_PUBLIC, PUBLIC_SHA))
    require(set(public) == {'algorithm','e','n_hex'} and public['algorithm'] == 'RSA-SHA256'
            and public['e'] == 65537, 'Wrong task local public key')
    modulus = subprocess.check_output(['openssl','rsa','-in',str(LOCAL_KEY),'-noout','-modulus'],
                                      stderr=subprocess.PIPE, text=True).strip().split('=',1)[1]
    require(int(modulus,16) == int(public['n_hex'],16), 'Local signer key does not match the actual public key')
    return public


def validate_review(review, ready, ready_pin, job, run):
    require(review.get('status') == REVIEW_STATUS, 'Actual independent initialization-only PASS required')
    source = review.get('source')
    require(source == SOURCE and review.get('job_id') == job, 'Actual review source/job changed')
    expected = {'wrapper_pid':ready['pid'], 'process_start_ticks':ready['process_start_ticks'],
                'nonce':ready['nonce'], 'ready_sha256':ready_pin,
                'binding_sha256':ready['binding_sha256'], 'public_key_sha256':PUBLIC_SHA,
                'uuids':ready['nvml_uuids']}
    for key, value in expected.items():
        require(type(review.get(key)) is type(value) and review[key] == value,
                'Independent review current ready identity differs: '+key)
    require(type(review.get('all_seven_binding_hashes')) is dict
            and set(review['all_seven_binding_hashes']) == ROLES,
            'Independent seven-role binding evidence missing')
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') is not None
                for v in review['all_seven_binding_hashes'].values()), 'Invalid bound evidence SHA')
    require(review.get('native_closure_sha256') == review['all_seven_binding_hashes']['import_closure'],
            'Native closure review SHA differs')
    for key in ('actual_profile_sha256', 'actual_import_proof_sha256'):
        require(type(review.get(key)) is str
                and re.fullmatch('[0-9a-f]{64}', review[key]) is not None,
                'Exact reviewed actual profile/import proof SHA required: '+key)
    require(ready['source_commit'] == SOURCE and ready['job_id'] == job and ready['uid'] == 0
            and ready['run_root'] == str(run) and type(ready['pid']) is int
            and ready['pid'] > 0 and re.fullmatch('[0-9a-f]{64}',ready['nonce']) is not None,
            'Actual ready identity differs')


LIVE_TEMPLATE = r'''
import csv,datetime,hashlib,io,json,math,os,pathlib,re,socket,subprocess,time
D=__DATA_LITERAL__
def need(ok,msg):
 if not ok:raise ValueError(msg)
def unique(raw):
 def pairs(items):
  d={}
  for k,v in items:need(k not in d,'duplicateJSON');d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfiniteJSON')))
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def physical(p,file=True):
 p=pathlib.Path(p);need(p.is_absolute() and p.resolve(strict=True)==p and p.stat().st_uid==0,'physicalownedpath')
 need(p.is_file() if file else p.is_dir(),'wrongpathkind');return p
def command(args):return subprocess.check_output(args,text=True,timeout=20)
def validate_live():
 need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','wrongnodeUID')
 r=physical(D['run_root'],False);rp=physical(r/'approval-ready.json');need(hashfile(rp)==D['ready_sha256'],'readybyteschanged');v=unique(rp.read_bytes())
 need(v==D['ready'],'readychanged');job=D['job_id'];need(v['source_commit']==D['source_commit'] and v['uid']==0 and v['job_id']==job and v['run_root']==str(r),'foreignready')
 p=pathlib.Path('/proc')/str(v['pid']);status=p.joinpath('status').read_text();cg=p.joinpath('cgroup').read_text();start=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
 need(set(map(int,re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()))=={0},'fourUIDchanged')
 need(re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',cg,re.M) is not None and start==str(v['process_start_ticks'])==str(v['process']['start']) and cg==v['process']['cgroup'],'readyprocesschanged')
 state=re.search(r'^State:\s+(\S+)',status,re.M).group(1);need(state in ('R','S','D','I'),'readyprocessnotlive')
 argv=p.joinpath('cmdline').read_bytes().split(b'\0');need(str(pathlib.Path(D['formal_tools'])/'formal_supervisor.py').encode() in argv and b'--prepare' in argv,'wrongwrapperargv')
 s=command(['scontrol','show','job',job,'-o']);fields=dict(x.split('=',1) for x in s.split() if '=' in x)
 need(fields.get('JobId')==job and fields.get('JobState')=='RUNNING' and fields.get('NodeList')=='h100-10' and fields.get('TresPerNode')=='gres:gpu:h100:8' and fields.get('NumNodes')=='1' and fields.get('OverSubscribe')=='NO','wrongactualallocation')
 need(int(fields.get('NumCPUs','0'))>=192 and fields.get('MinMemoryNode') in ('1T','1024G','1048576M') and fields.get('TimeLimit')=='7-00:00:00' and re.fullmatch(r'\S+\(0\)',fields.get('UserId','')) is not None,'allocationresourcesUIDdiffer')
 repo=physical(D['repo'],False);need(command(['git','-C',str(repo),'rev-parse','HEAD']).strip()==D['source_commit'] and not command(['git','-C',str(repo),'status','--porcelain']).strip(),'sourcechanged')
 for name,pin in D['source_files'].items():need(hashfile(physical(repo/name))==pin,'sourcefilechanged:'+name)
 tools=physical(D['formal_tools'],False);tm=physical(tools/'manifest.json');need(hashfile(tm)==D['formal_manifest_sha256'],'formalmanifestchanged');toolmanifest=unique(tm.read_bytes())
 need(toolmanifest['source_commit']==D['source_commit'] and toolmanifest['source_files']==D['source_files'],'formalbindingchanged')
 for name,pin in toolmanifest['tools'].items():need(hashfile(physical(tools/name))==pin,'formaltoolchanged')
 bp=physical(r/'bindings/manifest.json');need(hashfile(bp)==v['binding_sha256'],'bindingchanged');binding=unique(bp.read_bytes())
 need(binding['source_commit']==D['source_commit'] and binding['run_root']==str(r) and set(binding['files'])==set(D['binding_hashes']),'bindingidentitychanged')
 for role,rec in binding['files'].items():
  x=physical(rec['path']);x.relative_to(r);need(set(rec)=={'path','sha256'} and rec['sha256']==D['binding_hashes'][role] and hashfile(x)==rec['sha256'],'bindingrolechanged:'+role)
 pub=physical(r/'bindings/public.json');need(hashfile(pub)==D['public_key_sha256']==v['public_key_sha256'],'publicchanged')
 source=unique(physical(binding['files']['source']['path']).read_bytes());need(source['source_commit']==D['source_commit'] and source['repo']==str(repo) and source['clean'] is True and source['source_files']==D['source_files'] and source['formal_manifest_sha256']==D['formal_manifest_sha256'],'boundsourcechanged')
 closure=unique(physical(binding['files']['import_closure']['path']).read_bytes());need(set(closure)=={'modules'} and bool(closure['modules']),'nativeclosuremissing')
 for name,rec in closure['modules'].items():
  need(set(rec)=={'path','sha256'} and hashfile(physical(rec['path']))==rec['sha256'],'nativeclosurebyteschanged:'+name)
  if name.startswith(('search_r1.','verl.','reproduction.')):physical(rec['path']).relative_to(repo)
 profilepath=physical(r/'evidence/pre-gpu-profile.json');need(hashfile(profilepath)==D['actual_profile_sha256'],'reviewedactualprofilechanged')
 proofpath=physical(r/'evidence/pre-gpu-import-check.json');need(hashfile(proofpath)==D['actual_import_proof_sha256'],'reviewedactualimportproofchanged')
 proof=unique(proofpath.read_bytes());need(proof['source_commit']==D['source_commit'] and proof['job_id']==job and proof['uid']==0 and proof['repo']==str(repo) and proof['closure_sha256']==binding['files']['import_closure']['sha256'] and proof['cuda_is_initialized'] is False and type(proof['function_code_count']) is int and proof['function_code_count']>0 and type(proof['module_count']) is int and proof['module_count']==len(proof['actual_loaded_modules']['modules']),'actualnativeCPUproofchanged')
 for name,rec in proof['actual_loaded_modules']['modules'].items():need(closure['modules'][name]==rec,'actualnativeimportoriginchanged')
 if D.get('review_records'):
  for rec in D['review_records']:
   x=physical(rec['path']);x.relative_to(r);need(hashfile(x)==rec['sha256'],'uploadedreviewchanged')
 for name in ('bindings/gpu-init-approval.json','bindings/gpu-init-approval.sha256','acceptance-5-receipt.json'):
  x=r/name;need(not x.exists() and not x.is_symlink(),'approvalalreadyexists:'+name)
 snapshots=[]
 for n in range(2):
  inv=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']);apps=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits']);need(not apps.strip(),'GPUcomputealreadyactive')
  rows=[dict(index=int(a[0]),uuid=a[1].strip(),name=a[2].strip(),used=float(a[3]),total=float(a[4]),util=float(a[5])) for a in csv.reader(io.StringIO(inv))]
  need(len(rows)==8 and {a['uuid'] for a in rows}==set(v['nvml_uuids']) and len({a['uuid'] for a in rows})==8 and all('H100' in a['name'] and all(math.isfinite(a[k]) for k in ('used','total','util')) and a['total']>0 and 0<=a['used']<=.15*a['total'] and 0<=a['util']<=15 for a in rows),'doubleidleactualUUIDguard')
  snapshots.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inventory_original=inv,compute_apps_original=apps,rows=rows))
  if n==0:time.sleep(2)
 return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,source_commit=D['source_commit'],pid=v['pid'],start=start,status_original=status,cgroup_original=cg,argv=[x.decode(errors='replace') for x in argv],slurm_original=s,native_module_count=len(closure['modules']),double_idle=snapshots,binding_sha256=v['binding_sha256'],ready_sha256=D['ready_sha256'])
'''


ATOMIC_PUBLISH = r'''
import os,pathlib,secrets,sys,hashlib,json
def exclusive(path,raw):
 path=pathlib.Path(path);parent=path.parent;need(parent.resolve(strict=True)==parent and parent.stat().st_uid==0,'foreignpublishparent')
 temp=parent/('.root-approval-'+secrets.token_hex(16)+'.tmp')
 fd=os.open(str(temp),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  with os.fdopen(fd,'wb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
  os.link(temp,path,follow_symlinks=False)
 finally:
  temp.unlink(missing_ok=True)
 d=os.open(str(parent),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(d)
 finally:os.close(d)
 need(hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest(),'publishedbyteschanged')
'''


def live_code(data, suffix):
    return LIVE_TEMPLATE.replace('__DATA_LITERAL__', repr(data)) + '\n' + suffix


def publish_review(path, raw):
    code = "import os,pathlib,secrets,sys,hashlib,json\n" \
           + "def need(x,m):\n if not x:raise ValueError(m)\n" + ATOMIC_PUBLISH \
           + '\nraw=sys.stdin.buffer.read();need(hashlib.sha256(raw).hexdigest()==' \
           + repr(sha(raw)) + ",'reviewtransportchanged');exclusive("+repr(str(path))+",raw)\n" \
           + "print(json.dumps({'published_sha256':hashlib.sha256(raw).hexdigest()}))\n"
    return remote('python3 -c '+shlex.quote(code), raw)


def final_publication_code(data, envelope):
    return live_code(data, ATOMIC_PUBLISH + '\nstate=validate_live()\nraw=sys.stdin.buffer.read();need(hashlib.sha256(raw).hexdigest()=='
               + repr(sha(envelope)) + ",'approvaltransportchanged')\nr=pathlib.Path(D['run_root'])\n"
               + "exclusive(r/'bindings/gpu-init-approval.json',raw)\n"
               + "exclusive(r/'bindings/gpu-init-approval.sha256',(hashlib.sha256(raw).hexdigest()+'\\n').encode())\n"
               + "print(json.dumps({'published_exclusively':True,'approval_sha256':hashlib.sha256(raw).hexdigest(),'actual_live_before_publication':state},sort_keys=True))\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--ready-sha256', required=True)
    parser.add_argument('--review-json', required=True, type=Path)
    parser.add_argument('--review-json-sha256', required=True)
    parser.add_argument('--review-report', required=True, type=Path)
    parser.add_argument('--review-report-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    require(args.publish, 'Draft only: explicit root --publish required to execute')
    require(re.fullmatch('[1-9][0-9]*', args.job_id) is not None
            and re.fullmatch('[0-9a-f]{64}',args.ready_sha256) is not None,
            'Observed actual job/ready pin required')
    require(args.output_dir.is_absolute() and args.output_dir.parent.resolve(strict=True)
            == args.output_dir.parent and not args.output_dir.exists(), 'Fresh physical local audit directory required')
    args.output_dir.mkdir(mode=0o700)
    def record(name, raw):
        with (args.output_dir/name).open('xb') as f:
            f.write(raw)
    review_raw = local_pinned(args.review_json, args.review_json_sha256)
    report_raw = local_pinned(args.review_report, args.review_report_sha256)
    review = unique(review_raw)
    run = ROOT/'runs'/('formal-'+args.job_id)
    ready_result = remote('cat '+shlex.quote(str(run/'approval-ready.json')))
    record('actual-ready-original.json',ready_result.stdout)
    require(sha(ready_result.stdout)==args.ready_sha256,'Actual ready SHA changed')
    ready = unique(ready_result.stdout)
    validate_review(review, ready, args.ready_sha256, args.job_id, run)
    spec = unique(BOUND_SPEC.read_bytes())
    require(spec['source_commit']==SOURCE and spec['remote_root']==str(ROOT)
            and spec['formal']['manifest_sha256']==FORMAL_SHA, 'Wrong local reviewed bound tools')
    public = owned_key()
    formal_dir = Path(spec['formal']['local_directory'])
    fm = unique(local_pinned(formal_dir/'manifest.json', FORMAL_SHA))
    require(fm['source_files']==spec['source_files'] and len(fm['source_files'])==19,
            'Exact source/scheduler inventory required')
    for name,pin in fm['tools'].items():local_pinned(formal_dir/name,pin)
    json_path = run/'evidence/independent-pre-gpu-review.json'
    report_path = run/'evidence/independent-pre-gpu-review.md'
    data = {'run_root':str(run),'repo':spec['remote_repo'],'source_commit':SOURCE,
            'formal_tools':spec['formal']['remote_directory'],'formal_manifest_sha256':FORMAL_SHA,
            'source_files':fm['source_files'],'job_id':args.job_id,'ready_sha256':args.ready_sha256,
            'ready':ready,'binding_hashes':review['all_seven_binding_hashes'],
            'actual_profile_sha256':review['actual_profile_sha256'],
            'actual_import_proof_sha256':review['actual_import_proof_sha256'],
            'public_key_sha256':PUBLIC_SHA,'review_records':[]}
    result = remote('python3 -c '+shlex.quote(live_code(data,'print(json.dumps(validate_live(),sort_keys=True))')))
    record('actual-pre-upload-live-original.json',result.stdout)
    for name,path,raw in [('json',json_path,review_raw),('report',report_path,report_raw)]:
        published = publish_review(path,raw)
        record('review-'+name+'-publish-original.json',published.stdout)
    data['review_records']=[{'path':str(json_path),'sha256':args.review_json_sha256},
                            {'path':str(report_path),'sha256':args.review_report_sha256}]
    before_sign = remote('python3 -c '+shlex.quote(live_code(data,'print(json.dumps(validate_live(),sort_keys=True))')))
    record('actual-pre-sign-live-original.json',before_sign.stdout)
    payload = {key:ready[key] for key in ('source_commit','job_id','uid','pid','process_start_ticks',
                                         'run_root','nonce','binding_sha256','public_key_sha256')}
    payload.update(version=1,authorization='EXPLICIT_OPSD_E1_GPU_INITIALIZATION_ONLY',
                   ready_sha256=args.ready_sha256,
                   independent_review={'path':str(json_path),'sha256':args.review_json_sha256})
    # Private key path is passed only to this LOCAL process. remote() receives
    # review bytes and the public signed envelope, never key bytes or credentials.
    signature = subprocess.run(['openssl','dgst','-sha256','-sign',str(LOCAL_KEY)],
                               input=canonical(payload),capture_output=True,check=True).stdout
    sys.path.insert(0,str(formal_dir))
    module_spec = importlib.util.spec_from_file_location('reviewed_formal_root_signer', formal_dir/'formal_supervisor.py')
    f = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(f)
    require(payload==f.approval_expected(dict(ready,ready_sha256=args.ready_sha256),payload['independent_review']),
            'Frozen supervisor expected authorization differs')
    f.verify_rsa(payload,signature.hex(),public)
    envelope = canonical({'payload':payload,'signature_hex':signature.hex()})
    record('actual-gpu-init-signed-envelope.json',envelope)
    final_code = final_publication_code(data, envelope)
    try:
        final = remote('python3 -c '+shlex.quote(final_code),envelope)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
        record('publication-failure-or-unknown-outcome.json',canonical({'error_type':type(e).__name__,
               'returncode':getattr(e,'returncode',None),'scope':'Inspect actual remote original files before any retry; never automatically republish'}))
        raise
    record('actual-exclusive-publication-original.json',final.stdout)
    print(json.dumps({'job_id':args.job_id,'source_commit':SOURCE,'approval_sha256':sha(envelope),
                      'scope':'Actual external pre-GPU initialization authorization only; no first5/200/effect acceptance'}))


if __name__=='__main__':
    main()
