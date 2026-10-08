"""Root-task-only candidate publisher. No collector, GPU, job control or signing.

Only main with --deploy-reviewed performs one remote publish; preparation and
CPU tests never invoke it. Any unknown outcome must be inspected, not retried.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import types

W = Path(__file__).resolve().parents[1]
RUNTIME = 'd83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
TRANSPORT = 'a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'

CODE = r'''
import base64,hashlib,json,os,re,socket,subprocess,sys,types
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN=ROOT/'runs/formal-3261';REPO=ROOT/'repo-188f4e5e7b63'
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
RUNTIME='d83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
BASE='90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
READY='24d876c05496678648fa1d19228fe87c19685bd35903c2f0d1f641ef73af4892'
def need(v,m):
 if not v:raise ValueError(m)
def unique(raw):
 def pairs(items):
  out={}
  for k,v in items:need(k not in out,'Duplicate JSON key');out[k]=v
  return out
 v=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:need(False,'Nonfinite JSON'))
 json.dumps(v,allow_nan=False);return v
def validate_packet(packet):
 need(set(packet)=={'manifest_sha256','manifest_b64','members'},'Packet schema differs')
 pin=packet['manifest_sha256'];need(type(pin)is str and re.fullmatch('[0-9a-f]{64}',pin),'Invalid manifest pin')
 raw=base64.b64decode(packet['manifest_b64'],validate=True)
 need(hashlib.sha256(raw).hexdigest()==pin,'Manifest bytes changed')
 m=unique(raw);need(type(m['version'])is int and m['version']==1,'Typed manifest version required')
 need(m['kind']=='AUTHOR200_FINAL_ORIGINAL_ARCHIVE_PREPARATION_TOOLS' and m['source_commit']==SOURCE,'Wrong archive/source kind')
 need(m['tools']=={'collect_actual3261_final_originals.py':RUNTIME},'Only exact d83 is runtime')
 need(not(set(m['tools'])&set(m['review_records'])) and len(m['review_records'])==21,'Review/member closure differs')
 contract=m['contract'];need(contract['training_job_id']=='3261' and contract['run_root']==str(RUN) and contract['source_root']==str(REPO),'Foreign job/run/source')
 need(type(contract['original_driver_pid'])is int and contract['original_driver_pid']==1533312 and contract['original_driver_start_ticks']=='1829085316','Original driver identity differs')
 need(contract['deployment_requires_original_driver_alive']is True and contract['deployment_requires_typed_running_allocation']is True and contract['requires_live_training_driver']is False,'Deployment/collector life scopes differ')
 need(contract['runtime_execution_authorized_by_freeze']is False,'Bundle cannot execute collector')
 need(contract['config_sha256']=='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60' and type(contract['horizon'])is int and contract['horizon']==200 and type(contract['world_size'])is int and contract['world_size']==8,'Typed original config/budget differs')
 need(type(contract['fresh_existing_dependency_count'])is int and contract['fresh_existing_dependency_count']==len(m['dependencies'])==31,'Exact31 collector dependencies required')
 pins={**m['tools'],**m['review_records']};need(set(packet['members'])==set(pins),'Member set differs')
 members={n:base64.b64decode(b,validate=True) for n,b in packet['members'].items()}
 for n,data in members.items():need(Path(n).name==n and n not in ('manifest.json','.','..') and hashlib.sha256(data).hexdigest()==pins[n],'Member name/SHA differs')
 return m,raw,members
def load_io(raw):
 need(hashlib.sha256(raw).hexdigest()==RUNTIME,'Collector byte pin differs')
 c=types.ModuleType('verified_packet_archive_collector_io');c.__file__='<exact-reviewed-d83-packet>'
 exec(compile(raw,c.__file__,'exec'),c.__dict__);return c
def read(c,path,pin):
 fd=c.open_physical(path,uid=0)
 try:
  before=c.metadata(os.fstat(fd))
  with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read()
  need(c.metadata(os.fstat(fd))==before==c.physical_metadata(path,0) and len(raw)==before['bytes'] and hashlib.sha256(raw).hexdigest()==pin,'Dependency changed: '+str(path))
  return raw
 finally:os.close(fd)
def expected_dependencies(c):
 basepath=ROOT/'tools/final-acceptance-90c2695a7f46/manifest.json'
 base=unique(read(c,basepath,BASE));need(base['source_commit']==SOURCE and len(base['dependencies'])==278,'Existing90 reference differs')
 sourcepath=RUN/'bindings/source.json'
 source=unique(read(c,sourcepath,base['dependencies'][str(sourcepath)]))
 need(source['source_commit']==SOURCE and source['repo']==str(REPO) and len(source['source_files'])==16,'Existing actual source16 identity differs')
 wanted={str(REPO/n):h for n,h in source['source_files'].items()}
 selected=[RUN/n for n in ('bindings/source.json','bindings/allocation.json','bindings/manifest.json','bindings/import_closure.json','bindings/public.json','full_state/global_step_5/resolved-config.json','full_state/global_step_5/manifest.json','acceptance-5-receipt.json')]
 formal=ROOT/'tools/formal-bb5f4872142f'
 selected += [formal/n for n in ('manifest.json','formal_supervisor.py','formal_job.sh','owned_runtime.py','resource_sample.py')]
 for path in selected:wanted[str(path)]=base['dependencies'][str(path)]
 wanted[str(basepath)]=BASE;wanted[str(RUN/'approval-ready.json')]=READY
 need(len(wanted)==31,'Exact31 existing collector paths differ')
 return wanted
def source_and_allocation(c,verified):
 need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Wrong actual UID/node')
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip()
 clean=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'])
 need(head==SOURCE and not clean,'Source188 changed or dirty')
 raw=subprocess.check_output(['scontrol','show','job','3261','-o'])
 fields=c.scheduler_fields(raw)['fields']
 original=unique(verified[str(RUN/'bindings/allocation.json')])
 old=c.scheduler_fields(original['scontrol'].encode())['fields']
 need(fields.get('JobId')=='3261' and fields.get('JobState')=='RUNNING' and fields.get('NodeList')=='h100-10' and fields.get('UserId')=='root(0)','Original typed RUNNING allocation missing')
 need(fields.get('NumNodes')=='1' and fields.get('TresPerNode')=='gres:gpu:h100:8' and fields.get('OverSubscribe')=='NO' and fields.get('Restarts')=='0','Original8GPU/exclusive/noRestart allocation differs')
 need(fields.get('NumCPUs')=='192' and fields.get('MinMemoryNode')=='1T' and fields.get('StartTime')==old['StartTime'] and fields.get('SubmitTime')==old['SubmitTime'],'Original allocation start/resources differ')
 identity=c.process_observation(1533312,'1829085316')
 need(identity['status']=='ORIGINAL_OR_CURRENT_PROCESS_ALIVE' and identity['uids']==[0,0,0,0] and all(type(v)is int for v in identity['uids']) and identity['cgroup']==c.CGROUP,'Original driver must actually remain alive at deployment')
 return {'scontrol_original':raw.decode(),'source_commit':head,'source_clean':True,'original_driver_current_identity':identity,'scope':'Fresh deployment identity only; no collector or terminal acceptance'}
def publish(packet):
 need(sys.flags.optimize==0,'Optimized Python forbidden')
 m,mraw,members=validate_packet(packet);c=load_io(members['collect_actual3261_final_originals.py'])
 need(m['dependencies']==expected_dependencies(c),'Publication cannot read unapproved dependency paths or pins')
 verified={p:read(c,p,h) for p,h in m['dependencies'].items()}
 need(len(verified)==31,'Dependency count differs')
 accepted=unique(verified[str(RUN/'full_state/global_step_5/manifest.json')])
 need(accepted['source_commit']==SOURCE and accepted['job_id']=='3261' and accepted['run_root']==str(RUN) and type(accepted['driver_pid'])is int and accepted['driver_pid']==1533312 and accepted['config_sha256']==m['contract']['config_sha256'],'Existing accepted5 identity differs')
 source=unique(verified[str(RUN/'bindings/source.json')])
 need(source['source_commit']==SOURCE and source['repo']==str(REPO) and len(source['source_files'])==16,'Existing source inventory differs')
 for n,h in source['source_files'].items():need(m['dependencies'].get(str(REPO/n))==h,'Exact source16 closure differs')
 before=source_and_allocation(c,verified)
 d=ROOT/'tools'/('final-archive-'+packet['manifest_sha256'][:12]);parent=d.parent
 need(parent.resolve(strict=True)==parent and parent.stat().st_uid==0,'Physical own tools parent required')
 os.mkdir(d,0o700);published=[]
 for name,data in list(members.items())+[('manifest.json',mraw)]:
  c.write_exclusive(d/name,data);(d/name).chmod(0o444)
  fd=c.open_physical(d/name,uid=0)
  try:os.fsync(fd)
  finally:os.close(fd)
  pin=hashlib.sha256(data).hexdigest();need(read(c,d/name,pin)==data,'Published member readback differs')
  published.append({'name':name,'bytes':len(data),'sha256':pin,'mode':oct((d/name).stat().st_mode&0o777)})
 need(published[-1]['name']=='manifest.json' and set(p.name for p in d.iterdir())==set(members)|{'manifest.json'},'Manifest ordering/member set differs')
 d.chmod(0o555)
 for path in (d,parent):
  fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(fd)
  finally:os.close(fd)
 verified_after={p:read(c,p,h) for p,h in m['dependencies'].items()}
 need(verified_after==verified,'Existing dependencies changed during publication')
 after=source_and_allocation(c,verified_after)
 return {'scope':'ACTUAL_NEW_ARCHIVE_SOFTWARE_PUBLICATION_NOT_COLLECTOR_OR_FINAL_ACCEPTANCE','directory':str(d),'manifest_sha256':packet['manifest_sha256'],'manifest_last':True,'published':published,'runtime_count':1,'existing_dependency_count':31,'source_commit':SOURCE,'allocation_before':before,'allocation_after':after,'verified_dependencies':{p:{'sha256':hashlib.sha256(r).hexdigest(),'bytes':len(r)} for p,r in verified.items()},'directory_mode':oct(d.stat().st_mode&0o777),'collector_executed':False}
if __name__=='__main__':print(json.dumps(publish(unique(sys.stdin.buffer.read())),sort_keys=True))
'''


def need(value, message):
    if not value:
        raise ValueError(message)


def helpers():
    # Loader is local-only and exact source; no stale-pyc or source rewriting.
    path = Path(__file__).parent / 'freeze_final_archive_bundle.py'
    raw = path.read_bytes()
    module = types.ModuleType('local_archive_freeze_preparation')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    io, _ = module.bootstrap_runtime()
    return module, io


def prepare_packet(bundle, pin, review_path, review_pin):
    f, io = helpers()
    bundle = Path(bundle)
    need(bundle.parent == W and bundle.name == 'final-archive-' + pin[:12], 'Wrong local immutable bundle name')
    raw = f.read(io, bundle / 'manifest.json', pin)
    manifest = io.unique_json(raw)
    peer_raw = f.read(io, review_path, review_pin)
    peer = io.unique_json(peer_raw)
    need(peer['status'] == 'PASS_INDEPENDENT_FINAL_ARCHIVE_IMMUTABLE_BUNDLE_REVIEW'
         and peer['manifest_sha256'] == pin and peer['collector_runtime_sha256'] == RUNTIME
         and peer['source_commit'] == SOURCE, 'Actual independent freeze review must bind exact bundle/runtime/source')
    members = {}
    for name, h in {**manifest['tools'], **manifest['review_records']}.items():
        need(Path(name).name == name and name not in ('manifest.json', '.', '..'), 'Unsafe local member')
        data = f.read(io, bundle / name, h)
        need((bundle / name).stat().st_mode & 0o777 == 0o444, 'Immutable file mode differs')
        members[name] = base64.b64encode(data).decode()
    need(bundle.stat().st_mode & 0o777 == 0o555, 'Immutable directory mode differs')
    packet = {'manifest_sha256': pin, 'manifest_b64': base64.b64encode(raw).decode(), 'members': members}
    # Validate the exact portable remote program locally without calling publish.
    remote_module = types.ModuleType('portable_archive_publication_validation_only')
    exec(compile(CODE, '<candidate-root-publisher>', 'exec'), remote_module.__dict__)
    remote_module.validate_packet(packet)
    return packet, peer_raw


def approved_transport(f, io):
    # Root's approved route is loaded from its same pinned bytes, never pyc.
    path = W / 'transport.py'
    raw = f.read(io, path, TRANSPORT)
    module = types.ModuleType('exact_approved_h100_transport')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module.remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-reviewed', action='store_true')
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--independent-review', required=True)
    parser.add_argument('--independent-review-sha256', required=True)
    args = parser.parse_args()
    need(args.deploy_reviewed, 'Root must explicitly choose reviewed publication; default never deploys')
    packet, peer_raw = prepare_packet(args.bundle, args.manifest_sha256,
                                     args.independent_review, args.independent_review_sha256)
    f, io = helpers()
    remote = approved_transport(f, io)
    out = W / ('final-archive-deployment-' + args.manifest_sha256[:12] + '-originals')
    os.mkdir(out, 0o700)
    io.write_json(out / 'publish-once-intent.json', {'manifest_sha256': args.manifest_sha256,
        'independent_freeze_review_sha256': hashlib.sha256(peer_raw).hexdigest(),
        'remote_publisher_source_sha256': hashlib.sha256(CODE.encode()).hexdigest(),
        'unknown_outcome_rule': 'Inspect this attempt and exact remote target; never remove/overwrite/retry publisher'})
    io.write_exclusive(out / 'publisher-remote-source.py', CODE.encode())
    try:
        result = remote('python3 -c ' + shlex.quote(CODE), json.dumps(packet).encode(), timeout=180)
    except Exception as error:
        for name, attr in (('publish-failed.stdout.original', 'stdout'), ('publish-failed.stderr.original', 'stderr')):
            raw = getattr(error, attr, None)
            if isinstance(raw, bytes):
                io.write_exclusive(out / name, raw)
        io.write_json(out / 'publish-unknown-failure-original.json', {'error': repr(error), 'must_not_retry': True})
        raise
    io.write_exclusive(out / 'publish-once-stdout.original.json', result.stdout)
    io.write_exclusive(out / 'publish-once-stderr.original', result.stderr)
    io.write_json(out / 'publish-once-exit-original.json', {'returncode': result.returncode})
    need(type(result.returncode) is int and result.returncode == 0, 'Actual publication command failed')
    observed = io.unique_json(result.stdout)
    need(observed['manifest_sha256'] == args.manifest_sha256 and observed['manifest_last'] is True
         and observed['collector_executed'] is False, 'Actual publication result differs')
    print(json.dumps({'scope': 'NEW_SOFTWARE_DEPLOYMENT_ONLY_NOT_FINAL_ACCEPTANCE',
                      'manifest_sha256': args.manifest_sha256, 'actual_original_sha256': hashlib.sha256(result.stdout).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
