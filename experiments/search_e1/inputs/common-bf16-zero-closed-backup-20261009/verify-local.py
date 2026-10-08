# Native closed-phase copy verification; uses the already published stdlib helper.
import argparse,hashlib,json,pathlib,runpy,time
p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--destination',required=True);p.add_argument('--source-helper',required=True);p.add_argument('--actual-transfer-returncode',type=int,required=True);p.add_argument('--after-metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args()
H=pathlib.Path(a.source_helper);assert hashlib.sha256(H.read_bytes()).hexdigest()=='d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57';b=runpy.run_path(str(H))
i=pathlib.Path(a.inventory);raw=i.read_bytes();x=json.loads(raw);D=pathlib.Path(a.destination);b['checked_path'](str(D),directory=True)
assert a.actual_transfer_returncode==0
afterpath=pathlib.Path(a.after_metadata);after=json.loads(afterpath.read_bytes());assert after['accepted'] is True and after['all_source_metadata_stable'] is True and after['source']==x['source'] and after['files']==len(x['files'])
paths=[]
for f in sorted(D.rglob('*')):
 assert not f.is_symlink()
 if f.is_dir():continue
 b['checked_path'](str(f));paths.append(f.relative_to(D).as_posix())
assert paths==[f['relative'] for f in x['files']]
for f in x['files']:
 name=b['safe_name'](f['relative']);b['fileproof'](D/name,dict(path=str(D/name),sha256=f['sha256'],bytes=f['bytes']))
proof=dict(schema='closed-bf16-zero-full-raw-backup-complete-v1',accepted=True,run_id=x['run_id'],version=x['version'],primary_slots=x['primary_slots'],strict_hits=x['strict_hits'],source=x['source'],destination=str(D),files=len(paths),bytes=x['bytes'],actual_rsync_returncode=a.actual_transfer_returncode,phase=x['phase'],actual_terminal=x['actual_terminal'],actual_worker_wait_codes=x['actual_worker_wait_codes'],inventory_sha256=hashlib.sha256(raw).hexdigest(),after_metadata_sha256=hashlib.sha256(afterpath.read_bytes()).hexdigest(),all_source_fresh_SHA_before=True,all_source_metadata_stable_after=True,all_local_full_SHA_match=True,node_originals_retained=True,backup_is_training_dependency=False,parent_controller_completion_not_required=True,no_model_or_GPU=True,observed_unix=time.time())
with pathlib.Path(a.output).open('x') as o:json.dump(proof,o,indent=2,sort_keys=True);o.write('\n')
print(json.dumps(dict(accepted=True,files=len(paths),bytes=x['bytes'],output=a.output)))
