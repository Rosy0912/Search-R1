"""Publish independently reviewed actual formal allocation permission, never submit."""
import pathlib,json,hashlib,subprocess,sys,re,argparse
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');source='a2568d0df306ead6e059760e28b5bd53ee3f6876'
args=argparse.ArgumentParser();args.add_argument('--hardware-receipt-sha',required=True);a=args.parse_args();assert a.hardware_receipt_sha=='e8625dabce8c9cf5a97498cb0989b399937fd9719acfa186357b1d29cebe1e2c'
hardware_path=root/'logs/hardware-acceptance.json';assert hardware_path.is_file()and not hardware_path.is_symlink()and hardware_path.resolve()==hardware_path
hardware_raw=hardware_path.read_bytes();assert hashlib.sha256(hardware_raw).hexdigest()==a.hardware_receipt_sha;hardware=json.loads(hardware_raw)
assert hardware['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and hardware['source_commit']==source and hardware['job_id']==3287 and hardware['actual_gpu_accepted']is True and hardware['production_training_permission_signed']is False
payload=json.load(sys.stdin);job=payload['job_id'];assert type(job)is int and job>0 and job!=3287
run=root/'runs'/f'bapo-instruct300-formal300-{job}';assert run==run.resolve()and not run.is_symlink()
assert (run/'WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION').is_file() and (run/'source-commit.txt').read_text().strip()==source
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True)
files=payload['files'];report_name=f'FORMAL{job}-RESUME25-INDEPENDENT-PERMISSION-REVIEW.md';assert set(files)=={report_name,'recovery-permission.json'}
permit=json.loads(files['recovery-permission.json']['text']);manifest=run/'recovery-manifest.json';m=json.loads(manifest.read_text())
assert m['target_job_id']==job and m['target_source_commit']==source and m['target_run_root']==str(run)
assert permit['status']=='PASS_INDEPENDENT_BAPO_RESUME25_REVIEW' and permit['target_job_id']==job and permit['target_run_root']==str(run) and permit['target_source_commit']==source
assert permit['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
assert permit['sampling_continuity']=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT'
assert permit['hardware_receipt_sha256']==a.hardware_receipt_sha and permit['gpu_witnesses']==hardware['gpu_witnesses']and permit['gpu_validation']==hardware['gpu_validation']
g=permit['gpu_validation'];assert g['job_id']==3287 and g['run_root']==str(root/'runs/bapo-instruct300-hardware-3287') and len(permit['gpu_witnesses'])==9 and len(g['native_v3_roundtrip'])==8
r=(files[report_name]['text']).encode();assert permit['independent_review']=={'path':str(run/report_name),'sha256':hashlib.sha256(r).hexdigest(),'bytes':len(r)}
for name in [report_name,'recovery-permission.json']:
 item=files[name];raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 path=run/name;assert not path.exists()and not path.is_symlink()
 with path.open('xb')as f:f.write(raw)
print(json.dumps({'status':'INDEPENDENT_NEW_FORMAL_PERMISSION_PUBLISHED_PENDING_ACTUAL_GATE','job_id':job,'source_commit':source}))
