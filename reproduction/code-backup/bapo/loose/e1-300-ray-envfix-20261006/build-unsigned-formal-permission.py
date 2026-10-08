"""Build UNSIGNED actual-allocation candidate. Never signs an independent report."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,re
B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('formal_checker',B/'check-formal-allocation.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
def build(capture,job,receipt_raw,receipt_sha,hardware_manifest,independent_report,waiting_environment=None):
 assert re.fullmatch('[0-9a-f]{64}',receipt_sha)and hashlib.sha256(receipt_raw).hexdigest()==receipt_sha
 result=checker.check(capture,job,receipt_sha,hardware_manifest,waiting_environment);hardware=json.loads(receipt_raw)
 assert receipt_raw==__import__('base64').b64decode(capture['hardware_acceptance']['hardware-acceptance.json']['base64'],validate=True)
 assert independent_report.name==f'FORMAL{job}-RESUME25-INDEPENDENT-PERMISSION-REVIEW.md'and independent_report.is_file()and not independent_report.is_symlink()
 raw=independent_report.read_bytes();assert raw
 run=checker.ROOT+'/runs/bapo-instruct300-formal300-'+str(job)
 return dict(status='UNSIGNED_CANDIDATE_REQUIRES_ACTUAL_INDEPENDENT_ALLOCATION_REVIEW',target_source_commit=checker.SOURCE,target_run_root=run,target_job_id=job,manifest_sha256=result['manifest_sha256'],sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT',independent_review={'path':run+'/'+independent_report.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},gpu_witnesses=hardware['gpu_witnesses'],gpu_validation=hardware['gpu_validation'],scope=f'ACTUAL_FORMAL{job}_COMPLETE25_CONTINUATION26_TO300_ONLY',actual_outer_updates_restored=25,next_step=26,target_actual_outer_updates=300,hardware_receipt_sha256=receipt_sha,legacy_checkpoint50_complete=False,duplicate_or_fresh_training_authorized=False,actual_runtime_restore_verified=False,production_permission_signed=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--job-id',type=int,required=True);p.add_argument('--hardware-receipt',type=Path,required=True);p.add_argument('--hardware-receipt-sha',required=True);p.add_argument('--hardware-manifest',type=Path,required=True);p.add_argument('--independent-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument("--waiting-environment",type=Path,required=True);a=p.parse_args()
 candidate=build(json.loads(a.capture.read_text()),a.job_id,a.hardware_receipt.read_bytes(),a.hardware_receipt_sha,a.hardware_manifest,a.independent_report,json.loads(a.waiting_environment.read_text()))
 with a.output.open('x')as f:json.dump(candidate,f,indent=2);f.write('\n')
 print('UNSIGNED_ONLY_NOT_PERMISSION',a.output)
if __name__=='__main__':main()
