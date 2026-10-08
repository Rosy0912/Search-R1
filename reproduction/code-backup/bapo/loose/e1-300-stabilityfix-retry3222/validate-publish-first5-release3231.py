"""Run only after actual job3231 first5 review. Does not create PASS receipts."""
import argparse,json,os,subprocess
from pathlib import Path
from first5_guard3231 import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--evidence-sha256',required=True);p.add_argument('--signed-receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);a=p.parse_args()
    evidence=owned_file(a.evidence);receipt=owned_file(a.signed_receipt)
    require(receipt.parent==RUN/'logs'and receipt.name!='step5-independent-release.json','Fresh run/logs staging receipt required')
    require(digest(evidence)==a.evidence_sha256 and digest(receipt)==a.receipt_sha256,'Actual review hashes mismatch')
    e=read(evidence);r=read(receipt);signed_binding(r,evidence,a.evidence_sha256)
    tool=owned_file(Path(e['tool']['path']),ROOT);require(digest(tool)==EXTRACTOR_SHA,'Reviewed extractor changed')
    pid,processes=live_pause(e);check_processes(processes)
    def slurm():validate_slurm(subprocess.check_output(['scontrol','show','job','-dd',str(JOB)],text=True))
    slurm()
    manifest=RUN/'checkpoints/full_state/global_step_5/manifest.json'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'repo'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
    code='from search_r1.baselines.checkpoint_gate import validate_release; import sys; validate_release(sys.argv[1],sys.argv[2],sys.argv[3],expected_driver_pid=int(sys.argv[4]))'
    subprocess.run([str(ROOT/'env/bin/python'),'-c',code,str(receipt),str(manifest),str(ROOT/'repo'),str(pid)],env=env,cwd=str(ROOT/'repo'),check=True)
    require(digest(evidence)==a.evidence_sha256 and digest(receipt)==a.receipt_sha256 and digest(tool)==EXTRACTOR_SHA,'Bindings changed during durable gate')
    live_pause(e);check_processes(processes);slurm()
    # The same newly signed inode is atomically installed, with no overwrite.
    publish_link(receipt,RUN/'logs/step5-independent-release.json',a.receipt_sha256)
    print(json.dumps(dict(status='ACTUAL_GATE_PASSED_AND_PUBLISHED',job_id=JOB,driver_pid=pid,source=SOURCE,receipt_sha256=a.receipt_sha256,release=str(RUN/'logs/step5-independent-release.json'))))
if __name__=='__main__':main()
