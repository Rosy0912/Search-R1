"""Publish only committed future cold-verifier code; never deploy or execute it."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import chain_contract as chain
REL='reproduction/tree-nested-final-hf-20261014'
FILES=('verify_final_hf.py','chain_contract.py','build_code_package.py','README.md')
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve()
 if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('Commit the standalone verifier first')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 if out.exists() or out.is_relative_to(repo):raise ValueError('Distinct immutable code output required')
 out.mkdir(parents=True);hashes={}
 for name in FILES:
  data=subprocess.check_output(['git','show',commit+':'+REL+'/'+name],cwd=repo)
  (out/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
 value=dict(schema='nested_final_hf_code_v1',training_source_commit=chain.TRAIN_SOURCE,
     verifier_wrapper_source_commit=commit,files=hashes,
     training_chain=dict(package_sha256=chain.PACKAGE_SHA,config_sha256=chain.CONFIG_SHA,
         native5_loaded_receipt_sha256=chain.RESTORE_RECEIPT_SHA,effective_training_segments=chain.effective_segments()),
     frozen_successful_hf_verifier_sha256='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97',
     reused_final_chain_contract_source_commit='5830f11efd6690bc894d1d07faad6f6c9fb372fa',
     operational_changes_independently_reviewed=False,training_source_modified=False,
     actual_model_or_native_loading_performed=False,readiness='PENDING_REAL_TRAIN200_AND_FINAL_NATIVE_HF_COLD_RELOAD')
 path=out/'code-manifest.json';path.write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(dict(code_manifest=str(path),code_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
     remote_tools=str(chain.ROOT/'instruct300'/('final-hf-nested-'+commit)),deployed=False,executed=False)))
if __name__=='__main__':main()
