"""Export only committed standalone evaluator code; never deploy or launch it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from contract import PACKAGE_SHA,CONFIG_SHA,RESTORE_RECEIPT_SHA,effective_segments

REL = 'reproduction/tree-nested-final-eval-20261014'
TRAIN_SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
FILES = ('contract.py','prepare.py','supervisor.py','dispatch_once.py','build_code_package.py',
         'README.md','eval_worker.py','eval_merge.py','shared_guard.py')
EXTERNAL = {'/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/tree_shared_owned_runtime_fixed.py':
            '4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0',
            '/dataset/shaoshitong/search-r1-rl/tree-grpo/repo/reproduction/h20/retriever_local.py':
            'a78f0b493c538dddc6619e0e462995fd9945c25a187c2165e99d4c10622ce459'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('Commit the standalone wrapper first')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if commit==TRAIN_SOURCE or out.exists() or out.is_relative_to(repo):raise ValueError('Distinct committed immutable output required')
    out.mkdir(parents=True);hashes={}
    for name in FILES:
        data=subprocess.check_output(['git','show',commit+':'+REL+'/'+name],cwd=repo)
        (out/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    value=dict(schema='nested_final_eval_code_v1',training_source_commit=TRAIN_SOURCE,
        eval_wrapper_source_commit=commit,
        training_chain=dict(package_sha256=PACKAGE_SHA,config_sha256=CONFIG_SHA,
            native5_restore_receipt_sha256=RESTORE_RECEIPT_SHA,effective_training_segments=effective_segments()),files=hashes,external_files=EXTERNAL,
        operational_changes_independently_reviewed=False,training_source_modified=False,
        actual_gpu_evaluation_performed=False,readiness='PENDING_REAL_TRAIN200_NATIVE_HF_ARCHIVE_AND_EXPLICIT_HANDOFF')
    path=out/'code-manifest.json';path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(code_manifest=str(path),code_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        remote_tools='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/eval-nested-'+commit,
        deployed=False,launched=False)))

if __name__=='__main__':main()
