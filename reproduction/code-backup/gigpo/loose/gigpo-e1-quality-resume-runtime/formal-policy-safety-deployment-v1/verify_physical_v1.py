"""Template: target CPU-only physical proof; refuses null calibration before asset/import work."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from deployment_contract_v1 import *


def metadata(path):
    p=Path(path);s=p.stat()
    require(p.is_file() and not p.is_symlink(),'Non-original file metadata')
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)


def verify(candidate_record,output):
    c=read_bound(candidate_record);proposal(c)  # Current null draft stops here.
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only physical probe required')
    require(sys.executable==str(ROOT/'env/bin/python'),'Wrong target interpreter')
    inherited=read_bound(c['inherited_physical']);old={r['path']:r for r in inherited['metadata']}
    files=[];hashed=[]
    prefixes=(str(REPO)+'/',str(HELPERS)+'/',str(CONTROLS)+'/',str(ROOT/'tools/policy-safety-native-52befe9-v2')+'/')
    for item in c['files']:
        m=metadata(item['path']);require(m['bytes']==item['bytes'],'Physical file size changed')
        if item['path'].startswith(prefixes):
            require(sha(item['path'])==item['sha256'],'Actual new source/helper SHA differs');hashed.append(item['path'])
        else:require(m==old[item['path']],'Inherited asset metadata differs; no new asset hash claimed')
        files.append(dict(m,sha256=item['sha256']))
    evidence=[]
    for item in c['evidence']:
        require(record(item['path'])==item,'Actual original proof closure differs');evidence.append(item)
    require(subprocess.check_output(['git','-C',c['repo'],'rev-parse','HEAD'],text=True).strip()==SOURCE
        and not subprocess.check_output(['git','-C',c['repo'],'status','--porcelain'],text=True).strip(),'Dirty/different target source')
    import torch
    import search_r1.baselines.policy_safety as safety
    import search_r1.baselines.recovery as recovery
    require(torch.__version__=='2.4.0+cu121' and not torch.cuda.is_initialized(),'Wrong target Torch/CUDA state')
    require(inspect.getfile(safety)==str(REPO/'search_r1/baselines/policy_safety.py') and inspect.getfile(recovery)==str(REPO/'search_r1/baselines/recovery.py'),'Actual source import path differs')
    from prepare_candidate_v1 import preview_config
    actual=preview_config(c,REPO/'verl/trainer/config');preview=read_bound(c['expected_config_preview'])
    require(actual==preview['resolved'],'Actual target resolved config differs')
    from verl.utils import hf_tokenizer
    tokenizer=hf_tokenizer(actual['actor_rollout_ref']['model']['path'])
    require(tokenizer.pad_token_id==c['probability_identity']['pad_token_id']
        and tokenizer.eos_token_id==c['probability_identity']['eos_token_id'] and not torch.cuda.is_initialized(),
        'Actual official tokenizer differs or physical probe initialized CUDA')
    value=dict(status='PHYSICAL_FORMAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',candidate_sha256=candidate_record['sha256'],
        code_commit=SOURCE,files=files,evidence=evidence,resolved_config=actual,config_preview_sha256=c['expected_config_preview']['sha256'],
        executable=sys.executable,torch=torch.__version__,cuda_initialized=False,actual_formal_GPU_accepted=False,
        newly_hashed_paths=hashed,inherited_physical_sha256=c['inherited_physical']['sha256'],runtime_full_SHA_before_GPU_required=True,
        guard_identity=c['guard_identity'],pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    write_once(output,value);return record(output)


if __name__=='__main__':
    require(len(sys.argv)==3,'Need complete finalized candidate record JSON and new CPU-only physical output')
    print(json.dumps(verify(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])))
