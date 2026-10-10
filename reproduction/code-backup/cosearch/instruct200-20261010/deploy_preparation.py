"""Prepare a new remote code deployment. No Slurm submission or GPU initialization."""
import base64,hashlib,json,shutil,sys
from pathlib import Path
from prepare_source import patch_source,patch_trainer
from configure import configure,BASE,RUN

def main():
    source=Path(BASE)/'source/author-native/repo'
    run=Path(RUN)
    if run.exists():
        raise RuntimeError('Existing deployment: inspect manifest instead of overwriting')
    targets=['main_co_search_ppo.py','verl/verl/workers/fsdp_workers.py','verl/verl/trainer/ppo/search_r1_reranker_ray_trainer.py']
    texts=[(source/p).read_text() for p in targets]
    original_hashes={p:hashlib.sha256(t.encode()).hexdigest() for p,t in zip(targets,texts)}
    changed=[*patch_source(*texts[:2]),patch_trainer(texts[2])]
    run.mkdir()
    shutil.copytree(source,run/'repo',symlinks=True)
    for p,t in zip(targets,changed):(run/'repo'/p).write_text(t)
    payload=json.loads(sys.stdin.read())
    tools=run/'tools';tools.mkdir()
    for name,data in payload['tools'].items():
        if Path(name).name!=name:raise ValueError(name)
        (tools/name).write_bytes(base64.b64decode(data))
    overrides=configure(payload['author_cli'])
    (run/'author-cli.json').write_text(json.dumps(overrides,indent=2)+'\n')
    record={'status':'SOURCE_AND_CLI_DEPLOYED_NOT_RUNTIME_ACCEPTED','source':str(source),'run':str(run),'original_hashes':original_hashes,'patched_hashes':{p:hashlib.sha256((run/'repo'/p).read_bytes()).hexdigest() for p in targets},'training_started':False}
    (run/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
if __name__=='__main__':main()
