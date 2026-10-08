"""Local-only immutable tool binding. Does not start, upload, sign or inspect a job."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
FIRST5='12424d01025e9759d495500eef87de37712a29e1dfe989d3772d05cfa3bfe330'
NATIVE='1f209cab7e025439584a938b4aab7d71358db66ee3f7d63ab348285247ebbd65'
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    tools={'first5_overlap.py':(HERE/'first5_overlap.py').read_bytes()}
    manifest={'version':1,'source_commit':SOURCE,'first5_manifest_sha256':FIRST5,
              'native_manifest_sha256':NATIVE,'scope':'CPU_FIRST5_INSPECTION_ONLY_NO_ACCEPTANCE_SIGNATURE_RELEASE_OR_GPU',
              'contract':{'global_mini_batch':32,'adam_steps_per_rank_per_outer':16,
                          'saved_adam_steps':80,'horizon':200,'warmup_steps':0,'learning_rate':5e-6},
              'tools':{name:sha(raw) for name,raw in tools.items()}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    name='first5-overlap-'+sha(raw)[:12];directory=HERE/name
    directory.mkdir(exist_ok=True)
    assert directory.resolve(strict=True)==directory
    files=dict(tools);files['manifest.json']=raw
    for filename,content in files.items():
        target=directory/filename
        if target.exists():assert target.is_file() and not target.is_symlink() and target.read_bytes()==content
        else:
            with target.open('xb') as handle:handle.write(content)
    print(json.dumps({'scope':'LOCAL_TOOL_BINDING_ONLY_NO_ACTUAL_EXECUTION','source_commit':SOURCE,
         'local_directory':str(directory),'remote_directory':ROOT+'/tools/'+name,
         'manifest_sha256':sha(raw),'tools':manifest['tools']},sort_keys=True,indent=2))


if __name__=='__main__':main()
