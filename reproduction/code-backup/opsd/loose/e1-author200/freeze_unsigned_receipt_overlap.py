"""Local-only immutable unsigned runner binding, never a job or receipt launch."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    script=(HERE/'unsigned_receipt_overlap.py').read_bytes()
    manifest={'version':1,'source_commit':'188f4e5e7b63cf479290d0888a4a075499886750',
       'scope':'UNSIGNED_FIRST5_PAYLOAD_PREPARATION_ONLY_NO_REVIEW_SIGNATURE_RELEASE_OR_GPU',
       'first5_manifest_sha256':'12424d01025e9759d495500eef87de37712a29e1dfe989d3772d05cfa3bfe330',
       'overlap_manifest_sha256':'1cfb9379063254ae242c7fdcfa4a57fb9927cc9d5fe00b97a49455aa4f652fea',
       'overlap_script_sha256':'7ec32fbc071bcdb10de8824efa26d845588eb0431e14eaed834f4f5aaa4f98b2',
       'producer_sha256':'79f29673f4dee991f295e9915ff163ada5073776638a330529571dabf007a64d',
       'tools':{'unsigned_receipt_overlap.py':sha(script)}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    name='first5-unsigned-overlap-'+sha(raw)[:12];directory=HERE/name;directory.mkdir(exist_ok=True)
    assert directory.resolve(strict=True)==directory
    for filename,content in {'unsigned_receipt_overlap.py':script,'manifest.json':raw}.items():
        target=directory/filename
        if target.exists():assert target.is_file() and not target.is_symlink() and target.read_bytes()==content
        else:
            with target.open('xb') as handle:handle.write(content)
    print(json.dumps({'scope':'LOCAL_TOOL_BINDING_ONLY_NO_ACTUAL_UNSIGNED_PRODUCTION','local_directory':str(directory),
       'remote_directory':ROOT+'/tools/'+name,'manifest_sha256':sha(raw),'tools':manifest['tools']},sort_keys=True,indent=2))


if __name__=='__main__':main()
