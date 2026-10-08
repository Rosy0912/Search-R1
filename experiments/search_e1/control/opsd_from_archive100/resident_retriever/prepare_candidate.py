"""Owner-run CPU preparation only. No CUDA, GPU activation or route mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path('/data/shaoshitong/search-e1-retriever-resident-20261008')
OLD=Path('/data/shaoshitong/search-e1-retriever-segmented-20261007')
UPSTREAM=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runtime/opsd/upstream/Search-R1')
ORIGINAL_ENCODER_SHA='fc528a2df9d2927f7dbf21a92d4671405259cba90e8e02dca661a4ad784c1a97'
ORIGINAL_SCAN_SHA='9b5e50257261ce16d363e58900465ad154d6158f014b7c9707e9880f0a2e84ca'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--allow-cpu-preparation',action='store_true')
    args=p.parse_args()
    if not args.allow_cpu_preparation:raise RuntimeError('Owner must explicitly allow CPU preparation; nothing created')
    if args.base!=BASE or args.base.exists():raise RuntimeError('Use exactly the new, nonexistent candidate directory')
    assets=OLD/'receipts/asset-identity-evidence.json'
    evidence=json.loads(assets.read_text())
    if evidence.get('accepted') is not True:raise RuntimeError('Original asset proof is unaccepted')
    for item in evidence['comparisons']:
        if item['k1_sha256']!=item['k4_sha256']:raise RuntimeError('Asset source SHA identity differs')
    old_identity_path=OLD/'receipts/source-identity.json'
    old_identity=json.loads(old_identity_path.read_text())
    if old_identity['pid']!=2239529 or old_identity['source_identity_sha256']!='b2626bd30e2c9ca12b94f17d4464ade6e012bbef36d7c2dd051c36e4f14fb614':
        raise RuntimeError('Original private oracle PID/source identity changed')
    def file_identity(path):
        st=path.stat()
        return dict(path=str(path),realpath=str(path.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino,device=st.st_dev)
    bindings=[]
    for kind,key,relative in [('index','index_file','index/e5_Flat.index'),('corpus','corpus_file','corpus/wiki-18.jsonl')]:
        previous=old_identity['identity'][key]
        current=file_identity(OLD/'assets'/relative)
        if any(current[name]!=previous[name] for name in ['realpath','size','mtime_ns','inode','device']):
            raise RuntimeError('Historical hashed index/corpus identity is no longer unchanged')
        proof=next(item for item in evidence['comparisons'] if item['kind']==kind)
        if current['size']!=proof['bytes'] or current['realpath']!=str(Path(proof['path']).resolve()):
            raise RuntimeError('Historical full hash refers to another asset')
        bindings.append(dict(**{**current,'path':str(args.base/'assets'/relative)},kind=kind,sha256=proof['k1_sha256'],sha_scope='Historical full-file SHA with unchanged realpath/size/mtime_ns/inode/device, not a fresh full-file hash'))
    encoder_files=[item for item in evidence['comparisons'] if item['kind']=='encoder']
    if len(encoder_files)!=18:raise RuntimeError('All original eighteen encoder files required')
    for proof in encoder_files:
        relative=Path(proof['path']).relative_to('/data/shaoshitong/search-e1-config-study-20261004/assets/encoder')
        path=OLD/'assets/encoder'/relative
        if sha(path)!=proof['k1_sha256']:raise RuntimeError('Original encoder bytes changed')
        current=file_identity(path)
        bindings.append(dict(**{**current,'path':str(args.base/'assets/encoder'/relative)},kind='encoder',sha256=proof['k1_sha256'],sha_scope='Fresh complete file SHA during CPU-only preparation'))
    encoder=UPSTREAM/'search_r1/search/retrieval_server.py'
    if sha(encoder)!=ORIGINAL_ENCODER_SHA:raise RuntimeError('Original CPU encoder source changed')
    if sha(args.source_dir/'torch_streaming_flat.py')!=ORIGINAL_SCAN_SHA:raise RuntimeError('Original scanner source changed')
    # These are source/runtime files only. A complete source snapshot is copied
    # independently; old source, env, assets and receipts are never modified.
    forbidden={'.git','.venv','env','__pycache__','wandb','checkpoints','data','assets'}
    allowed={'.py','.sh','.json','.yaml','.yml','.toml','.ini','.cfg','.md','.txt','.in','.cpp','.c','.h','.cu','.hpp','.rst'}
    files=[p for p in UPSTREAM.rglob('*') if p.is_file() and not p.is_symlink()
           and not forbidden.intersection(p.relative_to(UPSTREAM).parts)
           and p.suffix in allowed]
    if any(p.stat().st_size>32*1024*1024 for p in files):raise RuntimeError('Unexpected large non-source runtime file')
    source_files=[p for p in args.source_dir.iterdir() if p.is_file() and p.suffix in {'.py','.json','.md'}]
    required={'server.py','resident_scan.py','request_queue.py','operator_trace.py','torch_streaming_flat.py','full_live_gate.py','memory_observer.py','cpu_fixture.py','launch_cpu.py','candidate-config.json'}
    if not required <= {p.name for p in source_files}:raise RuntimeError('Candidate source package is incomplete')
    args.base.mkdir()
    for d in ['source','runtime','receipts','logs']:(args.base/d).mkdir()
    for src in source_files:shutil.copy2(src,args.base/'source'/src.name)
    target=args.base/'runtime/upstream/Search-R1'
    for src in files:
        dst=target/src.relative_to(UPSTREAM);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    (args.base/'env').symlink_to(OLD/'env',target_is_directory=True)
    (args.base/'assets').symlink_to(OLD/'assets',target_is_directory=True)
    binding_path=args.base/'receipts/asset-bindings.json'
    binding_path.write_text(json.dumps(dict(accepted_asset_binding=True,files=bindings,old_oracle_identity_path=str(old_identity_path),old_oracle_identity_sha256=sha(old_identity_path),historical_full_asset_sha_evidence_path=str(assets),historical_full_asset_sha_evidence_sha256=sha(assets),scope='Index/corpus historical full SHA plus unchanged file identity; all eighteen encoder file SHA freshly recomputed. No GPU work.'),indent=2)+'\n')
    # Freeze independently copied source bytes. Runtime identity does not claim
    # a new environment; the original installed env is used without upgrading.
    for directory in [args.base/'source',args.base/'runtime']:
        for path in directory.rglob('*'):
            if path.is_file():path.chmod(0o444)
    manifest={str(p.relative_to(args.base)):dict(bytes=p.stat().st_size,sha256=sha(p))
              for directory in [args.base/'source',args.base/'runtime'] for p in sorted(directory.rglob('*')) if p.is_file()}
    receipt=dict(prepared_cpu_files=True,gpu_initialized=False,gpu_activated=False,production_admitted=False,routing_changed=False,base=str(args.base),asset_manifest=str(args.base/'assets/manifest.json'),asset_bindings_path=str(binding_path),asset_bindings_sha256=sha(binding_path),original_asset_identity_path=str(assets),original_asset_identity_sha256=sha(assets),original_cpu_encoder_source_sha256=sha(target/'search_r1/search/retrieval_server.py'),source_runtime_files=manifest,links=dict(env=str(OLD/'env'),assets=str(OLD/'assets')),scope='Independent source/runtime copied locally on node only after owner authorizes this preparation command; no GPU initialization or process signal')
    path=args.base/'receipts/cpu-prepared.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(prepared_cpu_files=True,files=len(manifest),receipt=str(path),sha256=sha(path))))


if __name__=='__main__':main()
