"""Actual archived baseline GPU retrieval; never manufactures a training receipt."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, runpy, subprocess, sys

ROOT = Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE = '080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
ARCHIVE_SHA256 = 'e85ab42b9032d66a35239a36e8cd73c6f3e3efde929541150fe7c5bfdb4936d9'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def passages(contents):
    result = ''
    for i, content in enumerate(contents):
        parts = content.split('\n'); title, text = parts[0], '\n'.join(parts[1:])
        result += f'Doc {i+1}(Title: {title}) {text}\n'
    return result

def main():
    if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        raise RuntimeError('Optimized Python disables mandatory guard checks')
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    a = p.parse_args(); output = Path(a.output)
    run = Path(os.environ['BAPO_RUN_ID'])
    assert run.parent == ROOT/'runs' and run.resolve() == run
    assert run.name.startswith('hardware-aligned-em200-') and run.name.endswith('-'+os.environ['SLURM_JOB_ID'])
    assert sys.prefix == str(ROOT/'env')
    sys.path.insert(0, str(ROOT/'repo'))
    assert output.parent == run and not output.exists()
    assert os.uname().nodename == 'h100-5' and os.environ['SLURM_JOB_ID']
    assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip() == SOURCE
    assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
    import search_r1.baselines.preflight as preflight
    assert Path(preflight.__file__).resolve() == ROOT/'repo/search_r1/baselines/preflight.py'
    validate_inherited_gpu_leases = preflight.validate_inherited_gpu_leases
    leases = validate_inherited_gpu_leases(os.environ['CUDA_VISIBLE_DEVICES'].split(','), require_guard=True)
    assert len(leases) == 8
    claim = dict(source=SOURCE,run=str(run),job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),
        start_ticks=(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        collector_sha256=sha(__file__),scope='INDEPENDENT_ARCHIVE_GPU_GOLDEN_CAPTURE')
    with (run/'GPU-GOLDEN-CAPTURE-CLAIM.json').open('x') as f:
        json.dump(claim,f);f.flush();os.fsync(f.fileno())
    import torch, faiss
    assert torch.cuda.is_available() and torch.cuda.device_count() == faiss.get_num_gpus() == 8
    here = Path(__file__).resolve().parent
    ref = here/'reference/retrieval_server.py'
    provenance = json.loads((here/'reference/retrieval-provenance.json').read_text())
    assert sha(ref) == provenance['sha256'] == ARCHIVE_SHA256
    namespace = runpy.run_path(str(ref))  # __main__ server launch intentionally excluded.
    assets = (ROOT/'assets-root/assets').resolve(strict=True)
    config = namespace['Config'](retrieval_method='e5', retrieval_topk=3,
        index_path=str(assets/'wiki/e5_Flat.index'), corpus_path=str(assets/'wiki/wiki-18.jsonl'),
        faiss_gpu=True, retrieval_model_path=str(assets/'e5-base-v2'),
        retrieval_pooling_method='mean', retrieval_query_max_length=256,
        retrieval_use_fp16=True, retrieval_batch_size=512)
    import search_r1.baselines.baseline_alignment_retrieval as identity
    assert Path(identity.__file__).resolve() == ROOT/'repo/search_r1/baselines/baseline_alignment_retrieval.py'
    asset_paths, file_identity = identity.asset_paths, identity.file_identity
    paths = asset_paths(config, ROOT/'repo/search_r1/search/retrieval_server.py')
    before = {k: file_identity(v) for k,v in paths.items()}
    pins = {k: dict(path=before[k]['path'],sha256=sha(v)) for k,v in paths.items()}
    for name in ('corpus','index'):
        mount = subprocess.check_output(['findmnt','-T',pins[name]['path'],'-n','-o','FSTYPE'],text=True).strip()
        assert mount in ('ext4','xfs'), 'production retrieval assets must be node-local'
    common = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data')
    assert sha(common/'manifest.json') == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    import pandas as pd
    prepared_path=ROOT/'inputs/baseline-aligned-data.json'
    assert sha(prepared_path)=='e5e47bdfbfdffcc9a5cf0feccc39e52f32aa093fa039f7464e9baa5422d21d15'
    import search_r1.baselines.baseline_alignment_data as recorded_order
    assert Path(recorded_order.__file__).resolve()==ROOT/'repo/search_r1/baselines/baseline_alignment_data.py'
    expected=json.loads(prepared_path.read_text())['splits']['validation']
    observed=recorded_order.dataset_identity([str(common/'validation.parquet')])
    assert observed==expected, 'validation bytes/semantics/IDs differ from fixed prepared input'
    validation_before=sha(common/'validation.parquet')
    rows = pd.read_parquet(common/'validation.parquet'); assert len(rows)==512
    indices = [0,1,64,128,192,255,256,257,320,384,448,511]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725'),local_files_only=True)
    queries = []
    for i in indices:
        row=rows.iloc[i]
        extra=row['extra_info']
        query=extra['question'] if 'question' in extra else row['prompt'][0]['content']
        queries.append(str(query))
    assert len(set(queries))==len(queries) and all(q.strip() for q in queries)
    query_sources=[dict(row=i,question_id=observed['question_ids'][i],query=q) for i,q in zip(indices,queries)]
    retriever = namespace['DenseRetriever'](config)
    parameter = next(retriever.encoder.model.parameters())
    assert str(parameter.device).startswith('cuda') and parameter.dtype == torch.float16
    gpu = retriever.index
    children = [gpu.at(i) for i in range(gpu.count())] if hasattr(gpu,'at') else [gpu]
    devices = [int(faiss.downcast_index(c).getDevice()) for c in children]
    assert len(devices)==8 and sorted(devices)==list(range(8))
    documents, scores = [], []
    for query in queries:
        docs, values = retriever.batch_search([query],num=3,return_score=True)
        assert len(docs)==len(values)==1
        documents.extend(docs);scores.extend(values)
    import math
    assert len(documents)==len(queries) and all(len(d)==3 for d in documents)
    assert all(math.isfinite(float(s)) for batch in scores for s in batch)
    probes=[]
    for i,query,docs in zip(indices,queries,documents):
        contents=[d['contents'] for d in docs]; text=passages(contents)
        observation='\n\n<information>'+text.strip()+'</information>\n\n'
        tokens=tokenizer([observation],add_special_tokens=False,return_tensors='pt',padding='longest')['input_ids'][0][:500].tolist()
        probes.append(dict(validation_row=i,query=query,contents=contents,formatted_passages=text,observation_token_ids=tokens))
    assert {k:file_identity(v) for k,v in paths.items()} == before, 'asset mutation during capture'
    assert sha(common/'validation.parquet')==validation_before, 'validation mutation during capture'
    manifest=dict(schema_version=1,topk=3,max_obs_length=500,format='Doc N(Title: title) text\\n',
        observation_truncation='batch_tokenize_then_head',assets=pins,probes=probes,
        runtime=dict(retrieval_method='e5',topk=3,encoder_path=str(Path(config.retrieval_model_path).resolve()),
            encoder_device=str(parameter.device),encoder_dtype=str(parameter.dtype),pooling_method='mean',
            query_max_length=256,query_prefix='query: ',faiss_gpu=True,faiss_gpu_devices=devices,
            faiss_index_class=type(gpu).__name__,faiss_dimension=int(gpu.d),faiss_ntotal=int(gpu.ntotal),
            faiss_clone_use_float16=True,faiss_clone_shard=True),
        provenance=dict(source=SOURCE,archive=provenance,capture_claim=claim,query_sources=query_sources,validation_file_sha256=validation_before,prepared_manifest_sha256=sha(prepared_path),query_batch_size=1,collector_sha256=sha(__file__),run=str(run),
            slurm_job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,leases=leases,
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='ACTUAL_ARCHIVE_GPU_GOLDEN_NOT_TRAINING_OR_LIVE_SERVICE_ACCEPTANCE'))
    with output.open('x') as f:
        json.dump(manifest,f,ensure_ascii=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(path=str(output),sha256=sha(output),probes=len(probes),scope=manifest['provenance']['scope'])),flush=True)

if __name__ == '__main__': main()
