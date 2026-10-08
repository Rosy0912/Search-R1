"""Pinned local assets plus live retriever identity and baseline golden probes."""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def asset_paths(config, source_path):
    paths = dict(corpus=Path(config.corpus_path), index=Path(config.index_path),
                 service_source=Path(source_path), identity_source=Path(__file__))
    model = Path(config.retrieval_model_path).expanduser().resolve(strict=True)
    if not model.is_dir():
        raise ValueError('Retriever encoder must be an offline local model directory')
    for path in sorted(model.rglob('*')):
        if path.is_file() and '.cache' not in path.parts and not path.name.endswith('.lock'):
            paths['encoder/' + str(path.relative_to(model))] = path
    return paths


def loaded_identity(config, retriever, source_path, initial_assets):
    """Read live objects; this is neither a manifest assertion nor a GPU test."""
    parameter = next(retriever.encoder.model.parameters())
    index = retriever.index
    devices = []
    inspection_error = None
    if config.faiss_gpu:
        try:
            import faiss
            children = [index.at(i) for i in range(index.count())] if hasattr(index, 'at') else [index]
            for child in children:
                child = faiss.downcast_index(child)
                devices.append(int(child.getDevice()))
        except (AttributeError, TypeError, RuntimeError) as error:
            inspection_error = str(error)
    runtime = dict(retrieval_method=retriever.encoder.model_name, topk=retriever.topk,
        encoder_path=str(Path(retriever.encoder.model_path).expanduser().resolve()),
        encoder_device=str(parameter.device), encoder_dtype=str(parameter.dtype),
        pooling_method=retriever.encoder.pooling_method,
        query_max_length=retriever.encoder.max_length,
        query_prefix='query: ' if 'e5' in retriever.encoder.model_name.lower() else None,
        faiss_gpu=bool(config.faiss_gpu), faiss_gpu_devices=devices,
        faiss_gpu_inspection_error=inspection_error,
        faiss_index_class=type(index).__name__, faiss_dimension=int(index.d),
        faiss_ntotal=int(index.ntotal),
        faiss_clone_use_float16=bool(getattr(retriever, 'alignment_gpu_clone_fp16', False)),
        faiss_clone_shard=bool(getattr(retriever, 'alignment_gpu_clone_shard', False)))
    return dict(schema_version=1, pid=os.getpid(), runtime=runtime,
                loaded_assets=initial_assets,
                current_assets={name: file_identity(path) for name, path in asset_paths(config, source_path).items()})


def attach_identity_endpoint(app, config, retriever, source_path):
    if not hasattr(retriever, 'alignment_loaded_assets'):
        raise ValueError('Missing asset identities captured before retriever loading')
    @app.get('/identity')
    async def identity_endpoint():
        return loaded_identity(config, retriever, source_path, retriever.alignment_loaded_assets)


def _loopback(url):
    parsed = urlsplit(url)
    try:
        local = ipaddress.ip_address(parsed.hostname or '').is_loopback
    except ValueError:
        local = parsed.hostname == 'localhost'
    if parsed.scheme != 'http' or not local or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Baseline-aligned retrieval requires a plain local loopback HTTP URL')
    if parsed.path != '/retrieve':
        raise ValueError('Expected the local /retrieve endpoint')
    return parsed


def _passages(contents):
    return ''.join(f'Doc {i+1}(Title: {content.split(chr(10))[0]}) '
                   + '\n'.join(content.split('\n')[1:]) + '\n' for i, content in enumerate(contents))


def validate_retriever_identity(config, tokenizer, transport=None):
    """Read-only preflight. Never reserves GPUs, changes leases, or starts services.

    identity_manifest_sha256 must come from the reviewed baseline independently
    of the manifest file being checked. Hash each local asset once per invocation.
    """
    import requests
    transport = transport or requests
    retriever_config = config.retriever
    parsed = _loopback(retriever_config.url)
    path = Path(retriever_config.identity_manifest).expanduser().resolve(strict=True)
    expected_hash = retriever_config.identity_manifest_sha256
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or file_sha256(path) != expected_hash:
        raise ValueError('Baseline retriever manifest SHA256 mismatch')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('assets'), dict):
        raise ValueError('Invalid baseline retriever manifest schema')
    if retriever_config.topk != 3 or manifest.get('topk') != 3:
        raise ValueError('Baseline retriever requires topk=3')
    if manifest.get('max_obs_length') != config.data.max_obs_length:
        raise ValueError('Observation truncation budget differs from the frozen baseline')
    if manifest.get('format') != 'Doc N(Title: title) text\\n' or manifest.get('observation_truncation') != 'batch_tokenize_then_head':
        raise ValueError('Baseline passage formatting/truncation contract differs')
    assets = manifest['assets']
    if not {'corpus', 'index', 'service_source', 'identity_source'} <= assets.keys() or not any(k.startswith('encoder/') for k in assets):
        raise ValueError('Incomplete corpus/index/encoder/server asset pins')
    local = {}
    hashes = {}
    for name, asset in assets.items():
        identity = file_identity(asset['path'])
        source = identity['path']
        if source not in hashes:
            hashes[source] = file_sha256(source)
        if hashes[source] != asset['sha256'] or file_identity(source) != identity:
            raise ValueError('Retriever asset hash mismatch or concurrent modification: ' + name)
        local[name] = identity
    response = transport.get(f'{parsed.scheme}://{parsed.netloc}/identity', timeout=(5, 30))
    response.raise_for_status()
    actual = response.json()
    if actual.get('schema_version') != 1 or actual.get('loaded_assets') != local or actual.get('current_assets') != local:
        raise ValueError('Live retriever assets differ from hashed baseline assets or changed since loading')
    runtime = actual.get('runtime', {})
    expected = manifest.get('runtime', {})
    required = {'retrieval_method', 'encoder_path', 'encoder_device', 'encoder_dtype', 'pooling_method',
                'query_max_length', 'query_prefix', 'faiss_gpu', 'faiss_gpu_devices', 'faiss_index_class',
                'faiss_dimension', 'faiss_ntotal', 'faiss_clone_use_float16', 'faiss_clone_shard', 'topk'}
    if not required <= expected.keys() or any(runtime.get(key) != value for key, value in expected.items()):
        raise ValueError('Actual loaded retriever runtime differs from baseline pins')
    if (expected['retrieval_method'] != 'e5' or expected['query_prefix'] != 'query: '
            or not str(runtime.get('encoder_device', '')).startswith('cuda')
            or runtime.get('faiss_gpu') is not True or not runtime.get('faiss_gpu_devices')
            or runtime.get('faiss_gpu_inspection_error') is not None):
        raise ValueError('Live CUDA encoder/GPU FAISS identity is unverified')
    probes = manifest.get('probes')
    if not isinstance(probes, list) or not probes:
        raise ValueError('Missing baseline golden retrieval probes')
    for probe in probes:
        contents = probe.get('contents')
        if not isinstance(contents, list) or len(contents) != 3 or not all(isinstance(c, str) for c in contents):
            raise ValueError('Golden probe must pin three ordered document contents')
        response = transport.post(retriever_config.url,
            json={'queries': [probe['query']], 'topk': 3, 'return_scores': True}, timeout=(5, 30))
        response.raise_for_status()
        result = response.json().get('result')
        if not isinstance(result, list) or len(result) != 1 or len(result[0]) != 3:
            raise ValueError('Golden probe response shape differs')
        actual_contents = [doc['document']['contents'] for doc in result[0]]
        if actual_contents != contents:
            raise ValueError('Golden retrieval contents/order differs from frozen baseline')
        passage = _passages(actual_contents)
        if passage != probe.get('formatted_passages'):
            raise ValueError('Golden passage formatting differs')
        observation = '\n\n<information>' + passage.strip() + '</information>\n\n'
        ids = tokenizer([observation], add_special_tokens=False, return_tensors='pt', padding='longest')['input_ids'][0]
        if ids[:config.data.max_obs_length].tolist() != probe.get('observation_token_ids'):
            raise ValueError('Golden tokenized/truncated observation differs')
    return dict(manifest_sha256=expected_hash, live_pid=actual.get('pid'), verified_assets=len(local),
                verified_probes=len(probes), gpu_identity_verified=True,
                full_gpu_parity_claimed=False)
