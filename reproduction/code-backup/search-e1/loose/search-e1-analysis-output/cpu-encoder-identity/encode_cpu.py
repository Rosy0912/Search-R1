"""Bounded, CPU-only comparison of the original live E5 encoder on two hosts."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['E1_ENCODER_DEVICE'] = 'cpu'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', required=True)
parser.add_argument('--encoder', required=True)
parser.add_argument('--manifest', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
os.nice(10)
sys.path.insert(0, args.upstream)
import numpy as np
import torch
import faiss
import transformers
torch.set_num_threads(16)
faiss.omp_set_num_threads(32)
from search_r1.search import retrieval_server as original

output = Path(args.output)
output.mkdir(exist_ok=False)
query_path = Path(args.queries)
fixture = json.loads(query_path.read_text())
manifest = json.loads(Path(args.manifest).read_text())
encoder_path = Path(args.encoder)
encoder_checks = {}
for item in manifest['encoder']['files']:
    p = encoder_path / item['path']
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual == item['sha256'], str(p)
    encoder_checks[item['path']] = actual

config = original.Config(retrieval_method='e5', retrieval_model_path=str(encoder_path),
                         retrieval_use_fp16=False, retrieval_batch_size=128)
assert config.retrieval_query_max_length == 256
assert config.retrieval_pooling_method == 'mean'
encoder = original.Encoder(model_name=config.retrieval_method,
                           model_path=config.retrieval_model_path,
                           pooling_method=config.retrieval_pooling_method,
                           max_length=config.retrieval_query_max_length,
                           use_fp16=config.retrieval_use_fp16)
assert next(encoder.model.parameters()).device.type == 'cpu'
assert next(encoder.model.parameters()).dtype == torch.float32
assert not encoder.model.training
assert not torch.cuda.is_initialized()
proof = {'time': time.time(), 'pid': os.getpid(), 'host': os.uname().nodename,
         'queries_path': str(query_path),
         'queries_sha256': hashlib.sha256(query_path.read_bytes()).hexdigest(),
         'original_module_path': original.__file__,
         'original_module_sha256': hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
         'encoder_path': str(encoder_path.resolve()), 'encoder_sha256': encoder_checks,
         'runtime': {'torch': torch.__version__, 'faiss': faiss.__version__,
                     'transformers': transformers.__version__,
                     'torch_num_threads': torch.get_num_threads(),
                     'faiss_num_threads': faiss.omp_get_max_threads()},
         'config': {'max_length': config.retrieval_query_max_length,
                    'pooling': config.retrieval_pooling_method,
                    'use_fp16': config.retrieval_use_fp16,
                    'encoder_batch_size': config.retrieval_batch_size,
                    'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']},
         'cases': []}
for case in fixture['cases']:
    queries = case['queries']
    t = time.monotonic()
    pieces = [encoder.encode(queries[start:start + config.retrieval_batch_size])
              for start in range(0, len(queries), config.retrieval_batch_size)]
    array = np.concatenate(pieces, axis=0)
    assert array.dtype == np.float32 and array.shape == (len(queries), 768)
    assert np.isfinite(array).all() and array.flags.c_contiguous
    assert not torch.cuda.is_initialized()
    np.save(output / (case['id'] + '.npy'), array, allow_pickle=False)
    record = {'id': case['id'], 'n': len(queries), 'shape': list(array.shape),
              'encoder_boundaries': [len(piece) for piece in pieces],
              'embedding_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
              'npy_sha256': hashlib.sha256((output / (case['id'] + '.npy')).read_bytes()).hexdigest(),
              'seconds': time.monotonic() - t}
    proof['cases'].append(record)
    print(json.dumps(record), flush=True)
proof.update(cuda_initialized=torch.cuda.is_initialized(),
             max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             complete_time=time.time())
(output / 'identity.json').write_text(json.dumps(proof, indent=2))
print('CPU_ENCODER_IDENTITY_COMPLETE', str(output / 'identity.json'), flush=True)
