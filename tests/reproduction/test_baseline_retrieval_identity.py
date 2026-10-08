"""CPU file/transport fixtures test fail-closed identity; no GPU parity evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('baseline_alignment_retrieval',
    ROOT / 'search_r1/baselines/baseline_alignment_retrieval.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class Tokenizer:
    def __call__(self, texts, **kwargs):
        return {'input_ids': torch.tensor([[ord(c) for c in texts[0]]])}

class Transport:
    def __init__(self, receipt, contents):
        self.receipt = receipt; self.contents = contents; self.posts = []
    def get(self, url, timeout):
        return NS(raise_for_status=lambda: None, json=lambda: copy.deepcopy(self.receipt))
    def post(self, url, json, timeout):
        self.posts.append(json)
        return NS(raise_for_status=lambda: None, json=lambda: {'result':
            [[{'document': {'contents': text}, 'score': .2} for text in self.contents]]})

class RetrievalIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        assets = {}
        for role in ('corpus', 'index', 'service_source', 'identity_source', 'encoder/config.json'):
            path = self.root / role; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(role)
            assets[role] = dict(path=str(path), sha256=module.file_sha256(path))
        self.runtime = dict(retrieval_method='e5', topk=3, encoder_path=str(self.root / 'encoder'),
            encoder_device='cuda:0', encoder_dtype='torch.float16', pooling_method='mean',
            query_max_length=256, query_prefix='query: ', faiss_gpu=True, faiss_gpu_devices=[0, 1],
            faiss_index_class='IndexShards', faiss_dimension=768, faiss_ntotal=3,
            faiss_clone_use_float16=True, faiss_clone_shard=True)
        self.contents = ['one\nfirst', 'two\nsecond', 'three\nthird']
        formatted = module._passages(self.contents)
        observation = '\n\n<information>' + formatted.strip() + '</information>\n\n'
        self.manifest = dict(schema_version=1, assets=assets, runtime=self.runtime,
            topk=3, max_obs_length=30, format='Doc N(Title: title) text\\n',
            observation_truncation='batch_tokenize_then_head',
            probes=[dict(query='golden', contents=self.contents, formatted_passages=formatted,
                         observation_token_ids=[ord(c) for c in observation][:30])])
        path = self.root / 'manifest.json'
        self.config = NS(retriever=NS(url='http://127.0.0.1:18080/retrieve', topk=3,
            identity_manifest=str(path), identity_manifest_sha256=''), data=NS(max_obs_length=30))
        identities = {name: module.file_identity(asset['path']) for name, asset in assets.items()}
        self.receipt = dict(schema_version=1, pid=42, runtime=copy.deepcopy(self.runtime),
            loaded_assets=identities, current_assets=copy.deepcopy(identities))
        self.pin()

    def pin(self):
        path = Path(self.config.retriever.identity_manifest); path.write_text(json.dumps(self.manifest))
        self.config.retriever.identity_manifest_sha256 = module.file_sha256(path)

    def validate(self):
        self.transport = Transport(self.receipt, self.contents)
        return module.validate_retriever_identity(self.config, Tokenizer(), self.transport)

    def test_pinned_assets_live_identity_exact_order_format_and_truncation(self):
        receipt = self.validate()
        self.assertEqual(receipt['verified_assets'], 5)
        self.assertEqual(receipt['verified_probes'], 1)
        self.assertFalse(receipt['full_gpu_parity_claimed'])
        self.assertEqual(self.transport.posts, [dict(queries=['golden'], topk=3, return_scores=True)])

    def test_manifest_and_asset_tampering_are_rejected_before_network(self):
        Path(self.config.retriever.identity_manifest).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
            self.validate()
        self.pin(); Path(self.manifest['assets']['index']['path']).write_text('changed index')
        with self.assertRaisesRegex(ValueError, 'asset hash mismatch'):
            self.validate()

    def test_server_identity_and_loaded_gpu_configuration_cannot_be_asserted_by_manifest(self):
        for mutate in (lambda r: r['loaded_assets']['index'].update(inode=-1),
                       lambda r: r['current_assets']['corpus'].update(size=0),
                       lambda r: r['runtime'].update(encoder_device='cpu'),
                       lambda r: r['runtime'].update(faiss_gpu=False),
                       lambda r: r['runtime'].update(faiss_gpu_devices=[]),
                       lambda r: r['runtime'].update(faiss_gpu_inspection_error='unsupported')):
            original = copy.deepcopy(self.receipt); mutate(self.receipt)
            with self.assertRaises(ValueError): self.validate()
            self.receipt = original

    def test_probe_order_format_token_or_budget_drift_is_rejected(self):
        self.contents = list(reversed(self.contents))
        with self.assertRaisesRegex(ValueError, 'contents/order'): self.validate()
        self.contents = self.manifest['probes'][0]['contents']
        self.manifest['probes'][0]['formatted_passages'] += 'drift'; self.pin()
        with self.assertRaisesRegex(ValueError, 'formatting'): self.validate()
        self.manifest['probes'][0]['formatted_passages'] = module._passages(self.contents)
        self.manifest['probes'][0]['observation_token_ids'][0] = -1; self.pin()
        with self.assertRaisesRegex(ValueError, 'tokenized'): self.validate()
        self.config.data.max_obs_length += 1
        with self.assertRaisesRegex(ValueError, 'budget'): self.validate()

    def test_remote_urls_and_missing_baseline_probes_are_rejected(self):
        for url in ('http://10.0.0.1/retrieve', 'https://localhost/retrieve', 'http://127.0.0.1/other'):
            self.config.retriever.url = url
            with self.assertRaises(ValueError): self.validate()
        self.config.retriever.url = 'http://127.0.0.1/retrieve'
        self.manifest['probes'] = []; self.pin()
        with self.assertRaisesRegex(ValueError, 'golden'): self.validate()

    def test_identity_reads_loaded_parameters_and_gpu_children_not_command_claims(self):
        config = NS(corpus_path=str(self.root / 'corpus'), index_path=str(self.root / 'index'),
            retrieval_model_path=str(self.root / 'encoder'), retrieval_method='e5',
            retrieval_topk=3, faiss_gpu=True, encoder_device='cuda')
        encoder = NS(model=NS(parameters=lambda: iter([NS(device='cpu', dtype=torch.float32)])),
                     pooling_method='mean', max_length=256, model_name='e5', model_path=str(self.root / 'encoder'))
        index = NS(d=768, ntotal=3, count=lambda: 2,
                   at=lambda i: NS(getDevice=lambda: i))
        retriever = NS(encoder=encoder, index=index, topk=3, alignment_gpu_clone_fp16=True,
                       alignment_gpu_clone_shard=True)
        source = self.root / 'service_source'
        before = {name: module.file_identity(path) for name, path in module.asset_paths(config, source).items()}
        with patch.dict(sys.modules, {'faiss': NS(downcast_index=lambda index: index)}):
            receipt = module.loaded_identity(config, retriever, source, before)
        self.assertEqual(receipt['runtime']['encoder_device'], 'cpu')
        self.assertEqual(receipt['runtime']['encoder_dtype'], 'torch.float32')
        self.assertEqual(receipt['runtime']['faiss_gpu_devices'], [0, 1])
        self.assertTrue(receipt['runtime']['faiss_clone_use_float16'])
        self.assertEqual(receipt['current_assets'], before)
        (self.root / 'index').write_text('modified after index loading')
        with patch.dict(sys.modules, {'faiss': NS(downcast_index=lambda index: index)}):
            receipt = module.loaded_identity(config, retriever, source, before)
        self.assertNotEqual(receipt['loaded_assets'], receipt['current_assets'])

if __name__ == '__main__': unittest.main()
