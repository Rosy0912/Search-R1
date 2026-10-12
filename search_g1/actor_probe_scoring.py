"""Two isolated binary classifiers read live actor features via a synchronous RPC."""

import hashlib
import json
import copy
from pathlib import Path

from search_g1.actor_probe_contract import validate_version
from search_g1.signals import contexts, validate_probe_reply


class ActorProbeScorer:
    def __init__(self, tokenizer, actor_workers, base, run):
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.actor_workers = actor_workers
        self.base = Path(base)
        self.run = run
        self._version = None
        self._heads = None
        self._manifest_bytes = None
        self.last_actor_binding = None

    def predict(self, items, *, logical_update, expected_version):
        import joblib
        import numpy as np
        import torch
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.utils.model import compute_position_id_with_mask
        live_path = self.base / 'artifacts/live/manifest.json'
        raw = live_path.read_bytes()
        version = validate_version(self.run, json.loads(raw), logical_update)
        if version['version'] != expected_version:
            raise ValueError('Actor readout version differs from the bound reward definition')
        if self._version == expected_version and raw != self._manifest_bytes:
            raise ValueError('Immutable head UUID was rebound to changed metadata')
        for key, filename in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
            if hashlib.sha256(Path(version[key]).read_bytes()).hexdigest() != version['head_sha256'][filename]:
                raise ValueError('Binary head bytes changed under its cached version')
        if self._version != expected_version:
            heads = []
            for key, filename in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
                path = Path(version[key])
                if hashlib.sha256(path.read_bytes()).hexdigest() != version['head_sha256'][filename]:
                    raise ValueError('Binary head bytes changed')
                head = joblib.load(path)
                if list(head['clf'].classes_) != [0, 1] or not np.isfinite(head['auc']):
                    raise ValueError('Invalid calibrated binary head')
                heads.append(head)
            self._heads = heads
            self._version = expected_version
            self._manifest_bytes = raw
        commit, boundary = self._heads
        if (commit['position'] != 'Xa' or commit['label'] != 'flip' or
                boundary['position'] != 'Xp' or boundary['label'] != 'closedbook'):
            raise ValueError('Probe target or context semantics changed')
        texts = []
        for item in items:
            ctx = contexts(item['prompt'], item.get('response', ''), item['model_ranges'])
            texts.append(ctx['answer_state'] if ctx is not None else item['prompt'] + item.get('response', ''))
        texts += [item['prompt'] for item in items]
        if not items:
            return []
        encoded = self.tokenizer(texts, return_tensors='pt', add_special_tokens=False,
                                 padding=True, truncation=True, max_length=4096)
        data = DataProto.from_dict(tensors=dict(input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            probe_row_ids=torch.arange(len(texts), dtype=torch.int64),
            position_ids=compute_position_id_with_mask(encoded['attention_mask'])),
            meta_info=dict(probe_layers=sorted({commit['layer'], boundary['layer']}),
                probe_logical_update=logical_update, probe_head_version=expected_version,
                probe_run_id=self.run['run_id'], probe_execution_id=self.run['execution_id'],
                probe_attempt_id=self.attempt_identity['attempt_id'],
                probe_attempt_index=self.attempt_identity['attempt_index']))
        padded, padding = pad_dataproto_to_divisor(data, self.actor_workers.world_size)
        result = unpad_dataproto(self.actor_workers.compute_probe_features(padded), padding)
        binding = result.meta_info['probe_actor_binding']
        if binding['completed_outer'] != logical_update - 1 or binding['head_version'] != expected_version:
            raise ValueError('Current actor identity differs from this reward batch')
        if not torch.equal(result.batch['probe_row_ids'], torch.arange(len(texts), dtype=torch.int64)):
            raise ValueError('Actor feature RPC changed readout row order')
        if live_path.read_bytes() != raw:
            raise ValueError('Head version changed inside an outer reward batch')
        self.last_actor_binding = dict(binding, row_order_verified=True, feature_rows=len(texts))
        n = len(items)
        features = result.batch
        flip = commit['clf'].predict_proba(features['probe_layer_' + str(commit['layer'])][:n].numpy())[:, 1]
        bnd = boundary['clf'].predict_proba(features['probe_layer_' + str(boundary['layer'])][n:].numpy())[:, 1]
        reply = dict(version=expected_version, results=[dict(id=item['id'], version=expected_version,
            flip=float(flip[i]), boundary=float(bnd[i])) for i, item in enumerate(items)])
        scores, _ = validate_probe_reply(reply, items)
        return scores
