from pathlib import Path
r=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/common-setup')
p=r/'search_r1/baselines/main.py';s=p.read_text().replace('    prepare_common_holdout(config)','    split_manifest = prepare_common_holdout(config)')
s=s.replace("        config.data.val_files = config.baselines.bapo.stage_validation_files", "        if config.baselines.method != 'grpo' or config.baselines.seed != 42:\n            raise ValueError('Teacher selection requires fresh GRPO and seed42')\n        if config.data.get('val_data_num') is not None:\n            raise ValueError('Holdout-only evaluation cannot subsample questions')\n        from .select_teacher import model_manifest\n        checkpoint_hash = model_manifest(config.actor_rollout_ref.model.path)\n        config.data.val_files = config.baselines.bapo.stage_validation_files")
s=s.replace('    ray.get(main_task.remote(config))','''    result = ray.get(main_task.remote(config))
    if config.baselines.holdout_eval_only:
        import json
        import re
        from pathlib import Path
        from .select_teacher import model_manifest, CANDIDATES
        model = Path(config.actor_rollout_ref.model.path).resolve()
        match = re.search(r'/global_step_(\\d+)/actor$', str(model))
        if not match or int(match.group(1)) not in CANDIDATES:
            raise ValueError('Holdout candidate must be a specified saved actor update')
        if model_manifest(model) != checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation')
        count = int(result['val/qa_count/all'])
        if count != split_manifest['holdout_rows']:
            raise ValueError('Evaluation dropped holdout questions')
        record = dict(outer_update=int(match.group(1)), evaluation_split='common-holdout', seed=42,
            evaluated_count=count, holdout_rows=split_manifest['holdout_rows'],
            mean_answer_f1=float(result['val/f1/all']),
            split_content_identity=split_manifest['content_identity']['sha256'],
            checkpoint_sha256=checkpoint_hash['sha256'], hf_directory=str(model), metrics=result)
        target = Path(config.baselines.holdout_record_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            json.dump(record, handle, indent=2)
        print('BASELINE_HOLDOUT_RECORD_JSON=' + json.dumps(record, sort_keys=True), flush=True)''')
p.write_text(s)
p=r/'verl/trainer/main_ppo.py';s=p.read_text().replace('    trainer.fit()','    return trainer.fit()');p.write_text(s)
p=r/'verl/trainer/ppo/ray_trainer.py';s=p.read_text().replace("            if self.config.trainer.get('val_only', False):\n                return", "            if self.config.trainer.get('val_only', False):\n                return val_metrics");p.write_text(s)
p=r/'verl/trainer/config/ppo_baselines.yaml';s=p.read_text().replace('  holdout_eval_only: false','  holdout_eval_only: false\n  holdout_record_path: null');p.write_text(s)
p=r/'search_r1/baselines/select_teacher.py';s=p.read_text().replace("    if not any(p.suffix == '.safetensors' or p.name.startswith('pytorch_model') for p in files):", "    if not any(p.suffix == '.safetensors' or (p.name.startswith('pytorch_model') and p.suffix == '.bin') for p in files):")
s=s.replace("    hashes = {str(p.relative_to(path)): file_hash(p) for p in files}", "    for index in path.glob('*.index.json'):\n        weight_map = json.loads(index.read_text()).get('weight_map')\n        if not weight_map:\n            raise ValueError('Invalid HF shard index')\n        for name in set(weight_map.values()):\n            shard = (path/name).resolve()\n            if not shard.is_relative_to(path.resolve()) or not shard.is_file():\n                raise ValueError('Missing/outside HF weight shard: ' + name)\n    hashes = {str(p.relative_to(path)): file_hash(p) for p in files}")
s=s.replace('def select(records, content_id):','def select(records, split_manifest):\n    content_id = split_manifest[\'content_identity\'][\'sha256\']\n    holdout_rows = split_manifest[\'holdout_rows\']')
s=s.replace("r['evaluated_count'] != r['holdout_rows'] or r['holdout_rows'] <= 0", "r['evaluated_count'] != holdout_rows or r['holdout_rows'] != holdout_rows or holdout_rows <= 0")
s=s.replace('def export(records, content_id, student, destination):','def export(records, split_manifest, student, destination):')
s=s.replace('winner = select(records, content_id)','winner = select(records, split_manifest)\n    for record in records:\n        if model_manifest(record[\'hf_directory\'])[\'sha256\'] != record[\'checkpoint_sha256\']:\n            raise ValueError(\'Candidate changed after evaluation\')')
s=s.replace('split_content_identity=content_id,',"split_content_identity=split_manifest['content_identity']['sha256'],")
s=s.replace("parser.add_argument('--split-content-identity', required=True)","parser.add_argument('--split-manifest', required=True)")
s=s.replace("report = export(json.loads(Path(args.records).read_text()), args.split_content_identity, args.student, args.destination)", "manifest_path = Path(args.split_manifest)\n    split_manifest = json.loads(manifest_path.read_text())\n    for name, expected in split_manifest['sha256'].items():\n        if file_hash(manifest_path.parent/name) != expected:\n            raise ValueError('Split data hash mismatch')\n    report = export(json.loads(Path(args.records).read_text()), split_manifest, args.student, args.destination)")
p.write_text(s)
p=r/'tests/baselines/test_common_setup.py';s=p.read_text().replace("select(rs,'abc')", "select(rs,dict(content_identity=dict(sha256='abc'),holdout_rows=100))").replace("select(bad,'abc')","select(bad,dict(content_identity=dict(sha256='abc'),holdout_rows=100))").replace("select(rs[:-1],'abc')","select(rs[:-1],dict(content_identity=dict(sha256='abc'),holdout_rows=100))")
s+='''

def test_hf_index_without_weights_refused(tmp_path):
    import json
    from search_r1.baselines.select_teacher import model_manifest
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'tokenizer_config.json').write_text('{}')
    (tmp_path/'pytorch_model.bin.index.json').write_text(json.dumps(dict(weight_map={'x':'missing.bin'})))
    with pytest.raises(ValueError):model_manifest(tmp_path)
    (tmp_path/'pytorch_model-00001.bin').write_bytes(b'weight')
    with pytest.raises(ValueError,match='shard'):model_manifest(tmp_path)
'''
p.write_text(s)
