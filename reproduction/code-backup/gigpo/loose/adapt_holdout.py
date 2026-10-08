from pathlib import Path
r=Path('work/common-setup')
p=r/'search_r1/baselines/holdout.py';s=p.read_text();at='def prepare_common_holdout(config):'
s=s.replace(at,'''def frame_identity(frame, prompt_key='prompt'):
    questions = sorted(set(frame[prompt_key].map(question_key)))
    rows = sorted(json.dumps(plain(row), sort_keys=True, ensure_ascii=False)
                  for row in frame.to_dict('records'))
    return dict(questions_sha256=hashlib.sha256('\\n'.join(questions).encode()).hexdigest(),
                rows_sha256=hashlib.sha256('\\n'.join(rows).encode()).hexdigest())


def content_identity(specification, train, holdout):
    content = dict(version=1, split_version=specification['version'],
                   train_sources=sorted(item['sha256'] for item in specification['train']),
                   excluded_sources=sorted(item['sha256'] for item in specification['final']),
                   fraction=specification['fraction'], seed=specification['seed'],
                   prompt_key=specification['prompt_key'],
                   train=frame_identity(train, specification['prompt_key']),
                   holdout=frame_identity(holdout, specification['prompt_key']))
    content['sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content


'''+at)
s=s.replace('manifest = dict(specification=specification, train_rows=', 'manifest = dict(content_identity=content_identity(specification, retained, holdout),\n                    specification=specification, train_rows=')
s=s.replace("    config.data.train_files = str(output / 'train.parquet')", "    if 'content_identity' not in manifest:\n        raise ValueError('Legacy split missing content_identity; use a new common-data directory')\n    print('BASELINE_SPLIT_MANIFEST_JSON=' + json.dumps(manifest, sort_keys=True), flush=True)\n    config.data.train_files = str(output / 'train.parquet')")
p.write_text(s)
p=r/'verl/trainer/config/ppo_baselines.yaml';s=p.read_text().replace('  acceptance_step: 5','  acceptance_step: 5\n  smoke_only: false').replace('  context_alignment: report','  context_alignment: report\n  holdout_eval_only: false');p.write_text(s)
p=r/'search_r1/baselines/main.py';s=p.read_text().replace('    validate_config(config)', '''    if config.trainer.smoke_only and config.trainer.total_training_steps > 6:
        raise ValueError('Smoke-only runs allow at most five outer updates')
    if config.baselines.holdout_eval_only:
        if not config.trainer.get('val_only', False) or not config.trainer.val_before_train:
            raise ValueError('Holdout evaluation requires val_only and val_before_train')
        config.data.val_files = config.baselines.bapo.stage_validation_files
        config.data.final_val_files = None
    validate_config(config)''');p.write_text(s)
p=r/'verl/trainer/ppo/ray_trainer.py';s=p.read_text().replace('        data_source_lst = []','''        data_source_lst = []
        qa_rows = []

        def collect_qa(batch):
            if not self.config.get('baselines'):
                return
            from search_r1.baselines.evaluation import batch_answer_metrics
            qa_rows.extend(batch_answer_metrics(batch, self.tokenizer))''',1)
s=s.replace('                    reward_tensor_lst.append(reward_tensor)', '                    collect_qa(test_batch)\n                    reward_tensor_lst.append(reward_tensor)',1)
s=s.replace('                reward_tensor_lst.append(reward_tensor)', '                collect_qa(test_batch)\n                reward_tensor_lst.append(reward_tensor)',1)
s=s.replace('        return metric_dict\n', '''        if qa_rows:
            from search_r1.baselines.evaluation import aggregate_answer_metrics
            metric_dict.update(aggregate_answer_metrics(qa_rows))
        if self.config.get('baselines'):
            print('BASELINE_EVALUATION_METRICS_JSON=' + json.dumps(metric_dict, sort_keys=True), flush=True)
        return metric_dict
''',1)
s=s.replace('                    if self.val_reward_fn is not None:\n                        self._prepare_final_validation()', "                    if self.val_reward_fn is not None and not getattr(self.config.trainer, 'smoke_only', False):\n                        self._prepare_final_validation()")
p.write_text(s)
