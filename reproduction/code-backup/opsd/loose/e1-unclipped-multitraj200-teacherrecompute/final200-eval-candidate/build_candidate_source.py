"""Create reviewable candidate copies only; never modify frozen source."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / 'opsd-unclipped-multitraj200-teacherrecompute'
PINS = {
    'search_r1/llm_agent/generation.py': '91e281dc853c077dfcda3f05b64a714c009c8993fe05b2069fcc2def8f114470',
    'search_r1/baselines/main.py': '2d4b8eb0fcc307c0ea74713505d53c673d55ac8b7e72a74e03ee21e6abd802c1',
    'verl/trainer/ppo/ray_trainer.py': 'f78dc2bebdf6b42b94d6845cd77cd1d723c2b6e33446f25295baf01f7e8d159d',
}


def one(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Exact source hunk absent or ambiguous: ' + old[:90])
    return text.replace(old, new, 1)


def normal_one(text, old, new):
    boundary = '        # final LLM rollout\n'
    if text.count(boundary) != 1:
        raise ValueError('Normal/final source boundary differs')
    normal, final = text.split(boundary)
    return one(normal, old, new) + boundary + final


def generate(source):
    text = one(source, 'import requests\n', 'import requests\nimport time\n')
    text = one(text, '    def _batch_tokenize(self,',
        "    def _observe(self, name, **values):\n"
        "        observer = getattr(self, 'final200_observer', None)\n"
        "        if observer is not None:\n"
        "            observer.event(name, **values)\n\n"
        '    def _batch_tokenize(self,')
    text = one(text, "        if next_obs_ids.shape[1] > self.config.max_obs_length:\n",
        "        untruncated_next_obs_ids = next_obs_ids\n"
        "        if next_obs_ids.shape[1] > self.config.max_obs_length:\n")
    text = one(text, '        return next_obs_ids\n',
        "        self._observe('observations', untruncated=untruncated_next_obs_ids, retained=next_obs_ids)\n"
        '        return next_obs_ids\n')
    text = one(text, '        original_left_side = {',
        "        self._observe('begin', gen_batch=gen_batch, initial_input_ids=initial_input_ids)\n"
        '        original_left_side = {')
    # Only normal tool rounds are instrumented. The final200 contract forbids a fifth final-answer round.
    text = normal_one(text, '            turns_stats[active_mask] += 1\n            gen_output = self._generate_with_gpu_padding(rollings_active)\n\n            meta_info = gen_output.meta_info            \n',
        "            turns_stats[active_mask] += 1\n"
        "            self._observe('before_generation', step=step, active_mask=active_mask, active_batch=rollings_active)\n"
        "            gen_output = self._generate_with_gpu_padding(rollings_active)\n"
        "            self._observe('generated', output=gen_output)\n\n"
        '            meta_info = gen_output.meta_info            \n')
    text = one(text, '            # Execute in environment and process observations\n',
        "            self._observe('postprocessed', responses=responses_ids, texts=responses_str)\n\n"
        '            # Execute in environment and process observations\n')
    text = one(text, "            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)\n            active_mask = active_mask * curr_active_mask\n            active_num_list.append(active_mask.sum().item())\n            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)\n            valid_search_stats += torch.tensor(is_search, dtype=torch.int)\n\n            next_obs_ids = self._process_next_obs(next_obs)\n",
        "            self._observe('environment', observations=next_obs, dones=dones, valid_actions=valid_action, searches=is_search)\n"
        "            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)\n"
        "            active_mask = active_mask * curr_active_mask\n"
        "            active_num_list.append(active_mask.sum().item())\n"
        "            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)\n"
        "            valid_search_stats += torch.tensor(is_search, dtype=torch.int)\n\n"
        "            next_obs_ids = self._process_next_obs(next_obs)\n")
    text = one(text, '        # final LLM rollout\n',
        "            self._observe('updated', rolling=rollings, right=original_right_side)\n\n"
        '        # final LLM rollout\n')
    text = one(text, '        return self._compose_final_output(original_left_side, original_right_side, meta_info)\n',
        "        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)\n"
        "        self._observe('end', output=final_output)\n"
        "        return final_output\n")
    text = one(text, '        return requests.post(self.config.search_url, json=payload).json()\n',
        "        started_ns = time.monotonic_ns()\n"
        "        result = requests.post(self.config.search_url, json=payload).json()\n"
        "        finished_ns = time.monotonic_ns()\n"
        "        self._observe('retrieval', request=payload, response=result, url=self.config.search_url,\n"
        "                      started_monotonic_ns=started_ns, finished_monotonic_ns=finished_ns)\n"
        "        return result\n")
    return text


def main_source(source):
    text = one(source, '    configure(config, split_manifest)\n',
        "    from .final200_evaluation import enabled as final200_enabled, configure as configure_final200\n"
        "    if final200_enabled(config):\n"
        "        configure_final200(config, split_manifest)\n"
        "    else:\n"
        "        configure(config, split_manifest)\n")
    text = one(text, '    task = main_task.options(runtime_env={\'env_vars\': audit_env}) if audit_env else main_task\n',
        "    if final200_enabled(config):\n"
        "        import os\n"
        "        audit_env['OPSD_FINAL200_EVAL_STAGE_SHA256'] = os.environ['OPSD_FINAL200_EVAL_STAGE_SHA256']\n"
        "        audit_env['OPSD_CORRECTED_EVAL_STAGE_SHA256'] = os.environ['OPSD_FINAL200_EVAL_STAGE_SHA256']\n"
        "        audit_env['SLURM_JOB_ID'] = os.environ['SLURM_JOB_ID']\n"
        "    task = main_task.options(runtime_env={'env_vars': audit_env}) if audit_env else main_task\n")
    return text


def trainer_source(source):
    text = one(source, "            if enabled(self.config):\n                corrected_writer, corrected_guard = start_capture(self)\n",
        "            from search_r1.baselines.final200_evaluation import enabled as final200_enabled, start_capture as final200_capture\n"
        "            if final200_enabled(self.config):\n"
        "                corrected_writer, corrected_guard = final200_capture(self)\n"
        "            elif enabled(self.config):\n"
        "                corrected_writer, corrected_guard = start_capture(self)\n")
    text = one(text, "        controller = getattr(self, 'baseline_controller', None)\n        if controller is not None and controller.ranker is not None:\n",
        "        if corrected_writer is not None and hasattr(corrected_writer, 'generation_observer'):\n"
        "            generation_manager.final200_observer = corrected_writer.generation_observer\n"
        "        controller = getattr(self, 'baseline_controller', None)\n"
        "        if controller is not None and controller.ranker is not None:\n")
    text = one(text, "            print('CORRECTED_SCORING_VIEW_MANIFEST_SHA256=' + transcript_sha, flush=True)\n",
        "            marker = ('FINAL200_FULL_TRAJECTORY_MANIFEST_SHA256=' if hasattr(corrected_writer, 'generation_observer')\n"
        "                      else 'CORRECTED_SCORING_VIEW_MANIFEST_SHA256=')\n"
        "            print(marker + transcript_sha, flush=True)\n")
    text = one(text, '        # we start from step 1\n',
        "        if self.config.get('baselines'):\n"
        "            from search_r1.baselines.final200_evaluation import reject_training\n"
        "            reject_training(self.config)\n"
        '        # we start from step 1\n')
    return text


def main():
    output = HERE / 'candidate-source'
    output.mkdir()
    records, patches = {}, []
    for relative, transform in (
        ('search_r1/llm_agent/generation.py', generate),
        ('search_r1/baselines/main.py', main_source),
        ('verl/trainer/ppo/ray_trainer.py', trainer_source)):
        path = REPO / relative
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if PINS[relative] and sha != PINS[relative]:
            raise ValueError('Frozen original differs')
        source = raw.decode()
        candidate = transform(source)
        compile(candidate, str(output / relative), 'exec')
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(candidate.encode())
        if path.read_bytes() != raw:
            raise ValueError('Frozen source changed')
        records[relative] = dict(original_sha256=sha, candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest())
        patches.extend(difflib.unified_diff(source.splitlines(True), candidate.splitlines(True),
                       fromfile='a/' + relative, tofile='b/' + relative))
    relative = 'search_r1/baselines/final200_evaluation.py'
    target = output / relative
    raw = (HERE / 'final200_evaluation.py').read_bytes()
    with target.open('xb') as stream:
        stream.write(raw)
    records[relative] = dict(original_sha256=None, candidate_sha256=hashlib.sha256(raw).hexdigest())
    patches.extend(difflib.unified_diff([], raw.decode().splitlines(True), fromfile='/dev/null', tofile='b/' + relative))
    relative = 'verl/trainer/config/ppo_opsd_final200_eval.yaml'
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (HERE / 'ppo_opsd_final200_eval.yaml').read_bytes()
    with target.open('xb') as stream:
        stream.write(raw)
    records[relative] = dict(original_sha256=None, candidate_sha256=hashlib.sha256(raw).hexdigest())
    patches.extend(difflib.unified_diff([], raw.decode().splitlines(True), fromfile='/dev/null', tofile='b/' + relative))
    with (HERE / 'source-adaptation.patch').open('x') as stream:
        stream.writelines(patches)
    with (HERE / 'source-adaptation-inventory.json').open('x') as stream:
        json.dump(dict(scope='LOCAL_SOURCE_CANDIDATE_NOT_DEPLOYED', training_source_commit='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6',
                       eval_source_commit=None, files=records), stream, indent=2)


if __name__ == '__main__':
    main()
