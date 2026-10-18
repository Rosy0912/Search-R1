"""Scientific recipe and common-data schema; no training dependencies."""
import hashlib
import json
from pathlib import Path
import re
import shlex

UPSTREAM = '7863ebcf7837e8b61722d6db5e5e697fb3f863a1'
DATA_SHA = '7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
REVISIONS = {'3b': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
             '7b': 'a09a35458c702b33eeacc393d103063234e8bc28'}
INSTRUCTION = ('Answer the given question. You must conduct reasoning inside <think> and </think> '
               'first every time you get new information. After reasoning, if you lack knowledge, '
               'call a search engine by <search> query </search>; results appear between '
               '<information> and </information>. You may search at most four times. '
               'Provide the final answer inside <answer> and </answer>, without detailed illustrations. '
               'For example, <answer> Beijing </answer>.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def job_name(size):
    if size not in REVISIONS:
        raise ValueError('Only the requested 3B and 7B comparison sizes are supported')
    return 'zhangjunming-branpo-' + size + ('-v6' if size=='3b' else '-v5')


def task_from_common(row):
    prompt = row['prompt']
    prefix = INSTRUCTION + '\nQuestion: '
    if (len(prompt) != 1 or prompt[0]['role'] != 'user'
            or not prompt[0]['content'].startswith(prefix)):
        raise ValueError('Common prompt identity changed')
    question = prompt[0]['content'][len(prefix):]
    answers = row['reward_model']['ground_truth']['target']
    identifier = row['extra_info']['index']
    if (not question or not isinstance(identifier, str) or not identifier
            or not isinstance(answers, list) or not answers
            or any(not isinstance(a, str) for a in answers)):
        raise ValueError('Incomplete exact question/gold/ID')
    # Never add a question mark, strip question text, normalize gold, or drop an ID.
    return {'question': question, 'ground_truth': list(answers),
            'data_source': row['data_source'], 'index': identifier}


def training_overrides(script, size, model, output):
    """Keep every available upstream launcher override except input/output names."""
    values = dict(MODEL_NAME=str(model), TRAIN_DATA='common-train',
                  EX_NAME=job_name(size), NUM_GPU='8', MAX_RES_LEN='3072',
                  MAX_TURNS='4', START_STEP='100', TOTAL_STEPS='300', LR='1e-6')
    result = {}
    for raw in script.splitlines():
        line = raw.strip().removesuffix('\\').strip()
        if not re.match(r'[a-z][a-z_]*(?:\.[a-z_]+)+=', line):
            continue
        key, value = line.split('=', 1)
        if key in result:
            raise ValueError('Duplicate upstream CLI override: ' + key)
        if key == 'trainer.logger':
            value = "['console']"  # execution logging only; no external service
        else:
            parts = shlex.split(value)
            if len(parts) != 1:
                raise ValueError('Unparsed upstream value: ' + line)
            value = parts[0]
            value = re.sub(r'\$\{(\w+)\}|\$(\w+)',
                           lambda m: values[m[1] or m[2]], value)
            if '$' in value:
                raise ValueError('Unresolved official variable: ' + line)
        result[key] = value
    required = {'algorithm.adv_estimator': 'grpo',
                'algorithm.rollout_correction.rollout_is': 'token',
                'algorithm.rollout_correction.rollout_is_threshold': '2.0',
                'algorithm.rollout_correction.bypass_mode': 'True',
                'data.train_batch_size': '256',
                'actor_rollout_ref.actor.ppo_mini_batch_size': '512',
                'actor_rollout_ref.rollout.n': '4',
                'actor_rollout_ref.rollout.temperature': '0.95',
                'actor_rollout_ref.actor.optim.lr': '1e-6',
                'rllm.agent.max_steps': '4',
                'trainer.start_step': '100', 'trainer.total_steps': '300',
                'trainer.total_epochs': '1'}
    for key, value in required.items():
        if result.get(key) != value:
            raise ValueError('Official scientific recipe changed: ' + key)
    result.update({'data.test_data': 'common-validation',
                   'trainer.project_name': 'zhangjunming-branpo-fair',
                   'trainer.default_local_dir': str(Path(output) / 'checkpoints'),
                   'trainer.resume_mode': 'disable',
                   'rllm.env.env_args.max_steps': '4'})
    # env_args is an empty dict upstream; Hydra requires an explicit new field.
    overrides = [('+' if k == 'rllm.env.env_args.max_steps' else '') + k + '=' + v
                 for k, v in result.items()]
    return overrides
