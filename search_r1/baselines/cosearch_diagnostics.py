"""Optional evidence for the existing ranker update, without changing selection."""
from collections import Counter
import json
import hashlib
import os
from pathlib import Path
import time


def plain(value):
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save_update_evidence(root, all_events, outcomes, trajectories, retained, keys, chosen, metrics, provenance=None):
    """Write original actions and actual grouping; neither gold nor diagnostics enters inference."""
    rows = []
    positions = {id(e): i for i, e in enumerate(retained)}
    chosen = set(chosen)
    for e in all_events:
        main, targets, legal = outcomes[e['trajectory_uid']]
        i = positions.get(id(e))
        rows.append(dict(question_uid=e['question_uid'], trajectory_uid=e['trajectory_uid'],
                         query=e['query'], ranking=e['ranking'], response_length=e['response_length'],
                         main_reward=main, targets=targets, main_eligible=legal,
                         retained=i is not None, group_key=None if i is None else keys[i],
                         ranker_reward=None if i is None else e['reward'], chosen=i in chosen))
        rows[-1]['candidates'] = e['candidates']
        length = e['response_length']
        rows[-1]['ranking_response_ids'] = e['input_ids'][-length:].tolist() if length else []
    groups = Counter(map(str, keys))
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('cosearch.py', 'controller.py', 'cosearch_diagnostics.py', 'bapo.py')}
    decoder = Path(__file__).resolve().parents[2] / 'verl/trainer/main_ppo.py'
    source_hashes['verl/trainer/main_ppo.py'] = hashlib.sha256(decoder.read_bytes()).hexdigest()
    record = plain(dict(probe_only=False, selection_unchanged=True, events=rows,
                        provenance=dict(job=os.environ.get('SLURM_JOB_ID'), pid=os.getpid(),
                                        source_sha256=source_hashes, **(provenance or {})),
                        trajectories=trajectories, group_size_histogram=Counter(groups.values()),
                        selection_metrics_before_gradients=metrics))
    destination = Path(root) / 'ranker-update-evidence'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f'{time.time_ns()}-{os.getpid()}.json'
    with path.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False)
    return path
