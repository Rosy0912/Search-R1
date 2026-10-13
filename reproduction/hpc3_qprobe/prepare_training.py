"""Prepare and admit the real eight-GPU allocation, then extra teacher warmup."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();base=Path(a.base).resolve()
    from reproduction.hpc3_qprobe.control import own_base
    own_base(base)
    features=json.loads((base/'preparation/features-ready.json').read_text())
    if features['actual_visible_cuda_count']!=1 or features['completed_training_outers']!=0:
        raise ValueError('Fixed feature preparation is incomplete or mixed with actor training')
    from reproduction.common8.prepare_run import prepare
    model=Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725')
    run=prepare(base,model,base/'preparation/official-model-manifest.json',a.commit,
        rollout_memory_utilization=0.1,
        local_retrieval_assets='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki',
        execution_site='hpc3')
    from reproduction.common8.qprobe_bind import bind
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    run=bind(base,base/'question-features',retriever_version=sha_json(ASSETS),
        exploration_questions=64,seed=20261013,teacher_warmup_cache=base/'teacher-warmup-cache')
    from reproduction.common8.configuration import resolved
    resolved(base)
    # The existing pipeline owns the one durable resource admission. Preparing
    # a second admission here would collide with its immutable receipt.
    print(json.dumps(dict(stage='hpc3_question_probe_configuration_prepared',run_id=run['run_id'],
        extra_frozen_teacher_warmup_outers=1,target_successful_training_outers=200,
        actual_training_outers_completed=0,actual_allocation_admission_pending=True)),flush=True)


if __name__=='__main__':main()
