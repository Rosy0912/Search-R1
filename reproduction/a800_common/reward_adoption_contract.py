"""Exact source delta for the reviewed reward schedule; no generic exception."""
import json
from pathlib import Path
from reproduction.common8.verify_data import digest
from search_g1.reward_schedule import SCHEDULE

DOCUMENT='reproduction/a800_common/reward-adoption-source-delta.json'
ORIGIN_RUN='6de29bcb7be741cfa1bfacdf9e34efc9'
ORIGIN_COMMIT='f820ba6cb2db4f5cbe2ffc370cf515c3fe22d43c'
ORIGIN_BASE='/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'
ORIGIN_ARCHIVE='eb54e4ac27035260e22e30788937dcdaa9208acde6b1d0a1b5d3ce413b946a5d'


def verify_delta(previous,current,run,old,source,repo):
    assert run['reward_schedule']==SCHEDULE and run['reward_schedule_boundary']==50
    assert old.get('reward_schedule') is None
    assert old['run_id']==ORIGIN_RUN and old['source_commit']==ORIGIN_COMMIT
    assert str(source)==ORIGIN_BASE
    assert run['training_world_size']==4 and run['precision_profile']=='bf16_policy_fp32_master'
    path=Path(repo)/DOCUMENT
    assert digest(path)==current[DOCUMENT]
    proof=json.loads(path.read_text())
    assert proof['origin_run']==ORIGIN_RUN and proof['origin_commit']==ORIGIN_COMMIT
    assert proof['origin_archive']==ORIGIN_ARCHIVE and proof['schedule']==SCHEDULE
    actual={name:{'old_sha256':previous.get(name),'new_sha256':current.get(name)}
            for name in set(previous)|set(current)
            if name!=DOCUMENT and previous.get(name)!=current.get(name)}
    assert actual==proof['changes'], 'Unreviewed initial0 source delta'
    assert DOCUMENT not in previous
    return True
