"""Explicit scheduler retirement evidence for human-requested renamed allocations."""
from .contracts import require


def validate_retirement(evidence, job):
    require(evidence.get('job')==job and evidence.get('authority')=='human_rename_and_resume_20261017',
            'Retirement job or authority differs')
    snapshots=evidence.get('snapshots',[])
    require(len(snapshots)>=2 and snapshots[-1]['at']>snapshots[0]['at'],
            'Need two independent scheduler observations')
    for s in snapshots:
        require(s.get('state') in ('NOT_FOUND_404','TERMINATED') and s.get('active_pods')==0,
                'Previous allocation still active or unverified')
    return True
