"""Scheduler-specific identity without representing a PT job as Slurm."""
import os


def job_id(environ=None, required=True):
    env = os.environ if environ is None else environ
    if env.get('SEARCH_R1_EXECUTION_BACKEND') == 'pj_pt':
        if env.get('SLURM_JOB_ID'):
            raise ValueError('PJ PT must not carry a fabricated Slurm job')
        value = env.get('SEARCH_R1_JOB_ID')
    else:
        value = env.get('SLURM_JOB_ID')
    if required and not value:
        raise ValueError('Actual scheduler job identity is missing')
    return value


def provenance_keys(environ, legacy):
    if environ.get('SEARCH_R1_EXECUTION_BACKEND') == 'pj_pt':
        job_id(environ)
        return tuple(k for k in legacy if k != 'SLURM_JOB_ID') + ('SEARCH_R1_EXECUTION_BACKEND','SEARCH_R1_JOB_ID')
    return legacy
