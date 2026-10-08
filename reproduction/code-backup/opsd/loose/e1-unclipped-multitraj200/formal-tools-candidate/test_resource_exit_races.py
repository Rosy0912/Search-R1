"""CPU event fixtures preserve absent/foreign/unknown distinctions."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from test_formal_tools import f, resource_row

SPEC = importlib.util.spec_from_file_location('new_resource_sampler',Path(__file__).parent/'resource_sample.py')
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)


def test_prior_readable_owned_then_confirmed_exit_is_not_foreign(tmp_path):
    a = resource_row('2026-10-06T00:00:00+00:00')
    b = resource_row('2026-10-06T00:00:02+00:00')
    c = resource_row('2026-10-06T00:00:04+00:00')
    live = c['gpu_processes'][-1]
    key = (live['uuid'],live['pid'])
    sampler.prior_owned.clear()
    sampler.resolve_exit_race(copy.deepcopy(live),b['time'],set(),True)
    exited = {'uuid':live['uuid'],'pid':live['pid'],'unavailable':True,
              'read_error':{'type':'FileNotFoundError','errno':2,'message':'fixture ENOENT'}}
    result = sampler.resolve_exit_race(exited,c['time'],set(),False)
    assert result['unavailable'] is True  # Never relabel it as currently owned.
    c['gpu_processes'][-1] = result
    c['nvml_apps_recheck_original'] = ''
    p = tmp_path/'raw.ndjson'
    p.write_text('\n'.join(json.dumps(v) for v in [a,b,c]))
    summary = f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100)
    assert summary['resolved_exit_races'] == 1
    assert summary['eight_worker_plus_service_samples'] == 2
    for mutated in [
        dict(result,exit_race_resolution=dict(result['exit_race_resolution'],proc_exists_after=True)),
        dict(result,exit_race_resolution=dict(result['exit_race_resolution'],prior={'time':'fake','record':live})),
        dict(result,exit_race_resolution={}),
    ]:
        bad = copy.deepcopy(c);bad['gpu_processes'][-1] = mutated
        p.write_text('\n'.join(json.dumps(v) for v in [a,b,bad]))
        with pytest.raises(ValueError):
            f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100)
    bad = copy.deepcopy(c);bad['nvml_apps_recheck_original'] = f'{key[0]}, {key[1]}, 10\n'
    p.write_text('\n'.join(json.dumps(v) for v in [a,b,bad]))
    with pytest.raises(ValueError):f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100)


@pytest.mark.parametrize('prior,still_present,proc_exists',[(False,False,False),(True,True,False),(True,False,True)])
def test_missing_prior_or_live_reused_pid_remains_unresolved(prior,still_present,proc_exists):
    sampler.prior_owned.clear()
    key = ('GPU-0',12)
    live = dict(uuid=key[0],pid=key[1],uid=str(os.getuid()),in_slurm_job=True)
    if prior:sampler.resolve_exit_race(live,'original_time',set(),True)
    app = dict(uuid=key[0],pid=key[1],unavailable=True,read_error={'errno':2})
    result = sampler.resolve_exit_race(app,'later_time',{key} if still_present else set(),proc_exists)
    assert 'exit_race_resolution' not in result


def test_foreign_gpu_process_never_becomes_owned(tmp_path):
    a = resource_row('2026-10-06T00:00:00+00:00')
    b = resource_row('2026-10-06T00:00:02+00:00')
    b['gpu_processes'][0]['in_slurm_job'] = False
    p = tmp_path/'raw.ndjson';p.write_text('\n'.join(json.dumps(v) for v in [a,b]))
    with pytest.raises(ValueError):f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100)


@pytest.mark.parametrize('read_error',[
    {'type':'PermissionError','errno':13},
    {'type':'KeyError','errno':None},
    {},
])
def test_only_explicit_enoent_read_can_resolve_exit(read_error):
    sampler.prior_owned.clear()
    live = dict(uuid='GPU-0',pid=12,uid=str(os.getuid()),in_slurm_job=True)
    sampler.resolve_exit_race(live,'original_time',set(),True)
    result = sampler.resolve_exit_race(dict(uuid='GPU-0',pid=12,unavailable=True,
        read_error=read_error),'later_time',set(),False)
    assert 'exit_race_resolution' not in result


@pytest.mark.parametrize('unavailable,owned',[(True,True),(False,False)])
def test_confirmation_query_failure_does_not_hide_ownership_failure(tmp_path,unavailable,owned):
    a=resource_row('2026-10-06T00:00:00+00:00')
    b=resource_row('2026-10-06T00:00:02+00:00')
    c=resource_row('2026-10-06T00:00:04+00:00')
    c['query_error']='TimeoutExpired'
    c['gpu_processes'][-1]['unavailable']=unavailable
    c['gpu_processes'][-1]['in_slurm_job']=owned
    p=tmp_path/'raw.ndjson';p.write_text('\n'.join(json.dumps(v) for v in [a,b,c]))
    with pytest.raises(ValueError):
        f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100)


def test_usage_retains_exact_permission_error(monkeypatch):
    def denied(_self,*_args,**_kwargs):
        raise PermissionError(13,'fixture denied')
    monkeypatch.setattr(Path,'read_text',denied)
    result=sampler.usage(12)
    assert result['unavailable'] is True
    assert result['read_error']['type']=='PermissionError'
    assert result['read_error']['errno']==13


def test_malformed_nvml_application_rows_fail_closed():
    with pytest.raises(ValueError):sampler.nvml_app_rows('GPU-0, 12\n')
