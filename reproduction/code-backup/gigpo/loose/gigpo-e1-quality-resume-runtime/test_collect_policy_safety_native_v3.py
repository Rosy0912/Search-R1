import importlib.util
from pathlib import Path
import errno
import pytest


def module():
    p = Path(__file__).with_name('collect_policy_safety_native_v3.py')
    s = importlib.util.spec_from_file_location('collector3', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def simulate(monkeypatch, stat_error=None, readable=True):
    m = module()
    calls = []
    monkeypatch.setattr(m.Path, 'is_dir', lambda p: readable)
    monkeypatch.setattr(m.os, 'listdir', lambda p: calls.append(str(p)) or [])
    def stat(path):
        if stat_error == 'present':
            return object()
        if stat_error:
            raise stat_error
        raise FileNotFoundError(errno.ENOENT, 'Removed', str(path))
    monkeypatch.setattr(m.os, 'stat', stat)
    return m, calls


def test_removed_job_and_step_with_readable_slurm_hierarchy(monkeypatch):
    m, calls = simulate(monkeypatch)
    r = m.cgroup_release('3267', '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3267/step_0')
    assert r['removed'] and r['stat_outcome'] == r['scope_stat_outcome'] == 'ENOENT'
    assert calls == ['/sys/fs/cgroup/freezer', '/sys/fs/cgroup/freezer/slurm_h100-8']


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'Unreadable'), OSError(errno.EIO, 'IO failure'), 'present'])
def test_stat_failure_or_live_cgroup_never_counts_as_released(monkeypatch, error):
    m, _ = simulate(monkeypatch, error)
    with pytest.raises((OSError, AssertionError)):
        m.cgroup_release('3267', '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3267/step_0')


def test_missing_slurm_hierarchy_is_not_an_empty_released_job(monkeypatch):
    m, _ = simulate(monkeypatch, readable=False)
    with pytest.raises(AssertionError):
        m.cgroup_release('3267', '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3267/step_0')


def test_unreadable_ancestor_is_not_an_empty_released_job(monkeypatch):
    m, _ = simulate(monkeypatch)
    def deny(path):
        raise PermissionError(errno.EACCES, 'Cannot list hierarchy')
    monkeypatch.setattr(m.os, 'listdir', deny)
    with pytest.raises(PermissionError):
        m.cgroup_release('3267', '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3267/step_0')


def test_other_job_scope_cannot_prove_current_job_release(monkeypatch):
    m, _ = simulate(monkeypatch)
    with pytest.raises(AssertionError):
        m.cgroup_release('3267', '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3259/step_0')
