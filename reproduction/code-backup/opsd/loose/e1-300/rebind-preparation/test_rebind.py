import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rebind_candidate',HERE/'rebind.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
REPO=HERE.parents[1]/'opsd-e1-300';SOURCE='a0968c2d24beeaa0400f60e4e34677fe6deef9c9'

def test_actual_clean_generation_deterministic_and_no_overwrite(tmp_path):
 a=r.generate(REPO,SOURCE,tmp_path/'one',HERE.parent);b=r.generate(REPO,SOURCE,tmp_path/'two',HERE.parent)
 assert a['output_manifests']==b['output_manifests'] and a['actual_first5_accepted'] is False
 with pytest.raises(FileExistsError):r.generate(REPO,SOURCE,tmp_path/'one',HERE.parent)

@pytest.mark.parametrize('source',['future',r.OLD,'0'*40])
def test_wrong_old_or_guessed_source_rejected(tmp_path,source):
 with pytest.raises(ValueError):r.generate(REPO,source,tmp_path/'out',HERE.parent)
 assert not (tmp_path/'out').exists()


def test_dirty_head_rejected_before_output(tmp_path,monkeypatch):
 original=r.git
 def git(repo,*args):return 'M unexpected.py' if args==('status','--porcelain') else original(repo,*args)
 monkeypatch.setattr(r,'git',git)
 with pytest.raises(ValueError):r.generate(REPO,SOURCE,tmp_path/'out',HERE.parent)
 assert not (tmp_path/'out').exists()


def test_extra_algorithm_delta_rejected(tmp_path,monkeypatch):
 original=r.git
 def git(repo,*args):return original(repo,*args)+'\nsearch_r1/baselines/distillation.py' if args[:2]==('diff','--name-only') else original(repo,*args)
 monkeypatch.setattr(r,'git',git)
 with pytest.raises(ValueError):r.generate(REPO,SOURCE,tmp_path/'out',HERE.parent)
 assert not (tmp_path/'out').exists()
