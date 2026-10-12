"""Real local Git bundles of tiny code fixtures; no A800 or GPU claim."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from reproduction.a800_common.stage_published_source import create_source_bundle


def git(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip()


def fixture(root):
    repo=Path(root)/'repo';repo.mkdir();git(repo,'init','-b','actor-probes-ten')
    git(repo,'config','user.name','CPU Git fixture');git(repo,'config','user.email','fixture@example.invalid')
    p=repo/'code.py';p.write_text('# fixture baseline\n');git(repo,'add','code.py');git(repo,'commit','-m','fixture baseline')
    baseline=git(repo,'rev-parse','HEAD');p.write_text('# fixture published source\n');git(repo,'commit','-am','fixture update')
    commit=git(repo,'rev-parse','HEAD')
    return repo,baseline,commit


class SourceStagingTests(unittest.TestCase):
    def test_actual_bundle_ref_and_offline_clone_restore_exact_commit_and_file(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);repo,baseline,commit=fixture(root);bundle=root/'source.bundle'
            raw=create_source_bundle(repo,commit,bundle,baseline_commit=baseline)
            self.assertGreater(len(raw),0)
            self.assertEqual(git(repo,'bundle','list-heads',str(bundle)),commit+' refs/heads/actor-probes-ten')
            clone=root/'clone';git(repo,'clone','--local','--no-hardlinks','--no-checkout',str(repo),str(clone))
            git(clone,'bundle','verify',str(bundle));git(clone,'fetch',str(bundle),'refs/heads/actor-probes-ten')
            self.assertEqual(git(clone,'rev-parse','FETCH_HEAD'),commit)
            git(clone,'checkout','--detach',commit)
            self.assertEqual((clone/'code.py').read_bytes(),(repo/'code.py').read_bytes())
            self.assertEqual(git(repo,'rev-parse','HEAD'),commit)
            self.assertEqual(git(clone,'status','--porcelain'),'')

    def test_detached_head_with_stale_named_ref_rejects_before_creating_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);repo,baseline,commit=fixture(root)
            git(repo,'checkout','--detach',commit);git(repo,'update-ref','refs/heads/actor-probes-ten',baseline)
            self.assertEqual(git(repo,'rev-parse','HEAD'),commit)
            bundle=root/'source.bundle'
            with self.assertRaises(AssertionError):create_source_bundle(repo,commit,bundle,baseline_commit=baseline)
            self.assertFalse(bundle.exists())

    def test_existing_or_dangling_bundle_path_cannot_be_replaced(self):
        for symlink in (False,True):
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory() as root:
                root=Path(root);repo,baseline,commit=fixture(root);bundle=root/'source.bundle'
                if symlink:bundle.symlink_to(root/'missing')
                else:bundle.write_text('fixture existing')
                with self.assertRaises(AssertionError):create_source_bundle(repo,commit,bundle,baseline_commit=baseline)
                if symlink:self.assertTrue(bundle.is_symlink())
                else:self.assertEqual(bundle.read_text(),'fixture existing')


if __name__=='__main__':unittest.main()
