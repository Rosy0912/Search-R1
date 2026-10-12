"""Tiny scalar files only; no fitted classifier/tensor files or GPU admission."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.a800_common.stage_actor_probe_recovery_heads import stage_context, _publish_live
from reproduction.a800_common.prepare_actor_probe_recovery import candidate_manifest
from search_g1.actor_probe_contract import MODE, validate_version
from tests.reproduction.test_actor_probe_recovery_preparation import saved_documents


def fixture(root, step):
    root = Path(root).resolve()
    parent,receipt,driver,args = saved_documents(root,step)
    base = root/'heads-child'; (base/'logs').mkdir(parents=True); (base/'artifacts').mkdir()
    directory = root/'checkpoint'; probes = directory/'probe-state'; probes.mkdir(parents=True)
    receipt['probe_head_hashes'] = {}
    for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
        # JSON scalar placeholders exercise the actual immutable file-copy path.
        payload=json.dumps(dict(fixture=True,classifier=key,step=step)).encode()
        (probes/name).write_bytes(payload)
        receipt['probe_head_hashes'][name]=hashlib.sha256(payload).hexdigest()
    manifest=dict(run_id=parent['run_id'],version=receipt['probe_version'],
        round=receipt['probe_actor_step'],actor_step=receipt['probe_actor_step'],
        representation_source=MODE,calibration_data_sha256=parent['input_sha256']['calibration-300.parquet'],
        head_sha256=receipt['probe_head_hashes'],commit=str(probes/'commit_probe.pkl'),
        boundary=str(probes/'boundary_probe.pkl'),calibration_report={'fixture':True})
    (probes/'version-manifest.json').write_text(json.dumps(manifest))
    receipt['probe_manifest_sha256']=hashlib.sha256((probes/'version-manifest.json').read_bytes()).hexdigest()
    args['base']=str(base);child=candidate_manifest(parent,receipt,driver,**args)
    context=dict(directory=directory,parent=parent,receipt=receipt,
        binding=child['actor_probe_recovery'],probe_manifest=manifest)
    return base,child,context


class RecoveryHeadsTests(unittest.TestCase):
    def test_pair_preserves_saved_version_hashes_and_fitting_step_at_every_milestone(self):
        for step in (5,50,100,150):
            with self.subTest(step=step),tempfile.TemporaryDirectory() as root:
                base,child,ctx=fixture(root,step);before=copy.deepcopy((child,ctx))
                parent_bytes={p:p.read_bytes() for p in (ctx['directory']/'probe-state').iterdir()}
                proof=stage_context(base,child,ctx,'1'*64)
                self.assertEqual((child,ctx),before)
                self.assertEqual(parent_bytes,{p:p.read_bytes() for p in parent_bytes})
                live=json.loads((base/'artifacts/live/manifest.json').read_text())
                validate_version(child,live,step+1)
                self.assertEqual(live['version'],ctx['receipt']['probe_version'])
                self.assertEqual(live['head_sha256'],ctx['receipt']['probe_head_hashes'])
                self.assertEqual(live['restored_from']['run_id'],ctx['parent']['run_id'])
                self.assertEqual(proof['live_manifest_sha256'],proof['archive_manifest_sha256'])
                self.assertFalse(proof['production_admission'])
                for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
                    path=Path(live[key]);self.assertTrue(path.is_relative_to(base/'artifacts'))
                    self.assertFalse(path.is_symlink())
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),live['head_sha256'][name])
                with self.assertRaises(AssertionError):stage_context(base,child,ctx,'1'*64)

    def test_second_head_corruption_and_symlink_fail_before_any_copy_or_live_switch(self):
        for symlink in (False,True):
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory() as root:
                base,child,ctx=fixture(root,100)
                path=ctx['directory']/'probe-state/boundary_probe.pkl'
                if symlink:
                    original=path.with_name('source.json');path.rename(original);path.symlink_to(original)
                else:path.write_text('{"fixture":"wrong version"}')
                with self.assertRaises(AssertionError):stage_context(base,child,ctx,'1'*64)
                self.assertEqual(list((base/'artifacts').iterdir()),[])
                self.assertFalse((base/'logs/actor-probe-recovery-heads-staged.json').exists())

    def test_partial_copy_failure_cannot_publish_pair_or_staged_marker(self):
        with tempfile.TemporaryDirectory() as root:
            base,child,ctx=fixture(root,50)
            from reproduction.a800_common.stage_actor_probe_recovery_heads import _copy_head as actual_copy
            copies=[]
            def fail_second(source,target,sha):
                copies.append(target)
                if len(copies)==2:raise OSError('fixture disk failure')
                actual_copy(source,target,sha)
            with patch('reproduction.a800_common.stage_actor_probe_recovery_heads._copy_head',fail_second):
                with self.assertRaises(OSError):stage_context(base,child,ctx,'1'*64)
            self.assertEqual(len(copies),2)
            self.assertTrue(copies[0].exists())
            self.assertFalse((base/'artifacts/live/manifest.json').exists())
            self.assertFalse((base/'logs/actor-probe-recovery-heads-staged.json').exists())

    def test_exclusive_atomic_live_publication_cannot_replace_existing_version(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root).resolve()/'manifest.json';original=b'{"fixture":"existing"}'
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):_publish_live(path,{'fixture':'replacement'})
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
