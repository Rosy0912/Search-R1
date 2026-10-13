"""Tiny scalar files and injected I/O faults; no model/native admission."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.common8 import actor_probe_publication as publisher
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import validate_version
from tests.reproduction.test_actor_probe_calibration_acceptance import original_fixture, write


def fixture(root, step=0):
    base, run, payload, archive = original_fixture(root)
    archive.unlink()
    (archive.parent/'calibration.log').write_text('scalar fit output\n')
    live = base/'artifacts/live/manifest.json'
    live.write_text('{"fixture_old_pair":"unchanged"}\n')
    if step:
        official = Path(run['initial_model'])
        write(official/'tokenizer.json', {'model': {'type':'BPE', 'merges':['a b']}})
        write(official/'tokenizer_config.json', {'fixture':'tokenizer'})
        run['checkpoint_dir'] = str(base/'checkpoints')
        encoder = base/'checkpoints/actor'/f'global_step_{step}'
        encoder.mkdir(parents=True)
        (encoder/'config.json').write_bytes((official/'config.json').read_bytes())
        write(encoder/'tokenizer.json', {'model': {'type':'BPE', 'merges':[['a','b']], 'ignore_merges':False}})
        (encoder/'tokenizer_config.json').write_bytes((official/'tokenizer_config.json').read_bytes())
        write(encoder/'complete.json', dict(run_id=run['run_id'], step=step,
            finite_all_saved_tensors=True, files_sha256={'config.json':digest(encoder/'config.json')}))
        payload.update(round=step, actor_step=step, encoder=str(encoder),
            version=run['run_id']+':'+str(step)+':'+'7'*32,
            encoder_sha256={name:digest(encoder/name) for name in
                            ('config.json','tokenizer.json','tokenizer_config.json')})
        new = base/'artifacts'/('round_'+str(step)+'-attempt-'+'8'*32)
        archive.parent.rename(new)
        archive = new/'version-manifest.json'
        for key in ('commit','boundary'):
            payload[key] = str(new/'heads'/(key+'_probe.pkl'))
    return base, run, payload, archive, live


class PublicationTests(unittest.TestCase):
    def test_dependencies_and_complete_archive_are_synced_before_live_commit(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, payload, archive, live = fixture(root, 10)
            before = copy.deepcopy((run,payload)); old = live.read_bytes(); events = []
            sync_file, sync_dir, replace = publisher._sync_file, publisher._sync_dir, publisher.os.replace
            def file(path):
                events.append(('file',Path(path))); return sync_file(path)
            def directory(path):
                events.append(('dir',Path(path))); return sync_dir(path)
            def commit(source,target):
                self.assertEqual(live.read_bytes(),old)
                self.assertEqual(json.loads(archive.read_bytes()),payload)
                required = [Path(payload['commit']),Path(payload['boundary']),
                    archive.parent/'heads/calibration-report.json',archive.parent/'dumps/calibration-label-proof.json',
                    archive.parent/'calibration.log',Path(payload['encoder'])/'complete.json']
                required += [Path(payload['encoder'])/name for name in payload['encoder_sha256']]
                self.assertTrue(all(('file',p) in events for p in required))
                self.assertIn(('dir',archive.parent),events)
                self.assertIn(('dir',base/'checkpoints/actor'),events)
                self.assertIn(('dir',base.parent),events)
                events.append(('commit',Path(target))); return replace(source,target)
            with patch.object(publisher,'_sync_file',file), patch.object(publisher,'_sync_dir',directory), \
                    patch.object(publisher.os,'replace',commit):
                self.assertEqual(publisher.publish_fitted_pair(base,run,archive.parent,payload),payload['version'])
            self.assertEqual(events[-1],('dir',live.parent))
            self.assertEqual(json.loads(live.read_bytes()),payload)
            self.assertEqual(json.loads(archive.read_bytes()),payload)
            self.assertEqual((run,payload),before)

    def test_failed_temp_archive_never_exposes_partial_final_json(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, payload, archive, live = fixture(root)
            old = live.read_bytes()
            def failed(directory,label,raw):
                (directory/'.failed-version-manifest.tmp').write_bytes(raw[:17])
                raise OSError('injected interrupted archive write')
            with patch.object(publisher,'_write_temp',failed),self.assertRaises(OSError):
                publisher.publish_fitted_pair(base,run,archive.parent,payload)
            self.assertFalse(archive.exists())
            self.assertEqual(list((base/'artifacts').glob('round_*-attempt-*/version-manifest.json')),[])
            self.assertEqual(live.read_bytes(),old)

    def test_sync_and_archive_exclusive_link_failures_leave_old_live(self):
        for phase in ('file','archive_link','live_temp'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as root:
                base, run, payload, archive, live = fixture(root)
                old = live.read_bytes(); original = publisher._write_temp
                def failed_temp(directory,label,raw):
                    if label == 'manifest': raise OSError('injected live candidate failure')
                    return original(directory,label,raw)
                if phase=='file': context=patch.object(publisher,'_sync_file',side_effect=OSError('sync failure'))
                elif phase=='archive_link': context=patch.object(publisher.os,'link',side_effect=FileExistsError('race'))
                else: context=patch.object(publisher,'_write_temp',failed_temp)
                with context,self.assertRaises(OSError):
                    publisher.publish_fitted_pair(base,run,archive.parent,payload)
                self.assertEqual(live.read_bytes(),old)
                if archive.exists(): self.assertEqual(json.loads(archive.read_bytes()),payload)

    def test_post_rename_failure_propagates_without_false_rollback(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, payload, archive, live = fixture(root)
            sync = publisher._sync_dir
            def failure(path):
                if Path(path)==live.parent: raise OSError('injected live directory sync failure')
                return sync(path)
            with patch.object(publisher,'_sync_dir',failure),self.assertRaises(OSError):
                publisher.publish_fitted_pair(base,run,archive.parent,payload)
            self.assertEqual(json.loads(live.read_bytes()),payload)
            self.assertEqual(json.loads(archive.read_bytes()),payload)

    def test_original_archive_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, payload, archive, live = fixture(root)
            archive.write_text('{"existing_original":"retain"}')
            old, original = live.read_bytes(),archive.read_bytes()
            with self.assertRaises(AssertionError):publisher.publish_fitted_pair(base,run,archive.parent,payload)
            self.assertEqual((live.read_bytes(),archive.read_bytes()),(old,original))

    def test_bad_artifacts_or_noncanonical_dependencies_fail_before_selection(self):
        for kind in ('head_hash','escape','symlink','report','encoder_hash','canonical_step','tokenizer_semantics'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                base, run, payload, archive, live = fixture(root,10);old=live.read_bytes()
                if kind=='head_hash':payload['head_sha256']['commit_probe.pkl']='0'*64
                elif kind=='escape':payload['dump_sha256']['../escape']='0'*64
                elif kind=='symlink':
                    head=Path(payload['commit']);other=archive.parent/'scalar-head';head.rename(other);head.symlink_to(other)
                elif kind=='report':payload['calibration_report']['commit_cv_auc']=.8
                elif kind=='encoder_hash':payload['encoder_sha256']['config.json']='0'*64
                elif kind=='canonical_step':
                    path=Path(payload['encoder'])/'complete.json';receipt=json.loads(path.read_bytes());receipt['step']=20;write(path,receipt)
                else:write(Path(payload['encoder'])/'tokenizer_config.json',{'fixture':'changed'})
                with self.assertRaises(AssertionError):publisher.publish_fitted_pair(base,run,archive.parent,payload)
                self.assertFalse(archive.exists());self.assertEqual(live.read_bytes(),old)

    def test_terminal200_is_published_without_inventing_prediction201(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, payload, archive, live = fixture(root,200)
            publisher.publish_fitted_pair(base,run,archive.parent,payload)
            self.assertEqual(json.loads(live.read_bytes())['actor_step'],200)
            with self.assertRaises(ValueError):validate_version(run,payload,201)
            with self.assertRaises(ValueError):validate_version(run,payload,200)

    def test_full_state5_syncs_canonical_receipt_and_parents_before_completion(self):
        from reproduction.common8 import actor_probe_checkpoint as checkpoint
        with tempfile.TemporaryDirectory() as root:
            base, run, payload, archive, live = fixture(root,5)
            # Step5 saves the initial pair; it performs no scheduled refit.
            payload.update(round=0,actor_step=0,version=run['run_id']+':0:'+'7'*32)
            live.write_text(json.dumps(payload))
            directory=Path(run['checkpoint_dir'])/'actor/global_step_5'
            attempts=SimpleNamespace(pending=None,completed_outer=5,cumulative_writes=100)
            workers=[dict(rank=r,completed_outer=5,optimizer_writes=100) for r in range(4)]
            trainer=SimpleNamespace(global_steps=5,actor_rollout_wg=SimpleNamespace(save_probe_training_state=lambda *a:workers))
            events=[];sync=publisher._sync_file;sync_dir=publisher._sync_dir
            def file(path):events.append(('file',Path(path)));return sync(path)
            def folder(path):events.append(('dir',Path(path)));return sync_dir(path)
            def driver(value,target,base):
                self.assertIn(('file',directory/'complete.json'),events)
                self.assertIn(('dir',directory.parent),events)
                self.assertIn(('dir',Path(run['checkpoint_dir'])),events)
                self.assertIn(('dir',base.parent),events)
                self.assertFalse((directory/'full-state-complete.json').exists())
                target.write_text('scalar driver stand-in');return {'path':str(target)}
            with patch.dict('os.environ',{'SEARCH_G1_BASE':str(base)}), \
                    patch.object(publisher,'_sync_file',file),patch.object(publisher,'_sync_dir',folder), \
                    patch.object(checkpoint,'driver_payload',return_value={'fixture':'driver'}), \
                    patch.object(checkpoint,'durable_torch_save',driver):
                result=checkpoint.save_full_state(trainer,run,attempts,0,4)
            self.assertEqual(result['canonical_model_receipt_sha256'],digest(directory/'complete.json'))
            self.assertTrue((directory/'full-state-complete.json').exists())
