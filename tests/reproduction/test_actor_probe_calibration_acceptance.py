"""Scalar provenance fixtures only; no real fitted model or native acceptance."""
import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from reproduction.common8.actor_probe_calibration_acceptance import select_versions, verify_original, verify_rebound
from search_g1.actor_probe_recovery import identity
from tests.reproduction.test_actor_probe_chain import segment
from tests.reproduction.test_actor_probe_head_service import fixture


def write(path, document):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_chain():
    from tests.reproduction.test_actor_probe_recovery import documents
    parent,child,_,_ = documents()
    a = segment(parent['run_id'],0,53,50)
    b = segment(child['run_id'],50,200,200,a['checkpoint_writes'],a['checkpoint_attempt_index'])
    entries = []; segments = []
    for run,counter in ((parent,a),(child,b)):
        start = counter['start_outer']; base = '/scalar-fixture/'+run['run_id']
        for row in counter['retained_updates']+counter['discarded_updates']:
            step = row['step']; round_id = (step-1)//10*10
            owner = parent['run_id'] if round_id<=50 else child['run_id']
            row['probe_version'] = owner+':'+str(round_id)+':'+'7'*32
            row['probe_actor_binding'] = dict(run_id=run['run_id'],execution_id=run['execution_id'],
                head_version=row['probe_version'],logical_update=step,completed_outer=step-1,
                optimizer_writes=row['optimizer_writes_before'])
        entries.append(dict(run=run,base=base,last=counter['retained_through'],
            origin=dict(completed_outer=start,actual_optimizer_writes=counter['start_writes'],
                attempt_index=counter['start_attempt_index'],aborted_attempts=0)))
        segments.append(dict(identity=identity(run),base=base,first=start+1,last=counter['retained_through'],
            counter_segment=counter,actual_interval_accepted=True))
    return entries,segments


def original_fixture(root):
    base,run,manifest,_,_ = fixture(root)
    run.update(started_unix=1,source_sha256={'scorer.py':'c'*64})
    official=base/'official'; official.mkdir()
    model=official/'config.json';model.write_text('{"fixture":"scalar model stand-in"}')
    run['initial_model']=str(official)
    run['initial_model_sha256']={'config.json':hashlib.sha256(model.read_bytes()).hexdigest()}
    run['initial_probe_version']=run['run_id']+':0:'+'7'*32
    archive=base/'artifacts'/('round_0-attempt-'+'8'*32);heads=archive/'heads';dump=archive/'dumps'
    heads.mkdir(parents=True);dump.mkdir()
    manifest.update(version=run['initial_probe_version'],published_unix=2,encoder=str(official),
        encoder_sha256=run['initial_model_sha256'],
        fitting_input_semantics='immutable current-actor export; production features come from current actor RPC',
        fitted_state_semantics='serialized sklearn classifier/scaler parameters; no GRPO optimizer ownership')
    for key in ('commit','boundary'):
        source=Path(manifest[key]);target=heads/source.name;target.write_bytes(source.read_bytes())
        manifest[key]=str(target)
    manifest['calibration_report_sha256']=write(heads/'calibration-report.json',manifest['calibration_report'])
    label_sha=write(dump/'calibration-label-proof.json',manifest['label_proof'])
    manifest['dump_sha256']={'calibration-label-proof.json':label_sha}
    path=archive/'version-manifest.json';sha=write(path,manifest)
    from reproduction.common8.policy_calibration_plan import document_sha256
    write(base/'logs/formal-inline-initial-probe.json',dict(schema_version=1,role='fresh200_actual_initial_probe',
        run_id=run['run_id'],version=manifest['version'],source_map_sha256=document_sha256(run['source_sha256']),
        model_revision=run['model_revision'],initial_model_sha256=run['initial_model_sha256'],
        input_sha256=run['input_sha256'],archive=path.relative_to(base).as_posix(),archive_sha256=sha))
    return base,run,manifest,path


class CalibrationAcceptanceTests(unittest.TestCase):
    def test_actual_restored_pair_files_remain_bound_to_saved_parent_and_staging_hash(self):
        from tests.reproduction.test_actor_probe_recovery import documents
        from search_g1.actor_probe_recovery import rebound_probe_manifest, binding_digest
        from reproduction.common8.verify_data import digest
        from search_g1.actor_probe_contract import MODE
        with tempfile.TemporaryDirectory() as root:
            root=Path(root).resolve();parent,child,binding,receipt=documents()
            oldbase=root/'parent';base=root/'child';parent['checkpoint_dir']=str(oldbase/'checkpoints')
            olddir=Path(parent['checkpoint_dir'])/'actor/global_step_5/probe-state'
            version=parent['run_id']+':0:'+'7'*32;receipt.update(probe_version=version)
            hashes={}
            for name in ('commit_probe.pkl','boundary_probe.pkl'):
                path=olddir/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{"fixture":"saved scalar head stand-in"}')
                hashes[name]=digest(path)
            receipt['probe_head_hashes']=hashes
            saved=dict(run_id=parent['run_id'],version=version,round=0,actor_step=0,representation_source=MODE,
                calibration_data_sha256=parent['input_sha256']['calibration-300.parquet'],head_sha256=hashes,
                commit=str(olddir/'commit_probe.pkl'),boundary=str(olddir/'boundary_probe.pkl'))
            receipt['probe_manifest_sha256']=write(olddir/'version-manifest.json',saved)
            archive=base/'artifacts'/('recovered-heads-'+child['execution_id']);archive.mkdir(parents=True)
            paths={key:archive/(key+'_probe.pkl') for key in ('commit','boundary')}
            for key,path in paths.items():path.write_bytes((olddir/path.name).read_bytes())
            actual=rebound_probe_manifest(saved,binding,child,parent,receipt,paths)
            sha=write(archive/'version-manifest.json',actual)
            prep_sha=write(base/'logs/actor-probe-recovery-preparation.json',{'identity':identity(child)})
            staged=dict(role='actor_probe_restored_heads_staged',staged=True,identity=identity(child),
                production_admission=False,recovery_binding_sha256=binding_digest(binding),
                parent_full_state_receipt_sha256=binding['full_state_receipt_sha256'],
                parent_probe_manifest_sha256=receipt['probe_manifest_sha256'],archive_manifest=str(archive/'version-manifest.json'),
                archive_manifest_sha256=sha,preparation_sha256=prep_sha,
                **{key:actual[key] for key in ('version','actor_step','round','head_sha256','calibration_data_sha256')})
            marker=base/'logs/actor-probe-recovery-heads-staged.json';write(marker,staged)
            entries=[dict(run=parent,base=str(oldbase),last=5),dict(run=child,base=str(base),origin={'completed_outer':5})]
            segment={'full_states':{'5':receipt}}
            adopted,proof=verify_rebound(*entries,segment)
            self.assertEqual(adopted,actual);self.assertEqual(proof['child_archive']['sha256'],sha)
            staged['parent_probe_manifest_sha256']='0'*64;write(marker,staged)
            with self.assertRaises(AssertionError):verify_rebound(*entries,segment)
            staged['parent_probe_manifest_sha256']=receipt['probe_manifest_sha256'];write(marker,staged)
            paths['commit'].write_text('{"fixture":"different child classifier"}')
            with self.assertRaises(AssertionError):verify_rebound(*entries,segment)

    def test_retained_version_owner_follows_saved_boundary_not_last_observed_update(self):
        entries,segments=retained_chain();before=copy.deepcopy((entries,segments))
        versions,owners,namespaces=select_versions(entries,segments)
        self.assertEqual(versions[50],entries[0]['run']['run_id']+':50:'+'7'*32)
        self.assertEqual(owners[50],entries[0]['run']['run_id'])
        self.assertEqual(owners[60],entries[1]['run']['run_id'])
        self.assertEqual(namespaces[(entries[1]['run']['run_id'],50)],versions[50])
        self.assertEqual(len(versions),20);self.assertEqual((entries,segments),before)

    def test_changed_within_outer_window_and_child_refit_of_parent_boundary_rejected(self):
        for kind in ('mid_window','wrong_owner','actor_counter'):
            entries,segments=retained_chain();row=segments[1]['counter_segment']['retained_updates'][0]
            if kind=='mid_window':row=segments[1]['counter_segment']['retained_updates'][1];row['probe_version']=row['probe_version'][:-32]+'9'*32
            elif kind=='wrong_owner':row['probe_version']=entries[1]['run']['run_id']+':50:'+'7'*32
            else:row['probe_actor_binding']['optimizer_writes']+=1
            if kind!='actor_counter':row['probe_actor_binding']['head_version']=row['probe_version']
            with self.subTest(kind=kind),self.assertRaises(AssertionError):select_versions(entries,segments)

    def test_actual_owned_hash_files_precede_separate_calibration_decoders(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,manifest,path=original_fixture(root)
            with ExitStack() as stack:
                arrays=stack.enter_context(patch('reproduction.common8.adopt_initial_probes._calibration_arrays'))
                fitted=stack.enter_context(patch('reproduction.common8.adopt_initial_probes._fitted_heads'))
                loader=stack.enter_context(patch('reproduction.common8.actor_probe_head_service.load_heads'))
                actual,proof=verify_original(base,run,path,0,manifest['version'])
                self.assertEqual(actual,manifest);arrays.assert_called_once();fitted.assert_called_once();loader.assert_called_once()
                self.assertFalse(proof.get('full_training_or_final_acceptance',False))
                target=Path(manifest['commit']);target.write_text('{"fixture":"tampered classifier"}')
                with self.assertRaises(Exception):verify_original(base,run,path,0,manifest['version'])
                self.assertEqual(arrays.call_count,1)

    def test_changed_encoder_namespace_or_initial_archive_hash_rejected(self):
        for kind in ('encoder','archive_sha','rebound_original'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                base,run,manifest,path=original_fixture(root)
                if kind=='encoder':(Path(run['initial_model'])/'config.json').write_text('{"fixture":"replaced"}')
                elif kind=='archive_sha':
                    p=base/'logs/formal-inline-initial-probe.json';v=json.loads(p.read_text());v['archive_sha256']='0'*64;write(p,v)
                else:manifest['restored_from']={};write(path,manifest)
                with self.assertRaises(AssertionError):verify_original(base,run,path,0,manifest['version'])


if __name__=='__main__':unittest.main()
