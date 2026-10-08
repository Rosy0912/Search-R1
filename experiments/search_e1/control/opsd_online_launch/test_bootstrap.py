"""CPU-only negative/control fixtures; every pidfd signal and HTTP call is mocked."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import bootstrap as b
spec=importlib.util.spec_from_file_location('bootstrap_fixture_checkpoint',HERE.parent/'opsd_online/checkpoint.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class BootstrapFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.run='a'*32
        self.human=dict(cluster='HKUST',run_id=self.run,component_clip=10,outer_iterations=200,
            explicit_exception=True,eight_gpu_execution=True,text='fixture actual choices',thread_id='fixture-thread',observed_unix=1,durable_target='/fixture/mac-backup')
    def tearDown(self):self.tmp.cleanup()

    def test_human_choices_not_inferred(self):
        b.human_admission(self.human,self.run,'/fixture/mac-backup')
        for key,value in [('component_clip',None),('outer_iterations',None),('explicit_exception',False),('run_id','b'*32),('eight_gpu_execution',False)]:
            changed=dict(self.human);changed[key]=value
            with self.assertRaises(RuntimeError):b.human_admission(changed,self.run,'/fixture/mac-backup')

    def inventory(self,extra=None):
        devices=[dict(index=i,uuid='gpu-'+str(i),actual_free_mib=9000 if i<4 else 24000) for i in range(8)]
        apps=[dict(pid=b.RESIDENT['pid'],gpu_uuid='gpu-'+str(i),used_mib=16000) for i in range(4)]
        apps.append(dict(pid=b.ORACLE['pid'],gpu_uuid='gpu-7',used_mib=574))
        if extra:apps.append(extra)
        a=SimpleNamespace(inventory=lambda:(devices,apps),process=lambda pid:dict(b.ORACLE),same_process=lambda x,y:all(x[k]==y[k] for k in ('pid','uid','starttick')),require=b.require)
        return dict(owner=dict(gpu_uuids=['gpu-'+str(i) for i in range(8)])),a,devices,apps

    def test_foreign_gpu_app_and_oracle_overflow_rejected(self):
        config,a,_,_=self.inventory(dict(pid=99,gpu_uuid='gpu-5',used_mib=10))
        with self.assertRaisesRegex(RuntimeError,'Foreign'):b.inventory_admission(config,a,True)
        config,a,_,apps=self.inventory();apps[-1]['used_mib']=1025
        with self.assertRaisesRegex(RuntimeError,'sharing'):b.inventory_admission(config,a,True)

    def test_after_release_still_requires_all_eight_direct_free(self):
        config,a,devices,apps=self.inventory();apps[:]=[apps[-1]]
        for device in devices:device['actual_free_mib']=24000
        b.inventory_admission(config,a,False);devices[0]['actual_free_mib']=20479
        with self.assertRaisesRegex(RuntimeError,'memory.free'):b.inventory_admission(config,a,False)

    def backup(self):
        resident=self.root/'resident';base=self.root/'online';(base/'receipts').mkdir(parents=True)
        paths=[]
        for name in ['server','launch_cpu']+['fixture'+str(i) for i in range(8)]:
            p=resident/'source'/(name+'.py');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# CPU fixture\n');paths.append(p)
        log=resident/'logs/server.log';log.parent.mkdir();log.write_text('fixture preserved log\n');paths.append(log)
        archive=base/'receipts/resident-backup.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in paths:tar.add(path,arcname=str(path.relative_to(resident)))
        value=dict(accepted=True,pid=b.RESIDENT['pid'],uid=1005,tick=b.RESIDENT['starttick'],
            archive_sha256=c.fileproof(archive)['sha256'],local_archive='/fixture/retained.tar.gz',local_extracted='/fixture/unpacked',
            files=[dict(relative_path=str(path.relative_to(resident)),path=str(path),sha256=c.fileproof(path)['sha256'],bytes=path.stat().st_size) for path in paths])
        return resident,base,value,log

    def test_uploaded_archive_real_members_and_growing_log_prefix(self):
        resident,base,value,log=self.backup()
        with patch.object(b,'RESIDENT_BASE',resident):
            self.assertTrue(b.verify_backup(value,c,base)['accepted']);log.write_text(log.read_text()+'later log append\n')
            self.assertTrue(b.verify_backup(value,c,base)['accepted'])
            (resident/'source/server.py').write_text('# modified source\n')
            with self.assertRaises(ValueError):b.verify_backup(value,c,base)

    def test_backup_refuses_wrong_identity_missing_log_and_bad_archive_hash(self):
        resident,base,value,_=self.backup()
        with patch.object(b,'RESIDENT_BASE',resident):
            for change in [dict(pid=99),dict(archive_sha256='0'*64),dict(files=value['files'][:-1])]:
                bad=dict(value,**change)
                with self.assertRaises((RuntimeError,ValueError)):b.verify_backup(bad,c,base)

    def finalize_fixture(self):
        base=self.root/'online';(base/'receipts').mkdir(parents=True)
        human_ref=c.exclusive_json(base/'receipts/human.json',self.human)
        backup_ref=c.exclusive_json(base/'receipts/backup.json',dict(fixture=True))
        config=dict(run_id=self.run,source_commit='b'*40,paths=dict(output_root=str(base/'runs')),
            bootstrap_human_authorization=human_ref,bootstrap_resident_backup=backup_ref,owner=dict(gpu_uuids=['gpu-'+str(i) for i in range(8)]),
            resources=dict(shared_gpu_lease_paths=[str(base/('lease-'+str(i))) for i in range(4)]),gates=dict(retriever_admission={},publication={},storage={}))
        c.exclusive_json(base/'receipts/config.preliminary.json',config)
        args=SimpleNamespace(base=str(base),run_id=self.run,sourcecommit='b'*40,durable_target='/fixture/mac-backup',timeout=1)
        a=SimpleNamespace(process=lambda pid:dict(b.RESIDENT,argv=[str(b.RESIDENT_BASE/'source/server.py')]),
            same_process=lambda x,y:all(x[k]==y[k] for k in ('pid','uid','starttick')),retriever_admission=Mock(),bound=lambda x:{},
            publication_admission=Mock(),storage_admission=Mock())
        h=dict(b.RESIDENT,source_identity_sha256=b.RESIDENT_DIGEST,pending_requests=0)
        return args,SimpleNamespace(validate_config=Mock()),a,h,base

    def test_finalize_signals_exact_pidfd_only_and_never_launches(self):
        args,p,a,h,base=self.finalize_fixture()
        with patch.object(b,'verify_backup',return_value={'accepted':True}),patch.object(b,'health',return_value=h),\
             patch.object(b,'inventory_admission',return_value={'devices':[]}),patch.object(b.os,'pidfd_open',return_value=99,create=True) as opened,\
             patch.object(b.signal,'pidfd_send_signal',create=True) as sent,patch.object(b.os,'close') as closed:
            result=b.retire_finalize(args,p,c,a)
        opened.assert_called_once_with(b.RESIDENT['pid']);sent.assert_called_once_with(99,signal.SIGTERM);closed.assert_called_once_with(99)
        self.assertTrue(Path(result['path']).is_file());self.assertFalse((base/'runs').exists())

    def test_busy_resident_is_rejected_before_signal(self):
        args,p,a,h,base=self.finalize_fixture();h['pending_requests']=1
        with patch.object(b,'verify_backup',return_value={'accepted':True}),patch.object(b,'health',return_value=h),\
             patch.object(b.os,'pidfd_open',create=True) as opened,patch.object(b.signal,'pidfd_send_signal',create=True) as sent:
            with self.assertRaisesRegex(RuntimeError,'busy'):b.retire_finalize(args,p,c,a)
        opened.assert_not_called();sent.assert_not_called();self.assertFalse((base/'receipts/config.json').exists())

    def test_timeout_preserves_failure_without_sigkill_or_finalized_config(self):
        args,p,a,h,base=self.finalize_fixture();real_exists=Path.exists
        def exists(path):return True if str(path)=='/proc/'+str(b.RESIDENT['pid']) else real_exists(path)
        with patch.object(b,'verify_backup',return_value={'accepted':True}),patch.object(b,'health',return_value=h),\
             patch.object(b,'inventory_admission',return_value={'devices':[]}),patch.object(b.os,'pidfd_open',return_value=99,create=True),\
             patch.object(b.signal,'pidfd_send_signal',create=True) as sent,patch.object(b.os,'close'),\
             patch.object(b.time,'monotonic',side_effect=[0,2]),patch.object(Path,'exists',exists):
            with self.assertRaisesRegex(RuntimeError,'no SIGKILL'):b.retire_finalize(args,p,c,a)
        sent.assert_called_once_with(99,signal.SIGTERM)
        self.assertTrue((base/'receipts/resident-retirement-timeout.json').exists());self.assertFalse((base/'receipts/config.json').exists())


if __name__=='__main__':unittest.main()
