"""Two focused local CPU regressions; no production archive/migration."""
from pathlib import Path
import tempfile
import types
import unittest

HERE=Path(__file__).absolute().parent
r=types.ModuleType('recovery_fixture');r.__file__=str(HERE/'recover.py')
exec(compile((HERE/'recover.py').read_bytes(),r.__file__,'exec'),r.__dict__)
corepath=HERE.parent/'opsd_ablation_compact_backup_v1/compact.py'


class Recovery(unittest.TestCase):
    def fixture(self,root,resumed=False,foreign=False):
        c=types.ModuleType('compact_fixture');c.__file__=str(corepath)
        exec(compile(corepath.read_bytes(),c.__file__,'exec'),c.__dict__)
        b,_=c.legacy_modules(HERE.parent/'opsd_ablation_outer_backup_v2',
            '4f9ee158ec1c326498898d26ee5b6a1d3212e94d5ecbbd24c8ef7864fa079b69')
        queue=root/'legacy';queue.mkdir()
        package=queue/('a'*64+'-attempt-'+'b'*32);package.mkdir()
        extraction=package/'verified-files';source=extraction/'closure/common/payload-000/adapter.bin'
        source.parent.mkdir(parents=True);source.write_bytes(b'original-toy')
        archive=package/'checkpoint.tar';archive.write_bytes(b'original-tar-toy')
        files={'closure/common/payload-000/adapter.bin':b.fileproof(source)}
        ap=b.fileproof(archive)
        receipt=b.write_json(package/'complete-sha-verified.json',dict(archive=ap,files=files))
        scope=root/'compact';scope.mkdir()
        ip=b.write_json(scope/'node-inventory.json',dict(files=[dict(relative=n,**fp) for n,fp in files.items()]))
        ref=b.write_json(scope/'compact-complete.json',dict(accepted=True))
        value=dict(historical_v2=dict(complete_receipt=receipt,archive=ap,files=files),node_inventory=ip)
        # This isolates the directory regression after the already accepted
        # independent archive proof; it does not claim a new real-XZ audit.
        c.verified_compact=lambda *args,**kwargs:value
        job=dict(run_id='toy',outer_index=0);args=types.SimpleNamespace(legacy_queue_root=str(queue))
        if resumed:
            b.write_json(scope/'legacy-reclaim-plan.json',dict(schema='compact-legacy-local-reclaim-plan-v1',
                compact=ref,job=job,files=[ap]+list(files.values()),preserve_original_receipt_inventory=True,node_files_touched=False))
            source.unlink();archive.unlink()
        extra=extraction/'closure/common/payload-000/foreign.txt'
        if foreign:extra.write_bytes(b'foreign')
        return r.patched_reclaim(c),b,ref,job,args,extraction,scope,package,extra

    def test_normal_and_durable_plan_missing_payload_resume(self):
        for resumed in (False,True):
            with self.subTest(resumed=resumed), tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as name:
                fn,b,ref,job,args,tree,scope,package,_=self.fixture(Path(name),resumed=resumed)
                original=b.fileproof(package/'complete-sha-verified.json')
                result=fn(b,ref,job,args)
                self.assertFalse(tree.exists());self.assertFalse((package/'checkpoint.tar').exists())
                self.assertTrue((scope/'legacy-reclaimed.json').exists())
                b.stable(original);self.assertEqual(result['path'],str(scope/'legacy-reclaimed.json'))
                # An already completed cleanup is idempotent, not rewritten.
                self.assertEqual(fn(b,ref,job,args),result)

    def test_foreign_residual_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as name:
            fn,b,ref,job,args,tree,scope,package,extra=self.fixture(Path(name),resumed=True,foreign=True)
            with self.assertRaises(b.Rejected):fn(b,ref,job,args)
            self.assertEqual(extra.read_bytes(),b'foreign');self.assertTrue(tree.exists())
            self.assertFalse((scope/'legacy-reclaimed.json').exists())

    def test_foreign_empty_directory_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as name:
            fn,b,ref,job,args,tree,scope,package,_=self.fixture(Path(name),resumed=True)
            foreign=tree/'unplanned-empty';foreign.mkdir()
            with self.assertRaises(b.Rejected):fn(b,ref,job,args)
            self.assertTrue(foreign.is_dir());self.assertTrue((tree/'closure/common/payload-000').is_dir())
            self.assertFalse((scope/'legacy-reclaimed.json').exists())


if __name__=='__main__':unittest.main()
