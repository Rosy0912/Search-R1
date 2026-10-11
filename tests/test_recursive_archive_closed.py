import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('archive_closed',Path(__file__).parents[1]/'release/recursive_bran_b/archive_closed.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def inv(data=b'known exact payload'):
    import hashlib
    return dict(protocol=a.PROTOCOL,run_id=a.RUN,source_commit=a.SOURCE,config_sha256=a.CONFIG,
                data_identity=a.DATA,kind='output',step=1,directories=['output/trajectories/outer-000001-attempt-000001'],
                members=[dict(name='output/trajectories/outer-000001-attempt-000001/actor-inputs.pt',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())])


def archive(j, entries):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w',format=tarfile.PAX_FORMAT) as t:
        b=a.raw(j);h=tarfile.TarInfo('inventory.json');h.size=len(b);t.addfile(h,io.BytesIO(b))
        for name,data,kind in entries:
            h=tarfile.TarInfo(name);h.size=len(data);h.type=kind
            t.addfile(h,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
    stream.seek(0);return stream


class ArchiveTests(unittest.TestCase):
    def receive(self,j,entries):
        with tempfile.TemporaryDirectory() as d:
            result=a.receive(Path(d),archive(j,entries))
            self.assertEqual(result['status'],'ALL_MEMBERS_FRESH_SHA_ACCEPTED')
            self.assertEqual(a.sha(Path(result['destination'])/'files'/j['members'][0]['name']),j['members'][0]['sha256'])
            return result

    def test_exact_receive_and_readback(self):
        j=inv();self.receive(j,[(j['members'][0]['name'],b'known exact payload',tarfile.REGTYPE)])

    def test_paths(self):
        for n in ('../x','/x','a/../x','a//x','a/./x','a\\x',''):
            with self.subTest(n=n),self.assertRaises(ValueError):a.safe_name(n)

    def test_duplicate_inventory(self):
        j=inv();j['members']*=2
        with self.assertRaises(ValueError):a.validate_inventory(j)

    def test_foreign_identity(self):
        for key in ('run_id','source_commit','config_sha256','data_identity'):
            j=inv();j[key]='foreign'
            with self.subTest(key=key),self.assertRaises(ValueError):a.validate_inventory(j)

    def test_bad_members(self):
        j=inv();n=j['members'][0]['name'];data=b'known exact payload'
        for entries in ([],[(n,data,tarfile.REGTYPE)]*2,[(n,b'wrong same length!!',tarfile.REGTYPE)],
                        [('../x',data,tarfile.REGTYPE)],[(n,b'',tarfile.SYMTYPE)],[(n,b'',tarfile.LNKTYPE)]):
            with self.subTest(entries=entries),tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):
                a.receive(Path(d),archive(j,entries))

    def test_existing_destination_never_overwritten(self):
        j=inv();entries=[(j['members'][0]['name'],b'known exact payload',tarfile.REGTYPE)]
        with tempfile.TemporaryDirectory() as d:
            a.receive(Path(d),archive(j,entries))
            with self.assertRaises(ValueError):a.receive(Path(d),archive(j,entries))

    def test_reverify_and_changed_destination(self):
        j=inv();entries=[(j['members'][0]['name'],b'known exact payload',tarfile.REGTYPE)]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.assertFalse(a.verify_destination(root,j)['exists'])
            r=a.receive(root,archive(j,entries));self.assertTrue(a.verify_destination(root,j)['exists'])
            (Path(r['destination'])/'files'/j['members'][0]['name']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):a.verify_destination(root,j)

    def test_ancestor_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'actual').mkdir();(root/'actual/child').mkdir();(root/'link').symlink_to(root/'actual',target_is_directory=True)
            with self.assertRaises(ValueError):a.files_under(root,['link/child'])

    def test_pax_large_size_supported(self):
        # Test the >8GB header encoding without creating a local huge payload.
        h=tarfile.TarInfo('large.pt');h.size=12_343_852_982
        b=h.tobuf(format=tarfile.PAX_FORMAT)
        self.assertIn(b'size=12343852982',b)

    def test_current_previous_not_selected(self):
        with patch.object(a,'fixed_source'),patch.object(a,'latest',return_value=3):
            for step in (2,3):
                with self.assertRaises(ValueError):a.select_dirs(Path('/unused'),'output',step)

    def test_no_nonprotected_or_final_prune(self):
        with patch.object(a,'fixed_source'),patch.object(a,'latest',return_value=202),patch.object(a,'boundary',return_value={}):
            for step in (1,49,200):
                with self.assertRaises(ValueError):a.select_dirs(Path('/unused'),'checkpoint',step)

    def test_bad_receipt_prevents_source_touch(self):
        with self.assertRaises((ValueError,KeyError)):a.prune(Path('/unused'),inv(),{})

    def test_changed_source_prevents_delete(self):
        j=inv();ident=a.archive_id(j)
        r=dict(protocol=a.PROTOCOL,archive_id=ident,inventory_sha256=__import__('hashlib').sha256(a.raw(j)).hexdigest(),
               status='ALL_MEMBERS_FRESH_SHA_ACCEPTED',destination=str(a.DEST/ident),members=1,bytes=j['members'][0]['bytes'])
        with patch.object(a,'inventory',return_value={'changed':True}),self.assertRaises(ValueError):a.prune(Path('/unused'),j,r)


if __name__=='__main__':unittest.main()
