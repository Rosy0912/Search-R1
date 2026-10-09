"""Small temporary GNU/USTAR and unsafe-scope fixtures; no actual backup access."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import reclaim as r


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_file(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return sha(path.read_bytes())


class ReclaimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def package(self, fmt='GNU', entries=None):
        base = self.root/fmt; base.mkdir(); extracted = base/'verified-files'; extracted.mkdir()
        data = {'state/rank0.pt': b'exact Adam and RNG'}
        entries = [(n, v, tarfile.REGTYPE) for n, v in data.items()] if entries is None else entries
        for n, v in data.items():
            p = extracted/n; p.parent.mkdir(exist_ok=True); p.write_bytes(v)
        archive = base/'checkpoint.tar'
        with tarfile.open(archive, 'w', format=tarfile.GNU_FORMAT if fmt=='GNU' else tarfile.USTAR_FORMAT) as t:
            for n, v, kind in entries:
                info = tarfile.TarInfo(n); info.size = len(v); info.type = kind
                if kind == tarfile.SYMTYPE: info.linkname = '/outside'
                t.addfile(info, io.BytesIO(v))
        rows = [dict(relative=n, path='/original/'+n, bytes=len(v), sha256=sha(v)) for n, v in data.items()]
        invsha = json_file(base/'inventory.json', dict(accepted=True, run_id=r.RUN, files=rows))
        files = {n:dict(local_path=str(extracted/n), bytes=len(v), sha256=sha(v)) for n, v in data.items()}
        ap = dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()))
        rcsha = json_file(base/'complete-sha-verified.json', dict(accepted=True, run_id=r.RUN, files=files, archive=ap))
        return dict(name=fmt, root=str(base), receipt_sha=rcsha, inventory='inventory.json', inventory_sha=invsha,
                    archive='checkpoint.tar', archive_sha=ap['sha256'], archive_bytes=ap['bytes'], member_count=1,
                    payload_bytes=sum(map(len, data.values())), format=fmt)

    def plan(self, scopes):
        source = dict(fixture=True)
        with patch.object(r, 'SCOPES', tuple(scopes)), patch.object(r, 'no_open_files'):
            plan = r.prepare(self.root/'audit', source)
        return plan, source

    def test_GNU_and_USTAR_exact_reclamation_preserves_archives_and_receipts(self):
        scopes = [self.package('GNU'), self.package('USTAR')]; plan, source = self.plan(scopes)
        before = {s['name']:(Path(s['root'])/s['archive']).read_bytes() for s in scopes}
        with patch.object(r, 'SCOPES', tuple(scopes)), patch.object(r, 'no_open_files'):
            done = r.execute(plan['path'], plan['sha256'], source)
            again = r.execute(plan['path'], plan['sha256'], source)
        self.assertEqual(done, again)
        for s in scopes:
            root=Path(s['root']); self.assertEqual((root/s['archive']).read_bytes(), before[s['name']])
            self.assertTrue((root/'complete-sha-verified.json').is_file())
            self.assertTrue((root/'verified-files/state').is_dir())
            self.assertFalse((root/'verified-files/state/rank0.pt').exists())

    def test_path_escape_and_link_tar_refused(self):
        for name, kind in [('../outside', tarfile.REGTYPE), ('state/rank0.pt', tarfile.SYMTYPE)]:
            with self.subTest(name=name):
                # Separate fixture directories for each refused archive.
                target=self.root/('escape' if kind==tarfile.REGTYPE else 'link'); target.mkdir()
                with patch.object(self, 'root', target): s=self.package('GNU', [(name,b'x',kind)])
                with self.assertRaises(RuntimeError): r.verify_archive(s, r.inventory(s)[0])

    def test_extra_member_refused(self):
        s=self.package('USTAR', [('state/rank0.pt',b'exact Adam and RNG',tarfile.REGTYPE),('extra',b'x',tarfile.REGTYPE)])
        with self.assertRaises(RuntimeError): r.verify_archive(s, r.inventory(s)[0])

    def test_loose_symlink_hardlink_and_extra_file_refused(self):
        for kind in ('symlink','hardlink','extra'):
            case=self.root/kind;case.mkdir()
            with patch.object(self, 'root', case):
                s=self.package(); p=Path(s['root'])/'verified-files/state/rank0.pt'
                if kind=='symlink': p.unlink();p.symlink_to(case/'outside')
                elif kind=='hardlink': os.link(p, case/'alias')
                else: (p.parent/'extra').write_bytes(b'x')
                with patch.object(r,'SCOPES',(s,)),patch.object(r,'no_open_files'),self.assertRaises((RuntimeError,FileNotFoundError)):
                    r.prepare(case/'audit',{})

    def test_changed_metadata_or_hash_rejects_before_unlink(self):
        s=self.package(); plan,source=self.plan([s]);p=Path(s['root'])/'verified-files/state/rank0.pt'
        p.write_bytes(b'changed bytes')
        with patch.object(r,'SCOPES',(s,)),patch.object(r,'no_open_files'),self.assertRaises(RuntimeError):
            r.execute(plan['path'],plan['sha256'],source)
        self.assertTrue(p.exists());self.assertFalse((self.root/'audit/execution-intent.json').exists())

    def test_busy_or_unknown_lsof_refused(self):
        for rc,out,err in [(0,'p123\nnfile\n',''),(1,'','inspection warning')]:
            with patch.object(r.shutil,'which',return_value='/usr/sbin/lsof'),patch.object(r.subprocess,'run',return_value=type('R',(),dict(returncode=rc,stdout=out,stderr=err))()),self.assertRaises(RuntimeError):
                r.no_open_files(['/fixed/file'])

    def test_absence_needs_exact_durable_pending(self):
        s=self.package();plan,source=self.plan([s]);p=Path(s['root'])/'verified-files/state/rank0.pt';p.unlink()
        with patch.object(r,'SCOPES',(s,)),patch.object(r,'no_open_files'),self.assertRaises(FileNotFoundError):
            r.execute(plan['path'],plan['sha256'],source)
        self.assertFalse((self.root/'audit/execution-intent.json').exists())

    def test_crash_after_pending_records_absence_without_claiming_new_unlink(self):
        s=self.package();plan,source=self.plan([s]);value,pref=r.checked_json(plan['path'],plan['sha256'])
        output=self.root/'audit';journal=output/'journal';journal.mkdir()
        r.write_json(output/'execution-intent.json',dict(plan=pref,source=source,files=value['files']))
        r.write_json(journal/'000-pending.json',dict(plan=pref,index=0,file=value['files'][0]))
        Path(value['files'][0]['path']).unlink()
        with patch.object(r,'SCOPES',(s,)),patch.object(r,'no_open_files'):
            done=r.execute(plan['path'],plan['sha256'],source)
        accepted,_=r.checked_json(done['path'],done['sha256'])
        self.assertEqual(accepted['journal'][0]['outcome'],'absent_after_durable_pending_actual_unlink_not_inferred')

    def test_unknown_scope_and_changed_plan_sha_refused(self):
        s=self.package();plan,source=self.plan([s])
        with patch.object(r,'SCOPES',()),self.assertRaises(RuntimeError): r.execute(plan['path'],plan['sha256'],source)
        with self.assertRaises(RuntimeError): r.execute(plan['path'],'0'*64,source)


if __name__ == '__main__':
    unittest.main()
