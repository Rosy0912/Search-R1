"""Minimal deterministic-prefix CPU fixtures. No SSH or node/model work."""
import io,json,hashlib,types,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).absolute().parent
b=types.ModuleType('resume_backup');b.__file__=str(HERE/'backup.py')
exec(compile((HERE/'backup.py').read_bytes(),b.__file__,'exec'),b.__dict__)
class Fixture:
    def __init__(self, root):
        self.base = root / 'node'; self.base.mkdir()
        self.output = self.base / 'runs'; self.output.mkdir()
        self.cpdir = self.output / 'checkpoints/update-00000089-outer-00008'
        self.cpdir.mkdir(parents=True)
        self.outer = self.output / 'outers/outer-00008'; self.outer.mkdir(parents=True)
        self.configpath = self.base / 'receipts/config.json'; self.configpath.parent.mkdir()
        refs = {}
        for name in ('runtime_manifest', 'source_model_manifest', 'helper_source_manifest'):
            refs[name] = self.save(self.base / 'receipts' / (name + '.json'), {'files': {'source.py': 'bound'}})
        cfg = dict(run_id=b.RUN, paths=dict(base=str(self.base), output_root=str(self.output)), **refs)
        self.cref = self.save(self.configpath, cfg)
        self.rip = self.save(self.output / 'run-identity.json', dict(schema='opsd-online-run-identity-v1',
                             run_id=b.RUN, config_sha256=self.cref['sha256'], config=self.cref))
        pairs = self.save(self.outer / 'sampling-attempt-00001/pairs.jsonl', b'private-fixture\n')
        encoding = self.save(self.outer / 'training/encoding.json', {'rows': []})
        order = self.save(self.outer / 'training/order.json', {'order': []})
        outer = self.save(self.outer / 'outer-manifest.json', dict(run_id=b.RUN,
                         config_sha256=self.cref['sha256'], outer_index=8, pairs=pairs,
                         runtime_identity=refs['runtime_manifest']['sha256'],
                         source_model_identity=refs['source_model_manifest']['sha256']))
        ranks = [dict(rank=i, file=self.save(self.cpdir / ('rank-%d-state.pt' % i), ('state-%d' % i).encode())) for i in range(8)]
        adapter = [self.save(self.cpdir / 'adapter' / name, ('adapter-' + name).encode())
                   for name in ('README.md', 'adapter_config.json', 'adapter_model.safetensors')]
        self.cp = dict(schema='opsd-online-checkpoint-v1', accepted=True, run_id=b.RUN,
                       config_sha256=self.cref['sha256'], optimizer_updates=89,
                       runtime_identity=refs['runtime_manifest']['sha256'],
                       source_model_identity=refs['source_model_manifest']['sha256'],
                       cursor=dict(outer_index=8, usable_pair_count=0, next_pair_cursor=0),
                       protection_reason=['outer_end'], complete_rank_count=8, equal_optimizer_steps=True,
                       ranks=ranks, adapter=dict(path=str(self.cpdir / 'adapter'), files=adapter),
                       training_input=dict(outer_manifest=outer, pairs=pairs, encoding=encoding, order=order))
        self.cpref = self.save(self.cpdir / 'manifest.json', self.cp)

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else b.canonical(value) + b'\n')
        return b.fileproof(path)

    def args(self):
        return (str(self.configpath), self.cref['sha256'], str(self.cpdir / 'manifest.json'), self.cpref['sha256'], b.RUN, 89, 8)

    def inv(self):
        return b.inventory(*self.args())

    def rewrite_cp(self):
        self.cpref = self.save(self.cpdir / 'manifest.json', self.cp)


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.f=Fixture(self.root)
        self.pin=patch.object(b,'CP89_SHA',self.f.cpref['sha256']);self.pin.start()
        self.pub={'manifest':{'sha256':'a'*64},'publication':{'sha256':'b'*64},'source_commit':'c'*40}
        self.value=b.inventory(*self.f.args(),publication=self.pub)
        out=io.BytesIO();b.stream_archive(self.value,out);self.tar=out.getvalue()
    def tearDown(self):self.pin.stop();self.tmp.cleanup()
    def test_prefix_arbitrary_exact_byte_alignment_reconstructs_strict_tar(self):
        for offset in (1,511,512,1377,len(self.tar)):
            out=io.BytesIO();writer=b.PrefixWriter(out,offset,hashlib.sha256(self.tar[:offset]).hexdigest())
            b.stream_archive(self.value,writer);proof=writer.finish()
            self.assertEqual(out.getvalue(),self.tar[offset:]);self.assertEqual(proof['total_archive_bytes'],len(self.tar))
        full=self.root/'full.tar';full.write_bytes(self.tar)
        actual,_=b.extract_verified(full,self.root/'extract',{x['relative']:x for x in self.value['files']})
        self.assertEqual(len(actual),18)
    def test_wrong_prefix_sha_outputs_zero_suffix(self):
        out=io.BytesIO();writer=b.PrefixWriter(out,1377,'0'*64)
        with self.assertRaises(b.Rejected):b.stream_archive(self.value,writer)
        self.assertEqual(out.getvalue(),b'')
    def test_prefix_past_eof_and_interrupted_prefix_never_accept(self):
        out=io.BytesIO();writer=b.PrefixWriter(out,len(self.tar)+1,hashlib.sha256(self.tar).hexdigest())
        b.stream_archive(self.value,writer)
        with self.assertRaises(b.Rejected):writer.finish()
        self.assertEqual(out.getvalue(),b'')
        writer=b.PrefixWriter(io.BytesIO(),1377,'0'*64);writer.write(self.tar[:100])
        with self.assertRaises(b.Rejected):writer.finish()
    def test_changed_source_inventory_or_stream_reject(self):
        changed=dict(self.value,total_bytes=self.value['total_bytes']+1)
        with self.assertRaises(b.Rejected):b.match_inventory(changed,self.value)
        row=self.value['files'][0];Path(row['path']).write_bytes(b'changed')
        with self.assertRaises(b.Rejected):b.stream_archive(self.value,b.PrefixWriter(io.BytesIO(),1377,'0'*64))
    def test_safe_copy_preserves_original_and_disk_hash(self):
        original=self.root/'original.partial';original.write_bytes(self.tar[:1377]);fp=b.fileproof(original)
        before=original.read_bytes();copy=self.root/'copy.partial';src,dst=b.copy_prefix(original,fp,copy)
        self.assertEqual(src['sha256'],dst['sha256']);self.assertEqual(copy.read_bytes(),before);self.assertEqual(original.read_bytes(),before)
        with self.assertRaises(FileExistsError):b.copy_prefix(original,fp,copy)
    def test_unsafe_extract_rejects_suffix_after_end(self):
        path=self.root/'unsafe.tar';path.write_bytes(self.tar+b'X'*512)
        with self.assertRaises(b.Rejected):b.extract_verified(path,self.root/'unsafeout',{x['relative']:x for x in self.value['files']})
    def args(self):
        offset=1377;partial=self.root/'original.partial';partial.write_bytes(self.tar[:offset])
        raw=b.canonical(self.value)+b'\n';original=self.root/'inventory.json';original.write_bytes(raw)
        return types.SimpleNamespace(original_inventory=str(original),original_inventory_sha256=hashlib.sha256(raw).hexdigest(),
            original_partial=str(partial),prefix_bytes=offset,prefix_sha256=hashlib.sha256(partial.read_bytes()).hexdigest(),
            config=self.f.args()[0],config_sha256=self.f.args()[1],checkpoint=self.f.args()[2],checkpoint_sha256=self.f.args()[3],
            destination=str(self.root/'resumed'),margin_gib=5),raw
    def run_child(self,rc):
        args,raw=self.args();before=Path(args.original_partial).read_bytes()
        class Child:
            pid=123;stdin=None;returncode=rc
            def communicate(self,data):pass
            def wait(self):return self.returncode
        def popen(command,**kwargs):
            out=kwargs['stdout'];writer=b.PrefixWriter(out,args.prefix_bytes,args.prefix_sha256)
            b.stream_archive(self.value,writer);writer.finish();return Child()
        reply=types.SimpleNamespace(returncode=0,stdout=b.canonical(self.value),stderr=b'')
        with patch.object(b,'ORIGINAL_PARTIAL_BYTES',args.prefix_bytes),patch.object(b,'ORIGINAL_PARTIAL_SHA',args.prefix_sha256),patch.object(b,'command',return_value=['never-executed']),patch.object(b.subprocess,'run',return_value=reply),patch.object(b.subprocess,'Popen',side_effect=popen),patch.object(b.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**12)):
            if rc:
                with self.assertRaises(b.Rejected):b.resume(args,self.pub)
            else:self.assertTrue(b.resume(args,self.pub)['accepted'])
        self.assertEqual(Path(args.original_partial).read_bytes(),before)
        self.assertEqual(json.loads((Path(args.destination)/'suffix-wait.json').read_text())['actual_returncode'],rc)
        return args
    def test_simulated_zero_transport_rebuild_and_disk_readback(self):
        a=self.run_child(0);proof=json.loads((Path(a.destination)/'complete-sha-verified.json').read_text())
        self.assertTrue(proof['independent_local_disk_readback']);self.assertEqual(len(proof['files']),18)
    def test_simulated_255_keeps_both_partials_without_acceptance(self):
        a=self.run_child(255);root=Path(a.destination)
        self.assertTrue((root/'checkpoint.tar.partial').exists());self.assertTrue((root/'failed.json').exists())
        self.assertFalse((root/'complete-sha-verified.json').exists())

if __name__=='__main__':unittest.main()
