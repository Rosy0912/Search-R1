from pathlib import Path
import sys,types,unittest,tempfile,io,tarfile,gzip,json,hashlib,copy
from unittest.mock import patch
D=Path(sys.argv[1])
p=D/'test_outer26.py';m=types.ModuleType('independent_outer26_tests');m.__file__=str(p)
exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),m.__dict__)
a,c,k=m.a,m.c,m.k
class IndependentCompositionTests(unittest.TestCase):
 def test_exact21_member_tar_gzip_full_positive_roundtrip(self):
  authority=json.loads((D/'AUTHORITY-v2.original.json').read_bytes())
  contents={n:(b'{}\n'*(2560 if n=='student.jsonl' else 64)) if n.endswith('.jsonl') else b'{}\n' for n in c.NAMES}
  identity=dict(source_commit=c.SOURCE,config_sha256=c.CONFIG,common_manifest_sha256=c.COMMON,job_id='3304',driver_pid=c.DRIVER[0],outer_step=26,directory=str(c.DIR),profile='multitraj200')
  records={n:dict(origin_path=str(c.DIR/n),sha256=k.sha(raw),bytes=len(raw),uid=0,mode=0o444,device=1,inode=i+1) for i,(n,raw) in enumerate(contents.items())}
  ctx=dict(driver=dict(pid=c.DRIVER[0],start_ticks=c.DRIVER[1]),workers=[dict(pid=p,start_ticks=s) for p,s in c.WORKERS],git=dict(head=dict(exit_code=0,stdout=c.SOURCE+'\n'),clean=dict(exit_code=0,stdout='')))
  inv=dict(version=1,kind='OBSERVED_COMPLETED_OUTER26_ARCHIVE_INVENTORY_NOT_TRAINING_CHECKPOINT_MANIFEST',identity=identity,authority=authority,members=records,raw_bytes=sum(len(b) for b in contents.values()),original_context_before=ctx,original_context_after=ctx)
  buf=io.BytesIO()
  with tarfile.open(fileobj=buf,mode='w',format=tarfile.USTAR_FORMAT) as tar:
   for n,raw in list(contents.items())+[('observed-inventory.json',k.canonical(inv)+b'\n')]:
    h=tarfile.TarInfo(n);h.size=len(raw);h.mode=0o444;tar.addfile(h,io.BytesIO(raw))
  blob=gzip.compress(buf.getvalue())
  with tempfile.TemporaryDirectory() as tmp,patch.object(c,'PAIR_SHA',records['paired-manifest.json']['sha256']):
   root=Path(tmp).resolve();gz=root/'original.gz';gz.write_bytes(blob)
   wait=dict(kind='ACTUAL_READ_ONLY_OUTER26_COLLECTION_WAIT',waited=True,returncode=0,runtime_sha256='SYNTHETIC_NOT_REAL_COLLECTOR',authority_sha256=c.AUTHORITY_SHA,remote_host='h100-10',compressed_stdout_sha256=k.sha(blob))
   w=root/'wait.json';w.write_bytes(k.canonical(wait));out=root/'copied'
   a.extract_once(k,c,gz,out,w,k.sha(w.read_bytes()),'SYNTHETIC_NOT_REAL_COLLECTOR')
   for n,raw in contents.items():self.assertEqual((out/n).read_bytes(),raw)
   self.assertEqual(len([p for p in out.iterdir() if p.is_file()]),22)
   self.assertEqual(json.loads((out/'collection-copy-qualification.json').read_bytes())['kind'],'LOCAL_ORIGINAL_BYTE_COPY_ONLY')
 def test_unwaited_zero_rejects_before_output_reservation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();gz=root/'original.gz';gz.write_bytes(b'SYNTHETIC')
   wait=dict(kind='ACTUAL_READ_ONLY_OUTER26_COLLECTION_WAIT',waited=False,returncode=0,runtime_sha256='fixture',authority_sha256=c.AUTHORITY_SHA,remote_host='h100-10',compressed_stdout_sha256=k.sha(gz.read_bytes()))
   w=root/'wait.json';w.write_bytes(k.canonical(wait));out=root/'copied'
   with self.assertRaises(ValueError):a.extract_once(k,c,gz,out,w,k.sha(w.read_bytes()),'fixture')
   self.assertFalse(out.exists())
s=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(m.Boundaries),unittest.defaultTestLoader.loadTestsFromTestCase(IndependentCompositionTests)])
r=unittest.TextTestRunner(verbosity=2).run(s)
sys.exit(not r.wasSuccessful())
