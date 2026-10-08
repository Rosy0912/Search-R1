"""CPU concurrency/input-boundary counterexamples, no native admission."""
import ast,hashlib,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
import first5_helpers as h
import guard_snapshot as g
ORIGINAL=B.parent/'fresh200-deployment-controller-initfix/tools/guard_journal.py'
class Snapshot(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.patch=patch.object(h,'ROOT',self.root);self.patch.start();self.source=self.root/'live';self.target=self.root/'snapshot'
 def tearDown(self):self.patch.stop();self.tmp.cleanup()
 def test_partialline_captures_complete_prefix_and_metadata(self):
  self.source.write_bytes(b'{"line":1}\npartial');p=h.immutable_prefix(self.source,self.target)
  self.assertEqual(self.target.read_bytes(),b'{"line":1}\n');self.assertEqual(p['partial_tail_bytes'],7);self.assertEqual(p['observed_size'],18)
  h.verify_snapshot_prefix(p)
  with self.source.open('ab')as f:f.write(b' completed\n')
  h.verify_snapshot_prefix(p)
 def test_concurrent_append_does_not_require_writer_quiescence(self):
  self.source.write_bytes(b'{"start":1}\n'*10000);stop=threading.Event()
  def append():
   with self.source.open('ab',buffering=0)as f:
    while not stop.is_set():f.write(b'{"append":2}\n');time.sleep(.0001)
  t=threading.Thread(target=append);t.start()
  try:
   p=h.immutable_prefix(self.source,self.target);time.sleep(.002);h.verify_snapshot_prefix(p)
   self.assertGreaterEqual(self.source.stat().st_size,p['observed_size']);self.assertEqual(len(self.target.read_bytes()),p['copied_bytes'])
  finally:stop.set();t.join()
 def test_source_replacement_rejected(self):
  self.source.write_bytes(b'one\n');p=h.immutable_prefix(self.source,self.target);self.source.rename(self.root/'old');self.source.write_bytes(b'one\n')
  with self.assertRaisesRegex(ValueError,'replaced'):h.verify_snapshot_prefix(p)
 def test_source_truncation_rejected(self):
  self.source.write_bytes(b'one\ntail');p=h.immutable_prefix(self.source,self.target);self.source.write_bytes(b'one\n')
  with self.assertRaisesRegex(ValueError,'truncated'):h.verify_snapshot_prefix(p)
 def test_source_snapshot_symlink_and_snapshot_mutation_rejected(self):
  self.source.write_bytes(b'one\n');link=self.root/'link';link.symlink_to(self.source)
  with self.assertRaises(ValueError):h.immutable_prefix(link,self.target)
  p=h.immutable_prefix(self.source,self.target);self.target.write_bytes(b'two\n')
  with self.assertRaisesRegex(ValueError,'Snapshot bytes changed'):h.verify_snapshot_prefix(p)
 def test_sameinode_prefix_edit_rejected(self):
  self.source.write_bytes(b'one\n');p=h.immutable_prefix(self.source,self.target);self.source.write_bytes(b'two\n')
  with self.assertRaisesRegex(ValueError,'prefix differs'):h.verify_snapshot_prefix(p)
 def test_no_full_line_or_existing_destination_rejected(self):
  self.source.write_bytes(b'partial')
  with self.assertRaises(ValueError):h.immutable_prefix(self.source,self.target)
  self.source.write_bytes(b'full\n');h.immutable_prefix(self.source,self.target)
  with self.assertRaises(FileExistsError):h.immutable_prefix(self.source,self.target)
 def test_original_assessment_state_machine_body_exactly_preserved(self):
  old=ast.parse(ORIGINAL.read_text());new=ast.parse((B/'guard_snapshot.py').read_text())
  a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='replay');b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='replay_snapshot')
  def body(fn):
   start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='uuids'for t in n.targets))
   end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(x,ast.Constant)and x.value=='Artifacts changed during replay'for x in n.value.args))
   return ast.dump(ast.Module(body=fn.body[start:end],type_ignores=[]),include_attributes=False)
  self.assertEqual(body(a),body(b));self.assertEqual(g.ORIGINAL_GUARD_SHA,h.digest(ORIGINAL))
 def test_unsafe_history_not_removed_by_snapshot_adapter(self):
  run=self.root/'runs/fixture-3315';raw=run/'logs/capture';raw.mkdir(parents=True);(self.root/'tools').mkdir()
  self.source=run/'resource-watch.jsonl';self.source.write_text(json.dumps(dict(run=str(run),slurm_job_id='3315',time=1,event='UNSAFE_UNSUPPORTED'))+'\n')
  snapshot=raw/'journal';p=h.immutable_prefix(self.source,snapshot);watcher=self.root/'tools/resource_watch.py';watcher.write_text('# fixture watcher not GPU proof\n');original=self.root/'tools/guard_journal.py';original.write_bytes(ORIGINAL.read_bytes())
  with patch.object(g,'NEW_ROOT',self.root),patch.object(g,'WATCH_SHA',h.digest(watcher)),patch.object(g,'load_bound_watcher',return_value=SimpleNamespace(load_runtime_anchor=lambda *a,**kw:dict(source_commit=g.FRESH200_SOURCE))):
   with self.assertRaisesRegex(g.JournalError,'Unsupported/unsafe'):g.replay_snapshot(snapshot,3315,str(run),g.FRESH200_SOURCE,watcher,p)
if __name__=='__main__':unittest.main()
