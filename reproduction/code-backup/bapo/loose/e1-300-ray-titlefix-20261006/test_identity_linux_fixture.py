"""Identity fixture grammar and safety tests; actual /proc signals run later Linux only."""
import ast,copy,importlib.util,pathlib,unittest
P=pathlib.Path(__file__).with_name('identity-linux-fixture.py');spec=importlib.util.spec_from_file_location('fixture',P);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
class W:
 def certificate_identity(self,c,a):
  if c['command'][0]not in ('ray::IDLE','ray::WorkerDict.actor_rollout_update_actor'):raise ValueError('unapproved')
  return {k:v for k,v in c.items()if k not in ('command','state')}
 def identity_difference(self,b,a,anchor):return dict(immutable_fields={k:[b.get(k),a.get(k)]for k in set(b)|set(a)if k not in('command','state')and b.get(k)!=a.get(k)},observed_fields={'command':[b['command'],a['command']]}if b['command']!=a['command']else{})
class Test(unittest.TestCase):
 def events(self):
  c=dict(pid=20,start_ticks=100,command=['ray::IDLE',''],state='S');after=dict(c,command=['ray::WorkerDict.actor_rollout_update_actor','']);stopped=dict(after,state='T')
  return [dict(event='owned_title_transition',pid=20,before_certificate=c,after_certificate=after),dict(event='owned_freeze_signal',pid=20,certificate=after),dict(event='owned_freeze_stable',pids=[20],epochs={'20':100},before_certificates={'20':stopped},after_certificates={'20':stopped}),dict(event='owned_resume_signal',pid=20,signal=int(f.signal.SIGCONT),start_ticks=100)]
 def test_complete_grammar(self):self.assertEqual(f.strict_identity_replay(self.events(),W(),{},20,99),dict(stopped=1,resumed=1,title_transitions=1))
 def test_replay_rejects_epoch_foreign_unfinished_and_nonstopped(self):
  for mutate in [lambda e:e[1].update(pid=99),lambda e:e[0]['after_certificate'].update(start_ticks=101),lambda e:e.pop(),lambda e:e[2]['after_certificates']['20'].update(state='S'),lambda e:e[3].update(signal=9),lambda e:e[2].update(epochs={'20':101})]:
   rows=copy.deepcopy(self.events());mutate(rows)
   with self.assertRaises((ValueError,KeyError)):f.strict_identity_replay(rows,W(),{},20,99)
 def test_stable_then_new_stop_requires_new_full_snapshot(self):
  rows=self.events();cert=dict(rows[1]['certificate'],pid=21,start_ticks=101);rows.insert(3,dict(event='owned_freeze_signal',pid=21,certificate=cert));rows.append(dict(event='owned_resume_signal',pid=21,signal=int(f.signal.SIGCONT),start_ticks=101))
  with self.assertRaises(ValueError):f.strict_identity_replay(rows,W(),{},20,99)
 def test_stop_or_title_after_resume_refused(self):
  for extra in [dict(event='owned_freeze_signal',pid=21,certificate=dict(self.events()[1]['certificate'],pid=21)),self.events()[0]]:
   with self.subTest(extra=extra['event']),self.assertRaises(ValueError):f.strict_identity_replay(self.events()+[extra],W(),{},20,99)
 def test_hanging_transition_or_changed_following_stop_refused(self):
  for mutate in [lambda rows:rows[1]['certificate'].update(start_ticks=101),lambda rows:rows.pop(1),lambda rows:rows.insert(1,copy.deepcopy(rows[0]))]:
   rows=copy.deepcopy(self.events());mutate(rows)
   with self.assertRaises(ValueError):f.strict_identity_replay(rows,W(),{},20,99)
 def test_replay_cannot_grant_gpu_resource_permission(self):
  rows=self.events();rows.append(dict(event='recovery_sample'))
  with self.assertRaises(ValueError):f.strict_identity_replay(rows,W(),{},20,99)
 def test_actual_negative_reader_has_no_signal_operation(self):
  tree=ast.parse(P.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='raw_process_observation');self.assertNotIn('pidfd_send_signal',ast.unparse(node));self.assertIn("p / 'environ'",ast.unparse(node));self.assertIn("p / 'exe'",ast.unparse(node))
 def test_real_runtime_and_actual_setproctitle_required(self):
  text=P.read_text();self.assertIn('import setproctitle',text);self.assertIn('setproctitle.setproctitle',text);self.assertIn('verify_current=True',text);self.assertIn('freezer.freeze()',text);self.assertIn('freezer.thaw()',text);self.assertNotIn('timeout=60',text)
if __name__=='__main__':unittest.main(verbosity=2)
