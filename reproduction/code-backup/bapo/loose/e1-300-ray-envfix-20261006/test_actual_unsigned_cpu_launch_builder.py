import base64,copy,hashlib,importlib.util,json,pathlib,shutil,tempfile,unittest
B=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('builder',B/'build-unsigned-cpu-launch-candidate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RAW=pathlib.Path('work/evidence/ray-envfix-target-cpu-raw');TRANSFER=pathlib.Path('work/evidence/ray-envfix-target-cpu-raw-transfer.json');REPO=pathlib.Path('work/bapo-e1-300-ray-envfix');CONFIG=REPO/'verl/trainer/config/ppo_bapo_e1_300.yaml'
class Test(unittest.TestCase):
 def build(self,raw=RAW,transfer=TRANSFER,tools=B/'deployment-tools'):return m.build(tools,raw,REPO,CONFIG,transfer)
 def rebound(self,name,change):
  with tempfile.TemporaryDirectory()as td:
   d=pathlib.Path(td);raw=d/'raw';shutil.copytree(RAW,raw);capture=json.loads(TRANSFER.read_bytes());p=raw/name;j=json.loads(p.read_bytes());change(j);p.write_text(json.dumps(j));row=capture['files'][name];binary=p.read_bytes();row.update(bytes=len(binary),sha256=hashlib.sha256(binary).hexdigest(),base64=base64.b64encode(binary).decode());transfer=d/'transfer.json';transfer.write_text(json.dumps(capture))
   with self.assertRaises((ValueError,AssertionError,KeyError)):self.build(raw,transfer)
 def test_actual_unsigned_scopes(self):
  manifest,cpu,r=self.build();self.assertEqual(cpu['passed'],395);self.assertEqual(r['tools_file_count'],36);self.assertEqual(r['title_dependency_preflight']['sha256'],'d10fb8f5de5bbb282e5c7a5096bd563457de567e149b09eb71b7279f8a84fd61');self.assertFalse(r['actual_gpu_accepted']);self.assertFalse(r['production_training_permission_signed']);self.assertFalse(r['launch_authorized']);self.assertEqual(r['status'],'CANDIDATE_REQUIRES_INDEPENDENT_E1_LAUNCH_REVIEW')
 def test_rebound_cpu_nonzero_exit(self):self.rebound('logs/cpu-gate-durable-exit.json',lambda j:j.update(exit_code=1))
 def test_rebound_cpu_old_manifest(self):self.rebound('logs/cpu-gate-durable-exit.json',lambda j:j.update(tools_manifest_sha256='f0543703d34213b319f51ab93efb45a57c869372bc18596dee3eca93e1c5ee55'))
 def test_rebound_old_runtime_root(self):self.rebound('logs/runtime-executable.json',lambda j:j.update(root='/old/root'))
 def test_rebound_title_pid_reuse(self):self.rebound('logs/title-dependency-preflight.json',lambda j:j['acks'][12]['snapshot'].update(start_ticks=j['original_process']['start_ticks']+1))
 def test_rebound_title_extra_argv(self):
  def change(j):
   s=j['acks'][5]['snapshot'];s['command']=[j['acks'][5]['title'],'extra'];s['cmdline_base64']=base64.b64encode(('\0'.join(s['command'])).encode()).decode()
  self.rebound('logs/title-dependency-preflight.json',change)
 def test_rebound_title_missing_ack(self):self.rebound('logs/title-dependency-preflight.json',lambda j:j['acks'].pop())
 def test_rebound_guardian_nonzero_exit(self):self.rebound('logs/guardian-tests-durable-exit.json',lambda j:j.update(exit_code=42))
 def test_added_tool_rejected(self):
  with tempfile.TemporaryDirectory()as td:
   tools=pathlib.Path(td)/'tools';shutil.copytree(B/'deployment-tools',tools);(tools/'extra.py').write_text('pass')
   with self.assertRaises(ValueError):self.build(tools=tools)
 def test_source_guard_pin_mismatch_rejected(self):
  with tempfile.TemporaryDirectory()as td:
   repo=pathlib.Path(td)/'repo';p=repo/'search_r1/baselines/recovery.py';p.parent.mkdir(parents=True);p.write_text((REPO/'search_r1/baselines/recovery.py').read_text().replace('c0334988153496305b57aaecafb908d28d14705791b4632792f03e135c96e8c3','16e14271d0f7710d7c86ce00e2a5e10bd0bdd754900be7951b5a0224e9574555'))
   with self.assertRaisesRegex(ValueError,'pins do not close'):m.build(B/'deployment-tools',RAW,repo,CONFIG,TRANSFER)
 def test_rebound_clobbered_actual_proc_env(self):self.rebound('logs/title-dependency-preflight.json',lambda j:j['acks'][3]['snapshot']['process_markers'].update(original_selected_environ_base64='AA=='))
 def test_rebound_bool_capacity(self):self.rebound('logs/title-dependency-preflight.json',lambda j:j.update(argv_capacity_bytes=True))
 def test_rebound_foreign_env_value(self):self.rebound('logs/title-dependency-preflight.json',lambda j:j['expected_process_markers'].update(SLURM_JOB_ID='3284'))
 def test_rebound_transferred_missing_original(self):
  with tempfile.TemporaryDirectory()as td:
   p=pathlib.Path(td)/'transfer.json';j=json.loads(TRANSFER.read_bytes());j['files'].pop('logs/runtime-capture.log');p.write_text(json.dumps(j))
   with self.assertRaisesRegex(ValueError,'Exact11'):self.build(transfer=p)
if __name__=='__main__':unittest.main(verbosity=2)
