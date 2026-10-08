import ast,base64,importlib.util,json,pathlib,unittest,unittest.mock as mock
B=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('preflight',B/'deployment-tools/actual-title-dependency-preflight.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('fixture',B/'deployment-tools/identity-linux-fixture.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
MARK={'BAPO_RUN_ID':'/own/run','SLURM_JOB_ID':'1','SEARCH_R1_GPU_LEASES':json.dumps([{'uuid':'CPU-'+str(i)}for i in range(8)]),'SPT_NOENV':'1'}
def env(values=MARK):return b'\0'.join((k+'='+v).encode()for k,v in values.items())+b'\0'
class Test(unittest.TestCase):
 def snap(self,title='ray::IDLE'):
  raw=title.encode()+b'\0'*(160-len(title.encode()));markers=dict(values=MARK,original_selected_environ_base64=base64.b64encode(env()).decode());return dict(pid=123,start_ticks=456,uid=1000,executable=dict(resolved_path='/usr/bin/python3.10',device=1,inode=2,sha256='a'*64),cmdline_base64=base64.b64encode(raw).decode(),command=raw.decode().split('\0'),process_markers=markers)
 def test_protected_real_command_layout_contract(self):m.validate_snapshot(self.snap(),self.snap('ray::WorkerDict'),'ray::WorkerDict')
 def test_marker_mutation_rejected_after_real_title(self):
  a=self.snap('ray::WorkerDict');a['process_markers']['values']=dict(MARK,SLURM_JOB_ID='2')
  with self.assertRaises(ValueError):m.validate_snapshot(self.snap(),a,'ray::WorkerDict')
 def test_missing_or_duplicate_raw_markers_rejected(self):
  for raw in (env({k:v for k,v in MARK.items()if k!='BAPO_RUN_ID'}),env()+b'SLURM_JOB_ID=1\0',env(dict(MARK,SPT_NOENV='0')),b'\0'*200):
   for fn in (m.process_markers,f.process_markers):
    with self.subTest(fn=fn.__module__,raw=raw),mock.patch.object(pathlib.Path,'read_bytes',return_value=raw),self.assertRaises(ValueError):fn(123)
 def test_marker_raw_recomputed_not_flag(self):
  for fn in (m.process_markers,f.process_markers):
   with mock.patch.object(pathlib.Path,'read_bytes',return_value=env()):self.assertEqual(fn(123)['values'],MARK)
 def test_capacity_or_truncation_rejected(self):
  a=self.snap('ray::WorkerDict');a['cmdline_base64']=base64.b64encode(b'ray::WorkerDict\0').decode();a['command']=['ray::WorkerDict','']
  with self.assertRaisesRegex(ValueError,'capacity'):m.validate_snapshot(self.snap(),a,'ray::WorkerDict')
  with self.assertRaises(ValueError):m.validate_snapshot(self.snap(),self.snap('ray::Worker'),'ray::WorkerDict')
 def test_title_extra_tail_and_pid_reuse_rejected(self):
  for mutate in (lambda a:a.update(start_ticks=457),lambda a:a.update(pid=124)):
   a=self.snap('ray::WorkerDict');mutate(a)
   with self.assertRaises(ValueError):m.validate_snapshot(self.snap(),a,'ray::WorkerDict')
 def test_stage_policy_precedes_any_subprocess_and_ray_start(self):
  text=(B/'deployment-tools/stage.sh').read_text();self.assertEqual(text.count('export SPT_NOENV=1'),1);self.assertLess(text.index('export SPT_NOENV=1'),text.index('ulimit'));self.assertLess(text.index('export SPT_NOENV=1'),text.index('ROOT='))
 def test_production_guard_exact_unchanged(self):
  old=pathlib.Path('work/e1-300-ray-titlefix-r2-20261006/deployment-tools/resource_watch.py').read_bytes();self.assertEqual(old,(B/'deployment-tools/resource_watch.py').read_bytes())
 def test_no_signal_or_ray_init_new_preflight(self):
  text=(B/'deployment-tools/actual-title-dependency-preflight.py').read_text();tree=ast.parse(text);self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in {'kill','terminate','pidfd_send_signal'}for n in ast.walk(tree)));self.assertNotIn('ray.init(',text)
if __name__=='__main__':unittest.main(verbosity=2)
