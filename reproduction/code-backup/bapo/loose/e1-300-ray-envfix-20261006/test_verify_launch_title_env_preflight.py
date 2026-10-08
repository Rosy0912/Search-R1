"""Synthetic schema/semantic rebinding tests; no actual Linux dependency proof."""
import ast,base64,copy,hashlib,json,pathlib,shutil,tempfile,unittest
P=pathlib.Path(__file__).resolve().parent
T=P/'deployment-tools'
source=(T/'verify_launch.py').read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_title_dependency_preflight')
namespace=dict(pathlib=pathlib,json=json,hashlib=hashlib);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(T/'verify_launch.py'),'exec'),namespace)
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=pathlib.Path(self.temp.name).resolve();self.commit='484c0b30fa73fa5354bdb3452f007134ec5db13f'
  for p in ('logs','tools','env/lib/python3.10/site-packages/ray/thirdparty_files','env/lib/python3.10/site-packages/ray/_private'):(self.root/p).mkdir(parents=True,exist_ok=True)
  for n in ('resource_watch.py','actual-title-dependency-preflight.py','identity-linux-fixture.py'):shutil.copy2(T/n,self.root/'tools'/n)
  self.extension=self.root/'env/lib/python3.10/site-packages/ray/thirdparty_files/setproctitle.cpython-310-x86_64-linux-gnu.so';self.extension.write_bytes(b'X'*69104)
  def sha(p):return '9429a267fba07b2ec523ae906878e154dbe053065c4e89c7e450037866c2557b'if p==self.extension else hashlib.sha256(p.read_bytes()).hexdigest()
  self.sha=sha;namespace['sha']=sha
  self.manifest={'files':{n:sha(self.root/'tools'/n)for n in ('resource_watch.py','actual-title-dependency-preflight.py','identity-linux-fixture.py')}};(self.root/'logs/tools-manifest.json').write_text(json.dumps(self.manifest))
  init=self.root/'env/lib/python3.10/site-packages/ray/__init__.py';init.write_text('# syntheticRay\n');ws='def _changeproctitle(title):\n    setproctitle.setproctitle(title)\n';(init.parent/'_private/worker.py').write_text(ws)
  dep=dict(ray_version='2.40.0',same_as_ray_worker=True,function_builtin=True,function_name='setproctitle',function_module='setproctitle',ray_init_path=str(init),ray_init_sha256=sha(init),extension_path=str(self.extension),extension_bytes=69104,extension_sha256=sha(self.extension),worker_title_source=ws)
  self.runtime={'executable':dict(resolved_path='/usr/bin/python3.10',device=2,inode=3),'binary_sha256':'b'*64}
  original=self.snapshot([str(self.root/'env/bin/python'),'-B',str(self.root/'tools/actual-title-dependency-preflight.py'),'--child',''])
  import importlib.util
  spec=importlib.util.spec_from_file_location('preflight_test_watch',self.root/'tools/resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
  self.record=dict(status='PASS_ACTUAL_CPU25_TITLES_NOT_GPU_OR_PERMISSION',scope='ACTUAL_LINUX_RAY_VENDORED_TITLE_CPU_ONLY_NOT_GPU_OR_TRAINING',root=str(self.root),source_commit=self.commit,sys_prefix=str(self.root/'env'),tool_manifest_sha256=sha(self.root/'logs/tools-manifest.json'),self_sha256=self.manifest['files']['actual-title-dependency-preflight.py'],fixture_sha256=self.manifest['files']['identity-linux-fixture.py'],signals_sent=0,child_exit_code=0,child_stderr='',ray_init_called=False,actual_gpu_accepted=False,ready={'type':'ready','pid':1234,'dependency':dep},dependency_after=copy.deepcopy(dep),original_process=original,acks=[dict(type='ack',sequence=i,title=t,snapshot=self.snapshot([t,'','']))for i,t in enumerate(sorted(w.RAY_APPROVED_TITLES))])
  self.record.update(marker_mode='SYNTHETIC_CPU_MARKERS_NOT_SLURM_ALLOCATION',expected_process_markers=original['process_markers']['values'],argv_capacity_bytes=512)
  self.refresh()
 def snapshot(self,cmd):
  values={'BAPO_RUN_ID':str(self.root/'logs/title-dependency-owned-cpu-child'),'SLURM_JOB_ID':'1','SEARCH_R1_GPU_LEASES':json.dumps([{'uuid':'CPU-TITLE-PREFLIGHT-'+str(i)}for i in range(8)],sort_keys=True),'SPT_NOENV':'1'};raw='\0'.join(cmd).encode();raw+=b'\0'*(512-len(raw));env=b'\0'.join((k+'='+v).encode()for k,v in values.items())+b'\0';return dict(pid=1234,start_ticks=555,uid=501,executable=dict(self.runtime['executable'],sha256=self.runtime['binary_sha256']),command=raw.decode().split('\0'),cmdline_base64=base64.b64encode(raw).decode(),process_markers=dict(values=values,original_selected_environ_base64=base64.b64encode(env).decode()))
 def refresh(self):
  p=self.root/'logs/title-dependency-preflight.json';p.write_text(json.dumps(self.record));self.approval={'title_dependency_preflight':dict(path=str(p),bytes=p.stat().st_size,sha256=self.sha(p))}
 def validate(self):return namespace['validate_title_dependency_preflight'](self.root,self.commit,self.approval,self.manifest,self.runtime)
 def test_complete_synthetic_format(self):self.validate()
 def test_missing_original_binding(self):
  self.approval={}
  with self.assertRaises(KeyError):self.validate()
 def test_rebound_semantic_changes(self):
  old=copy.deepcopy(self.record)
  mutations=[lambda r:r.update(source_commit='c'*40),lambda r:r.update(child_exit_code=-9),lambda r:r.update(signals_sent=False),lambda r:r.update(ray_init_called=True),lambda r:r['acks'][0].update(title='ray::BAD'),lambda r:r['acks'][0]['snapshot'].update(start_ticks=556),lambda r:r['acks'][0]['snapshot']['executable'].update(inode=True),lambda r:r['acks'][0].update(sequence=False),lambda r:r['acks'][0]['snapshot'].update(cmdline_base64='AAAA')]
  for index,mutate in enumerate(mutations):
   self.record=copy.deepcopy(old);mutate(self.record);self.refresh()
   with self.subTest(index=index),self.assertRaises(AssertionError):self.validate()
 def test_rebound_dependency_and_source_function(self):
  old=copy.deepcopy(self.record)
  for field,bad in [('extension_sha256','0'*64),('ray_version','2.41.0'),('worker_title_source','def _changeproctitle(title):\n    pass\n')]:
   self.record=copy.deepcopy(old);self.record['ready']['dependency'][field]=bad;self.record['dependency_after'][field]=bad;self.refresh()
   with self.subTest(field=field),self.assertRaises(AssertionError):self.validate()
 def test_real_env_symlink_requires_exact_physical_dependency(self):
  physical=self.root/'physical-env';(self.root/'env').rename(physical);(self.root/'env').symlink_to(physical,target_is_directory=True)
  dep=self.record['ready']['dependency'];dep['ray_init_path']=str((self.root/'env/lib/python3.10/site-packages/ray/__init__.py').resolve());dep['extension_path']=str((self.root/'env/lib/python3.10/site-packages/ray/thirdparty_files/setproctitle.cpython-310-x86_64-linux-gnu.so').resolve())
  self.extension=pathlib.Path(dep['extension_path']);self.record['dependency_after']=copy.deepcopy(dep);self.refresh();self.validate()
  wrong=self.root/'wrong-physical-ray';shutil.copytree(pathlib.Path(dep['ray_init_path']).parent,wrong)
  dep['ray_init_path']=str(wrong/'__init__.py');dep['extension_path']=str(wrong/'thirdparty_files/setproctitle.cpython-310-x86_64-linux-gnu.so');self.record['dependency_after']=copy.deepcopy(dep);self.refresh()
  with self.assertRaises(AssertionError):self.validate()
 def test_marker_original_default_clobber_rejected(self):
  self.record['acks'][12]['snapshot']['process_markers']={'values':{},'original_selected_environ_base64':base64.b64encode(b'\0'*512).decode()};self.refresh()
  with self.assertRaises(AssertionError):self.validate()
 def test_synthetic_cpu_markers_not_misrepresented_as_gpu_allocation(self):
  self.record['marker_mode']='ACTUAL_GPU_ALLOCATION';self.refresh()
  with self.assertRaises(AssertionError):self.validate()
 def test_original_capacity_and_truncated_title_rejected(self):
  self.record['argv_capacity_bytes']=5;self.refresh()
  with self.assertRaises(AssertionError):self.validate()
if __name__=='__main__':unittest.main()
