"""Constructed title counterexamples only: cannot generate actual CPU approval."""
import base64,copy,importlib.util,json,pathlib,unittest
D=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('new_builder',D/'build-unsigned-cpu-launch-candidate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Test(unittest.TestCase):
 def setUp(self):
  old='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006';R=pathlib.Path('work/evidence/ray-titlefix-r2-source3f85-target-cpu-raw/logs');self.r=json.loads((R/'title-dependency-preflight.json').read_text().replace(old,m.ROOT));self.runtime=json.loads((R/'runtime-executable.json').read_text());self.tools=D/'deployment-tools';self.manifest=json.loads((D/'tools-manifest.json').read_text());self.r.update(source_commit=m.SOURCE,tool_manifest_sha256=m.MANIFEST,self_sha256=m.sha(self.tools/'actual-title-dependency-preflight.py'),fixture_sha256=m.sha(self.tools/'identity-linux-fixture.py'),marker_mode='SYNTHETIC_CPU_MARKERS_NOT_SLURM_ALLOCATION');expected=dict(BAPO_RUN_ID=m.ROOT+'/logs/title-dependency-owned-cpu-child',SLURM_JOB_ID='1',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':'CPU-TITLE-PREFLIGHT-'+str(i)}for i in range(8)],sort_keys=True),SPT_NOENV='1');self.r['expected_process_markers']=expected
  original=self.r['original_process'];raw=('\0'.join(original['command'])).encode();original['cmdline_base64']=base64.b64encode(raw).decode();capacity=len(raw);self.r['argv_capacity_bytes']=capacity;marker=dict(values=expected,original_selected_environ_base64=base64.b64encode(b'\0'.join((k+'='+v).encode()for k,v in expected.items())+b'\0').decode());original['process_markers']=copy.deepcopy(marker)
  for ack in self.r['acks']:
   s=ack['snapshot'];raw=ack['title'].encode()+b'\0'*(capacity-len(ack['title']));s.update(process_markers=copy.deepcopy(marker),command=raw.decode().split('\0'),cmdline_base64=base64.b64encode(raw).decode())
 def validate(self,r=None):return m.validate_title(r or self.r,self.runtime,self.tools,self.manifest)
 def reject(self,change):
  r=copy.deepcopy(self.r);change(r)
  with self.assertRaises((ValueError,KeyError)):self.validate(r)
 def test_constructed_complete_title_parses(self):self.validate()
 def test_default_clobbered_proc_env(self):self.reject(lambda r:r['acks'][1]['snapshot']['process_markers'].update(original_selected_environ_base64='AA=='))
 def test_changed_raw_marker_order(self):
  def change(r):
   x=r['acks'][1]['snapshot']['process_markers'];rows=base64.b64decode(x['original_selected_environ_base64']).split(b'\0');x['original_selected_environ_base64']=base64.b64encode(b'\0'.join(list(reversed(rows[:-1]))+[b''])).decode()
  self.reject(change)
 def test_duplicate_original_marker(self):
  def change(r):
   x=r['original_process']['process_markers'];x['original_selected_environ_base64']=base64.b64encode(base64.b64decode(x['original_selected_environ_base64'])+b'SPT_NOENV=1\0').decode()
  self.reject(change)
 def test_rebound_spt(self):self.reject(lambda r:r['expected_process_markers'].update(SPT_NOENV='0'))
 def test_boolean_capacity(self):self.reject(lambda r:r.update(argv_capacity_bytes=True))
 def test_changed_ack_capacity(self):
  def change(r):
   s=r['acks'][0]['snapshot'];raw=base64.b64decode(s['cmdline_base64'])+b'\0';s['cmdline_base64']=base64.b64encode(raw).decode();s['command']=raw.decode().split('\0')
  self.reject(change)
 def test_missing_synthetic_scope(self):self.reject(lambda r:r.update(marker_mode='ACTUAL_SLURM'))
 def test_old_source(self):self.reject(lambda r:r.update(source_commit='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4'))
 def test_old_manifest(self):self.reject(lambda r:r.update(tool_manifest_sha256='old'))
if __name__=='__main__':unittest.main()
