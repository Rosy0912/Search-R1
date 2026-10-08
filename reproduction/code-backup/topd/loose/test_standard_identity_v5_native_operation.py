import ast,hashlib,json,pathlib,tarfile
W=pathlib.Path(__file__).parent

def test_actual_package_union_full_manifest_and_regular_only():
 data={}
 for name in ('standard-search-r1-teacher-identity-v5.tar.gz','standard-identity-v5-native-cpu.tar.gz'):
  with tarfile.open(W/name) as tf:
   for m in tf.getmembers():
    assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts and m.name not in data
    data[m.name]=tf.extractfile(m).read()
 names=set()
 for name in ('standard-search-r1-teacher-identity-v5-files.json','native-files.json'):
  for n,v in json.loads(data[name])['files'].items():
   assert len(data[n])==v['size'] and hashlib.sha256(data[n]).hexdigest()==v['sha256'];names.add(n)
 assert set(data)==names|{'standard-search-r1-teacher-identity-v5-files.json','native-files.json'}
 assert hashlib.sha256(data['search-r1-32b-grpo100-source-v2/r1-files.json']).hexdigest()=='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'

def test_real_entry_six_counts_short_distinct_sockets_and_no_gres():
 p=W/'standard-identity-v5-native-cpu';tree=ast.parse((p/'native_cpu_runner.py').read_text());suites=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='suites' for t in n.targets)))
 assert [v[0] for v in suites.values()]==[36,49,26,18,62,34] and sum(v[0] for v in suites.values())==225
 assert len({'p' if n=='supervisor' else n[0] for n in suites})==6
 entry=(p/'native-cpu.sbatch').read_text();assert '--gres' not in entry and '--cpus-per-task=8' in entry and '--mem=8G' in entry and 'python" -B' in entry
 assert '[sys.executable,\'-B\',\'-m\',\'pytest\'' in (p/'native_cpu_runner.py').read_text()

def test_only_native_hf_expectation_changes_not_production_scan():
 d=W/'standard-identity-v5-native-cpu/r1-32b-hf-semantic-review-v4';old=(W/'r1-32b-hf-semantic-review-v4/test_review.py').read_text()
 expected=old.replace("assert m.job_id(actual['job'])==m.job_id(3163)","assert m.job_id(actual['job'])==int(actual['job'])").replace("assert m.job_id('3164')!=m.job_id(actual['job'])","assert m.job_id(str(int(actual['job'])+1))!=m.job_id(actual['job'])")
 assert (d/'test_review_native.py').read_text()==expected
 for n in ('review.py','original_semantic_gate.py','checkpoint_inventory.py'):assert (d/n).read_bytes()==(W/'r1-32b-hf-semantic-review-v4'/n).read_bytes()

def test_operation_default_only_static_and_unknown_not_retried():
 s=(W/'run_standard_identity_v5_native_cpu.py').read_text();tree=ast.parse(s);remote=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets)))
 compile(remote,'native-operation','exec')
 assert "if '--execute' not in sys.argv:" in s and "d['status']=='SUBMITTED'" in remote and 'UNKNOWN_BEFORE_SUBMISSION' in remote
 assert 'timeout=min(5,max(end-time.monotonic(),.1))' in remote and 'timeout=30' in remote
