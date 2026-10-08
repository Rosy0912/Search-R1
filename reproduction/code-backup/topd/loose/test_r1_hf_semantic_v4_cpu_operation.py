import pathlib,json,tarfile,hashlib,ast
W=pathlib.Path(__file__).parent

def test_both_whole_archives_exact_all_regular_members_and_original_scanner():
 data={}
 for name in ('r1-32b-hf-semantic-review-v4.tar.gz','r1-hf-semantic-v4-native-operation.tar.gz'):
  with tarfile.open(W/name) as tf:
   for m in tf.getmembers():
    assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts and m.name not in data
    data[m.name]=tf.extractfile(m).read()
 for folder in ('r1-32b-hf-semantic-review-v4','r1-hf-semantic-v4-native-operation'):
  manifest=json.loads(data[folder+'/files.json']);names={folder+'/'+n for n in manifest['files']}|{folder+'/files.json'}
  assert {n for n in data if n.startswith(folder+'/')}==names
  for n,v in manifest['files'].items():assert len(data[folder+'/'+n])==v['size'] and hashlib.sha256(data[folder+'/'+n]).hexdigest()==v['sha256']
 assert hashlib.sha256(data['r1-32b-hf-semantic-review-v4/original_semantic_gate.py']).hexdigest()=='10ecd9ac072a93903e55acb870b4962844c1bcfd9bf5f2440355885e24e7b759'

def test_cpu_entry_no_gres_no_model_forward_or_legacy_selector():
 d=W/'r1-hf-semantic-v4-native-operation';entry=(d/'native-cpu.sbatch').read_text();assert '--gres' not in entry and '--cpus-per-task=2' in entry and '--mem=64G' in entry and "CUDA_VISIBLE_DEVICES=''" in entry
 s=(d/'entry.py').read_text();assert 'review.run(q)' in s and 'R1_HF_OPERATION_MANIFEST' in s
 assert 'AutoModel' not in s and '.forward(' not in s and 'check_selection' not in s

def test_operation_compiles_unknown_no_repeat_and_bounded_terminal():
 s=(W/'run_r1_hf_semantic_v4_cpu_once.py').read_text();t=ast.parse(s);r=ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='REMOTE' for a in n.targets)))
 compile(r,'operation','exec');assert "d['status']=='SUBMITTED'" in r and 'UNKNOWN_BEFORE_SUBMISSION' in r and 'timeout=30' in r and 'deadline=time.monotonic()+10860' in r and 'time.sleep(2)' in r
 assert 'if not args.execute:' in s and 'a.check_request(REQUEST)' in r
