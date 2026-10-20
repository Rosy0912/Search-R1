"""CPU launch/namespace adapter; full worker/driver restoration uses original loaders."""
from pathlib import Path
import json,os,socket,sys
from .contract import verify,validate_gate,project_config,BASE,SCIENCE,ALLOCATION,PYTHON,COMPILER,sha
_BINDING='RECURSIVE_ABLATION_7B_NATIVE_RECOVERY_BINDING'

def install_binding():
 p=Path(os.environ[_BINDING]);plan=json.loads(p.read_text());parent=plan['parent']
 assert os.environ['SEARCH_R1_JOB_ID']==plan['job'] and os.environ['RECURSIVE_ABLATION_RESUME_MANIFEST']==parent['manifest_path'] and os.environ['RECURSIVE_ABLATION_RESUME_SHA256']==parent['manifest_sha256']
 assert os.environ['SEARCH_R1_BASELINE_COMMIT']==SCIENCE
 def manifest():
  assert sha(parent['manifest_path'])==parent['manifest_sha256'];m=json.loads(Path(parent['manifest_path']).read_text());assert m==parent['manifest']
  for r in m['worker_states']+[m['driver']]:
   f=Path(r['path']);assert f.parent==Path(parent['manifest_path']).parent and f.is_file() and not f.is_symlink()
  return m
 from release.recursive_bran_a.ablation150_v1 import native_resume
 native_resume.manifest=manifest

def worker_setup():
 install_binding()
 from release.recursive_bran_a.ablation150_v1.bootstrap import install
 install()

def record_for(plan,host):
 from omegaconf import OmegaConf
 from release.recursive_bran_a.ablation150_v1 import PROTOCOL,EVALUATION_STEPS
 from release.recursive_bran_a.ablation150_v1.contracts import validate_config_dict
 root=Path(plan['root']);parent=OmegaConf.to_container(OmegaConf.load(plan['parent_resolved']),resolve=True)
 cfg=project_config(parent,root,host);validate_config_dict(cfg)
 path=root/'receipts/resolved.yaml'
 with path.open('x') as f:f.write(OmegaConf.to_yaml(OmegaConf.create(cfg),resolve=True))
 record=dict(protocol=PROTOCOL,group=plan['group'],run_id=plan['run_id'],config=str(path),config_sha256=sha(path),source_commit=SCIENCE,
  model=plan['model'],data=plan['data_root'],data_identity=plan['data_identity'],base=str(root),output=plan['output'],fresh_authorized=False,
  parent_checkpoint=plan['parent']['manifest_path'],parent_manifest_sha256=plan['parent']['manifest_sha256'],entry_source_commit=plan['entry_source_commit'],
  execution_backend='pj_pt',allocation=plan['job'],scheduler_node=host,host=host,condition=plan['condition'],actual_update_limit=150,
  evaluation_steps=list(EVALUATION_STEPS),scheduler_scale=200,warmup_rounds=57,model_size='7B',model_revision=plan['model_revision'],
  retriever_receipt=plan['retriever_receipt'],retriever_receipt_sha256=plan['retriever_receipt_sha256'])
 receipt=root/'receipts/configuration.json'
 with receipt.open('x') as f:json.dump(record,f,indent=2)
 return record,receipt

def train(planfile):
 plan=json.loads(Path(planfile).read_text());root=verify(plan)
 os.environ.update({ _BINDING:str(Path(planfile).resolve()),'RECURSIVE_ABLATION_RESUME_MANIFEST':plan['parent']['manifest_path'],
  'RECURSIVE_ABLATION_RESUME_SHA256':plan['parent']['manifest_sha256']})
 # Original orchestration retains scientific modules, lease, fresh CUDA setup,
 # optimizer and complete driver restoration, Ray lifecycle and exit evidence.
 from reproduction.sevenb import ablations as original
 from release.recursive_bran_a.ablation150_v1 import bootstrap
 original.verify=lambda _:root;original.record_for=record_for;original.run_root=lambda group:root
 original.output_path=lambda _:Path(plan['output']);bootstrap.worker_setup=worker_setup
 # Called again in each worker; original load_worker/migrate_trainer stay byte-identical.
 original_install=bootstrap.install
 def install():
  install_binding();return original_install()
 bootstrap.install=install
 original.train(planfile)

import subprocess,time,threading

def launch(planfile):
    plan = json.loads(Path(planfile).read_text()); root = verify(plan)
    assert os.environ.get('SEARCH_R1_EXECUTION_BACKEND') == 'pj_pt'
    assert os.environ.get('SEARCH_R1_JOB_ID') == plan['job'] and not os.environ.get('SLURM_JOB_ID')
    intent = json.loads((root / 'submission-intent.json').read_text())
    assert intent['plan_sha256'] == sha(planfile) and intent['job'] == plan['job']
    deadline = time.monotonic() + 180
    while not ((root / 'submission.json').exists() and (root / 'scheduler-release-once.json').exists()):
        if time.monotonic() >= deadline:
            raise TimeoutError('No exact UID scheduler release; no training started')
        time.sleep(.5)
    sub = json.loads((root / 'submission.json').read_text()); gate = json.loads((root / 'scheduler-release-once.json').read_text())
    actual = validate_gate(plan, sub, gate)
    inv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(inv) == 8 and len({r.split(',')[1].strip() for r in inv}) == 8
    assert all('A100' in r and float(r.split(',')[3]) >= 80000 for r in inv)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not (root / 'checkpoints').exists() and not (root / 'output').exists()
    with (root / 'entry-once.json').open('x') as f:
        json.dump(dict(at=time.time(), actual=actual, gpus=inv, plan_sha256=sha(planfile)), f, indent=2)
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(r.split(',')[0].strip() for r in inv), TRITON_PTXAS_PATH=COMPILER['path'])
    for name in ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES']:
        os.environ.pop(name, None)
    subprocess.run([sys.executable, '-m', 'reproduction.sevenb.ablation_smoke', str(root / 'compiler-smoke.json')], check=True, timeout=120)
    from http.server import ThreadingHTTPServer
    from reproduction.recursive_lumia.proxy import Handler
    ready = json.loads(Path(plan['retriever_receipt']).read_text())
    server = ThreadingHTTPServer(('127.0.0.1', 29146), Handler); server.remote = ready['url']
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        child = subprocess.Popen([sys.executable, '-m', 'search_r1.baselines.preflight', '--', sys.executable,
                                  '-m', 'reproduction.sevenb_ablation_resume_v1.entry', 'train', str(planfile)])
        rc = child.wait()
        with (root / 'actual-parent-wait.json').open('x') as f:
            json.dump(dict(at=time.time(), pid=child.pid, rc=rc, job=plan['job']), f)
        raise SystemExit(rc if rc >= 0 else 128 - rc)
    finally:
        server.shutdown(); server.server_close()

if __name__=='__main__':
 (train if sys.argv[1]=='train' else launch)(sys.argv[2])
