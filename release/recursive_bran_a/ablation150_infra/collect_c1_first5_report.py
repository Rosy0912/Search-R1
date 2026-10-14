"""CPU-only native aggregate comparison; no prediction capture or GPU evaluation."""
import hashlib,json,re,sys,time,uuid
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-ablation150-20261014')
a=json.loads((b/'active-study.json').read_text());sys.path.insert(0,a['source'])
from release.recursive_bran_a.ablation150_reports_v1.report import build,evaluation,native_log,write_report
from release.recursive_bran_a.ablation150_v1.contracts import read_receipt

g=b/'C1-root-only';receipt=read_receipt(g/'receipts/configuration.json')
e=json.loads((g/'receipts/current-entry.json').read_text());out=Path(e['output'])
complete=[json.loads(line) for line in (out/'completed-outers.jsonl').read_text().splitlines()]
row=next(v for v in complete if v['completed_outer']==5)
assert row['run_id']==receipt['run_id'] and row['config_sha256']==receipt['config_sha256']
assert row['metrics']['val/qa_count/all']==512 and row['metrics']['val/qa_count/nq']==256 and row['metrics']['val/qa_count/hotpotqa']==256
key=Path(row['checkpoint']['path']);assert key.is_file() and hashlib.sha256(key.read_bytes()).hexdigest()==row['checkpoint']['sha256']
assert json.loads(key.read_text())['boundary_phase']=='post-evaluation'
reference=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/logs/training.log')
line=None
with reference.open(errors='replace') as stream:
 for raw in stream:
  if re.search(r'\bstep:5 - ',raw):line=raw;break
assert line and 'val/qa_count/all:512' in line
folder=b/'analysis'/('C1-first5-'+uuid.uuid4().hex);folder.mkdir(parents=True)
reference_log=folder/'reference-native-outer5.log';reference_log.write_text(line)
r=native_log([reference_log]);dev=evaluation(r[5]);assert dev['all']['hits']==96
aggregate=dict(dev512=dict(count=512,hits=dev['all']['hits'],nq=dev['nq'],hotpot=dev['hotpotqa']),
 source=dict(native_log=str(reference),captured_line_sha256=hashlib.sha256(line.encode()).hexdigest(),new_inference=False))
ref_file=folder/'A-outer5-aggregate.json';ref_file.write_text(json.dumps(aggregate,indent=2))
study=dict(reference_condition='A-original',milestones=[5],conditions=[
 dict(id='A-original',run_id='retained-A-reference',config_sha256=None,world_size=8,
      aggregate_files={'5':str(ref_file)},logs=[],output_roots=[],
      provenance_scope='Existing original A native aggregate from its early entry, not a fabricated single config identity'),
 dict(id='C1-root-only',run_id=receipt['run_id'],config_sha256=receipt['config_sha256'],world_size=8,
      logs=[str(p) for p in g.joinpath('logs').glob('training*.log')],output_roots=[str(p) for p in g.joinpath('output').glob('attempt-*')])])
(folder/'study-input.json').write_text(json.dumps(study,indent=2))
report=build(study);report['limitations'].extend([
 'C1 outer1 reuses a saved pre-update batch after a namespace fault; its replay timer excludes original sampling. Do not treat recorded prefix cost as full end-to-end GPU billing.',
 'Worker CUDA RNG and former vLLM executor state were not saved at the zero-write fault; no bitwise restart is claimed.',
 'A reference is read from original native logs, not replayed checkpoints; only aggregate EM is compared.'])
write_report(report,folder)
print(json.dumps(dict(time=time.time(),directory=str(folder),files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in folder.iterdir() if p.is_file()},aggregate=report,scope='Formal native C1 first5 vs retained A5; CPU only, no new inference'),ensure_ascii=False),flush=True)
