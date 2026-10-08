from pathlib import Path
import json,hashlib,importlib.util,copy
W=Path(__file__).resolve().parent;D=W/'first5-tools-a096-raydedup-candidate';S=W/'actual3210-independent-first5-20261004T171004Z';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('independent_dedup_producer',D/'receipt_payload.py');g=load('independent_same_frozen_gate',W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py');assert H(Path(g.__file__))=='74c3c13929c4f5cb784b90b7fd568ee80dfe1d37c8efaa28f97aae60b3492632'
mp=S/'full_state/global_step_5/manifest.json';assert H(mp)=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b';m=g.load_json(mp);saved={};workers=set()
for rank in range(8):
 rec=m['files'][f'actor/training_state/rank_{rank}.json'];p=S/rec['path'].removeprefix(m['run_root']+'/');assert H(p)==rec['sha256'];row=g.load_json(p);assert row['rank']==rank and row['world_size']==8 and row['step']==5 and row['status']=='EXACT_FULL_STATE_RELOAD_OK';saved[rank]=row;workers.add(row['pid'])
assert len(workers)==8
log=S/'train.log';lines=[l for l in log.read_text().splitlines() if 'FULL_STATE_RANK_JSON=' in l];assert len(lines)==2;observed=[]
for line in lines:
 parsed=r.parse_observed_rank_console(line,m,g,workers);assert r.canonical(parsed)==r.canonical(saved[parsed['rank']]);observed.append(parsed)
assert len({v['rank'] for v in observed})==2
negative=[];line=lines[-1]
for label,bad in [('tail',line+' arbitrary'),('secondjson',line+' {}'),('secondmarker',line+' FULL_STATE_RANK_JSON={}'),('foreignpid',line.replace(str(observed[-1]['pid']),'999999')),('ansi',line.replace('\x1b[32m','\x1b[31m')),('missingreset',line.removesuffix('\x1b[0m')),('repeat8',line.replace('repeated 7x','repeated 8x')),('repeat0',line.replace('repeated 7x','repeated 0x')),('repeatneg',line.replace('repeated 7x','repeated -1x')),('repeatdecimal',line.replace('repeated 7x','repeated 7.0x'))]:
 try:r.parse_observed_rank_console(bad,m,g,workers)
 except ValueError:negative.append(label)
 else:raise AssertionError('unexpected accepted '+label)
result={'scope':'ACTUAL3210_TWO_OBSERVED_CONSOLE_RECORDS_NOT_EIGHT_RANK_CONTENT_ACCEPTANCE','manifest_sha256':H(mp),'producer_sha256':H(D/'receipt_payload.py'),'gate_sha256':H(Path(g.__file__)),'log_sha256':H(log),'actual_observed_count':len(observed),'actual_observed_ranks':[v['rank'] for v in observed],'actual_workers_from_all_eight_pinned_reports':sorted(workers),'observed_exact_report_bodies':True,'dedup_suffixes':[r.RANK_DEDUP_SUFFIX.search(l).group(0) for l in lines if r.RANK_DEDUP_SUFFIX.search(l)],'negative_cases_rejected':negative,'native_full_payload_and_HF_still_pending':True}
(W/'ACTUAL3210_TWO_CONSOLE_REPLAY_INDEPENDENT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
