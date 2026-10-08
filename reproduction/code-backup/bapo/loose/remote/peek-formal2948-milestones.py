import pathlib,json,subprocess
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo'); r=root/'runs/bapo-formal500-2948'
rows=[json.loads(l.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for l in (r/'train.log').read_text().splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in l]
periods=[{k:v for k,v in x.items() if k.startswith(('val/qa_count/','val/f1/','val/em/','val/format_rate/','val/search_calls/'))} for x in rows if 'val/qa_count/triviaqa' in x]
states=[{'path':str(p),'value':json.loads(p.read_text())} for p in r.glob('output/**/algorithm/global_step_*/state.json')]
ck=[{'path':str(p),'file_count':sum(x.is_file() for x in p.rglob('*'))} for p in r.glob('output/**/actor/global_step_*')]
print(json.dumps({'periodic_evaluations':periods,'states':states,'checkpoints':ck,'source':subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True)}))
