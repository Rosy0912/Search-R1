import subprocess,sys,shlex
job=sys.argv[1]; assert job.isdigit()
code="""import pathlib,json,re,datetime
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-acceptance-JOB')
files=list((r/'runtime/ray/session_latest/logs').glob('worker-*.out'))
metric=[]
for p in files:
 for line in p.read_text(errors='replace').splitlines():
  line=re.sub(r'\x1b\[[0-9;]*m','',line)
  if re.match(r'^step:\d+ - ',line): metric.append(line)
log=(r/'training.log').read_text(errors='replace') if (r/'training.log').exists() else ''
print(json.dumps(dict(sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='JOB',metric_lines=metric,training_tail=log[-1400:],training_finished=(r/'training-finished.txt').exists(),exit=(r/'exit.txt').read_text() if (r/'exit.txt').exists() else None)))
""".replace('JOB',job)
raise SystemExit(subprocess.run([sys.executable,'work/remote.py','python3 -c '+shlex.quote(code)]).returncode)
