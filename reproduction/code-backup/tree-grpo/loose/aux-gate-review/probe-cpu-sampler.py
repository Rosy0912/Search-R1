import pathlib,tempfile,ast,sys,types,argparse,json,os,re,time,hashlib
r=pathlib.Path(__file__).resolve().parents[2];source=r/'work/sample_retriever_cpu.py';rows=[]
for name,cgroup,accepted in [('own_job','0::/slurm/uid_0/job_42/step_0\n',True),('foreign_job','0::/slurm/uid_0/job_43/step_0\n',False),('prefix_collision','0::/slurm/uid_0/job_420/step_0\n',False)]:
 with tempfile.TemporaryDirectory() as tmp:
  t=pathlib.Path(tmp);run=t/'run';run.mkdir();(run/'retriever.log').write_text('INFO: Started server process [777]\n');proc=t/'proc'/'777';proc.mkdir(parents=True);(proc/'cgroup').write_text(cgroup);(proc/'status').write_text('VmHWM:\t1000 kB\n')
  fields=['S']+['0']*21;fields[11]='123';fields[12]='456';fields[19]='1000';fields[21]='300';(proc/'stat').write_text('777 (fixture) '+ ' '.join(fields))
  def sleep(seconds):fields[19]='1001';(proc/'stat').write_text('777 (fixture) '+' '.join(fields))
  def path(value):return t/'proc' if str(value)=='/proc' else pathlib.Path(value)
  ns={'argparse':argparse,'json':json,'os':os,'pathlib':types.SimpleNamespace(Path=path),'re':re,'time':types.SimpleNamespace(time=time.time,sleep=sleep)};sys.argv=[str(source),'--run',str(run),'--job','42'];tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  try:exec(compile(tree,str(source),'exec'),ns);passed=True;err=None
  except Exception as e:passed=False;err=repr(e)
  assert passed==accepted
  samples=[]
  if passed:
   samples=[json.loads(l) for l in (run/'retriever-cpu-samples.jsonl').read_text().splitlines()];assert len(samples)==1 and samples[0]['pid']==777 and samples[0]['job_id']=='42' and samples[0]['process_cpu_seconds']==579/os.sysconf('SC_CLK_TCK') and samples[0]['rss_bytes']==300*os.sysconf('SC_PAGE_SIZE') and samples[0]['peak_rss_bytes']==1024000
  rows.append({'case':name,'accepted':passed,'error':err,'samples':samples})
(r/'work/aux-gate-review/cpu-sampler-fixtures.json').write_text(json.dumps({'scope':'actual source with fake proc tree; verifies parsing/units/job-segments/starttime reuse exit, not real Linux sampling or Slurm cleanup','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n');print('3 CPU sampler fixtures passed')
