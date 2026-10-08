python3 - <<'PY'
import pathlib,json,datetime,subprocess,re,hashlib,os
R=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004');out=R/'logs/ACTION-PREFIX-NATIVE-FAILURE3297-SLURMCTLD-v1.json';pat=re.compile(r'(?<!\d)3297(?!\d)');rows=[]
for p in sorted(pathlib.Path('/var/log/slurm').glob('*slurmctld*')):
 if not p.is_file():continue
 lines=[]
 try:
  with p.open(errors='replace') as f:
   for n,line in enumerate(f,1):
    if pat.search(line):lines.append({'line_number':n,'text':line.rstrip('\n')})
  if lines:rows.append({'path':str(p),'matching_original_lines':lines})
 except Exception as e:rows.append({'path':str(p),'read_error':repr(e)})
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':subprocess.check_output(['hostname'],text=True).strip(),'job':'3297','original_matches':rows,'scope':'Read-only original daemon lines. This is not a live scontrol terminal snapshot.'};b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'payload':v},ensure_ascii=False))
PY
