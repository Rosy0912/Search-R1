"""Submit exactly one new authorized fresh200 allocation; unknown outcomes never retry."""
import pathlib,json,datetime,shlex,subprocess,re,hashlib
H=pathlib.Path(__file__).resolve().parent
spec=json.loads((H/'runtime-tools-bound-667c68e0f91b.json').read_text());native=json.loads((H/'native-cpu-actual.json').read_text())
deployment=json.loads((H/'tools-deployment-actual.json').read_text())
root=spec['remote_root'];old='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';formal=spec['formal']
assert native['source_commit']==spec['source_commit']=='667c68e0f91b4bc439324d0e095ce13df10da559'
assert deployment['formal']['verified_files']['manifest.json']==formal['manifest_sha256']
assert hashlib.sha256((H/'native-source-inventory-667c68e0f91b-ACTUAL.json').read_bytes()).hexdigest()==native['inventory_sha256']
env={'SEARCH_R1_REPO':spec['remote_repo'],'E1_FORMAL_TOOLS':formal['remote_directory'],'E1_FORMAL_MANIFEST_SHA256':formal['manifest_sha256'],'E1_NATIVE_IMPORT_CLOSURE':native['inventory_path'],'E1_NATIVE_IMPORT_CLOSURE_SHA256':native['inventory_sha256'],'OPSD_E1_COMMON_DATA_ROOT':old+'/common-data','OPSD_E1_COMMON_MANIFEST':old+'/common-data/manifest.json','E1_TOOL_ASSET_MANIFEST':old+'/tools/tool-assets-full-verify-04f8-d1f36d3d5578/evidence/manifest.json','OPSD_E1_REVIEW_PUBLIC_KEY':root+'/review-signing/public.json','OPSD_E1_REVIEW_PUBLIC_KEY_SHA256':'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'}
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-10','--nodes=1','--ntasks=1','--cpus-per-task=192','--mem=1T','--gres=gpu:h100:8','--exclusive','--time=7-00:00:00','--job-name=opsd-multitraj200','--chdir='+env['SEARCH_R1_REPO'],'--output='+root+'/logs/formal-%j.out','--error='+root+'/logs/formal-%j.err','--wrap='+shlex.join(['env']+[k+'='+v for k,v in env.items()]+['srun','--ntasks=1','bash',formal['remote_directory']+'/formal_job.sh'])]
intent=H/'submit-intent.json'
with intent.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'env':env,'source_commit':spec['source_commit'],'actual_outer':0,'authorization':'Latest human unclipped multiresponse pure OPSD fresh5 independent gate then sameprocess200'},f,indent=2);f.write('\n')
try:r=subprocess.run(['python3',str(H.parent/'head.py'),shlex.join(args)],capture_output=True,timeout=60)
except subprocess.TimeoutExpired as e:
 (H/'submit-unknown.json').write_text(json.dumps({'outcome':'UNKNOWN inspect scheduler before any mutation','stdout':(e.stdout or b'').decode(),'stderr':(e.stderr or b'').decode()},indent=2)+'\n');raise
result={'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()};(H/'submit-original.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
assert r.returncode==0 and re.fullmatch(r'\d+(?:;[^\n]+)?\n?',result['stdout'])
job=result['stdout'].strip().split(';')[0];(H/'actual-job-id.txt').write_text(job+'\n')
