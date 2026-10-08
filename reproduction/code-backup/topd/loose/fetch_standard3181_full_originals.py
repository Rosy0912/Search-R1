"""Invoke already reviewed collectorV3 with actual terminal3181; readonly."""
import pathlib,json,subprocess,sys
r=json.loads(pathlib.Path('outputs/remote-evidence/standard-r1-profile3181-small/fetch-refs.json').read_text());refs={pathlib.Path(x['path']).name:x for x in r['files']}
args=[sys.executable,'-B','work/fetch_standard_terminal_evidence_v3.py','--job','3181','--evidence-root','/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v7/evidence','--output','outputs/remote-evidence/standard-r1-profile3181-full']
for key,name in [('terminal','standard3181-captured-controller-terminal.json'),('result','train-result.json'),('cleanup','owned-cleanup.json')]:
 ref=refs[name];args.extend(['--terminal-record' if key=='terminal' else '--'+key,ref['path'],'--'+key+'-sha256',ref['sha256']])
raise SystemExit(subprocess.run(args).returncode)
