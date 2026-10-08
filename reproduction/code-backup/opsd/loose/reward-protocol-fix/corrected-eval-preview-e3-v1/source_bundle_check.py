"""Exact activation relationship for a NEW independently reviewed corrected tool bundle."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ACTIVATED={'eval_entry.sh','eval_job.py','eval_supervisor.py','retrieval_audit_entry.py','post_eval_check.py','source_bundle_check.py','strict-tools/strict_stage_prerequisites.py'}

def activated_bytes(name,raw):
 lines=raw.splitlines(keepends=True)
 if name not in ACTIVATED:return raw
 matches=[i for i,line in enumerate(lines) if line.lstrip().startswith(b"echo 'BLOCKED") or line.lstrip().startswith(b"print('BLOCKED")]
 if len(matches)!=1:raise ValueError('Exact canonical single blocked guard required')
 i=matches[0]
 if i+1>=len(lines) or lines[i+1].strip() not in (b'exit 78',b'raise SystemExit(78)',b'return 78'):raise ValueError('Exact blocked78 line required')
 return b''.join(lines[:i]+lines[i+2:])

def verify_bundle(bundle,canonical=HERE):
 expected=json.loads((canonical/'source-manifest.json').read_text())
 if set(bundle)!={'schema','eval_commit','template_manifest_sha256','file_sources'} or bundle['schema']!='corrected-eval-activated-tool-bundle-v1' or bundle['eval_commit']!=expected['eval_commit']:raise ValueError('Exact reviewed activated bundle schema required')
 if bundle['template_manifest_sha256']!=hashlib.sha256((canonical/'source-manifest.json').read_bytes()).hexdigest() or set(bundle['file_sources'])!=set(expected['file_sources']):raise ValueError('Actual launcher bundle ancestry/file closure differs')
 for name,record in bundle['file_sources'].items():
  if set(record)!={'path','sha256'}:raise ValueError('Actual launcher file binding required')
  original=canonical/name
  if hashlib.sha256(original.read_bytes()).hexdigest()!=expected['file_sources'][name]:raise ValueError('Canonical source template mutated')
  actual=Path(record['path'])
  if not actual.is_absolute() or actual.read_bytes()!=activated_bytes(name,original.read_bytes()) or hashlib.sha256(actual.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Actual launcher bytes differ from exact reviewed activation')
 return bundle

if __name__=='__main__':
 print('BLOCKED source bundle preview: reviewed activated bundle unavailable',file=sys.stderr)
 raise SystemExit(78)
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--canonical',required=True)
 args=parser.parse_args();path=Path(args.bundle)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=args.sha256:raise ValueError('External activated bundle SHA differs')
 verify_bundle(json.loads(path.read_text()),Path(args.canonical))
