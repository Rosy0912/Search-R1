from pathlib import Path
import json
b=Path('work/search-e1-analysis-output/opsd-ablation-execution-20261008')
for suffix in ['0-7','8-9']:
 d=json.loads((b/f'raw-c0-closed{suffix}-direct-20261009/local-full-sha-private.json').read_text());print(suffix)
 for r in d['files']:
  if ('outer-00000' in r['relative'] or 'outer-00009' in r['relative']) and (r['relative'].endswith('teacher-diagnostics/complete.json') or r['relative'].endswith('teacher-diagnostics/selection.json') or r['relative'].endswith('teacher-diagnostics/child-complete.json') or 'vocabulary-' in r['relative'] or r['relative'].endswith('training/outer-complete.json')):print(r['relative'],r['bytes'])
