"""Verify installed package bytes and exact raw-to-derived retrieval identity."""
import gzip
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['AGENT_ROOT'])
manifest = json.loads((root/'logs/source-environment-file-hashes.json').read_text())
site = root/'env/lib/python3.10/site-packages'
skipped, bad, checked = [], [], 0
for name, expected in manifest['hashes'].items():
    first = Path(name).parts[0]
    if (first.startswith('__editable__') and 'verl' in first) or first == 'verl.egg-link' or (first.startswith('verl') and first.endswith('.dist-info')):
        skipped.append(name)
        continue
    path = site/name
    digest = hashlib.sha256()
    if not path.is_file():
        bad.append(name)
        continue
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(chunk)
    checked += 1
    if digest.hexdigest() != expected:
        bad.append(name)
result = {'manifest_sha256':manifest['manifest_sha256'], 'files_checked':checked,
          'failed':bad, 'intentionally_replaced_source_verl_editable_files':skipped}
(root/'logs/installed-environment-file-verification.json').write_text(json.dumps(result,indent=2))
assert not bad, bad
wiki = root/'assets-root/assets/wiki'
def digest(streams):
    value = hashlib.sha256()
    for stream in streams:
        with stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):
                value.update(chunk)
    return value.hexdigest()
expected_index = digest([(wiki/'part_aa').open('rb'), (wiki/'part_ab').open('rb')])
actual_index = digest([(wiki/'e5_Flat.index').open('rb')])
expected_corpus = digest([gzip.open(wiki/'wiki-18.jsonl.gz','rb')])
actual_corpus = digest([(wiki/'wiki-18.jsonl').open('rb')])
result = {'index':{'raw_concat_sha256':expected_index,'derived_sha256':actual_index},
          'corpus':{'raw_decompressed_sha256':expected_corpus,'derived_sha256':actual_corpus}}
(root/'logs/derived-retrieval-verification.json').write_text(json.dumps(result,indent=2))
assert expected_index == actual_index
assert expected_corpus == actual_corpus
(root/'logs/installed-and-derived-verification.ready').touch()
print('INSTALLED_ENVIRONMENT_AND_EXACT_RETRIEVAL_DERIVATION_PASS',flush=True)
