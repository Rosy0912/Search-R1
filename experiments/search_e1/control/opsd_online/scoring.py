"""Bind the existing strict scorer inside the exact frozen305 upstream tree."""
import importlib
from pathlib import Path
import sys

try:
    from .checkpoint import checked_json, checked_file, require
except ImportError:
    from checkpoint import checked_json, checked_file, require

QA_RELATIVE = 'upstream/Search-R1/search_r1/qa_protocol.py'
INIT_RELATIVE = 'upstream/Search-R1/search_r1/__init__.py'
QA_SHA256 = '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def verify_scoring(config):
    manifest, proof = checked_json(config['runtime_manifest'])
    files = manifest.get('files', manifest)
    root = Path(config['paths']['runtime']).resolve()
    actual = []
    for name, fixed_sha in ((QA_RELATIVE, QA_SHA256), (INIT_RELATIVE, EMPTY_SHA256)):
        entry = files[name]
        digest = entry['sha256'] if isinstance(entry, dict) else entry
        require(digest == fixed_sha, 'Frozen native strict scorer identity changed')
        actual.append(checked_file(dict(path=str(root/name), sha256=digest)))
    upstream = str(root/'upstream'/'Search-R1')
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    package = importlib.import_module('search_r1')
    qa = importlib.import_module('search_r1.qa_protocol')
    require(Path(package.__file__).resolve() == root/INIT_RELATIVE
            and Path(qa.__file__).resolve() == root/QA_RELATIVE,
            'Actual imported strict scorer is not the frozen305 upstream dependency')
    for reference in actual:
        checked_file(reference)
    require(callable(qa.extract_final_answer) and callable(qa.parse_model_action), 'Strict scorer API missing')
    return dict(accepted=True, runtime_manifest=proof, actual_imported_scoring_files=actual,
                scope='Existing strict terminal authored-action scorer; no reward/mining rule change')
