import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'candidates';archive=base/'tree-native-numeric-296217a.tar.gz';candidate=base/'tree-native-numeric-296217a'
assert not candidate.exists(),'Candidate already exists; do not replace'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='f27df70e328837264e44dfecf65fcdcf34061051d1df94d497ad7c2f2e5fff8e'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  assert (m.isdir() or m.isfile()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]==candidate.name
 tar.extractall(base)
assert hashlib.sha256((candidate/'manifest.json').read_bytes()).hexdigest()=='f1d34d7fec865f2c0ccb3c572b3cd17a896c295ca9fcd702c52070b63b01285e'
manifest=json.loads((candidate/'manifest.json').read_text())
for file,expected in manifest['files'].items():assert hashlib.sha256((candidate/file).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'tools/start_tree_native_numeric_296217a.sh').read_bytes()).hexdigest()=='cc1f6483a99bbe7d4c0b4e69e6e324c3f665672c088b76fc46bf2f3406199cc1'
for file in candidate.rglob('*'):
 if file.is_file():file.chmod(0o444)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo',text=True).strip()=='4d133efeb1f675139d2558640f1e21d50780f133'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root/'repo',text=True).strip()
print(json.dumps(dict(status='ISOLATED_CANDIDATE_FILES_VERIFIED',candidate=str(candidate),source=manifest['code_commit'],manifest_files=len(manifest['files']),production_commit_unchanged='4d133efeb1f675139d2558640f1e21d50780f133')))
