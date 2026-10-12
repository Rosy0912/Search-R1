"""One-shot native submission; an ambiguous receipt forbids another sbatch."""
import hashlib, json, os, pathlib, re, subprocess
root = pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
review_path = root/'logs/FRESH75-E5B30D7-NATIVE-LAUNCH-INDEPENDENT-REVIEW.json'
binding_path = root/'logs/fresh75-native-e5b30d7-v3-binding.json'
def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
review = json.loads(review_path.read_text())
binding = json.loads(binding_path.read_text())
assert review['status'] == 'READY_FOR_FRESH_NATIVE_ONLY'
assert review['binding_sha256'] == sha(binding_path)
assert binding['code_commit'] == 'e5b30d7cf35c30a83d07375733501392d6fa42f1'
repo = pathlib.Path(binding['repo'])
helpers = pathlib.Path(binding['helpers'])
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == binding['code_commit']
queued = subprocess.check_output(['squeue','-h','-w','h100-8','-n','gigpo-e1-fresh75-native','-o','%i %T'],text=True)
assert not queued.strip(), 'Native already queued; inspect it instead of submitting'
intent_path = root/'logs/fresh75-native-e5b30d7-v3-submit-intent.json'
receipt_path = root/'logs/fresh75-native-e5b30d7-v3-submit-receipt.json'
assert not intent_path.exists() and not receipt_path.exists(), 'Existing intent or receipt: inspect rather than submit'
command = ['sbatch','--parsable',str(helpers/'native_fresh_restore_v2.sbatch'),str(binding_path),str(repo),str(helpers)]
with intent_path.open('x') as output:
    json.dump(dict(command=command,binding_sha256=sha(binding_path),review_sha256=sha(review_path)),output)
    output.flush(); os.fsync(output.fileno())
directory_fd=os.open(str(intent_path.parent),os.O_RDONLY)
try: os.fsync(directory_fd)
finally: os.close(directory_fd)
result = subprocess.run(command,capture_output=True,text=True)
receipt = dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,intent=str(intent_path))
with receipt_path.open('x') as output:
    json.dump(receipt,output,indent=2)
    output.flush(); os.fsync(output.fileno())
directory_fd=os.open(str(receipt_path.parent),os.O_RDONLY)
try: os.fsync(directory_fd)
finally: os.close(directory_fd)
assert result.returncode == 0 and re.fullmatch(r'\d+(;[^\s]+)?',result.stdout.strip()), 'Ambiguous submission; inspect receipt/queue, never repeat'
print(json.dumps(dict(receipt=receipt,job=result.stdout.strip().split(';')[0],binding=str(binding_path),binding_sha256=sha(binding_path)),indent=2))
