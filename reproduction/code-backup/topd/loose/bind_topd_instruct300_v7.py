"""Bind delivered teacher100 and fixed official Instruct/common assets. No submit."""
import argparse,hashlib,importlib.util,json,os,pathlib,socket,sys
sys.dont_write_bytecode=True
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
OFFICIAL={'official_files': {'LICENSE': {'sha256': 'ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9', 'size': 7388}, 'README.md': {'sha256': '2c74693e6ae36ba878124d465195b4dad0f71035b54ece1c9fd1553cc497ef8c', 'size': 4930}, 'config.json': {'sha256': 'eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b', 'size': 661}, 'generation_config.json': {'sha256': 'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9', 'size': 242}, 'merges.txt': {'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3', 'size': 1671839}, 'model-00001-of-00002.safetensors': {'sha256': '67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2', 'size': 3968658944}, 'model-00002-of-00002.safetensors': {'sha256': 'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d', 'size': 2203268048}, 'model.safetensors.index.json': {'sha256': 'bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56', 'size': 35581}, 'tokenizer.json': {'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539', 'size': 7031645}, 'tokenizer_config.json': {'sha256': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583', 'size': 7305}, 'vocab.json': {'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'size': 2776833}}, 'public_source_api_sha256': '2182e6ac33d1c7f1116d1c85ea0531f1e14565bcdbb14668ddc69e07a37302f3', 'repo': 'Qwen/Qwen2.5-3B-Instruct', 'revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'verified_manifest_sha256': '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'}
PINS={'source_manifest': ('tools/topd-instruct300-source-v1/files.json', '0e4ae26a06b64881db9db2cc85a9c0dba55d5bea64b1b62f3e8c6f27e7d7fa34'), 'tools_manifest': ('tools/topd-instruct300-launch-v5/runtime/files.json', '8b90402d929fc516846f1a1384ea887dee0d8af13b935c8257a8180c4a1cace3'), 'launcher_manifest': ('tools/topd-instruct300-launch-v5/files.json', '4d3b9cc84fa1ce82d9c07c06342f46f2d437fb2fd487f691cdd3de381015ce40'), 'teacher_backend_manifest': ('tools/topd-vllm-teacher-v3/files.json', 'd99f21ec6b548a67854488035e2db1264e96201794fc2b85de6dcf5d301bb98f')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--infrastructure',required=True);p.add_argument('--teacher-result',required=True);p.add_argument('--teacher-result-sha256',required=True);p.add_argument('--run-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 if socket.gethostname()!='h100-11':raise ValueError('Same-node h11 only')
 package=ROOT/'tools/topd-instruct300-launch-v5';spec=importlib.util.spec_from_file_location('actual_student_prepare',package/'prepare.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 c=module.derive(json.loads(pathlib.Path(a.infrastructure).read_text()),a.teacher_result,a.teacher_result_sha256,a.run_directory)
 for key,(relative,expected) in PINS.items():
  path=ROOT/relative
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Source manifest drift '+key)
  c[key]={'path':str(path),'sha256':expected}
 c['teacher_backend']=str(ROOT/'tools/topd-vllm-teacher-v3')
 c['training_overrides']=[('trainer.save_freq=50' if value=='trainer.save_freq=5' else value) for value in c['training_overrides']]
 if c['training_overrides'].count('trainer.save_freq=50')!=1 or c['training_overrides'].count('trainer.acceptance_step=5')!=1:raise ValueError('Student save50/acceptance5 configuration differs')
 model=pathlib.Path(c['base_model']);c['student_model_identity']={'repo':OFFICIAL['repo'],'revision':OFFICIAL['revision'],'verified_manifest':{'path':str(model/'VERIFIED-MODEL-MANIFEST.json'),'sha256':OFFICIAL['verified_manifest_sha256']},'public_source_api':{'path':str(model/'PUBLIC-SOURCE-API.json'),'sha256':OFFICIAL['public_source_api_sha256']},'official_files':OFFICIAL['official_files']}
 output=pathlib.Path(a.output).resolve()
 if ROOT/'logs' not in output.parents:raise ValueError('Ownlogs output only')
 output.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 with output.open('x') as f:json.dump(c,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(output.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 print(json.dumps({'config':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'mode':'topd300','no_submission':True}))
if __name__=='__main__':main()
