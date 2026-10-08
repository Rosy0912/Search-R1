"""Actual new diagnostic runtime/context precheck; never CUDA or optimizer."""
from diagnostic_contract import *
require(not sys.flags.optimize and sys.platform=='linux'and os.uname().nodename=='h100-5'and sys.prefix==str(SOURCE_ROOT/'env'),'Actual target diagnostic CPU environment required')
require(subprocess.check_output(['git','-C',str(SOURCE_ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Fixed source differs')
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';sys.path.insert(0,str(SOURCE_ROOT/'repo'))
import torch
from verl import DataProto
from search_r1.llm_agent.sampling_tokens import audit_batch
from search_r1.baselines.fresh200_contract import critical_source_hashes,verify_official_model_assets
require(sha(RAW)==RAW_SHA,'Actual3307 saved raw bytes differ')
payload=torch.load(RAW,map_location='cpu',weights_only=False)
data=DataProto.from_dict(tensors=payload['tensors'],non_tensors=payload['non_tensor_batch'],meta_info=payload['meta_info'])
audit=audit_batch(data)
require(audit['context_verified']and audit['row_count']==2560 and audit['probability_role']=='stochastic_sampling_policy','Actual2560 context/journal audit required')
require(not torch.cuda.is_initialized(),'CPU precheck initialized CUDA')
pre=dict(status='ACTUAL_ZERO_ADAM_CPU_CONTEXT_INPUT_PRECHECK_NOT_NATIVE',source_commit=SOURCE,raw_sha256=RAW_SHA,critical_source_sha256=critical_source_hashes(),rows=2560,audit=audit,official_model_manifest_sha256=verify_official_model_assets(MODEL))
exe=Path(sys.executable).resolve();stat=exe.stat();require(Path('/proc/self/exe').resolve()==exe,'Actual executable identity differs')
runtime=dict(source_commit=SOURCE,root=str(ROOT),sys_prefix=sys.prefix,host=os.uname().nodename,scope='ACTUAL_ZERO_ADAM_DIAGNOSTIC_INTERPRETER_NOT_NATIVE_ACCEPTANCE',executable=dict(resolved_path=str(exe),device=stat.st_dev,inode=stat.st_ino),executable_sha256=sha(exe))
for name,value in [('diagnostic-cpu-precheck.json',pre),('runtime-executable.json',runtime)]:
 with (ROOT/'logs'/name).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(precheck=bind(ROOT/'logs/diagnostic-cpu-precheck.json'),runtime=bind(ROOT/'logs/runtime-executable.json'))))
