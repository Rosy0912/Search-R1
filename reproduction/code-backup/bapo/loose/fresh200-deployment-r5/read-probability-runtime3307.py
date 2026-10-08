from pathlib import Path
import os,sys,json,inspect
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006');assert os.uname().nodename=='h100-5'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(root/'repo'));os.chdir(root/'repo')
import torch,transformers
import verl.utils.torch_functional as tf
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer
model=root/'assets/qwen2.5-3b-instruct-aa8e725'
tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
p=root/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307/checkpoints/diagnostics/formal-sampling-attempt_1/formal-first-sampling-batch.pt'
a=torch.load(p,map_location='cpu',weights_only=False)
rows=[]
for row,pos in [(1595,575),(1865,588),(500,40),(119,106)]:
 t=a['tensors']; tokens=t['responses'][row,max(0,pos-12):pos+8].tolist();rows.append({'row':row,'position':pos,'nearby_tokens':tokens,'nearby_text':tok.decode(tokens),'token':tok.decode([int(t['responses'][row,pos])])})
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'flash_cross_entropy_available':tf.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,'torch_functional_path':tf.__file__,'qwen_forward_source':inspect.getsource(Qwen2ForCausalLM.forward),'meta_info':a['meta_info'],'examples':rows},default=str))
