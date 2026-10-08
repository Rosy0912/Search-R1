import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
import inspect,json,importlib.metadata
from tensordict import TensorDict
print(json.dumps({'tensordict_version':importlib.metadata.version('tensordict'),'cuda_source':inspect.getsource(TensorDict.cuda),'to_source':inspect.getsource(TensorDict.to)},indent=2))
