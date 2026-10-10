"""Exercise the migrated FAISS GPU extension only inside the owned four-GPU job."""
import ctypes,json,os,time,uuid
from pathlib import Path

def visible_cuda_uuids(count):
    # CUDA driver ordinals obey the process visibility mask, unlike physical NVML indices.
    driver=ctypes.CDLL('libcuda.so.1')
    def checked(code):
        if code:raise RuntimeError('CUDA driver identity query failed: '+str(code))
    checked(driver.cuInit(0))
    result=[]
    for ordinal in range(count):
        device=ctypes.c_int();checked(driver.cuDeviceGet(ctypes.byref(device),ordinal))
        raw=(ctypes.c_ubyte*16)();checked(driver.cuDeviceGetUuid(ctypes.byref(raw),device))
        result.append('GPU-'+str(uuid.UUID(bytes=bytes(raw))))
    return result

def main():
    import torch,numpy as np,faiss
    leased=os.environ['COSEARCH_LEASED_UUIDS'].split(',')
    assert os.environ['SLURM_JOB_ID'] and len(set(leased))==4
    assert torch.cuda.device_count()==4 and faiss.get_num_gpus()==4
    actual=visible_cuda_uuids(4)
    assert set(actual)==set(leased)
    # Clear-margin, exactly representable vectors avoid false FP16 ranking ambiguity.
    vectors=np.eye(64,dtype=np.float32);queries=vectors[[0,7,31,63]]
    cpu=faiss.IndexFlatIP(64);cpu.add(vectors)
    options=faiss.GpuMultipleClonerOptions();options.shard=True;options.useFloat16=True
    gpu=faiss.index_cpu_to_all_gpus(cpu,co=options)
    scores,indices=gpu.search(queries,1);expected_scores,expected_indices=cpu.search(queries,1)
    assert np.array_equal(indices,expected_indices)
    assert np.array_equal(scores,expected_scores)
    receipt={'status':'FOUR_OWN_GPU_FAISS_CLEAR_MARGIN_PARITY_PASS','epoch':time.time(),'job':os.environ['SLURM_JOB_ID'],'uuids':actual,'faiss_version':faiss.__version__,'full_index_runtime_pending':True,'e5_runtime_pending':True}
    (Path(os.environ['COSEARCH_RUN_ROOT'])/'faiss-small-gpu-parity.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
