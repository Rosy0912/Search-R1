"""Native hardware function entry; caller owns freshHF gate/Slurm/leases/cleanup.

No CLI, automatic approval, training, remote RPC, or base fallback. Calling this
requires an independently constructed actual32B FrozenTeacher and owned runtime.
"""
import hashlib
import inspect
from pathlib import Path
import time
import torch
import capacity_reference as reference
from pipeline import PipelineScorer, SOURCE_SHA, require


def compare_serial_and_pipeline(teacher, runtime, tensors, allowed_ids, workers=(1,2,4,8)):
    """Whole phase timings + strict original GPU oracle, alloffload before nextphase.

    Caller must separately record current retriever/actor/vLLM residence, own
    actual NVML/UUID/cgroup/foreign15% sampling and cleanup. This function alone
    cannot prove coexistence, teacher64quality or permit TOP-D actor update.
    """
    require(runtime.kind=='NATIVE_UNVALIDATED_PROTOTYPE', 'Real leased native runtime required')
    require(teacher.model.config.num_hidden_layers==64 and teacher.model.config.vocab_size==152064,
            'Require actual32B fullvocab model')
    require(hashlib.sha256(Path(inspect.getfile(inspect.unwrap(teacher.log_probs))).read_bytes()).hexdigest()==SOURCE_SHA,
            'Original source drift')
    mapping=reference.device_map_32b()
    # Serial independent original method on the identical FP32 partition, one row.
    serial=reference.SharedScorer(teacher,runtime,mapping,allowed_ids)
    start=time.monotonic();expected,serial_record=serial.score_round(tensors,batch_size=1)
    records=[dict(scope='ORIGINAL_SERIAL_GPU_FUNCTION_REFERENCE_NO_TRAINING',
                  whole_phase_wall_seconds=time.monotonic()-start,record=serial_record)]
    candidate=PipelineScorer(teacher,runtime,mapping,allowed_ids)
    outputs=[]
    for count in workers:
        for repeat in range(2):
            start=time.monotonic();actual,record=candidate.score_round(tensors,workers=count)
            delta=(actual-expected).abs()
            close=torch.allclose(actual,expected,atol=1e-4,rtol=1e-4)
            records.append(dict(scope='PIPELINE_NATIVE_HARDWARE_ONLY_UNREVIEWED',workers=count,repeat=repeat,
                                whole_phase_wall_seconds=time.monotonic()-start,
                                allclose=close,atol=1e-4,rtol=1e-4,max_abs_delta=float(delta.max()),record=record))
            require(close, 'Original GPU function parity failure; preserve tensors and failure record')
            outputs.append(actual)
    return expected,outputs,records
