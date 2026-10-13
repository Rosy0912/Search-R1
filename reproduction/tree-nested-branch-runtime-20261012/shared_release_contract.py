"""Explicit release binding for the separate own shared training adapter."""
import math
from production_contract import require,descriptor
from shared_identity import PROTOCOL

def verify_shared_release(package,release):
    spec=package['dispatch']
    require(release['dispatch_kind']=='shared_systemd' and release['resource_protocol']==PROTOCOL
            and release['physical_gpus']==8 and release['slurm_reserved_gpus']==0
            and release['shared_memory_contract']=={key:spec[key] for key in
                 ('initial_required_free_gib','runtime_required_headroom_gib','own_memory_cap_gib','measured_training_budget_gib','expected_gpu_uuids','initial_required_host_mem_available_gib','minimum_host_cpu_capacity')},
            'Shared release exact physical/memory protocol differs')
    require(spec['initial_required_free_gib']==68 and spec['runtime_required_headroom_gib']==8
            and spec['own_memory_cap_gib']==72 and spec['measured_training_budget_gib']==60
            and spec['initial_required_host_mem_available_gib']==320 and spec['minimum_host_cpu_capacity']==32
            and spec['exclusive'] is False and spec['physical_gpus']==8 and spec['slurm_reserved_gpus']==0,
            'Shared training admission/headroom/cap differs')
    require(release['completed200_memory_peak_report']==spec['completed200_memory_peak_report'], 'Final memory audit descriptor differs from reviewed source/package')
    report,_,_=descriptor(release['completed200_memory_peak_report'])
    require(report['job_id'] in ('3360',3360) and report['completed_outer']==200
            and type(report['resource_samples']) is int and report['resource_samples']>0
            and set(report['whole_card_peaks'])==set(spec['expected_gpu_uuids']),
            'Actual complete200 sampled memory evidence required')
    require(report['code_commit']==package['predecessor_gate']['source_commit']
            and report['config_sha256']==package['predecessor_gate']['config_sha256'],
            'Actual memory/cost evidence source/config differs')
    evidence=report.get('evidence_sha256',report.get('input_sha256'))
    require(type(evidence) is dict and evidence,'Final200 cost report underlying evidence bindings missing')
    from predecessor_gate import hash_bound
    for path,expected in evidence.items():hash_bound(dict(path=path,sha256=expected))
    maximum=max(row['total_used_memory_bytes'] for row in report['whole_card_peaks'].values())
    require(type(maximum) is int and 0<maximum<=60*1024**3,
            'Final sampled peak needs more than60GiB budget; raise reviewed admission without shrinking training')
    require(type(release['gpu_sharing_authority']) is str and release['gpu_sharing_authority'].strip(),
            'Latest explicit user sharing authorization must be bound')
    require(release['sampled_peak_caveat']=='Five-second whole-card sampled occupancy is a lower bound; admission8GiB margin and continuous1s physical free guard',
            'Sampled versus continuous peak caveat required')
    return maximum
