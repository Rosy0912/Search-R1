import datetime, hashlib, json
from pathlib import Path

base=Path.cwd();out=base/'outputs/gigpo-e1-policy-safety'
proofs={
 'runtime_manifest':('CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json','e1fb643f082eac44d3e91c6151c1956f635ff50f4810caa57a93c97243f67143'),
 'backend_static_review':('CALIBRATION-BACKEND-INDEPENDENT-REVIEW-v9.json','d8ede54b5c579eeb8ede3fd6de2be32cd77b838d2d3a2ba2ed78489f578bd39e'),
 'target_cpu':('CALIBRATION-DATAROW-TARGET-CPU-v9.json','577b50da6d91b89fe84a558e72b6eb0d55077a309551a89ceb458662a2d5022b'),
 'target_cpu_raw':('CALIBRATION-DATAROW-TARGET-CPU-RAW-v9.json','305598c03c794885ff6bcc4380d65d05abe15444baa85987a38de8921f4bc851'),
 'target_cpu_review':('CALIBRATION-DATAROW-TARGET-CPU-INDEPENDENT-REVIEW-v9.json','678250416095fb19d3c87c2cd477bb92278ef4f39435ce0b0e10394dfe9baaa2'),
 'analyzer_static_review':('PROBABILITY-ANALYZER-STATIC-INDEPENDENT-REVIEW-v3.json','455b18a58be1cf7f7263618233284a0071676da3b22a4c213bae3d14416f2c42'),
 'analyzer_actual_install':('PROBABILITY-ANALYZER-ACTUAL-INSTALL-v3.json','9fae4301a975120bf90abe0a67edae6dd3bd9a79cc1b39a031c131cb4f293601'),
 'optional_ab_clarification':('FORMAL-V3-OPTIONAL-AB-QUALIFICATION-CLARIFICATION-v1.json','8f8d713fe804a78e58fb4137e322868c8655cb4484fe32a58485e3b0d2ab723a'),
}
r={}
for key,(name,expected) in proofs.items():
 p=out/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 r[key]=dict(path=str(p),bytes=p.stat().st_size,sha256=expected)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
v9=dict(status='FROZEN_LOCAL_RUNTIME_STATIC_REVIEW_AND_TARGET_CPU_PASS_DEPLOYMENT_CONTROL_PREPARATION',
        proofs=r,source_commit='52befe968c856162d1bc79cd0171bb38309f4f9e',
        changed_runtime_filenames=['calibration_driver_v1.py','calibration_run_v1.py','calibration_worker_v1.py'],
        driver_sha256='31f191711844c0c875149502e9c2ddfc9569f40ef88163baf860aa46dbd762b9',
        local_backend_cpu_tests=47,target_datarow_cpu_tests=9,
        target_gpu_backend_accepted=False,probability_accepted=False,
        production_tolerances_selected=False,production_thresholds_selected=False,
        formal_iterations_contributed=0,submitted=False,job=None,
        v8_frozen_unused_preserved=True,
        optional_ab_diagnostic_not_mandatory=True,
        required_remaining='Actual v9 backend/probability and 10Adam calibration plus production26-30 same-process full acceptance; no extra optional20Adam prerequisite')
state_p=base/'work/gigpo-e1-instruct300-state.json';state=json.loads(state_p.read_text())
state['policy_safety']['backend_v9_preparation']=v9
state['policy_safety']['probability_analyzer_v3_deployed']=dict(target='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/tools/probability-analysis-52befe9-v3',
    analyzer_sha256='e95c15cfaca0544c8800b1285ee57d945512854b3ffdb86a8553cc2e0d3db619',
    failure_contract_sha256='c4659c602574c1c29b44501f804a34c94286d81a95f62d6e63c7f6cf0d9a2edb',
    install=r['analyzer_actual_install'],actual_raw_material_analyzed=False)
state['active_job']=None;state['latest_calibration_job_status']='FAILED';state['last_checked_utc']=now
state['progress_evidence']=str(out/'CALIBRATION-V9-RESOURCE-READONLY-20261006T003903Z.json')
state['phase']='POLICY_SAFETY_V9_RUNTIME_STATIC_AND_TARGET_CPU_PASS_DEPLOYMENT_PREPARATION'
state['next_action']='Finish exact isolated v9 deployment/package independent closure, actual physical READY and one Slurm GPU diagnostic; preserve unused v8/no repeat or assumed probability thresholds.'
state_p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
status_p=out/'IMPLEMENTATION-STATUS.json';status=json.loads(status_p.read_text())
status['backend_v9_preparation']=v9;status['phase']=state['phase'];status['last_checked_utc']=now
status['probability_analyzer_v3_deployed']=state['policy_safety']['probability_analyzer_v3_deployed']
status_p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
report=Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct300/REPORT.md')
with report.open('a') as f:f.write('''\n\n2026-10-06 v9后端修复准备：三个外部文件显式请求及传播FLASH_ATTN，并记录每rank实际runner与36层attention实现；任何回退或未知属性在恢复/基线/Adam前拒绝。冻结manifest e1fb643f、代码/47CPU独审d8ede54b；新driver31f191实际目标Torch2.4数据容器回归9项通过，独审67825041。上述是源码与CPU接口验收，尚未代表真实八卡Flash执行或概率一致性。只读分析器v3已实际隔离安装，新backend原件纳入精确库存而不允许任意extra；安装原件9fae4301，actualPT尚未分析。可选额外20Adam A/B草案不构成新增强制门禁，原native、新实际校准及正式首五的各自真实范围仍须接受。v9部署控制正在闭合，未提交；v8 READY原件保留不用。正式仍25/300、新0。\n''')
print(json.dumps(dict(utc=now,phase=state['phase'],actual_iterations=25,new_formal_iterations=0)))
