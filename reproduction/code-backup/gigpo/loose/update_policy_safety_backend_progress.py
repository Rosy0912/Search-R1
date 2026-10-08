import datetime, hashlib, json
from pathlib import Path

base = Path.cwd()
out = base / 'outputs/gigpo-e1-policy-safety'
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
proofs = {
    'v8_diagnostic_ready': ('policy-safety-calibration-52befe9-reviewed-binding-v8.json', '180431b9fe8d747e5f33f7e756b7d32181cb55c94cc623ebe788c3d4273d72d5'),
    'v8_actual_freeze': ('CALIBRATION-V8-ACTUAL-FREEZE-v1.json', 'f210a8af505521618ce592150ad5faf21dcba78828c8bbca59ece22ebcac841c'),
    'v8_actual_deployment_review': ('CALIBRATION-V8-ACTUAL-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json', '59a57c65f8b41ff0e86ea885df9a9f5ac25d3861f441452147c632cbfcd8ba4c'),
    'perturn_probability_review': ('PROBABILITY3281-PERTURN-PT-CPU-INDEPENDENT-REVIEW-v1.json', '6dc447d5de0fe763000b677f2626616710e54c8a526352a04660e329fda576d4'),
    'restore25_rope_buffer_review': ('PROBABILITY-MISMATCH-RESTORE25-BUFFERS-CPU-v2.json', '6cd9763760a888b2825a60c396f1b1042812d9d6cfde504228f0b065f13eae93'),
}
actual_proofs = {}
for key, (name, expected) in proofs.items():
    p = out / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, (key, p)
    actual_proofs[key] = {'path': str(p), 'sha256': expected}

update = {
    'repair_v8_frozen_diagnostic_ready': actual_proofs['v8_diagnostic_ready'],
    'repair_v8_freeze_actual': actual_proofs['v8_actual_freeze'],
    'repair_v8_actual_deployment_review': actual_proofs['v8_actual_deployment_review'],
    'repair_v8_submitted': False,
    'repair_v8_job': None,
    'repair_v8_submission_scope': 'Frozen diagnostic-only deployment preserved unused; no submit intent or receipt created',
    'probability_identity_accepted': False,
    'probability_backend_investigation': {
        'status': 'PREPARING_ISOLATED_V9_ACTUAL_BACKEND_DIAGNOSTIC',
        'actual3281_hf_vllm_max_abs_logprob': 3.1895965933799744,
        'actual3281_vllm_backend': 'XFORMERS: actual selector INFO, not an inferred default',
        'training_backend': 'HF FlashAttention2',
        'explicit_forcing_locations': ['frozen calibration_run_v1.py os.environ VLLM_ATTENTION_BACKEND=XFORMERS', 'frozen source52 verl/trainer/runtime_env.yaml env_vars VLLM_ATTENTION_BACKEND=XFORMERS'],
        'propagation_scope_pending': 'Ray driver/worker env and actual all8 model_runner backend identity must be established; wrapper flag alone is not evidence',
        'cause_proven': False,
        'tolerance_or_threshold_relaxed': False,
        'rope_buffers_repaired': False,
        'restore25_rope_buffers': 'All37 original inv_freq buffers FP32 and exact official canonical values; no buffer repair justified',
        'proofs': {key: actual_proofs[key] for key in ('perturn_probability_review', 'restore25_rope_buffer_review')},
        'v9_actual_gpu_supported_or_parity_accepted': False,
    },
}
state_path = base / 'work/gigpo-e1-instruct300-state.json'
state = json.loads(state_path.read_text())
state['phase'] = 'POLICY_SAFETY_V8_DIAGNOSTIC_READY_UNUSED_V9_BACKEND_PREPARATION'
state['actual_iterations'] = 25
state['new_formal_iterations'] = 0
state['last_checked_utc'] = now
state['next_action'] = 'Preserve frozen unused v8; isolate v9 explicit backend propagation and actual all8 backend identity, independently review then unique Slurm diagnostic. No probability tolerance or production thresholds chosen before actual accepted evidence.'
state['policy_safety'].update(update)
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
status_path = out / 'IMPLEMENTATION-STATUS.json'
status = json.loads(status_path.read_text())
status.update(update)
status['phase'] = state['phase']
status['last_checked_utc'] = now
status['retained_iterations'] = 25
status['new_formal_iterations'] = 0
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')

def replace_marker(path, marker, end_marker, content):
    text = path.read_text()
    start = text.index(marker) + len(marker)
    end = text.index(end_marker, start)
    path.write_text(text[:start] + '\n' + content + '\n' + text[end:])

replace_marker(out / 'DESIGN.md', '<!-- CURRENT-IMPLEMENTATION -->', '<!-- END-CURRENT-IMPLEMENTATION -->', '''当前实施状态（北京时间2026-10-06）：正式保留25/300，新增0。三项防护源52befe968c856162d1bc79cd0171bb38309f4f9e已提交，八卡数值/分布式/完整恢复诊断3267已独立接受。校准3281在记录接口失败前完成健康25完整恢复、512/64基线和四轮采样；实际训练反向、Adam、新outer均为0，原件和资源释放保留。
v8外部记录格式修复已实际部署、独立核对并冻结为诊断READY（SHA180431b9fe8d747e5f33f7e756b7d32181cb55c94cc623ebe788c3d4273d72d5），没有提交，不存在v8作业。它不能代表概率一致性、校准阈值或正式训练接受。
对3281原始逐轮概率文件的只读核查发现HF与vLLM最大log-prob差3.1896；tokens、输入前缀和mask一致，37个原始RoPE buffer也为官方精确FP32值。实际vLLM日志显示XFormers，外部启动器与source52运行配置明确强制XFORMERS，而HF训练侧使用FlashAttention2。后端差异是有证据的排查方向，尚未证明全部误差的因果来源。正在准备独立v9显式传播与实际八rank后端记录；不得放宽容差、修改旧buffer或冒称GPU Flash支持。下方早期快照是保留历史。''')

report = Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct300/REPORT.md')
replace_marker(report, '<!-- CURRENT-POLICY-SAFETY-STATUS -->', '<!-- END-CURRENT-POLICY-SAFETY-STATUS -->', '''当前状态（北京时间2026-10-06 08:35）：正式训练保留 **25/300轮，新增0轮**。三项防护代码已加入，八卡数值与完整状态诊断通过；正式续训尚未开始，刹车阈值仍须实际校准。

校准3281已经退出并释放资源。健康第25轮完整状态实际恢复通过；同512题F1 **32.3198%**、EM **26.1719%**，64题F1 **37.9861%**。四轮采样后，新增记录字段的数组格式不兼容，在第一次参数更新前拒绝，实际新增Adam和训练反向均0。该接口修复v8已部署并独立冻结为仅诊断READY，但没有提交新作业。

原始概率记录还显示生成端与训练端最大log-prob差 **3.1896**。输入token、前缀、mask和保存RoPE buffer的只读核查通过；实际生成日志显示XFormers，而HF训练用FlashAttention2，启动配置存在明确强制XFormers的设置。正在独立v9修正后端传播并准备实际八卡核验，尚未断言该差异就是全部误差原因，也没有放宽容差。只有实际概率一致性和校准接受后才开始正式第26轮。

集群：武汉H100，授权节点 **h100-8**。恢复来源：`/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25`。失败诊断目录：`/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-policy-safety-calibration25-job3281-1791243032869507143`。新源码提交：`52befe968c856162d1bc79cd0171bb38309f4f9e`；本地防护实现与证据目录：`/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/`。全部旧实验、代码、日志、轨迹和checkpoint原位保留，下方历史轮数不拼入当前25/300。''')
print(json.dumps({'phase': state['phase'], 'actual_iterations': 25, 'new_formal_iterations': 0, 'updated_utc': now}))
