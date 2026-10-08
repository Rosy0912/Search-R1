import shutil,json,hashlib
from pathlib import Path
src=Path('work/tree-fixed-five-296217a');dst=Path('work/tree-fixed-formal-296217a-draft');dst.mkdir(exist_ok=True)
for f in src.iterdir():
 if f.is_file() and f.name!='manifest.json': shutil.copyfile(f,dst/f.name)
p=dst/'supervisor.py';s=p.read_text().replace("choices=['five']","choices=['formal']")
s=s.replace("    a=p.parse_args()","    p.add_argument('--actual-five-review',type=Path,required=True); p.add_argument('--actual-five-review-sha',required=True)\n    a=p.parse_args()")
marker="    lock=(ROOT/'shared-gpu-job.lock').open('w');"
gate='''    require(sha(a.actual_five_review)==a.actual_five_review_sha,'Actual five independent receipt missing/changed')
    five=json.loads(a.actual_five_review.read_text())
    require(five['status']=='PASS' and five['stage']=='five' and five['candidate_commit']==COMMIT and five['actual_outer_updates']==5,'Actual five candidate gate missing')
    require(five['scientific_five_accepted'] and five['operational_five_accepted'] and five['hf5_full_cpu_reload'],'Actual five scientific/operational/HF gate incomplete')
    require(five['original_slurm_state']=='COMPLETED' and five['original_slurm_exit_code']=='0:0' and five['training_body_exit_code']==0 and five['entry_config_verified'] and five['cleanup_residual']=={},'Actual five terminal gate incomplete')
    require(five['direct_rank_outer_records']==40 and five['tree_audit_files']==10 and five['actual_memory_profile_records']==40,'Actual five coverage incomplete')
    roots=[ROOT/'runs/tree-fixed-five-3103',ROOT/'runs/tree-fixed-five-cpu-3103-3104']
    require(bool(five['evidence']) and bool(five['small_review_bundle_mapping']),'Actual five proof lists empty')
    for item in five['evidence']:
        path=Path(item['path']); require(any(str(path.resolve()).startswith(str(root)+'/') for root in roots),'Prior evidence outside actual five owned runs')
        require(sha(path)==item['sha256'],'Actual five immutable evidence changed')
    for item in five['small_review_bundle_mapping']:
        path=Path(item['path']); require(str(path.resolve()).startswith(str(ROOT/'logs/tree-fixed-five-3103-review')+'/'),'Prior independent five review outside owned bundle')
        require(sha(path)==item['sha256'],'Actual five independent review changed')
'''
assert marker in s;s=s.replace(marker,gate+marker);p.write_text(s)
p=dst/'validation_cost_hooks_fixed_five.py';new=dst/'validation_cost_hooks_fixed_formal.py';new.write_text(p.read_text().replace('tree-fixed-five-','tree-fixed-formal-').replace('fixed-five Slurm job','fixed-formal Slurm job'));p.unlink()
p=dst/'validation_cost_entry.py';p.write_text(p.read_text().replace('validation_cost_hooks_fixed_five','validation_cost_hooks_fixed_formal'))
p=dst/'run_tree_with_validation_cost_fixed_short.sh';p.write_text(p.read_text().replace('tree-fixed-five-296217a','tree-fixed-formal-296217a'))
(dst/'DRAFT_NOT_DISPATCHABLE.json').write_text(json.dumps({'status':'DRAFT_ONLY_WAITING_ACTUAL5_AND_INDEPENDENT_FORMAL_PRE_DISPATCH','fresh_base':True,'actual_target_outer':500,'counter_budget':501,'source':'296217a4b676a26a92fc729f5d9882b785dbfa96','no_manifest_or_wrapper_yet':True},indent=2)+'\n')
p=Path('work/check_fixed_formal_afterany_draft.py');p.write_text(Path('work/check_fixed_five_afterany.py').read_text().replace('HF5','HF500').replace('fullHF5','fullHF500').replace('hf5','hf500').replace('tree-fixed-five','tree-fixed-formal').replace('global_step_5','global_step_500').replace('not5 independent acceptance or500','not independent formal acceptance'))
print('Formal local draft prepared; no upload, manifest, launch wrapper or submission.')
