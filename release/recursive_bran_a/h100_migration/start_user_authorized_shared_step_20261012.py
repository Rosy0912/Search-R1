import pathlib,subprocess,json,time,os
b=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert not (b/'receipts/train-start.json').exists()
ready=b/'control/launch-ready.sh';assert ready.exists()
launch=b/'control'/('launch-user-shared-step-'+str(int(time.time()))+'.sh')
ready.rename(launch)
logpath=b/'logs/shared-h10011-step.log';log=logpath.open('ab')
command=['srun','--jobid=3389','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=48','--mem=256G','--gres=gpu:h100:8','--nodelist=h100-11','bash','-c','export RBA_H100_ALLOCATION_ID="$SLURM_JOB_ID"; bash "$1"; rc=$?; echo "ACTUAL_SHARED_TRAIN_ENTRY_EXIT=$rc"; exit "$rc"','bash',str(launch)]
p=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
d=dict(time=time.time(),controller_pid=p.pid,own_entry=str(launch),log=str(logpath),requested_existing_allocation='3389',mode='scheduler_authorized_overlapping_step',new_allocation_created=False,foreign_tasks_stopped=False,not_claiming_training_started=True)
(b/'receipts/shared-h10011-step-launch.json').write_text(json.dumps(d));print(json.dumps(d),flush=True)
