from pathlib import Path
import shutil,json,hashlib
base=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004')
source=base/'runtime/r1';target=base/'runtime/r1-full-resume-20261004'
assert not target.exists();shutil.copytree(source,target,symlinks=True)
shutil.copy2(base/'source/experiments/search_e1/control/full_resume.py',target/'search_e1/full_resume.py')
p=target/'search_e1/grpo_main.py';s=p.read_text().replace("'E1_VIRTUAL_DP_WORLD']", "'E1_VIRTUAL_DP_WORLD', 'E1_FULL_RESUME_STATE']");assert 'E1_FULL_RESUME_STATE' in s;p.write_text(s)
p=target/'upstream/Search-R1/verl/trainer/ppo/ray_trainer.py';s=p.read_text()
s=s.replace('        self.global_steps = 0\n','''        self.global_steps = 0
        resume_directory = os.environ.get('E1_FULL_RESUME_STATE')
        if resume_directory:
            from search_e1.full_resume import restore_group
            self.global_steps = restore_group(self.actor_rollout_wg, resume_directory)
''',1)
needle='''            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')'''
replacement='''            for batch_index, batch_dict in enumerate(self.train_dataloader):
                if resume_directory and epoch == 0 and batch_index < 2:
                    print('RESUME_SKIP_ALREADY_TRAINED_BATCH', batch_index, flush=True)
                    continue
                print(f'epoch {epoch}, step {self.global_steps}')'''
assert needle in s;s=s.replace(needle,replacement)
# One normal checkpoint at first resumed update, then original cadence50.
needle='self.global_steps % self.config.trainer.save_freq == 0'
assert needle in s;s=s.replace(needle,"(self.global_steps % self.config.trainer.save_freq == 0 or (resume_directory and self.global_steps == 3))")
p.write_text(s)
files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(base/'receipts/r1-full-resume-runtime.sha256.json').write_text(json.dumps(files,indent=2));print('SEPARATE_RESUME_RUNTIME_BUILT',len(files),flush=True)
