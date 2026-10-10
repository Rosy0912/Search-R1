from pathlib import Path
import shutil,ast,json,hashlib
pub=Path('work/search-e1-first20-report-publish-20261010/experiments/search_e1')
src=pub/'control/opsd_j_bf16_200_v1';dst=pub/'control/opsd_teacher_offpolicy200_v1'
dst.mkdir(exist_ok=False)
for p in src.glob('*.py'):shutil.copyfile(p,dst/p.name)
def edit(n,a,b):
 p=dst/n;s=p.read_text();assert a in s,(n,a);p.write_text(s.replace(a,b))
edit('miner_v1.py','outer_index<10','outer_index<200') if 'outer_index<10' in (dst/'miner_v1.py').read_text() else None
p=dst/'miner_v1.py';s=p.read_text().replace('outer_index < 10','outer_index < 200');p.write_text(s)
edit('controller.py',"no_old_adam_restored=False,continuation=self.config['continuation']","no_old_adam_restored=True,trajectory_generator='fixed_source100_ordinary'")
p=dst/'controller.py';s=p.read_text();a=s.index("        prefix=self.config['continuation']");z=s.index('            outer=self.output',a);s=s[:a]+"        previous=None;updates=0;cursor=dict(cycle=0,offset=0)\n        for index in range(self.config['scientific']['outer_iterations']):\n"+s[z:]
a=s.index("                if previous is None and self.config['precision']['mode']=='bf16':");z=s.index("                self.stage('SAMPLING'",a)
s=s[:a]+"                model_identity=dict(kind='raw-source100',path=self.config['paths']['model'],sha256=self.config['source_model_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256'],optimizer_updates=0,adapter=None,checkpoint=None,adapter_tensor_sha256=self.config['common_initial_adapter_tensor_sha256'],variant=self.config['variant'],precision=self.config['precision'],common_initial_adapter=self.config['common_initial_adapter_manifest'])\n"+s[z:]
s=s.replace("if (index+1)%10==0:self.evaluate(index+1,previous)","")
s=s.replace("cursor=manifest['train_cursor']['after'];continue","cursor=manifest['train_cursor']['after']\n                if (index+1)%10==0:self.evaluate(index+1,previous)\n                continue")
s=s.replace("            from checkpoint import prune_merged_caches","            if (index+1)%10==0:self.evaluate(index+1,previous)\n            from checkpoint import prune_merged_caches")
p.write_text(s)
p=dst/'trainer.py';s=p.read_text();a=s.index("    if identity['checkpoint'] != previous");z=s.index('    return identity',a);s=s[:a]+"    if identity['kind']!='raw-source100' or identity['checkpoint'] is not None or identity['optimizer_updates']!=0 or identity['adapter_tensor_sha256']!=config['common_initial_adapter_tensor_sha256']:\n        raise ValueError('Off-policy sampling must remain fixed source100 ordinary teacher')\n    if previous is not None and (prior_checkpoint is None or prior_checkpoint['optimizer_updates']!=outer['cumulative_optimizer_updates_before']):\n        raise ValueError('Student optimizer continuation differs from preceding completed outer')\n"+s[z:];p.write_text(s)
# Reuse exact periodic evaluation implementation with new base and local worker.
old=pub/'control/opsd_j_bf16_periodic200_v1/entry.py';tree=ast.parse(old.read_text());cl=next(x for x in tree.body if isinstance(x,ast.ClassDef));method=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='evaluate');lines=old.read_text().splitlines();block='\n'.join(lines[method.lineno-1:method.end_lineno]);block=block.replace("destination=B/'periodic512'", "destination=Path(self.config['paths']['base'])/'periodic512'").replace("str(R/'worker.py')","str(Path(self.config['paths']['helper_root'])/'eval_worker.py')")
p=dst/'controller.py';s=p.read_text().replace('    def run(self):',block+'\n\n    def run(self):');p.write_text(s)
shutil.copyfile(pub/'control/opsd_j_bf16_periodic200_v1/worker.py',dst/'eval_worker.py')
for p in dst.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
print(dst)
