from pathlib import Path
import json,csv,hashlib,tempfile,os
p=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/search-e1-pr100-20261010/public/first10-training')
s=json.loads((p/'summary.json').read_text()); rows=s['train_outer_records']
with (p/'committed-adam-loss.csv').open() as f: updates=list(csv.DictReader(f))
with tempfile.TemporaryDirectory(prefix='pr100-plot-cache-') as cfg:
 os.environ['MPLCONFIGDIR']=cfg
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import MaxNLocator
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
 fig,axes=plt.subplots(3,1,figsize=(10.5,9.3),gridspec_kw={'height_ratios':[1,1,1.1]})
 x=[r['outer_number'] for r in rows]
 axes[0].plot(x,[100*r['train_em'] for r in rows],color='#1f6f8b',marker='o',linewidth=2,markersize=5)
 axes[0].set(title='A  TRAIN sampled EM before each outer (different cohorts)',ylabel='Strict EM (%)',xlabel='Online outer')
 axes[0].set_ylim(42,53);axes[0].set_xticks(x)
 for r in rows: axes[0].annotate(f"{100*r['train_em']:.2f}",(r['outer_number'],100*r['train_em']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
 axes[1].plot(x,[r['raw_loss'] for r in rows],color='#777777',marker='o',linewidth=1.7,label='Raw reverse KL')
 axes[1].plot(x,[r['post_loss'] for r in rows],color='#c45d1c',marker='o',linewidth=1.9,label='Post-clipping loss')
 axes[1].set(title='B  Recorded TRAIN loss per outer: equal trajectory mean',ylabel='Recorded loss',xlabel='Online outer')
 axes[1].set_xticks(x);axes[1].legend(loc='upper right',frameon=False,ncol=2)
 u=[int(r['actual_adam']) for r in updates]
 axes[2].plot(u,[float(r['raw_loss']) for r in updates],color='#777777',linewidth=1.1,alpha=.8,label='Raw reverse KL')
 axes[2].plot(u,[float(r['post_loss']) for r in updates],color='#c45d1c',linewidth=1.3,label='Post-clipping loss')
 for r in rows[:-1]: axes[2].axvline(r['actual_adam_cumulative']+.5,color='#bbbbbb',linewidth=.8,linestyle=':')
 axes[2].set(title='C  Recorded loss for each committed Adam update (99 updates)',ylabel='Recorded loss',xlabel='Actual cumulative Adam updates')
 axes[2].set_xlim(1,99);axes[2].legend(loc='upper right',frameon=False,ncol=2)
 for ax in axes: ax.grid(axis='y',alpha=.2);ax.yaxis.set_major_locator(MaxNLocator(5))
 fig.suptitle('PR100 | accepted first 10 online outers',fontsize=16,fontweight='bold',x=.09,ha='left')
 fig.text(.09,.014,'EM: 512 different TRAIN questions × 5 sampled trajectories per outer. Loss: recorded trajectory token means,\nweighted equally across actual trajectories; tail batches included. Heldout512 evaluations completed at this snapshot: 0.',fontsize=9,color='#555555',va='bottom')
 fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.105,hspace=.65)
 out=p/'first10-train-em-loss.png';fig.savefig(out,dpi=140);plt.close(fig)
 version=matplotlib.__version__

def ref(q): return {'path':str(q),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
provenance={'schema':'pr100-static-training-plot-v1','plot':ref(out),'matplotlib_version':version,'inputs':[ref(p/'summary.json'),ref(p/'committed-adam-loss.csv'),ref(p/'complete.json')],'scope':'Aggregate-only plot. No model scoring, bootstrap, trajectory/token reads, or smoothing. Varying TRAIN cohorts; pre-update sampled EM. Loss values are recorded before optimizer update, averaged equally across actual trajectories. Heldout512 completion count is zero at this snapshot.','training_run_id':s['training_run_id'],'reporter_commit':'176f50f41659bdf0361e64eb30b76eaaacb4d556'}
(p/'plot-provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
print(json.dumps({'plot':ref(out),'provenance':ref(p/'plot-provenance.json'),'file_count':len(list(p.iterdir()))}))
