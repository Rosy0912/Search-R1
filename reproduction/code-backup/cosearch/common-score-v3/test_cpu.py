from common_em import score
from score_saved_boundaries import canonical
assert score('<answer>wrong</answer><answer>gold</answer>',['gold'])['em']==1
assert score('<answer>gold</answer><answer>wrong</answer>',['gold'])['em']==0
assert score('<search><answer>wrong</answer></search><answer>gold</answer>',['gold'])['em']==1
assert score('<answer><answer>gold</answer></answer>',['gold'])['em']==0
assert score('<answer>gold</answer><search>q</search>',['gold'])['em']==0
assert score('<answer>2 > 1</answer>',['2 > 1'])['em']==1
assert score('</reason><answer>gold</answer>',['gold'])['em']==1
assert score('<reason>x</think><answer>gold</answer>',['gold'])['em']==0
assert score('<search><answer>gold</answer></search>',['gold'])['em']==0
r=lambda i:{'question_uid':str(i),'question_answer_sha256':str(i),'model_actions':[{'author':'main_model','text':'<answer>gold</answer>','fully_retained':True}],'references':['gold'],'data_source':'nq','em':1}
rows=[r(i) for i in range(51713)];rows+= [r(51712) for _ in range(7)]
kept,pad=canonical(rows,51713);assert len(kept)==51713 and pad==7
try:canonical(rows[:-1],51713)
except RuntimeError:pass
else:raise AssertionError('Malformed native padding accepted')
assert len(canonical([r(i) for i in range(512)],512)[0])==512
print('PASS orphan-format independent EM,model-only terminal answers,exact native final padding and512coverage')

rows=[r(i) for i in range(512)];rows[0]['model_actions'][0]['fully_retained']=False
assert canonical(rows,512)[0][0]['em']==1
rows[0]['model_actions'][0]['text']='<answer>gold'
assert canonical(rows,512)[0][0]['em']==0
print('PASS full actually-generated model action scored independently of downstream tensor retention')
