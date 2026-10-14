"""TRAIN-only empirical curriculum; no model, test labels, or tree-selected rewards."""
import copy
from collections import Counter
import hashlib
import random

PROTOCOL = 'recursive_question_curriculum_v1'

def quotas(total, sizes):
    n = sum(sizes.values())
    result = {k: total*v//n for k,v in sizes.items()}
    order = sorted(sizes, key=lambda k: (-(total*sizes[k] % n), k))
    for k in order[:total-sum(result.values())]:result[k] += 1
    return result

class Selector:
    def __init__(self, sources, seed=42, candidate_count=4096, pool_count=2048,
                 chosen_count=384, uniform_count=128, cap=4, window=16):
        self.sources = list(sources)
        self.strata = {k: [i for i,s in enumerate(sources) if s==k] for k in sorted(set(sources))}
        self.sizes = {k:len(v) for k,v in self.strata.items()}
        self.seed, self.candidate_count, self.pool_count = seed,candidate_count,pool_count
        self.chosen_count,self.uniform_count,self.cap,self.window = chosen_count,uniform_count,cap,window
        self.rng = random.Random(seed)
        self.pool = None
        self.history = []
        self.refresh_step = None
        self.steps = 0

    def candidates(self, step):
        rng=random.Random(int.from_bytes(hashlib.sha256(f'{self.seed}:screen:{step}'.encode()).digest()[:8],'big'))
        counts=quotas(self.candidate_count,self.sizes)
        return [i for s in sorted(self.strata) for i in rng.sample(self.strata[s],counts[s])]

    def update(self, step, candidates, rewards):
        assert len(candidates)==self.candidate_count and len(set(candidates))==len(candidates)
        assert set(rewards)==set(candidates)
        assert all(len(v)==5 and all(type(r) in (int,float) and r in (0,1) for r in v) for v in rewards.values())
        scores={i:4*(sum(rewards[i])/5)*(1-sum(rewards[i])/5) for i in candidates}
        counts=quotas(self.pool_count,self.sizes)
        self.pool=None if not any(scores.values()) else {
            s:sorted([i for i in candidates if self.sources[i]==s],key=lambda i:(-scores[i],
                hashlib.sha256(f'{self.seed}:{step}:{i}'.encode()).hexdigest()))[:counts[s]] for s in self.strata}
        self.refresh_step=step
        return scores

    def choose(self):
        exposure=Counter(i for batch in self.history[-(self.window-1):] for i in batch)
        eligible={s:[i for i in rows if exposure[i]<self.cap] for s,rows in self.strata.items()}
        selected=[]
        if self.pool is not None:
            counts=quotas(self.chosen_count,self.sizes)
            for s in sorted(self.strata):
                candidates=[i for i in self.pool[s] if exposure[i]<self.cap]
                if len(candidates)<counts[s]:
                    raise RuntimeError('Exposure cap leaves insufficient curriculum candidates; no silent oversampling')
                selected.extend(self.rng.sample(candidates,counts[s]))
            uniform=self.uniform_count
        else:uniform=self.chosen_count+self.uniform_count
        blocked=set(selected)
        for s,n in quotas(uniform,self.sizes).items():
            selected.extend(self.rng.sample([i for i in eligible[s] if i not in blocked],n))
        self.rng.shuffle(selected)
        assert len(selected)==self.chosen_count+self.uniform_count and len(set(selected))==len(selected)
        assert all(exposure[i]<self.cap for i in selected)
        self.history=(self.history+[selected])[-(self.window-1):]
        self.steps+=1
        return selected

    def state_dict(self):
        return copy.deepcopy(dict(protocol=PROTOCOL, sources=self.sources, pool=self.pool, history=self.history,
            refresh_step=self.refresh_step,steps=self.steps,rng=self.rng.getstate()))

    def load_state_dict(self,state):
        assert state['protocol']==PROTOCOL and state['sources']==self.sources
        for key in ('pool','history','refresh_step','steps'):setattr(self,key,copy.deepcopy(state[key]))
        self.rng.setstate(state['rng'])
