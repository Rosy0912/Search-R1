import copy
import pytest
from prepare import derive
from test_standard_r1 import prior
from source_binding import check_fresh_base
from supervise import verify_recipe

def config():return derive(prior(),'profile2','/dataset/shaoshitong/search-r1-rl/topd/logs/freshbase-v12')
def test_actual_caller_accepts_fresh_zero_none():
 c=config();assert check_fresh_base(c) and verify_recipe(c)==2
 c['training_command'].append('+actor_rollout_ref.model.actor_checkpoint_path=null')
 assert check_fresh_base(c) and verify_recipe(c)==2
@pytest.mark.parametrize('kind',['paired_continuation','checkpoint_only','duplicate_override'])
def test_actual_caller_rejects_weight_continuation_checkpoint_or_duplicate(kind):
 c=config()
 if kind=='paired_continuation':c['training_command']=[x.replace('weight_continuation_step=0','weight_continuation_step=1') for x in c['training_command']];c['training_command'].append('+actor_rollout_ref.model.actor_checkpoint_path=/own/unapproved-model')
 if kind=='checkpoint_only':c['training_command'].append('+actor_rollout_ref.model.actor_checkpoint_path=/own/unapproved-model')
 if kind=='duplicate_override':c['training_command'].append('trainer.weight_continuation_step=0')
 with pytest.raises(ValueError):check_fresh_base(c)
 with pytest.raises(ValueError):verify_recipe(c)
