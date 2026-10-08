"""Read-only synthetic CPU demonstration; not native vLLM/GPU evidence."""
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / 'tree-behavior-mu-ppo'
sys.path.insert(0, str(SOURCE))
import test_tree_behavior_mu_data_cpu as fixture

factory = fixture.BehaviorMuDataTests()
factory.setUp()
try:
    original = fixture.params
    def selected_params():
        value = original()
        value.best_of = 2
        return value
    fixture.params = selected_params
    output = factory.output()
    result = factory.validate(output)
    settings = output.non_tensor_batch['tree_raw_semantics'][0]['raw_ledger'][0]['action']['native_emission']['sampling_params']
    print(json.dumps(dict(actual_SamplingParams_best_of=2,
        recorded_best_of=settings.get('best_of', 'MISSING'), recorded_fields=len(settings),
        validator_status=result['status'], production_frozen_config_best_of_enabled=False,
        scope='Synthetic CPU attribute accepted and omitted by actual capture/compose/validate; not native vLLM distribution behavior'), indent=2))
finally:
    factory.tearDown()
