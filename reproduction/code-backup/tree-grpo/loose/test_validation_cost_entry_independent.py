import hashlib,types,pathlib
import pytest
import validation_cost_entry as entry

@pytest.mark.parametrize('polluted,options,accepted',[(False,{},True),(True,{},False),(False,{'num_cpus':2},False)])
def test_actual_loader_pins_source_and_preserves_default_remote_options(tmp_path,monkeypatch,polluted,options,accepted):
 source=pathlib.Path(__file__).parent/'tree-grpo/verl/trainer/main_ppo.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==entry.EXPECTED
 module=types.SimpleNamespace(__file__=str(tmp_path/'main_ppo.py'));pathlib.Path(module.__file__).write_bytes(b'polluted' if polluted else source.read_bytes())
 calls=[];original_function=lambda config:config;replacement=object()
 class OriginalLoader:
  def exec_module(self,module):calls.append('standard_exec');module.main_task=types.SimpleNamespace(_default_options=options,_function=original_function)
 def measured(function):assert function is original_function;calls.append('replace_remote');return replacement
 monkeypatch.setattr(entry,'measured_remote',measured);loader=entry.Loader(OriginalLoader())
 if accepted:loader.exec_module(module);assert module.main_task is replacement and calls==['standard_exec','replace_remote']
 else:
  with pytest.raises(AssertionError):loader.exec_module(module)
  assert calls==([] if polluted else ['standard_exec'])
