"""Original Controller, with one exact torchrun script-path replacement."""
import argparse
import importlib
import os
from pathlib import Path
import sys
from recovery_common import bind, install_original, receipt, require


def rewrite(command, phase, config, config_ref, original_root, launcher):
    original = str(original_root / 'trainer.py')
    if phase != 'training-running':
        require(original not in command, 'Trainer appeared outside the original training phase')
        return list(command), False
    prefix = [config['paths']['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', original]
    require(command[:6] == prefix and command.count(original) == 1, 'Training command is not the exact original eight-rank torchrun')
    for option, expected in [('--config', config_ref['path']), ('--config-sha256', config_ref['sha256'])]:
        require(command.count(option) == 1 and command[command.index(option) + 1] == expected, 'Training original config changed')
    result = list(command)
    result[5] = str(launcher)
    require(result[:5] == command[:5] and result[6:] == command[6:], 'Replacement changed original options')
    return result, True


def main():
    parser = argparse.ArgumentParser()
    for field in ('repair-manifest', 'repair-manifest-sha256', 'config', 'config-sha256', 'run-id'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only overlay controller required')
    repair_ref = dict(path=args.repair_manifest, sha256=args.repair_manifest_sha256)
    config_ref = dict(path=args.config, sha256=args.config_sha256)
    value, config, root, manifest, _ = bind(repair_ref, config_ref, args.run_id, resume=args.resume)
    install_original(root, manifest)
    original = importlib.import_module('controller')
    launcher = Path(value['helper_root']).resolve() / 'trainer_launcher.py'
    base = original.Controller

    class RepairedController(base):
        def spawn(self, command, log, phase, extra_env=None):
            replacement, changed = rewrite(command, phase, self.config, config_ref, root, launcher)
            env = dict(extra_env or {})
            if changed:
                require(not any(k in env for k in ('E1_DTYPE_REPAIR_MANIFEST', 'E1_DTYPE_REPAIR_MANIFEST_SHA256')),
                        'Original caller already supplied a repair override')
                env.update(E1_DTYPE_REPAIR_MANIFEST=repair_ref['path'], E1_DTYPE_REPAIR_MANIFEST_SHA256=repair_ref['sha256'])
                receipt(config, repair_ref, 'torchrun-outer-' + str(Path(command[command.index('--outer-manifest') + 1]).parent.name),
                        dict(config=config_ref, original_command=command, actual_command=replacement,
                             original_options_and_environment_preserved=True, original_controller=str(root / 'controller.py'),
                             repair_launcher=str(launcher)))
            return super().spawn(replacement, log, phase, env)

    original.Controller = RepairedController
    receipt(config, repair_ref, 'controller-start', dict(config=config_ref, resume=args.resume,
            original_controller=str(root / 'controller.py'), original_helper_manifest=value['original_helper_manifest'],
            cpu_only=True, repair_source_manifest=value['source_manifest'], publication=value['publication']))
    sys.argv = [str(root / 'controller.py'), '--config', args.config, '--config-sha256', args.config_sha256,
                '--run-id', args.run_id] + (['--resume'] if args.resume else [])
    original.main()


if __name__ == '__main__':
    main()
