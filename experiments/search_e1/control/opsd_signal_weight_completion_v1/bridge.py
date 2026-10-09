"""Independent input-format adapter for the already published TRAIN analysis."""
import argparse
import json
import os
from pathlib import Path
import sys
try:
    from .assemble import bridge
    from .common import require, STUDY_ROOT
except ImportError:
    from assemble import bridge
    from common import require, STUDY_ROOT


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'sampling-complete', 'sampling-complete-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); target = Path(args.output).resolve()
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
            and target.is_relative_to(Path(STUDY_ROOT) / 'analysis'), 'Bridge must remain CPU/authorized remote analysis output')
    proof = bridge(dict(path=args.config, sha256=args.config_sha256),
                   dict(path=args.sampling_complete, sha256=args.sampling_complete_sha256), target)
    print(json.dumps(dict(accepted=True, output=proof, input_format_bridge_only=True), sort_keys=True))


if __name__ == '__main__':
    main()
