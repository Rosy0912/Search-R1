"""Enable non-destructive SIGUSR1 stack dumps in this project's Ray GPU workers."""
import faulthandler
import signal
import sys
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

# Opt in only for validated multi-trajectory batching; existing runs stay unchanged.
import os
if os.environ.get('SEARCH_E1_TRAJECTORY_BATCHING') == '1':
    from search_e1.grpo_batching import install
    install()
    print({'trajectory_mean_batching': 'enabled'}, flush=True)
