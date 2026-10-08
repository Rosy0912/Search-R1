"""Install only the explicitly bound cohosting resource admission in fresh processes."""
import os
if 'GIGPO_COHOST_BINDING' in os.environ:
    from cohosting_v21 import authorize, check_selected_gpus
    authorize()
    import search_r1.baselines.preflight as preflight
    preflight.check_selected_gpus = check_selected_gpus
