from search_r1.baselines import preflight
from cohosting_v21 import authorize, check_selected_gpus
authorize()
preflight.check_selected_gpus = check_selected_gpus
preflight.main()
