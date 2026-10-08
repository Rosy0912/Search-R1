import sys
from pathlib import Path
from launch_transaction import durable
run=Path(sys.argv[1])
durable(run/'DRIVER_RETURNED.json',{'status':'DRIVER_RETURNED_ZERO','scope':'NOT_TRAINING_ACCEPTANCE'})
