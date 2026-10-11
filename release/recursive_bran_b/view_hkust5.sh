#!/usr/bin/env bash
set -euo pipefail
# Map paths only; this entry does not invoke the stopped OPSD experiment.
B=/home/shaoshitong/search-e1-migration-20261011
exec bwrap --tmpfs / --ro-bind /usr /usr --ro-bind /etc /etc --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /bin /bin --ro-bind /sbin /sbin --ro-bind /var /var --bind /home /home --bind /tmp /tmp --bind /run /run --proc /proc --dev /dev --bind /dev/shm /dev/shm --dev-bind /dev/nvidiactl /dev/nvidiactl --dev-bind /dev/nvidia-uvm /dev/nvidia-uvm --dev-bind /dev/nvidia-uvm-tools /dev/nvidia-uvm-tools --dev-bind /dev/nvidia0 /dev/nvidia0 --dev-bind /dev/nvidia1 /dev/nvidia1 --dev-bind /dev/nvidia2 /dev/nvidia2 --dev-bind /dev/nvidia3 /dev/nvidia3 --ro-bind /sys /sys --bind "$B/weka_3" /weka_3 "$@"
