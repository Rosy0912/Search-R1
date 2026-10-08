#!/usr/bin/env bash
# Prepared future submission only. Do not run while2994/2996 are active/unaccepted.
set -euo pipefail
: "${OPSD_STAGE_MANIFEST_SHA256:?Externally reviewed FINAL manifest SHA required}"
: "${OPSD_ACCEPTANCE_RECEIPT:?Completed acceptance receipt required}"
root=/mnt/data/shaoshitong/search-r1-rl/opsd
[[ -f "$OPSD_ACCEPTANCE_RECEIPT" && -f "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20/formal_entry.sh" ]]
# Recheck current Slurm time/QOS policy before this prepared submission.
exec sbatch --parsable --partition=gpu --nodelist=h100-10 --nodes=1 --ntasks=1 \
 --cpus-per-task=192 --mem=1T --gres=gpu:h100:8 --exclusive --time=7-00:00:00 \
 --job-name=opsd-formal500 --output="$root/logs/slurm-%j-formal.out" \
 --export=ALL,OPSD_ACCEPTANCE_RECEIPT="$OPSD_ACCEPTANCE_RECEIPT",OPSD_STAGE_MANIFEST_SHA256="$OPSD_STAGE_MANIFEST_SHA256" \
 --wrap="srun --ntasks=1 bash '$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20/formal_entry.sh'"
