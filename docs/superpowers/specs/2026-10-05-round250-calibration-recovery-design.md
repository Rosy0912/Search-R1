# Round250 calibration recovery

The user has authorized reproduction fixes, real probe training, strict review, weights-only recovery and continuation to300. This fixes the observed calibration/executor mismatch within that scope. Original run source/data/dumps/checkpoints stay immutable.

## Evidence
250 finite435, fixed64 EM7/64; failed calibration only5 searched/1500, flip4/1 with one positive question group. Paired32 first actions at256/500 all invalid. Real full correction diagnostic at500: both top_p1 and.95 perform32 actual searches and32 strict answers; EM4/32 and3/32. Correcting calibration cannot be claimed to repair model accuracy.

## Options and selected design
Merely raising token budget is contradicted by paired32 results. Reusing old heads or forcing CV labels violates encoder/label identity. Select an explicit fresh calibration protocol with500 token actions, existing executor correction feedback, strict terminal action validation, unchanged300 questions/G5/top_p1/Xa/flip/binary groupedCV. Original default256/no-correction helpers remain backward compatible.

## Provenance
Record separately model character ranges, retrieved-evidence ranges and correction-feedback ranges. Evidence deletion replaces only recorded retrieval spans with empty information markers; it preserves correction feedback and every model-authored reasoning span. Never use removal of all nonmodel gaps after adding corrections. Refuse invalid ranges; tests include model or correction text with information tags. No fake labels, no selected-question oversampling, no seed/split search.

## Admission and continuation
Fresh checkpoint250 encoder generates genuine interventions for all reserved300 questions. Require both binary classes supported across fixed groupedCV and finite hidden arrays/heads, exact dump hashes and HTTP identity before GRPO. If insufficient, design bounded independent full300/G5 repeats before collection, never select by outcomes. New segment251–300 uses newAdam/noRNG-sampler, original base KL and LR position250. Accept real251–255 and fresh255 parameter delta; final chain verifies1–200,201–250,251–300, real probe identities and all boundary hashes plus same64 final EM/fresh300finite. Share original /data/search-g1/gpu-job.lock across every segment. Never touch foreign4/7.

## Validation
CPU regression checks correction→search→answer, terminal action budgets, evidence-only deletion with corrections retained, malformed bounds, old default compatibility, initial dual-head admission and three-segment ancestry. Independent review before GPU calibration and newtrainer. Real32 full diagnostic is evidence for interaction only, not final64 accuracy.

## Independent review clarifications
Retain exactly four search-capable turns plus a final answer-only turn; corrections on nonterminal invalid actions cannot extend this budget. Require both fixed groupedCV folds valid for every admitted candidate, not a finite mean from just one surviving fold. Range validation rejects cross-type overlaps, gaps and out-of-bounds segments. Independent reviewer approved this design with these explicit contracts before implementation.
