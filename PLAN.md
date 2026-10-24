# Faithful scaled LHFM-V learning stage1: frozen before training/test inspection

Prior40update functional-verification outputs remain unchanged. This is the authorized first meaningful-learning stage, not the separate6hour/3seed proposal. No forced superiority, no tuning after seeing results.

## Fixed models/data/seed

- Core architecture.py/operators.py/dataset.py/metrics.py byte-identical to the validated prior faithful implementation (core_identity.json). Full64×64, original28×28 MNIST,2digits,10observed+10future; all original components/depths/loss terms, second-order upwind/zero exterior, unbounded transport, degree24 adaptive shifted exponential/source integration preserved.
- Same scaled widths65,893 parameters. Original-width count18,637,493 verified previously. Full official-model same-weight equivalence remains unverified; official source is read-only pinned commit979a41de542f7b190549fc94a8803a56530e4a00.
- One predeclared initialization seed1017, modes transport then transport_off,3000updates each,batch1. Same initial tensors and same training sequences across modes. Transport_off is the exact frozen w=0 source-only limit, with unused transport parameters disclosed.
- Training/validation digit indices: originalMNIST55k/5k split seed271100. Training sequence absolute IDs0..2999/data seed270829. Validation first32sequences from heldout pool/data seed271109; fixed before training. No heldout test is opened until both models finish and validation checkpoints are fixed.
- Official final test sequence IDs16..143 inclusive(128), disjoint from already-inspected0..15. Full-file SHA256 has been verified; checksum reading is not sample evaluation. No test-based hyperparameter/early-stop/checkpoint selection.

## Optimizer/EMA/schedule/selection

- AdamW wd1e-4 eps1e-8 beta2.999,gradient global norm clip1. beta1/lr use unchanged operators.schedule(update,3000): linear mapping of1..3000 onto original1..1,250,000, retaining OneCycle beta1 and post400000 half-lr/cosine policy. FP32/MPS/eager as explicitly scaled implementation; no changed metric or EMA to favor results.
- EMA.999 per optimizer update preserved. Initial EMA weight at3000 is.999^3000≈4.97%, unlike40update smoke≈96.1%.
- Validation selection remains minimum raw pixel MSE EMA. Original validation milestones are rescaled by ceil(original_update×3000/1250000), deduplicated:35fixed updates listed in config. No early stop for winning/losing. Selected EMA plus last EMA/raw state retained; last/EMA validation and train loss histories distinguish optimization and smoothing.
- All original objective terms retained: prediction MSE+.001source-energy+.0001transport-TV+.05coarse-plan MSE.

## Budget and safety

- First-stage only, no automatic3seed follow-on. Core measured steady transport pilot~1.15sec/update; prior validation-inclusive extrapolation for3000/model pair~91min. Hard global cap7200sec including training, validation, final evaluation/latency; reserve enough time for evaluation but never reduce planned updates just to complete.
- RSS1.5GiB, MPS driver1GiB, global swap growth256MiB from runner start. Guard every25updates and during validation/test. Nonfinite/128internal-piece guard/resource/time failure stops and records exact phase/model/update. No automatic retry, clipping of velocity, loss omission, test fallback, or selected run exclusion.
- Progress JSONL includes actual update/time/resource and validation results. No performance-based stopping; all negatives/failures retained.

## Final reporting and delivery

- Primary raw spatial-sum MSE/MAE, with pixel means and SSIM range2/FP32 uniform7/sample covariance/transpose/crop/channel-cast and clipped per-frame PSNR unchanged. Same fixed128sequences for both models, last-frame baseline. Single seed cannot establish robust superiority.
- Save full raw predictions locally, validation-selected EMA, last/raw optimizer state, train/validation curves, EMA state/bias, selected update, test/horizon metrics, synchronized batch1 latency and memory/time.
- Native delivery lean source/results archive excludes official raw downloads/caches and large prediction arrays. Includes source/config/hash/checksum/download instructions and small checkpoints/example. Full raw predictions stay local and can be sharded into supported-size attachments. Library success alone is not assumed to satisfy client attachment limits; target lean ZIP<5MB and raw chunks<8MB.
