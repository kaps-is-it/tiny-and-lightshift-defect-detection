#!/usr/bin/env python
"""Round 7: champion ensemble finalization — {42e, 7e, 99e} re-verify + 4-seed variants."""
from __future__ import annotations



from experiments.refinements.hybrid_maps import ensemble_eval
from local_contrast.run_experiment import load_splits
from local_contrast.detail_branch import BranchFusion
from local_contrast.ensemble_maps import DATASET, DEVICE, eval_champion, fit_champion, report, test_maps


def main():
    print(f"device: {DEVICE}", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)

    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    deep_test42 = test_maps(det42, captures)
    fusion42e = BranchFusion(radius=25, eps=1.0).fit(deep_val42, val_images)
    s42e, base42e = eval_champion(fusion42e, deep_test42, captures)
    report("seed 42 eps=1.0", s42e)

    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    deep_test7 = test_maps(det7, captures)
    fusion7e = BranchFusion(radius=25, eps=1.0).fit(deep_val7, val_images7)
    s7e, base7e = eval_champion(fusion7e, deep_test7, captures)
    report("seed 7 eps=1.0", s7e)

    det99, fusion99, val_images99, deep_val99 = fit_champion(training, validation, 99)
    deep_test99 = test_maps(det99, captures)
    fusion99e = BranchFusion(radius=25, eps=1.0).fit(deep_val99, val_images99)
    s99e, base99e = eval_champion(fusion99e, deep_test99, captures)
    report("seed 99 eps=1.0", s99e)

    det3, fusion3, val_images3, deep_val3 = fit_champion(training, validation, 3)
    deep_test3 = test_maps(det3, captures)
    fusion3e = BranchFusion(radius=25, eps=1.0).fit(deep_val3, val_images3)
    s3e, base3e = eval_champion(fusion3e, deep_test3, captures)
    report("seed 3 eps=1.0", s3e)

    ensemble_eval([base42e, base7e, base99e], "ensemble {42e, 7e, 99e} (champion)")
    ensemble_eval([base42e, base7e, base99e, base3e], "ensemble {42e, 7e, 99e, 3e}")
    ensemble_eval([base42e, base99e, base3e], "ensemble {42e, 99e, 3e}")
    ensemble_eval([base7e, base99e, base3e], "ensemble {7e, 99e, 3e}")
    print("\nDONE7", flush=True)


if __name__ == "__main__":
    main()
