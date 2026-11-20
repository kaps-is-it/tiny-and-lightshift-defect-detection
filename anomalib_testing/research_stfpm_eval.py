"""Evaluate the already-trained Stfpm checkpoint with the project harness.

The main run was stopped because Engine() without explicit trainer arguments
falls back to Lightning's default max_epochs=1000 (anomalib's 100-epoch value
only applies to its CLI configs). The saved checkpoint is from epoch 130, which
is already past the intended 100 epochs, so it can be scored directly.

From the repository root: python -m anomalib_testing.research_stfpm_eval
"""

from anomalib.models import Stfpm

from anomalib_testing import research_extra12 as r

CKPT = r.EXPERIMENT_DIR / "results" / "Stfpm" / "MVTecAD2" / "can" / "v0" / "weights" / "lightning" / "model.ckpt"
SIZE = (256, 558)


def main():
    print(f"checkpoint: {CKPT}", flush=True)
    model = r.make_model(Stfpm, SIZE)
    r.run_anomalib("Stfpm", model, ckpt_path=CKPT)
    print("\nDONE_STFPM", flush=True)


if __name__ == "__main__":
    main()
