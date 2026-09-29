"""Fig. 3 (thesis Fig. 7.3): a randomly initialised CNN learns rule 54.

A seeded re-run of the 2024 training recipe (``ca_emulators.training``), with
the published example configuration as the first training sample. The
published figure came from an unseeded run, so the heatmap differs in detail;
the script reports how the re-run compares (pretraining restarts, final loss,
exactness after thresholding) in ``output/fig3_training.json``.
"""
import json

import numpy as np
import tensorflow as tf

import _style
from ca_emulators.reference import eca_step
from ca_emulators.rules import de_bruijn_configuration
from ca_emulators.plotting import plot_training_history
from ca_emulators.training import train_2024_recipe

STEM = "plot_configs-ECA-32cells-rule54-40epochs_bs64_lr0p005"
SEED = 2024  # fixed before the first run, not selected afterwards


def main(argv=None):
    p = _style.parser(__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    _style.apply(args)
    tf.config.experimental.enable_op_determinism()
    fig3 = _style.load_npz(_style.DATA / "published_inputs" / "fig3_training_example_rule54.npz")
    rule = int(fig3["rule"])

    run = train_2024_recipe(rule, N=len(fig3["x"]), seed=args.seed, example=fig3["x"])
    x_example, y_example = run.x_train[0, :, 0], run.y_train[0, :, 0]
    assert np.array_equal(y_example, fig3["y"])

    certificate = de_bruijn_configuration(len(x_example))
    prediction = run.model.predict(certificate[np.newaxis, :, np.newaxis].astype(np.float32),
                                   verbose=0)[0, :, 0]
    summary = {
        "seed": args.seed,
        "pretraining_restarts": run.n_restarts,
        "pretraining_best_loss": min(run.pretrain_losses),
        "final_loss": float(run.history.history["loss"][-1]),
        "final_val_loss": float(run.history.history["val_loss"][-1]),
        "exact_after_thresholding": bool(np.array_equal(prediction > 0.5,
                                                        eca_step(certificate, rule) == 1)),
        "margin": float(np.min(np.abs(prediction - 0.5))),
    }
    fig, _ = plot_training_history(run.model, run.weights_history, x_example, y_example, rule)
    _style.save(fig, STEM, args)
    (args.out / "fig3_training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
