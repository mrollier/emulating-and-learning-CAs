"""Cross-check the ensemble's 'baseline_2024' against the real Keras 2024 recipe.

Runs ``ca_emulators.training.train_2024_recipe`` (a fixed training set of
4096 configurations, Keras ``fit``, the pretraining loop capped at
``--max-restarts`` networks) for a few rules and seeds, and scores the trained
Keras networks with the same metrics as the ensemble engine. Slow (about
0.5-2.5 minutes per run on a laptop CPU), hence a small sample.

    python validate_2024.py --rules 1 30 54 105 110 150 --seeds 6
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rules", type=int, nargs="+", default=[1, 30, 54, 105, 110, 150])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--max-restarts", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "validate_2024.csv")
    args = parser.parse_args(argv)

    import numpy as np
    import tensorflow as tf

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    sys.path.insert(0, str(HERE))
    import ensemble as E
    from ca_emulators.training import train_2024_recipe

    cfg = E.Config(name="keras_2024")
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as fh:
            done = {(int(r["rule"]), int(r["seed"])) for r in csv.DictReader(fh)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        for rule in args.rules:
            if (rule, seed) in done:
                continue
            t0 = time.time()
            run = train_2024_recipe(rule, seed=seed, max_restarts=args.max_restarts)
            det = run.model.get_layer("detectors").get_weights()
            tab = run.model.get_layer("rule_tables").get_weights()
            params = {"W1": det[0][None, :, 0, :], "b1": det[1][None],
                      "wo": tab[0][None, 0, :, 0], "bo": tab[1][None, 0]}
            targets = E.rule_tables([rule])
            metrics = E.pattern_metrics(params, cfg, targets)
            first, _ = E.closed_loop(params, cfg, [rule], metrics["exact"])
            losses = run.pretrain_losses
            row = {"rule": rule, "seed": seed, "exact_final": bool(metrics["exact"][0]),
                   "err_mask": int(metrics["err_mask"][0]),
                   "margin_final": float(metrics["margin"][0]),
                   "cl_exact": bool(first[0] == cfg.cl_steps),
                   "pretrain_restarts": run.n_restarts,
                   "pretrain_pass": int(sum(v < 0.1 for v in losses)),
                   "loss_final": float(run.history.history["loss"][-1]),
                   "seconds": round(time.time() - t0, 1)}
            new = not args.out.exists()
            with open(args.out, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
