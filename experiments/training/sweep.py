"""Run a sweep of training configurations over rules and seeds, resumably.

    python sweep.py configs/onestep.json                 # the whole sweep
    python sweep.py configs/onestep.json --smoke         # a few rules/seeds/steps (minutes)
    python sweep.py configs/onestep.json --chunk 0/4     # every 4th work unit, from unit 0
    python sweep.py configs/onestep.json --only baseline_2024,best_minimal
    python sweep.py configs/onestep.json --list          # show configurations and work units

A sweep file (JSON) holds ``name``, ``base_seed``, ``rules`` ("all",
"representatives", "odd", "even" or a list), ``seeds`` (a count or [start,
stop]), ``defaults`` (Config fields shared by all configurations),
``configs`` (a list of Config overrides, each with a ``name``; an entry with a
``grid`` expands into the Cartesian product, its name being a format string)
and optionally ``smoke`` (overrides for --smoke: ``rules``, ``seeds`` and
Config fields under ``overrides``) and ``members_per_part``.

The members of a configuration (seed-major: all rules for seed 0, then seed
1, ...) are split into parts; a work unit is one part of one configuration,
and each writes ``results/raw/<sweep>/<config>/part_XXXX.csv`` atomically.
Existing parts are skipped, so an interrupted sweep resumes where it stopped.
Every member's run depends only on (base_seed, rule, seed, config name), not
on the part or chunk it falls in.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"


def _parse_chunk(text: str) -> tuple[int, int]:
    i, n = (int(v) for v in text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("--chunk must be i/n with 0 <= i < n")
    return i, n


def resolve_rules(spec) -> list[int]:
    from ca_emulators import rules as ca_rules

    if spec == "all":
        return list(range(256))
    if spec == "representatives":
        return ca_rules.representatives()
    if spec == "odd":
        return list(range(1, 256, 2))
    if spec == "even":
        return list(range(0, 256, 2))
    return [ca_rules.check_rule(r) for r in spec]


def resolve_seeds(spec) -> list[int]:
    if isinstance(spec, int):
        return list(range(spec))
    start, stop = spec
    return list(range(int(start), int(stop)))


def expand_configs(sweep: dict, overrides: dict | None = None):
    from ensemble import Config

    defaults = dict(sweep.get("defaults", {}))
    out = []
    for entry in sweep["configs"]:
        entry = dict(entry)
        grid = entry.pop("grid", None)
        if grid is None:
            combos = [{}]
        else:
            keys = list(grid)
            combos = [dict(zip(keys, values)) for values in itertools.product(*grid.values())]
        for combo in combos:
            fields = {**defaults, **entry, **combo, **(overrides or {})}
            fields["name"] = entry["name"].format(**{**defaults, **entry, **combo})
            out.append(Config.from_dict(fields))
    names = [c.name for c in out]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate configuration names: {sorted(duplicates)}")
    return out


def members_per_part(cfg, n_rules: int, n_members: int, sweep: dict, args) -> int:
    from ensemble import estimate_member_bytes

    if args.members_per_part:
        return int(args.members_per_part)
    if "members_per_part" in sweep:
        return int(sweep["members_per_part"])
    budget = args.mem_mb * (1 << 20)
    per_part = max(1, min(n_members, budget // estimate_member_bytes(cfg), args.max_members))
    if per_part >= n_rules:  # whole seeds per part
        per_part -= per_part % n_rules
    return int(per_part)


def write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _spawn_workers(args, argv) -> int:
    """Re-run this command as ``args.workers`` processes splitting the chunk's units."""
    import subprocess

    argv = list(sys.argv[1:] if argv is None else argv)
    clean, skip = [], False
    for a in argv:  # drop --workers K / --workers=K
        if skip:
            skip = False
            continue
        if a == "--workers":
            skip = True
            continue
        if a.startswith("--workers="):
            continue
        clean.append(a)
    if not any(a == "--threads" or a.startswith("--threads=") for a in clean):
        clean += ["--threads", "1"]
    procs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *clean,
                               "--worker", f"{w}/{args.workers}"])
             for w in range(args.workers)]
    codes = [p.wait() for p in procs]
    print(f"workers finished with exit codes {codes}; logs in {args.out}")
    return max(codes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep", type=Path, help="sweep definition (JSON)")
    parser.add_argument("--smoke", action="store_true", help="tiny version (a couple of minutes)")
    parser.add_argument("--chunk", type=_parse_chunk, default=(0, 1), metavar="i/n")
    parser.add_argument("--only", default="", help="comma-separated configuration names")
    parser.add_argument("--members-per-part", type=int, default=0,
                        help="members trained together (default: from memory budget)")
    parser.add_argument("--mem-mb", type=int, default=2000,
                        help="memory budget per ensemble in MB (use ~1200 on a 2 GB GPU)")
    parser.add_argument("--max-members", type=int, default=1024,
                        help="upper bound on members per part (1024 suits a laptop CPU; "
                             "use 8192 or more on a GPU)")
    parser.add_argument("--out", type=Path, default=RAW, help="root of the raw output")
    parser.add_argument("--jit", choices=("auto", "on", "off"), default="auto",
                        help="XLA-compile the training loops (auto: only without 1x1 layers)")
    parser.add_argument("--threads", type=int, default=0, help="TF intra-op threads (0: default)")
    parser.add_argument("--workers", type=int, default=1,
                        help="run this chunk in K parallel processes (one core each)")
    parser.add_argument("--worker", type=_parse_chunk, default=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--seeds", default="",
                        help="override the sweep's seeds: a count N or a range a:b")
    parser.add_argument("--rules", default="",
                        help="override the sweep's rules: all, representatives, or a comma list")
    parser.add_argument("--tag", default="",
                        help="suffix of the output folder (e.g. 'ws' writes <sweep>-ws)")
    parser.add_argument("--require-gpu", action="store_true",
                        help="abort unless TensorFlow sees a GPU (use on the workstation)")
    parser.add_argument("--list", action="store_true", help="list configurations and exit")
    args = parser.parse_args(argv)

    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    name = sweep["name"]
    rules = resolve_rules(sweep.get("rules", "all"))
    seeds = resolve_seeds(sweep.get("seeds", 32))
    overrides = None
    if args.smoke:
        smoke = sweep.get("smoke", {})
        rules = resolve_rules(smoke.get("rules", [0, 1, 30, 54, 105, 110, 150, 232]))
        seeds = resolve_seeds(smoke.get("seeds", 4))
        overrides = smoke.get("overrides", {"steps": 256, "eval_every": 32})
        name = f"{name}-smoke"
    if args.seeds:
        seeds = resolve_seeds(int(args.seeds) if ":" not in args.seeds
                              else [int(v) for v in args.seeds.split(":")])
    if args.rules:
        rules = resolve_rules(args.rules if args.rules in ("all", "representatives", "odd", "even")
                              else [int(r) for r in args.rules.split(",")])
    if args.tag:
        name = f"{name}-{args.tag}"
    configs = expand_configs(sweep, overrides)
    if args.only:
        wanted = set(args.only.split(","))
        missing = wanted - {c.name for c in configs}
        if missing:
            parser.error(f"unknown configurations: {sorted(missing)}")
        configs = [c for c in configs if c.name in wanted]
    base_seed = int(sweep.get("base_seed", 2024))
    members = [(r, s) for s in seeds for r in rules]
    out_root = args.out / name

    units = []
    for cfg in configs:
        size = members_per_part(cfg, len(rules), len(members), sweep, args)
        meta_path = out_root / cfg.name / "meta.json"
        has_parts = any((out_root / cfg.name).glob("part_*.csv"))
        if meta_path.exists() and has_parts:  # resume with the part size used before
            size = json.loads(meta_path.read_text(encoding="utf-8"))["members_per_part"]
        n_parts = -(-len(members) // size)
        units += [(cfg, size, p, n_parts) for p in range(n_parts)]

    if args.list:
        for cfg in configs:
            parts = [u for u in units if u[0] is cfg]
            print(f"{cfg.name:40s} params={cfg.n_params:6d} members={len(members)} "
                  f"parts={len(parts)} x {parts[0][1]}")
        print(f"{len(units)} work units; this chunk: {len(units[args.chunk[0]::args.chunk[1]])}")
        return 0

    if args.workers > 1:
        return _spawn_workers(args, argv)

    import tensorflow as tf  # after argument parsing: slow import

    tf.get_logger().setLevel("ERROR")  # retracing warnings: one tf.function per ensemble
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        print("--require-gpu: TensorFlow sees no GPU (check `docker run --gpus all` and the "
              "NVIDIA container toolkit)", file=sys.stderr)
        return 2
    import ensemble

    out_root.mkdir(parents=True, exist_ok=True)
    i, n = args.chunk
    w, k_workers = args.worker
    log_path = out_root / (f"log_chunk{i}of{n}.txt" if k_workers == 1
                           else f"log_chunk{i}of{n}_worker{w}of{k_workers}.txt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    mine = units[i::n][w::k_workers]
    log(f"sweep {name}: {len(configs)} configs, {len(rules)} rules x {len(seeds)} seeds, "
        f"{len(units)} units, chunk {i}/{n} runs {len(mine)}; TensorFlow {tf.__version__}, "
        f"GPUs: {[g.name for g in gpus] or 'none (CPU)'}")
    t0 = time.time()
    for k, (cfg, size, p, n_parts) in enumerate(mine):
        cfg_dir = out_root / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cfg_dir / "meta.json"
        stored = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if stored.get("members_per_part") != size:  # new, or no part existed yet
            meta = {"sweep": name, "config": cfg.to_dict(), "rules": rules, "seeds": seeds,
                    "base_seed": base_seed, "members_per_part": size, "n_parts": n_parts,
                    "tensorflow": tf.__version__, "jit": args.jit}
            meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        part_path = cfg_dir / f"part_{p:04d}.csv"
        if part_path.exists():
            log(f"({k + 1}/{len(mine)}) {cfg.name} part {p + 1}/{n_parts}: exists, skipped")
            continue
        chunk_members = members[p * size:(p + 1) * size]
        jit = {"auto": None, "on": True, "off": False}[args.jit]
        try:
            rows, extra = ensemble.run_members(cfg, chunk_members, base_seed=base_seed,
                                               log=log, jit=jit)
        except tf.errors.ResourceExhaustedError:
            log(f"out of memory in {cfg.name} part {p + 1} ({len(chunk_members)} networks). "
                "Rerun with a smaller --mem-mb or --max-members; the part size of a "
                "configuration can change as long as none of its parts exists yet.")
            return 3
        for row in rows:
            row["seconds_per_part"] = round(extra["seconds"], 2)
        write_csv(part_path, rows)
        exact = sum(r["exact_final"] for r in rows) / len(rows)
        closed = sum(r["cl_exact"] for r in rows) / len(rows)
        log(f"({k + 1}/{len(mine)}) {cfg.name} part {p + 1}/{n_parts}: {len(rows)} members, "
            f"exact {exact:.3f}, closed-loop {closed:.3f}, {extra['seconds']:.1f} s")
    log(f"chunk done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
