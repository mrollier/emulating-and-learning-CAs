"""Driver of the emulation benchmark (see README.md).

Every (method, scenario point, thread setting) is timed in a fresh Python
subprocess, so no method inherits another's threads, caches, traces or
memory. The worker (this file with ``--worker``) builds the method, checks
its diagram against :func:`ca_emulators.reference.evolve`, and times it:

- cold: set-up (model build, analytic weights) + first call (tracing, XLA
  compilation, Keras's predict function, ...), measured once;
- warm: later calls, ``--repeats`` repeats (at least 10 in a full run) of a
  timeit-style loop of ``number`` calls, with ``number`` chosen so that a
  repeat lasts at least ``--min-time``; median and interquartile range.

A timing only counts if the method's diagram is bit-identical to the
reference, both after the cold call and after the last warm call.

Examples::

    python experiments/benchmarks/run.py --smoke             # tiny sizes, plumbing
    python experiments/benchmarks/run.py --all --dry-run     # list the full run
    python experiments/benchmarks/run.py --all               # full run (hours)
    python experiments/benchmarks/run.py --all --resume      # continue a full run
    python experiments/benchmarks/run.py --scenarios N --methods numpy,xla:elem --threads single
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import methods as bm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
MARKER = "@@BENCHMARK-RESULT@@"
THREAD_SETTINGS = ("default", "single")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
#: Pure-Python methods, run in the default thread setting only.
THREAD_INSENSITIVE = frozenset({"cellpylib"})
#: Statuses after which the larger points of the same chain are not attempted.
BLOCKING = frozenset({"over_budget", "timeout", "error", "mismatch"})


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One varying parameter; ``core`` is the range of thesis Tab. 7.2."""

    name: str
    vary: str  # attribute of methods.Case
    fixed: dict
    core: tuple
    extended: tuple
    label: str

    def points(self, ranges=("core", "extended")):
        out = [(v, "core") for v in self.core] if "core" in ranges else []
        out += [(v, "extended") for v in self.extended] if "extended" in ranges else []
        return sorted(out)

    def case(self, value) -> bm.Case:
        return bm.Case(**{**self.fixed, self.vary: int(value)})


def _powers(lo, hi):
    return tuple(2 ** k for k in range(lo, hi + 1))


SCENARIOS = {
    "NR": Scenario("NR", "n_rules", dict(N=256, T=32, S=32), _powers(0, 8), (), "number of rules $N_R$"),
    "T": Scenario("T", "T", dict(N=64, n_rules=4, S=32), tuple(range(10, 101, 10)), (200, 500, 1000),
                  "rows of the diagram $T$"),
    "N": Scenario("N", "N", dict(n_rules=4, T=32, S=32), tuple(range(32, 257, 32)),
                  (512, 1024, 2048, 4096, 8192, 16384), "number of cells $N$"),
    "S": Scenario("S", "S", dict(N=32, n_rules=4, T=32), _powers(0, 10), _powers(11, 14),
                  "number of samples $S$"),
}

SMOKE_SCENARIOS = {
    "NR": Scenario("NR", "n_rules", dict(N=16, T=4, S=2), (3,), (), SCENARIOS["NR"].label),
    "T": Scenario("T", "T", dict(N=16, n_rules=2, S=2), (4,), (), SCENARIOS["T"].label),
    "N": Scenario("N", "N", dict(n_rules=2, T=4, S=2), (16,), (32, 64), SCENARIOS["N"].label),
    "S": Scenario("S", "S", dict(N=16, n_rules=2, T=4), (2,), (), SCENARIOS["S"].label),
}


@dataclass(frozen=True)
class Settings:
    repeats: int = 10
    min_time: float = 0.1          # s, minimum duration of one warm repeat
    budget: float = 60.0           # s, warm phase of an extended point
    cold_budget: float = 120.0     # s, cold phase of an extended point
    core_budget: float = 600.0     # s, warm phase of a Tab. 7.2 point
    core_cold_budget: float = 600.0
    timeout_margin: float = 180.0  # s, added to the budgets for the hard timeout
    max_selector_mb: float = 512.0

    def budgets(self, rng: str):
        return (self.core_budget, self.core_cold_budget) if rng == "core" else (
            self.budget, self.cold_budget)


FULL = Settings()
#: Tiny budgets and memory cap, so that the smoke run also exercises the
#: over-budget, predicted-skip and memory-skip paths.
SMOKE = Settings(repeats=2, min_time=0.0, budget=0.25, cold_budget=60.0, core_budget=60.0,
                 core_cold_budget=120.0, timeout_margin=120.0, max_selector_mb=0.02)


# --------------------------------------------------------------------------
# Worker (runs in the subprocess)
# --------------------------------------------------------------------------

def _peak_memory_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "rss", None)
        return round(peak / 2 ** 20, 1) if peak else None
    except Exception:  # psutil is optional
        return None


def _timed(run, number):
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(number):
            out = run()
        return time.perf_counter() - start, out
    finally:
        if gc_was_enabled:
            gc.enable()


def measure_warm(run, repeats: int, min_time: float, budget: float):
    """Warm timings: (per-call times, number, status, last output).

    One call estimates the cost; if ``repeats`` calls would exceed the
    budget the point is abandoned ("over_budget", with that one timing).
    Otherwise ``number`` is increased (1, 2, 5, 10, 20, ...) until one
    repeat of ``number`` calls lasts ``min_time``, as ``timeit`` does.
    """
    gc.collect()
    first, out = _timed(run, 1)
    if first * repeats > budget:
        return [first], 1, "over_budget", out
    number = 1
    if first < min_time:
        base = 1
        while True:
            for factor in (1, 2, 5):
                number = base * factor
                elapsed, out = _timed(run, number)
                if elapsed >= min_time:
                    break
            else:
                base *= 10
                continue
            break
    times = []
    for _ in range(repeats):
        elapsed, out = _timed(run, number)
        times.append(elapsed / number)
    return times, number, "ok", out


def worker(spec: dict) -> dict:
    """Build, validate and time one method on one case (in a fresh process)."""
    row = {"status": "error", "note": "", "correct": ""}
    t0 = time.perf_counter()
    import numpy as np
    import tensorflow as tf
    if spec["threads"] == "single":
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        import ca_emulators  # noqa: F401
    except ImportError:
        sys.path.append(str(REPO / "src"))
        import ca_emulators  # noqa: F401
    if spec["method"] == "cellpylib":
        import cellpylib  # noqa: F401  (imports matplotlib: an import cost, not a cold-call cost)
    row["import_s"] = time.perf_counter() - t0
    row["intra_threads"] = tf.config.threading.get_intra_op_parallelism_threads()
    row["inter_threads"] = tf.config.threading.get_inter_op_parallelism_threads()

    case = bm.Case(**spec["case"])
    method = bm.METHODS[spec["method"]]
    np.random.seed(case.seed % 2 ** 32)
    tf.random.set_seed(case.seed)
    rules, alloc, x0 = case.data()
    expected = bm.expected_diagram(case, rules, alloc, x0)

    try:
        t = time.perf_counter()
        prepared = method.setup(case, rules, alloc, x0)
        row["build_s"] = time.perf_counter() - t
        t = time.perf_counter()
        out = prepared.run()
        row["first_call_s"] = time.perf_counter() - t
        row["cold_s"] = row["build_s"] + row["first_call_s"]
        row.update(prepared.info)
        if not np.array_equal(prepared.to_states(out), expected):
            row.update(status="mismatch", correct=False, note="cold output differs from reference")
            return row
        row["correct"] = True
        if row["cold_s"] > spec["cold_budget"]:
            row.update(status="over_budget", note=f"cold {row['cold_s']:.1f} s > {spec['cold_budget']} s")
            return row
        times, number, status, out = measure_warm(prepared.run, spec["repeats"], spec["min_time"],
                                                  spec["budget"])
        if not np.array_equal(prepared.to_states(out), expected):
            row.update(status="mismatch", correct=False, note="warm output differs from reference")
            return row
        q1, median, q3 = np.percentile(times, [25, 50, 75])
        row.update(status=status, warm_median_s=median, warm_q1_s=q1, warm_q3_s=q3,
                   warm_min_s=min(times), warm_max_s=max(times), repeats=len(times), number=number)
        if status == "over_budget":
            row["note"] = f"one warm call {times[0]:.2f} s x {spec['repeats']} > {spec['budget']} s"
    except Exception as exc:  # recorded, the driver carries on
        row.update(status="error", note=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                   if str(exc) else type(exc).__name__)
    finally:
        row["peak_memory_mb"] = _peak_memory_mb()
    return row


def worker_info() -> dict:
    """Software metadata, collected in a subprocess (the driver never imports TensorFlow)."""
    import cellpylib
    import keras
    import numpy as np
    import tensorflow as tf
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "tensorflow": tf.__version__, "keras": keras.__version__,
            "numpy": np.__version__, "cellpylib": getattr(cellpylib, "__version__", "?"),
            "tf_default_intra_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "tf_default_inter_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "(unset)")}
    try:
        import ca_emulators
        info["ca_emulators"] = ca_emulators.__version__
    except ImportError:
        info["ca_emulators"] = "not importable"
    try:
        info["tf_build"] = {k: str(v) for k, v in tf.sysconfig.get_build_info().items()}
    except Exception:
        pass
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    info["numpy_config"] = _numpy_blas(np)
    return info


def _numpy_blas(np):
    try:
        cfg = np.show_config(mode="dicts")
        return {k: v.get("name", "?") for k, v in cfg.get("Build Dependencies", {}).items()}
    except Exception:
        return "?"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

COLUMNS = [
    "tag", "timestamp", "scenario", "vary", "value", "range", "method", "driver", "selector",
    "threads", "N", "n_rules", "T", "n_updates", "S", "seed", "status", "note", "correct",
    "import_s", "build_s", "first_call_s", "cold_s", "warm_median_s", "warm_q1_s", "warm_q3_s",
    "warm_min_s", "warm_max_s", "repeats", "number", "n_params", "selector_mb",
    "intra_threads", "inter_threads", "peak_memory_mb", "wall_s",
]


def child_env(threads: str) -> dict:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="2")
    for var in THREAD_VARS:
        if threads == "single":
            env[var] = "1"
        else:
            env.pop(var, None)
    return env


def run_subprocess(args, stdin: str, env: dict, timeout: float):
    """(stdout, stderr, returncode or None on timeout, wall seconds)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], input=stdin,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=env, cwd=str(REPO), timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode, time.perf_counter() - start
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return out, err, None, time.perf_counter() - start


def parse_result(stdout: str):
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    return None


@dataclass
class Chain:
    """State of one (scenario, method, threads) chain, whose points run in ascending order."""

    last_value: float | None = None
    last_warm: float | None = None
    last_cold: float | None = None
    blocked: str = ""

    def update(self, row):
        status = row.get("status")
        if status == "ok":
            self.last_value = float(row["value"])
            self.last_warm = float(row["warm_median_s"])
            self.last_cold = float(row["cold_s"])
        elif status in BLOCKING and not self.blocked:
            self.blocked = f"{status} at {row['vary']}={row['value']}"


def predicted_skip(chain: Chain, value, repeats: int, budget: float, cold_budget: float) -> str:
    """Reason to skip an extended point, extrapolating linearly from the last good one."""
    if chain.last_value is None:
        return ""
    ratio = float(value) / chain.last_value
    warm, cold = chain.last_warm * ratio, chain.last_cold * ratio
    if warm * repeats > budget:
        return f"predicted warm {warm:.3g} s x {repeats} > {budget} s"
    if cold > cold_budget:
        return f"predicted cold {cold:.3g} s > {cold_budget} s"
    return ""


def plan(scenarios: dict, ranges, method_names, threads):
    """Jobs in execution order: scenario, point (ascending), thread setting, method."""
    jobs = []
    for scenario in scenarios.values():
        for value, rng in scenario.points(ranges):
            for thread in threads:
                for name in method_names:
                    if thread != "default" and name in THREAD_INSENSITIVE:
                        continue
                    jobs.append((scenario, value, rng, thread, name))
    return jobs


def base_row(tag, scenario, value, rng, thread, method: bm.Method, case: bm.Case, settings):
    row = dict(tag=tag, timestamp=dt.datetime.now().isoformat(timespec="seconds"),
               scenario=scenario.name, vary=scenario.vary, value=value, range=rng,
               method=method.name, driver=method.driver, selector=method.selector, threads=thread,
               N=case.N, n_rules=case.n_rules, T=case.T, n_updates=case.n_updates, S=case.S,
               seed=case.seed)
    if method.uses_tf:
        row["selector_mb"] = round(method.kernel_bytes(case) / 2 ** 20, 3)
    return row


def run_job(tag, scenario, value, rng, thread, method, settings: Settings, chain: Chain, log_dir: Path):
    case = scenario.case(value)
    row = base_row(tag, scenario, value, rng, thread, method, case, settings)
    budget, cold_budget = settings.budgets(rng)
    if chain.blocked:
        row.update(status="skipped_after_failure", note=chain.blocked)
        return row
    if method.uses_tf and method.kernel_bytes(case) > settings.max_selector_mb * 2 ** 20:
        row.update(status="skipped_memory",
                   note=f"selector kernel {row['selector_mb']} MiB > {settings.max_selector_mb} MiB")
        return row
    if rng == "extended":
        reason = predicted_skip(chain, value, settings.repeats, budget, cold_budget)
        if reason:
            row.update(status="skipped_predicted", note=reason)
            return row
    spec = dict(method=method.name, case=asdict(case), threads=thread, repeats=settings.repeats,
                min_time=settings.min_time, budget=budget, cold_budget=cold_budget)
    timeout = budget + cold_budget + settings.timeout_margin
    stdout, stderr, code, wall = run_subprocess(["--worker"], json.dumps(spec), child_env(thread), timeout)
    result = parse_result(stdout)
    row["wall_s"] = round(wall, 2)
    if code is None:
        row.update(status="timeout", note=f"killed after {timeout:.0f} s")
    elif result is None:
        tail = " | ".join(stderr.strip().splitlines()[-3:])[-300:]
        row.update(status="error", note=f"worker exited with {code}: {tail}")
    else:
        row.update(result)
    if row["status"] not in ("ok",):
        log_dir.mkdir(parents=True, exist_ok=True)
        name = f"{scenario.name}-{value}-{thread}-{method.name.replace(':', '_')}.log"
        (log_dir / name).write_text(f"{json.dumps(spec)}\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
                                    encoding="utf-8")
    return row


def _compact(row: dict) -> dict:
    """Round floats to 5 significant digits (the CSV stays small; timings are not that precise)."""
    return {k: float(f"{v:.5g}") if isinstance(v, float) else v for k, v in row.items()}


def _fmt_seconds(s):
    if s is None or s == "":
        return "-"
    s = float(s)
    return f"{s * 1e6:.0f} us" if s < 1e-3 else f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s"


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "?"


def calibration() -> dict:
    """A fixed workload timed in the driver, before and after the run.

    If the two differ markedly, the machine was not equally quiet throughout.
    """
    import numpy as np
    a = np.random.default_rng(0).random((256, 256))
    matmul = min(_timed(lambda: a @ a, 50)[0] for _ in range(3))
    loop = min(_timed(lambda: sum(range(1_000_000)), 1)[0] for _ in range(3))
    return {"numpy_matmul_256_x50_s": round(matmul, 5), "python_sum_1e6_s": round(loop, 5)}


def machine_metadata() -> dict:
    meta = {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu_name(),
            "logical_cores": os.cpu_count()}
    try:
        import psutil
        meta["physical_cores"] = psutil.cpu_count(logical=False)
        meta["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        freq = psutil.cpu_freq()
        if freq:
            meta["cpu_mhz_max"] = freq.max
        battery = psutil.sensors_battery()
        if battery is not None:
            meta["on_ac_power"] = battery.power_plugged
            meta["battery_percent"] = battery.percent
        meta["cpu_load_percent_before"] = psutil.cpu_percent(interval=2.0)
    except Exception:
        pass
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        dirty = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True)
        meta["git_commit"] = commit.stdout.strip()
        meta["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        pass
    return meta


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = p.add_argument_group("what to run")
    what.add_argument("--all", action="store_true", help="full run: every scenario, core and extended "
                      "ranges, the default methods, both thread settings")
    what.add_argument("--smoke", action="store_true", help="tiny sizes, 2 repeats, tiny budgets; "
                      "output in results/raw/smoke/ (timings meaningless)")
    what.add_argument("--scenarios", help=f"comma-separated subset of {','.join(SCENARIOS)}")
    what.add_argument("--ranges", default="core,extended", help="core (Tab. 7.2), extended, or both")
    what.add_argument("--methods", default="default",
                      help="'default', 'all' or a comma-separated list, e.g. numpy,xla:elem")
    what.add_argument("--threads", default="default,single", help="default, single, or both")
    how = p.add_argument_group("protocol (defaults: full run)")
    how.add_argument("--repeats", type=int)
    how.add_argument("--min-time", type=float, help="s, minimum duration of one warm repeat")
    how.add_argument("--budget", type=float, help="s, warm budget of an extended point")
    how.add_argument("--cold-budget", type=float, help="s, cold budget of an extended point")
    how.add_argument("--core-budget", type=float, help="s, warm budget of a Tab. 7.2 point")
    how.add_argument("--core-cold-budget", type=float, help="s, cold budget of a Tab. 7.2 point")
    how.add_argument("--max-selector-mb", type=float, help="skip selectors with a larger kernel")
    out = p.add_argument_group("output")
    out.add_argument("--tag", help="name of the results file (default: full, smoke or custom)")
    out.add_argument("--resume", action="store_true", help="skip jobs already in the results file")
    out.add_argument("--overwrite", action="store_true", help="replace an existing results file")
    out.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--info", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def resolve(args):
    if args.smoke:
        scenarios, settings, tag = SMOKE_SCENARIOS, SMOKE, args.tag or "smoke"
    else:
        scenarios, settings, tag = SCENARIOS, FULL, args.tag or ("full" if args.all else "custom")
    if args.scenarios:
        names = args.scenarios.split(",")
        unknown = set(names) - set(scenarios)
        if unknown:
            raise SystemExit(f"unknown scenarios {sorted(unknown)}; choose from {list(scenarios)}")
        scenarios = {n: scenarios[n] for n in names}
    overrides = {k: getattr(args, k) for k in ("repeats", "min_time", "budget", "cold_budget",
                                               "core_budget", "core_cold_budget", "max_selector_mb")
                 if getattr(args, k) is not None}
    settings = Settings(**{**asdict(settings), **overrides})
    if args.methods == "default":
        names = list(bm.DEFAULT_METHODS)
    elif args.methods == "all":
        names = list(bm.METHODS)
    else:
        names = args.methods.split(",")
        unknown = [n for n in names if n not in bm.METHODS]
        if unknown:
            raise SystemExit(f"unknown methods {unknown}; available: {', '.join(bm.METHODS)}")
    threads = args.threads.split(",")
    if set(threads) - set(THREAD_SETTINGS):
        raise SystemExit(f"--threads takes {THREAD_SETTINGS}")
    ranges = args.ranges.split(",")
    folder = RESULTS / "raw" / "smoke" if args.smoke else RESULTS
    return scenarios, ranges, names, threads, settings, tag, folder


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        row = worker(json.loads(sys.stdin.read()))
        print(MARKER + json.dumps(row, default=str), flush=True)
        return
    if args.info:
        print(MARKER + json.dumps(worker_info(), default=str), flush=True)
        return
    if not (args.all or args.smoke or args.scenarios or args.methods != "default"):
        raise SystemExit("nothing selected: pass --smoke, --all, or --scenarios/--methods (see --help)")

    scenarios, ranges, names, threads, settings, tag, folder = resolve(args)
    jobs = plan(scenarios, ranges, names, threads)
    csv_path, meta_path = folder / f"benchmark_{tag}.csv", folder / f"benchmark_{tag}_meta.json"
    print(f"{len(jobs)} jobs ({len(names)} methods, {sum(len(s.points(ranges)) for s in scenarios.values())} "
          f"points, threads {threads}) -> {csv_path.relative_to(REPO)}")
    if args.dry_run:
        for scenario, value, rng, thread, name in jobs:
            print(f"  {scenario.name:>2} {scenario.vary}={value:<6} {rng:<8} {thread:<7} {name}")
        return

    done, chains = set(), {}
    if csv_path.exists():
        if args.resume:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done.add((row["scenario"], row["value"], row["threads"], row["method"]))
                    chains.setdefault((row["scenario"], row["method"], row["threads"]), Chain()).update(row)
            print(f"resuming: {len(done)} jobs already in {csv_path.name}")
        elif not args.overwrite:
            raise SystemExit(f"{csv_path} exists: pass --resume or --overwrite")
        else:
            csv_path.unlink()
    folder.mkdir(parents=True, exist_ok=True)

    stdout, stderr, code, _ = run_subprocess(["--info"], "", child_env("default"), 300)
    software = parse_result(stdout)
    if software is None:
        raise SystemExit(f"the worker cannot start (exit {code}):\n{stderr[-2000:]}")
    meta = {"tag": tag, "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "settings": asdict(settings), "methods": names, "threads": threads, "ranges": ranges,
            "scenarios": {k: {**asdict(s)} for k, s in scenarios.items()},
            "base_seed": bm.BASE_SEED, "machine": machine_metadata(), "software": software,
            "calibration_before": calibration()}
    if args.resume and meta_path.exists():
        earlier = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["earlier_sessions"] = earlier.pop("earlier_sessions", []) + [earlier]
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if meta["machine"].get("cpu_load_percent_before", 0) > 15:
        print(f"WARNING: the CPU is already {meta['machine']['cpu_load_percent_before']}% busy; "
              "timings will be distorted")

    new_file = not csv_path.exists()
    counts, start = {}, time.perf_counter()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for i, (scenario, value, rng, thread, name) in enumerate(jobs, 1):
            if (scenario.name, str(value), thread, name) in done:
                continue
            chain = chains.setdefault((scenario.name, name, thread), Chain())
            row = run_job(tag, scenario, value, rng, thread, bm.METHODS[name], settings, chain,
                          folder / "raw" / tag if folder == RESULTS else folder / "logs")
            chain.update(row)
            writer.writerow(_compact(row))
            f.flush()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"[{i}/{len(jobs)} {time.perf_counter() - start:7.0f} s] {scenario.name:>2} "
                  f"{scenario.vary}={value:<6} {thread:<7} {name:<22} {row['status']:<21} "
                  f"warm {_fmt_seconds(row.get('warm_median_s'))}, cold {_fmt_seconds(row.get('cold_s'))}"
                  + (f"  ({row['note']})" if row.get("note") and row["status"] != "ok" else ""),
                  flush=True)
            if row["status"] == "mismatch":
                print("!!! MISMATCH: this method does not reproduce the reference diagram", flush=True)

    meta.update(finished=dt.datetime.now().isoformat(timespec="seconds"),
                duration_s=round(time.perf_counter() - start), status_counts=counts,
                calibration_after=calibration())
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"done in {meta['duration_s']} s: {counts}")
    print(f"results: {csv_path}\nmetadata: {meta_path}")


if __name__ == "__main__":
    main()
