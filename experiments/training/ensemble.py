"""Ensemble engine: train thousands of independent tiny ECA emulators at once.

Every member of an ensemble is a translation-invariant network with receptive
field 3 (one width-3 convolution, optionally followed by 1x1 layers):

    u = enc(s_-, s_o, s_+)                        encoded neighbourhood, (3,)
    h_1 = act(u W1 + b1)                          W1: (3, H), the "detectors"
    h_{d+1} = act(h_d Wh[d] + bh[d])              D extra 1x1 layers, (H, H)
    z = h_{D+1} . wo + bo                         the "rule-table" layer, (H,)
    y = head(z)                                   the predicted next state

With H = 8, D = 0, ``enc="01"``, ``act="relu"`` and ``head="relu_tanh_mse"``
this is exactly ``ca_emulators.EcaEmulator(N, rule=None, activation="tanh")``
(40 parameters), the architecture of the 2024 training recipe.

Weights carry a leading member axis of size M, and the training loss is the
sum over members of each member's own mean loss. The gradient of that sum with
respect to member m's weights is the gradient of member m's loss alone, and
Adam acts elementwise, so members train independently, exactly as M separate
Keras networks would (see ``tests/test_ensemble.py``).

Two data paths:

``mode="patterns"`` (one-step training)
    A network with receptive field 3 sees a configuration only through its
    neighbourhoods, so the mean loss over the B*N cells of a batch equals
    sum_i f_i * loss_i, where f_i is the frequency of neighbourhood i in the
    batch. The forward pass therefore runs on the 8 neighbourhood patterns
    only, weighted either by the frequencies of a batch of random
    configurations (``data="random"``; drawn from a precomputed pool) or
    uniformly (``data="full"``: the complete set of 8 neighbourhoods as one
    full batch). This is exact, not an approximation.
``mode="configs"`` (spacetime training)
    Random configurations are rolled out ``T`` times through the network
    (backpropagation through time), with the loss on every frame or on the
    final frame only, optionally with a curriculum over T and with a
    straight-through binarisation between steps.

Everything uses plain TensorFlow ops (tf.Variable, tf.einsum, tf.GradientTape)
and a hand-written Adam identical to Keras's, so the code runs on TF 2.14 and
on newer versions.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import asdict, dataclass, fields

import numpy as np
import tensorflow as tf

from ca_emulators import rules as ca_rules

ENCODINGS = ("01", "pm1")
ACTIVATIONS = ("relu", "leaky_relu", "softplus")
HEADS = ("relu_tanh_mse", "tanh_mse", "sigmoid_mse", "sigmoid_bce", "linear_mse")
MODES = ("patterns", "configs")
DATA = ("random", "full")
FRAMES = ("all", "final")

N_PAT = ca_rules.N_NEIGHBOURHOODS
#: Row i is the neighbourhood (s_-, s_o, s_+) with index i = 4 s_- + 2 s_o + s_+.
PATTERNS = ca_rules.neighbourhood_patterns().astype(np.float32)
PARAM_NAMES = ("W1", "b1", "Wh", "bh", "wo", "bo")
BIT_WEIGHTS = (1 << np.arange(N_PAT)).astype(np.int64)
HE_NORMAL_CORRECTION = 0.87962566103423978  # std of a unit normal truncated at +-2


@dataclass(frozen=True)
class Config:
    """One training configuration (architecture, data, optimiser, schedule)."""

    name: str = "baseline_2024"
    # architecture
    enc: str = "01"                 # input encoding of the states: {0,1} or {-1,+1}
    act: str = "relu"               # hidden activation
    leaky_alpha: float = 0.1        # slope of leaky_relu for negative inputs
    head: str = "relu_tanh_mse"     # output head and loss
    width: int = 8                  # H, channels of the width-3 layer (and of 1x1 layers)
    depth: int = 0                  # D, extra 1x1 hidden layers
    bias_init: float = 0.0          # initial bias of every hidden unit (output bias: 0)
    # optimisation
    lr: float = 0.005
    steps: int = 2560               # 2024: 40 epochs x (4096 / 64) batches
    eval_every: int = 64            # one 2024 "epoch"
    mode: str = "patterns"
    data: str = "random"
    batch: int = 64                 # configurations per batch
    cells: int = 32                 # cells per configuration
    # the 2024 pretraining trick (patterns mode only)
    pretrain: bool = False
    pretrain_candidates: int = 32
    pretrain_steps: int = 32        # one epoch of 4096 configurations at batch size 128
    pretrain_batch: int = 128
    pretrain_threshold: float = 0.1
    # spacetime training (configs mode)
    T: int = 1
    frames: str = "all"
    curriculum: bool = False
    ste: bool = False
    # closed-loop test
    cl_steps: int = 100
    cl_configs: int = 8             # the first is the tiled de Bruijn configuration
    cl_cells: int = 32

    def __post_init__(self):
        checks = [(self.enc, ENCODINGS), (self.act, ACTIVATIONS), (self.head, HEADS),
                  (self.mode, MODES), (self.data, DATA), (self.frames, FRAMES)]
        for value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")
        if self.width < 1 or self.depth < 0 or self.T < 1 or self.steps < 0:
            raise ValueError("width >= 1, depth >= 0, T >= 1 and steps >= 0 are required")
        if self.mode == "patterns" and self.T != 1:
            raise ValueError("T > 1 needs mode='configs'")
        if self.pretrain and self.mode != "patterns":
            raise ValueError("the pretraining trick is implemented for mode='patterns'")
        if self.eval_every < 1:
            raise ValueError("eval_every must be positive")

    @property
    def config_id(self) -> int:
        """Stable integer derived from the name, used in the seed sequences."""
        return zlib.crc32(self.name.encode("utf-8"))

    @property
    def n_params(self) -> int:
        h, d = self.width, self.depth
        return 4 * h + d * (h * h + h) + h + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**d)


# --------------------------------------------------------------------------- seeds and init

def member_streams(base_seed: int, rule: int, seed: int, config_id: int):
    """SeedSequences (init, data, pretraining) of one member.

    The root is ``SeedSequence([base_seed, rule, seed, config_id])``, so a
    member's run does not depend on which ensemble or chunk it belongs to.
    """
    root = np.random.SeedSequence([int(base_seed), int(rule), int(seed), int(config_id)])
    return root.spawn(3)


def he_normal(rng: np.random.Generator, shape, fan_in: int) -> np.ndarray:
    """Keras's HeNormal: truncated normal (+-2 std) with variance 2 / fan_in."""
    out = rng.standard_normal(shape)
    bad = np.abs(out) > 2.0
    while bad.any():
        out[bad] = rng.standard_normal(int(bad.sum()))
        bad = np.abs(out) > 2.0
    return (out * np.sqrt(2.0 / fan_in) / HE_NORMAL_CORRECTION).astype(np.float32)


def init_member(cfg: Config, rng: np.random.Generator) -> dict:
    """Initial weights of one member (Keras layouts without the member axis)."""
    h, d = cfg.width, cfg.depth
    p = {"W1": he_normal(rng, (3, h), fan_in=3),
         "b1": np.full(h, cfg.bias_init, np.float32)}
    if d:
        p["Wh"] = np.stack([he_normal(rng, (h, h), fan_in=h) for _ in range(d)])
        p["bh"] = np.full((d, h), cfg.bias_init, np.float32)
    p["wo"] = he_normal(rng, (h,), fan_in=h)
    p["bo"] = np.float32(0.0)
    return p


def stack_params(members: list[dict]) -> dict:
    return {k: np.stack([np.asarray(p[k], np.float32) for p in members])
            for k in PARAM_NAMES if k in members[0]}


def take_params(params: dict, index) -> dict:
    return {k: np.asarray(v)[index] for k, v in params.items()}


# --------------------------------------------------------------------------- the network

def encode(s, enc: str):
    """States in [0, 1] (or real-valued network outputs) to network inputs."""
    return s if enc == "01" else 2.0 * s - 1.0


def activation_fn(cfg: Config):
    if cfg.act == "relu":
        return tf.nn.relu
    if cfg.act == "leaky_relu":
        return lambda x: tf.nn.leaky_relu(x, alpha=cfg.leaky_alpha)
    return tf.nn.softplus


def forward(params, u, cfg: Config):
    """Pre-activations of the hidden layers and the output logits.

    ``u`` holds encoded neighbourhoods, either (P, 3) shared by all members or
    (M, P, 3). Returns ``(pres, z)`` with ``pres`` a list of (M, P, H) tensors
    and ``z`` of shape (M, P).
    """
    act = activation_fn(cfg)
    if len(u.shape) == 2:
        pre = tf.einsum("pk,mkh->mph", u, params["W1"])
    else:
        pre = tf.matmul(u, params["W1"])
    pre = pre + params["b1"][:, None, :]
    pres = [pre]
    h = act(pre)
    for d in range(cfg.depth):
        pre = tf.matmul(h, params["Wh"][:, d]) + params["bh"][:, d][:, None, :]
        pres.append(pre)
        h = act(pre)
    z = tf.linalg.matvec(h, params["wo"]) + params["bo"][:, None]
    return pres, z


def forward_configs(params, x, cfg: Config):
    """Logits (M, S, N) for encoded periodic configurations ``x`` of shape (M, S, N)."""
    u = tf.stack([tf.roll(x, 1, axis=2), x, tf.roll(x, -1, axis=2)], axis=-1)
    _, z = forward(params, tf.reshape(u, [tf.shape(x)[0], -1, 3]), cfg)
    return tf.reshape(z, tf.shape(x))


def head_state(z, head: str):
    """The network's prediction of the next state (threshold 0.5)."""
    if head == "relu_tanh_mse":
        return tf.tanh(tf.nn.relu(z))
    if head == "tanh_mse":
        return tf.tanh(z)
    if head in ("sigmoid_mse", "sigmoid_bce"):
        return tf.sigmoid(z)
    return z


def head_loss(z, t, head: str):
    """Elementwise loss of logits ``z`` against binary targets ``t``."""
    if head == "sigmoid_bce":
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=z)
    return tf.square(head_state(z, head) - t)


class Adam:
    """Adam exactly as Keras 2.11+ implements it (epsilon outside the square root)."""

    def __init__(self, variables, learning_rate: float, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7):
        self.variables = list(variables)
        self.lr, self.beta_1, self.beta_2, self.epsilon = learning_rate, beta_1, beta_2, epsilon
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.variables]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.variables]
        self.iterations = tf.Variable(0, dtype=tf.int64, trainable=False)

    def reset(self):
        for s in self.m + self.v:
            s.assign(tf.zeros_like(s))
        self.iterations.assign(0)

    def apply(self, grads):
        step = tf.cast(self.iterations + 1, tf.float32)
        alpha = (self.lr * tf.sqrt(1.0 - tf.pow(tf.constant(self.beta_2, tf.float32), step))
                 / (1.0 - tf.pow(tf.constant(self.beta_1, tf.float32), step)))
        for var, g, m, v in zip(self.variables, grads, self.m, self.v):
            m.assign_add((g - m) * (1.0 - self.beta_1))
            v.assign_add((tf.square(g) - v) * (1.0 - self.beta_2))
            var.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        self.iterations.assign_add(1)


class EnsembleNet:
    """The trainable weights of M members and their optimiser."""

    def __init__(self, cfg: Config, params: dict):
        self.cfg = cfg
        self.vars = {k: tf.Variable(np.asarray(params[k], np.float32), name=k)
                     for k in PARAM_NAMES if k in params}
        self.var_list = [self.vars[k] for k in PARAM_NAMES if k in self.vars]
        self.adam = Adam(self.var_list, cfg.lr)
        self.M = int(params["W1"].shape[0])

    def numpy(self) -> dict:
        return {k: v.numpy() for k, v in self.vars.items()}


# --------------------------------------------------------------------------- data

def pattern_frequencies(configs) -> np.ndarray:
    """(..., 8) frequencies of the neighbourhoods in batches of periodic configurations.

    ``configs`` has shape (..., B, N); the frequencies of a batch are the
    weights that make the 8-pattern loss equal to the mean loss over its B*N cells.
    """
    x = np.asarray(configs).astype(np.uint8)
    lead, cells = x.shape[:-2], x.shape[-2] * x.shape[-1]
    idx = (np.roll(x, 1, axis=-1) << 2) | (x << 1) | np.roll(x, -1, axis=-1)
    idx = idx.reshape(-1, cells).astype(np.int64)
    offsets = N_PAT * np.arange(len(idx))[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=N_PAT * len(idx))
    return (counts.reshape(*lead, N_PAT) / cells).astype(np.float32)


_POOLS: dict = {}


def frequency_pool(batch: int, cells: int, size: int = 1 << 16, seed: int = 0) -> np.ndarray:
    """(size, 8) neighbourhood frequencies of ``size`` batches of random configurations."""
    key = ("freq", batch, cells, size, seed)
    if key not in _POOLS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED, batch, cells]))
        out = np.empty((size, N_PAT), np.float32)
        chunk = max(1, (1 << 22) // (batch * cells))
        for start in range(0, size, chunk):
            n = min(chunk, size - start)
            out[start:start + n] = pattern_frequencies(
                rng.integers(0, 2, size=(n, batch, cells), dtype=np.uint8))
        _POOLS[key] = out
    return _POOLS[key]


def config_pool(cells: int, size: int = 1 << 16, seed: int = 0) -> np.ndarray:
    """(size, cells) random configurations for spacetime training."""
    key = ("cfg", cells, size, seed)
    if key not in _POOLS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0F1, cells]))
        _POOLS[key] = rng.integers(0, 2, size=(size, cells), dtype=np.uint8)
    return _POOLS[key]


def closed_loop_configs(n_configs: int, cells: int, seed: int = 0) -> np.ndarray:
    """Test configurations: the tiled de Bruijn configuration, then random ones.

    The first row contains every neighbourhood, so a member that is not exact
    after one update fails the closed-loop test at the first update.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC105ED, cells]))
    rows = [ca_rules.de_bruijn_configuration(cells)]
    rows += list(rng.integers(0, 2, size=(n_configs - 1, cells), dtype=np.uint8))
    return np.stack(rows).astype(np.uint8)


def rule_tables(rules) -> np.ndarray:
    return np.stack([ca_rules.rule_table(r) for r in rules]).astype(np.float32)


# --------------------------------------------------------------------------- training loops

def make_pattern_trainer(net: EnsembleNet, targets, pool, jit: bool = False):
    """A tf.function that runs K optimisation steps on the 8 neighbourhoods.

    Called with ``idx`` of shape (K, M): member m draws the batch
    ``pool[idx[k, m]]`` in step k (ignored when ``pool`` is None: uniform
    weights, the full batch). Returns each member's mean loss over the K steps.
    """
    cfg = net.cfg
    u = tf.constant(encode(PATTERNS, cfg.enc))
    t = tf.constant(targets, tf.float32)
    pool_t = None if pool is None else tf.constant(pool, tf.float32)
    uniform = tf.fill([net.M, N_PAT], 1.0 / N_PAT)

    def body(k, acc, idx):
        w = uniform if pool_t is None else tf.gather(pool_t, idx[k])
        with tf.GradientTape() as tape:
            _, z = forward(net.vars, u, cfg)
            member_loss = tf.reduce_sum(w * head_loss(z, t, cfg.head), axis=1)
            total = tf.reduce_sum(member_loss)
        net.adam.apply(tape.gradient(total, net.var_list))
        return k + 1, acc + member_loss, idx

    return _step_loop(body, net.M, jit)


def _step_loop(body, m: int, jit: bool):
    """tf.function looping ``body`` over the first axis of idx (no autograph: fast tracing)."""
    def run(idx):
        n_steps = tf.shape(idx)[0]
        _, acc, _ = tf.while_loop(lambda k, acc, idx: k < n_steps, body,
                                  (tf.constant(0), tf.zeros([m], tf.float32), idx))
        return acc / tf.cast(n_steps, tf.float32)

    return tf.function(run, jit_compile=jit, autograph=False)


def eca_step_tf(s, tables):
    """One ECA update of integer states s (M, B, N) with per-member tables (M, 8)."""
    idx = 4 * tf.roll(s, 1, axis=2) + 2 * s + tf.roll(s, -1, axis=2)
    return tf.gather(tables, idx, batch_dims=1)


def make_config_trainer(net: EnsembleNet, targets, pool, T: int, jit: bool = False):
    """A tf.function running K BPTT steps on random configurations (spacetime training).

    ``idx`` has shape (K, M, B): member m trains in step k on the B
    configurations ``pool[idx[k, m]]`` rolled out T times.
    """
    cfg = net.cfg
    tables = tf.constant(np.asarray(targets, np.int32))
    pool_t = tf.constant(np.asarray(pool, np.int32))

    def body(k, acc, idx):
        s = tf.gather(pool_t, idx[k])                      # (M, B, N)
        x = encode(tf.cast(s, tf.float32), cfg.enc)
        refs = []
        for _ in range(T):
            s = eca_step_tf(s, tables)
            refs.append(tf.cast(s, tf.float32))
        with tf.GradientTape() as tape:
            member_loss = tf.zeros([net.M], tf.float32)
            for step in range(T):
                z = forward_configs(net.vars, x, cfg)
                if cfg.frames == "all" or step == T - 1:
                    member_loss += tf.reduce_mean(head_loss(z, refs[step], cfg.head),
                                                  axis=[1, 2])
                if step < T - 1:
                    y = head_state(z, cfg.head)
                    if cfg.ste:
                        y = y + tf.stop_gradient(tf.cast(y > 0.5, tf.float32) - y)
                    x = encode(y, cfg.enc)
            if cfg.frames == "all":
                member_loss /= float(T)
            total = tf.reduce_sum(member_loss)
        net.adam.apply(tape.gradient(total, net.var_list))
        return k + 1, acc + member_loss, idx

    return _step_loop(body, net.M, jit)


def curriculum_schedule(cfg: Config) -> list[tuple[int, int]]:
    """(T, steps) stages: T = 1, 2, 4, ... up to cfg.T with equal steps, or one stage."""
    if not cfg.curriculum or cfg.T == 1:
        return [(cfg.T, cfg.steps)]
    ts = []
    t = 1
    while t < cfg.T:
        ts.append(t)
        t *= 2
    ts.append(cfg.T)
    base, extra = divmod(cfg.steps, len(ts))
    return [(t, base + (1 if i < extra else 0)) for i, t in enumerate(ts)]


# --------------------------------------------------------------------------- metrics

def pattern_metrics(params, cfg: Config, targets) -> dict:
    """One-step metrics on the 8 neighbourhoods, per member (numpy arrays).

    exact: thresholded output correct on all 8 neighbourhoods (for
    translation-invariant networks with receptive field 3 this is equivalent
    to exactness on the de Bruijn certificate configuration 00010111);
    margin: min over neighbourhoods of (y - 0.5)(2t - 1), positive iff exact
    (and then equal to min |y - 0.5|).
    """
    u = tf.constant(encode(PATTERNS, cfg.enc))
    p = {k: tf.constant(np.asarray(v, np.float32)) for k, v in params.items()}
    pres, z = forward(p, u, cfg)
    y = head_state(z, cfg.head).numpy()
    z = z.numpy()
    t = np.asarray(targets) > 0.5
    pred = y > 0.5
    correct = pred == t
    out = {
        "exact": correct.all(axis=1),
        "margin": ((y - 0.5) * np.where(t, 1.0, -1.0)).min(axis=1),
        "err_mask": ((~correct) * BIT_WEIGHTS).sum(axis=1),
        "learned_rule": (pred * BIT_WEIGHTS).sum(axis=1),
        "y": y,
    }
    fire = [pre.numpy() > 0 for pre in pres]              # (M, 8, H) per layer
    out["dead_l1"] = (~fire[0].any(axis=1)).sum(axis=1)
    out["dead_hidden"] = sum((~f.any(axis=1)).sum(axis=1) for f in fire[1:]) if cfg.depth \
        else np.zeros(len(y), np.int64)
    out["stuck_out"] = ((z <= 0) & t).sum(axis=1) if cfg.head == "relu_tanh_mse" \
        else np.zeros(len(y), np.int64)
    # template recovery in the width-3 layer: "pure" units fire on one neighbourhood only
    n_fire = fire[0].sum(axis=1)                          # (M, H)
    pure = n_fire == 1
    detected = np.where(pure[:, None, :], fire[0], False).any(axis=2)  # (M, 8)
    out["n_pure_units"] = pure.sum(axis=1)
    out["cover_pure"] = detected.sum(axis=1)
    out["template_strict"] = (out["cover_pure"] == N_PAT) & (out["n_pure_units"] == cfg.width) \
        & (cfg.depth == 0)
    return out


def origin_gradient_norms(params, cfg: Config, targets) -> tuple[np.ndarray, np.ndarray]:
    """Norm of the gradient of the loss on neighbourhood 000 alone, per member.

    Returns (all parameters, width-3 layer only). Diagnoses the "dead origin"
    hypothesis H1: with {0,1} inputs and zero biases, 000 gives zero
    pre-activations everywhere and ReLU'(0) = 0 in TensorFlow.
    """
    u = tf.constant(encode(PATTERNS[:1], cfg.enc))
    vars_ = {k: tf.Variable(np.asarray(v, np.float32)) for k, v in params.items()}
    t = tf.constant(np.asarray(targets, np.float32)[:, :1])
    with tf.GradientTape() as tape:
        _, z = forward(vars_, u, cfg)
        total = tf.reduce_sum(head_loss(z, t, cfg.head))
    names = [k for k in PARAM_NAMES if k in vars_]
    grads = tape.gradient(total, [vars_[k] for k in names])
    sq = {k: np.square(g.numpy()).reshape(g.shape[0], -1).sum(axis=1)
          for k, g in zip(names, grads)}
    return np.sqrt(sum(sq.values())), np.sqrt(sq["W1"] + sq["b1"])


#: Radii tried by the interval certificate of closed-loop exactness.
CERT_EPS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)


def interval_certificate(params, cfg: Config, targets, eps_grid=CERT_EPS) -> np.ndarray:
    """Largest radius eps (0 if none) certified by interval bound propagation.

    The certificate holds for eps if every real-valued neighbourhood within
    eps (max norm, in state space) of a binary neighbourhood p is mapped to an
    output within eps of the correct next state f(p). By induction, iterating
    the network WITHOUT binarisation then stays within eps of the exact ECA
    trajectory from every binary configuration, of any size, forever. All
    activations and output heads are monotone, so propagating intervals layer
    by layer gives sound (if loose) bounds.
    """
    act = activation_fn(cfg)
    p = {k: tf.constant(np.asarray(v, np.float32)) for k, v in params.items()}
    t = np.asarray(targets) > 0.5
    centre = tf.constant(encode(PATTERNS, cfg.enc))
    scale = 1.0 if cfg.enc == "01" else 2.0
    best = np.zeros(len(t), np.float32)
    for eps in eps_grid:
        pc = tf.einsum("pk,mkh->mph", centre, p["W1"]) + p["b1"][:, None, :]
        pr = scale * eps * tf.reduce_sum(tf.abs(p["W1"]), axis=1)[:, None, :]
        lo, hi = act(pc - pr), act(pc + pr)
        for d in range(cfg.depth):
            c, r = (lo + hi) / 2, (hi - lo) / 2
            pc = tf.matmul(c, p["Wh"][:, d]) + p["bh"][:, d][:, None, :]
            pr = tf.matmul(r, tf.abs(p["Wh"][:, d]))
            lo, hi = act(pc - pr), act(pc + pr)
        c, r = (lo + hi) / 2, (hi - lo) / 2
        zc = tf.linalg.matvec(c, p["wo"]) + p["bo"][:, None]
        zr = tf.linalg.matvec(r, tf.abs(p["wo"]))
        y_lo = head_state(zc - zr, cfg.head).numpy()
        y_hi = head_state(zc + zr, cfg.head).numpy()
        ok = np.where(t, (y_lo >= 1 - eps) & (y_hi <= 1 + eps),
                      (y_hi <= eps) & (y_lo >= -eps)).all(axis=1)
        best = np.where(ok, np.float32(eps), best)
    return best


def _closed_loop_block(cfg: Config, n_steps: int, jit: bool):
    def run(p, x0, refs, rule_idx):
        m = tf.shape(rule_idx)[0]
        x = encode(tf.broadcast_to(x0[None], [m, tf.shape(x0)[0], tf.shape(x0)[1]]), cfg.enc)

        def body(step, x, first, maxdev):
            y = head_state(forward_configs(p, x, cfg), cfg.head)
            ref = tf.gather(refs[step + 1], rule_idx)
            wrong = tf.reduce_any((y > 0.5) != (ref > 0.5), axis=[1, 2])
            first = tf.where(wrong & (first == n_steps), step, first)
            maxdev = tf.maximum(maxdev, tf.reduce_max(tf.abs(y - ref), axis=[1, 2]))
            return step + 1, encode(y, cfg.enc), first, maxdev

        _, _, first, maxdev = tf.while_loop(
            lambda step, *_: step < n_steps, body,
            (tf.constant(0), x, tf.fill([m], n_steps), tf.zeros([m], tf.float32)))
        return first, maxdev
    return tf.function(run, jit_compile=jit, reduce_retracing=True, autograph=False)


_REFERENCES: dict = {}


def reference_trajectories(x0, n_steps: int) -> np.ndarray:
    """(256, n_steps + 1, S, N) uint8: every rule's spacetime diagrams from x0 (cached).

    Vectorised over rules; ``tests/test_ensemble.py`` checks it against
    ``ca_emulators.reference.evolve``.
    """
    x0 = np.asarray(x0, np.uint8)
    key = (x0.tobytes(), x0.shape, n_steps)
    if key not in _REFERENCES:
        tables = rule_tables(range(256)).astype(np.uint8)            # (256, 8)
        x = np.broadcast_to(x0, (256, *x0.shape)).copy()
        frames = [x]
        rows = np.arange(256)[:, None, None]
        for _ in range(n_steps):
            idx = (np.roll(x, 1, axis=-1) << 2) | (x << 1) | np.roll(x, -1, axis=-1)
            x = tables[rows, idx]
            frames.append(x)
        _REFERENCES[key] = np.stack(frames, axis=1)
    return _REFERENCES[key]


def closed_loop(params, cfg: Config, rules, member_mask=None, seed: int = 0,
                max_positions: int = 1 << 22, jit: bool | None = None):
    """Iterate each network cfg.cl_steps times WITHOUT binarisation.

    The test set is ``closed_loop_configs(cfg.cl_configs, cfg.cl_cells)``.
    Returns, per member, the number of consecutive updates whose thresholded
    output agrees with the reference ECA on every test configuration
    (``cfg.cl_steps`` means closed-loop exact), and the largest deviation
    |y - reference| seen along the way. Members outside ``member_mask`` are
    not simulated and get 0 (they are not exact after one update, and the
    tiled de Bruijn test configuration exposes that at the first update).
    """
    rules = np.asarray(rules)
    m_total = len(rules)
    first = np.zeros(m_total, np.int64)
    maxdev = np.full(m_total, np.nan, np.float32)
    todo = np.flatnonzero(np.ones(m_total, bool) if member_mask is None else member_mask)
    if len(todo) == 0 or cfg.cl_steps == 0:
        return first, maxdev
    if jit is None:
        jit = cfg.depth == 0
    x0 = closed_loop_configs(cfg.cl_configs, cfg.cl_cells, seed)
    unique = np.unique(rules[todo])
    ref = reference_trajectories(x0, cfg.cl_steps)[unique]              # (U, T+1, S, N)
    refs = tf.constant(np.moveaxis(ref, 1, 0).astype(np.float32))        # (T+1, U, S, N)
    lookup = {int(r): i for i, r in enumerate(unique)}
    run = _closed_loop_block(cfg, cfg.cl_steps, jit)
    per_member = cfg.cl_configs * cfg.cl_cells * cfg.width * (cfg.depth + 1)
    block = max(1, min(len(todo), max_positions // per_member))
    x0_t = tf.constant(x0.astype(np.float32))
    for start in range(0, len(todo), block):
        sel = todo[start:start + block]
        p = {k: tf.constant(np.asarray(v, np.float32)[sel]) for k, v in params.items()}
        idx = tf.constant([lookup[int(r)] for r in rules[sel]], tf.int32)
        f, d = run(p, x0_t, refs, idx)
        first[sel] = f.numpy()
        maxdev[sel] = d.numpy()
    return first, maxdev


# --------------------------------------------------------------------------- one ensemble run

def _pretrain(cfg: Config, rules, streams, base_seed: int, log, jit: bool):
    """The 2024 pretraining loop, run for all members and candidates in parallel.

    Candidate c of a member is a fresh network trained for one epoch
    (``pretrain_steps`` batches of ``pretrain_batch`` configurations). As in
    2024, the member keeps the first candidate whose epoch loss is below the
    threshold (the loop stopped there), or else the best of all candidates.
    All candidates of a member come from one generator (its pretraining
    stream), drawn in candidate order.
    """
    m, r = len(rules), cfg.pretrain_candidates
    pool = frequency_pool(cfg.pretrain_batch, cfg.cells, seed=base_seed)
    rngs = [np.random.default_rng(s[2]) for s in streams]
    cand = [[init_member(cfg, g) for _ in range(r)] for g in rngs]           # [m][r]
    idx_all = np.stack([g.integers(0, len(pool), size=(r, cfg.pretrain_steps)) for g in rngs])
    losses = np.empty((m, r), np.float32)
    trained = {}
    group = max(1, min(r, (1 << 16) // max(1, m)))
    for c0 in range(0, r, group):
        cs = list(range(c0, min(r, c0 + group)))
        pairs = [(i, c) for c in cs for i in range(m)]
        params = stack_params([cand[i][c] for i, c in pairs])
        net = EnsembleNet(cfg, params)
        trainer = make_pattern_trainer(net, rule_tables(rules[[i for i, _ in pairs]]), pool, jit)
        idx = np.stack([idx_all[i, c] for i, c in pairs], axis=1).astype(np.int32)
        loss = trainer(tf.constant(idx)).numpy()
        final = net.numpy()
        for j, (i, c) in enumerate(pairs):
            losses[i, c] = loss[j]
            trained[(i, c)] = (final, j)
        log(f"    pretraining candidates {cs[0]}-{cs[-1]} of {r} ({len(pairs)} networks)")
    below = losses < cfg.pretrain_threshold
    chosen = np.where(below.any(axis=1), below.argmax(axis=1), losses.argmin(axis=1))
    selected = []
    for i in range(m):
        final, j = trained[(i, int(chosen[i]))]
        selected.append(take_params(final, j))
    info = {"pretrain_restarts": np.where(below.any(axis=1), chosen + 1, r),
            "pretrain_loss": losses[np.arange(m), chosen],
            "pretrain_pass": below.sum(axis=1)}
    return stack_params(selected), info


def run_members(cfg: Config, members, base_seed: int = 2024, log=print, jit=None,
                keep_params: bool = False) -> tuple[list[dict], dict]:
    """Train one ensemble (all ``members`` = [(rule, seed), ...]) and return per-member rows.

    ``jit=None`` compiles the training loop with XLA only for one-step
    (patterns-mode) training of networks without hidden 1x1 layers: on a CPU
    this is 3-8x faster there and slower everywhere else.
    """
    t_start = time.time()
    if jit is None:
        jit = cfg.depth == 0 and cfg.mode == "patterns"
    rules = np.array([int(r) for r, _ in members])
    seeds = np.array([int(s) for _, s in members])
    m = len(rules)
    targets = rule_tables(rules)
    streams = [member_streams(base_seed, r, s, cfg.config_id) for r, s in zip(rules, seeds)]

    info = {}
    if cfg.pretrain:
        params, info = _pretrain(cfg, rules, streams, base_seed, log, jit)
    else:
        params = stack_params([init_member(cfg, np.random.default_rng(s[0])) for s in streams])
    data_rngs = [np.random.default_rng(s[1]) for s in streams]

    init = pattern_metrics(params, cfg, targets)
    g000, g000_l1 = origin_gradient_norms(params, cfg, targets)
    net = EnsembleNet(cfg, params)

    if cfg.mode == "patterns":
        pool = None if cfg.data == "full" else frequency_pool(cfg.batch, cfg.cells, seed=base_seed)
        stages = [(1, cfg.steps)]
        if pool is None:
            all_idx = np.zeros((cfg.steps, m), np.int32)
        else:
            all_idx = np.stack([g.integers(0, len(pool), size=cfg.steps) for g in data_rngs],
                               axis=1).astype(np.int32)
    else:
        pool = config_pool(cfg.cells, seed=base_seed)
        stages = curriculum_schedule(cfg)
    exact_hist = [init["exact"]]
    loss_last = np.full(m, np.nan, np.float32)
    done = 0
    for T, n_stage in stages:
        if cfg.mode == "patterns":
            trainer = make_pattern_trainer(net, targets, pool, jit)
        else:
            trainer = make_config_trainer(net, targets, pool, T, jit)
        stage_done = 0
        while stage_done < n_stage:
            k = min(cfg.eval_every - done % cfg.eval_every, n_stage - stage_done)
            if cfg.mode == "patterns":
                idx = all_idx[done:done + k]
            else:
                idx = np.stack([g.integers(0, len(pool), size=(k, cfg.batch)) for g in data_rngs],
                               axis=1).astype(np.int32)
            loss_last = trainer(tf.constant(idx)).numpy()
            stage_done += k
            done += k
            if done % cfg.eval_every == 0 or done == cfg.steps:
                exact_hist.append(pattern_metrics(net.numpy(), cfg, targets)["exact"])
        if len(stages) > 1:
            log(f"    curriculum stage T={T} done ({done}/{cfg.steps} steps)")
    t_train = time.time() - t_start

    final_params = net.numpy()
    final = pattern_metrics(final_params, cfg, targets)
    hist = np.stack(exact_hist)                          # (n_evals + 1, M), row 0 = init
    eval_steps = np.array([0] + [min(cfg.eval_every * (i + 1), cfg.steps)
                                 for i in range(len(hist) - 1)])
    ever = hist.any(axis=0)
    first_idx = np.where(ever, hist.argmax(axis=0), -1)
    first_step = np.where(ever, eval_steps[np.maximum(first_idx, 0)], -1)
    flips = (hist[:-1] & ~hist[1:]).sum(axis=0)
    persist = ever & np.array([hist[first_idx[j]:, j].all() if ever[j] else False
                               for j in range(m)])
    cl_first, cl_dev = closed_loop(final_params, cfg, rules, final["exact"], seed=base_seed)
    cert = interval_certificate(final_params, cfg, targets)

    rows = []
    for j in range(m):
        row = {
            "config": cfg.name, "rule": int(rules[j]), "seed": int(seeds[j]),
            "exact_init": bool(init["exact"][j]),
            "exact_final": bool(final["exact"][j]),
            "exact_ever": bool(ever[j]),
            "first_exact_step": int(first_step[j]),
            "persist": bool(persist[j]),
            "n_flips": int(flips[j]),
            "margin_final": round(float(final["margin"][j]), 6),
            "err_mask": int(final["err_mask"][j]),
            "learned_rule": int(final["learned_rule"][j]),
            "loss_final": float(loss_last[j]),
            "cl_steps": int(cl_first[j]),
            "cl_exact": bool(cl_first[j] == cfg.cl_steps),
            "cl_maxdev": round(float(cl_dev[j]), 6),
            "cl_cert_eps": float(cert[j]),
            "dead_l1_init": int(init["dead_l1"][j]),
            "dead_l1_final": int(final["dead_l1"][j]),
            "dead_hidden_final": int(final["dead_hidden"][j]),
            "stuck_out_init": int(init["stuck_out"][j]),
            "stuck_out_final": int(final["stuck_out"][j]),
            "g000_init": round(float(g000[j]), 6),
            "g000_l1_init": round(float(g000_l1[j]), 6),
            "n_pure_units": int(final["n_pure_units"][j]),
            "cover_pure": int(final["cover_pure"][j]),
            "template_strict": bool(final["template_strict"][j]),
        }
        for key in ("pretrain_restarts", "pretrain_loss", "pretrain_pass"):
            row[key] = (float(info[key][j]) if key == "pretrain_loss" else int(info[key][j])) \
                if key in info else ""
        rows.append(row)
    extra = {"seconds": time.time() - t_start, "seconds_train": t_train, "members": m}
    if keep_params:
        extra["params"] = final_params
    return rows, extra


def estimate_member_bytes(cfg: Config) -> int:
    """Rough peak memory per member (weights, Adam state, activations), in bytes."""
    floats = cfg.n_params * 5
    if cfg.mode == "patterns":
        floats += N_PAT * cfg.width * (cfg.depth + 1) * 8
        if cfg.pretrain:
            floats *= cfg.pretrain_candidates
    else:
        floats += cfg.batch * cfg.cells * cfg.width * (cfg.depth + 1) * cfg.T * 6
    return 4 * floats + 256
