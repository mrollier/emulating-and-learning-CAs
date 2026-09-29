"""Fast checks of the ensemble engine (run: pytest experiments/training/tests)."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import ensemble as E
from ca_emulators import EcaEmulator, PeriodicPadding1D
from ca_emulators import rules as ca_rules
from ca_emulators import weights as ca_weights
from ca_emulators.reference import eca_step

N_CELLS, BATCH, STEPS = 16, 4, 6


def _batches(seed=0, steps=STEPS, batch=BATCH, cells=N_CELLS):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(steps, batch, cells), dtype=np.uint8)


def _init(cfg, members, base_seed=7):
    streams = [E.member_streams(base_seed, r, s, cfg.config_id) for r, s in members]
    return E.stack_params([E.init_member(cfg, np.random.default_rng(s[0])) for s in streams])


def _keras_model(cfg, params, m):
    """A standalone Keras network with the architecture and weights of member m."""
    act = {"relu": "relu", "softplus": "softplus",
           "leaky_relu": lambda x: tf.nn.leaky_relu(x, alpha=cfg.leaky_alpha)}[cfg.act]
    head = {"relu_tanh_mse": None, "tanh_mse": "tanh", "sigmoid_mse": "sigmoid",
            "sigmoid_bce": None, "linear_mse": None}[cfg.head]
    inputs = tf.keras.Input((N_CELLS, 1))
    x = PeriodicPadding1D(1)(inputs)
    layer1 = tf.keras.layers.Conv1D(cfg.width, 3, activation=act)
    x = layer1(x)
    hidden = [tf.keras.layers.Conv1D(cfg.width, 1, activation=act) for _ in range(cfg.depth)]
    for layer in hidden:
        x = layer(x)
    out = tf.keras.layers.Conv1D(1, 1, activation="relu" if cfg.head == "relu_tanh_mse" else None)
    x = out(x)
    if cfg.head == "relu_tanh_mse":
        x = tf.keras.layers.Activation("tanh")(x)
    elif head:
        x = tf.keras.layers.Activation(head)(x)
    model = tf.keras.Model(inputs, x)
    layer1.set_weights([params["W1"][m][:, None, :], params["b1"][m]])
    for d, layer in enumerate(hidden):
        layer.set_weights([params["Wh"][m, d][None], params["bh"][m, d]])
    out.set_weights([params["wo"][m][None, :, None], params["bo"][m][None]])
    return model


def _train_keras(model, cfg, batches, rule):
    loss_fn = (tf.keras.losses.BinaryCrossentropy(from_logits=True)
               if cfg.head == "sigmoid_bce" else tf.keras.losses.MeanSquaredError())
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    for b in batches:
        x = E.encode(b.astype(np.float32), cfg.enc)[..., None]
        y = eca_step(b, rule).astype(np.float32)[..., None]
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))


def _train_ensemble_patterns(cfg, params, member_rules, batches):
    """Pattern mode, each member weighting the 8 neighbourhoods like the given batches."""
    net = E.EnsembleNet(cfg, params)
    pool = E.pattern_frequencies(batches)                       # (steps, 8)
    trainer = E.make_pattern_trainer(net, E.rule_tables(member_rules), pool)
    idx = np.repeat(np.arange(len(batches))[:, None], len(member_rules), axis=1)
    trainer(tf.constant(idx.astype(np.int32)))
    return net.numpy()


def test_package_architecture_is_the_2024_member():
    """The default Config is ca_emulators' trainable EcaEmulator with a tanh output.

    Trainable, the rule-table layer has a bias, so the network has 40 + 1 parameters.
    """
    cfg = E.Config(name="t")
    params = _init(cfg, [(54, 0)])
    model = EcaEmulator(N_CELLS, activation="tanh").model()
    assert model.count_params() == cfg.n_params == 41
    model.get_layer("detectors").set_weights([params["W1"][0][:, None, :], params["b1"][0]])
    model.get_layer("rule_tables").set_weights([params["wo"][0][None, :, None],
                                               params["bo"][0][None]])
    x = _batches(1)[0]
    keras_y = model(x[..., None].astype(np.float32)).numpy()[..., 0]
    p = {k: tf.constant(v) for k, v in params.items()}
    z = E.forward_configs(p, tf.constant(x[None].astype(np.float32)), cfg)
    np.testing.assert_allclose(E.head_state(z, cfg.head).numpy()[0], keras_y, atol=1e-6)


@pytest.mark.parametrize("cfg", [
    E.Config(name="baseline"),
    E.Config(name="variant", enc="pm1", act="leaky_relu", head="sigmoid_bce", width=6,
             depth=1, bias_init=0.1, lr=0.02),
    E.Config(name="softplus", act="softplus", head="linear_mse", width=5, lr=0.01),
])
def test_member_equals_standalone_keras_network(cfg):
    """One ensemble member follows the same trajectory as a Keras network with its weights."""
    member_rules = [110, 54, 1]
    params = _init(cfg, [(r, 0) for r in member_rules])
    batches = _batches(2)
    trained = _train_ensemble_patterns(cfg, params, member_rules, batches)
    for m, rule in enumerate(member_rules):
        model = _keras_model(cfg, params, m)
        _train_keras(model, cfg, batches, rule)
        weights = model.get_weights()
        np.testing.assert_allclose(trained["W1"][m], weights[0][:, 0, :], atol=2e-5)
        np.testing.assert_allclose(trained["b1"][m], weights[1], atol=2e-5)
        np.testing.assert_allclose(trained["wo"][m], weights[-2][0, :, 0], atol=2e-5)
        np.testing.assert_allclose(trained["bo"][m], weights[-1][0], atol=2e-5)
        for d in range(cfg.depth):
            np.testing.assert_allclose(trained["Wh"][m, d], weights[2 + 2 * d][0], atol=2e-5)


@pytest.mark.parametrize("head", ["relu_tanh_mse", "sigmoid_bce"])
def test_patterns_mode_equals_configs_mode(head):
    """Weighting the 8 neighbourhoods by batch frequencies = training on the configurations."""
    cfg = E.Config(name="t", head=head, mode="configs", T=1, batch=BATCH, cells=N_CELLS)
    member_rules = [30, 1]
    params = _init(cfg, [(r, 0) for r in member_rules])
    batches = _batches(3)
    via_patterns = _train_ensemble_patterns(cfg, params, member_rules, batches)
    net = E.EnsembleNet(cfg, params)
    pool = batches.reshape(-1, N_CELLS)                        # configuration pool
    trainer = E.make_config_trainer(net, E.rule_tables(member_rules), pool, T=1)
    idx = np.arange(len(pool)).reshape(STEPS, 1, BATCH).repeat(len(member_rules), axis=1)
    trainer(tf.constant(idx.astype(np.int32)))
    for k, v in net.numpy().items():
        np.testing.assert_allclose(v, via_patterns[k], atol=2e-5)


def test_analytic_template_is_exact_for_every_rule():
    cfg = E.Config(name="t", head="linear_mse")
    rules = np.arange(256)
    kernel = ca_weights.detector_kernel()[:, 0, :]
    params = {"W1": np.repeat(kernel[None], 256, 0),
              "b1": np.repeat(ca_weights.detector_bias()[None], 256, 0),
              "wo": E.rule_tables(rules), "bo": np.zeros(256, np.float32)}
    out = E.pattern_metrics(params, cfg, E.rule_tables(rules))
    assert out["exact"].all() and np.allclose(out["margin"], 0.5)
    assert out["template_strict"].all() and (out["learned_rule"] == rules).all()
    first, dev = E.closed_loop(params, cfg, rules)
    assert (first == cfg.cl_steps).all() and np.allclose(dev, 0)
    # the analytic network is exactly binary, but it amplifies off-binary inputs
    # (gain up to 3 * omega), so the interval certificate does not apply to it,
    # except for rule 0, whose rule-table weights (hence outputs) are all zero
    cert = E.interval_certificate(params, cfg, E.rule_tables(rules))
    assert cert[0] > 0 and (cert[1:] == 0).all()


def test_interval_certificate_is_sound():
    """A certified network stays exact in closed loop (here: a saturated sigmoid template)."""
    cfg = E.Config(name="t", head="sigmoid_bce", cl_steps=30)
    rules = np.arange(256)
    kernel = ca_weights.detector_kernel(omega=1.0)[:, 0, :]
    params = {"W1": np.repeat(kernel[None], 256, 0),
              "b1": np.repeat(ca_weights.detector_bias()[None] - 0.5, 256, 0),
              "wo": 40 * (2 * E.rule_tables(rules) - 1), "bo": np.zeros(256, np.float32)}
    eps = E.interval_certificate(params, cfg, E.rule_tables(rules))
    first, _ = E.closed_loop(params, cfg, rules)
    assert (eps > 0).any()
    assert (first[eps > 0] == cfg.cl_steps).all()


def test_member_does_not_depend_on_its_ensemble():
    cfg = E.Config(name="t", steps=64, eval_every=32, cl_steps=10)
    alone, _ = E.run_members(cfg, [(54, 3)], log=lambda *_: None)
    together, _ = E.run_members(cfg, [(1, 0), (54, 3), (110, 1)], log=lambda *_: None)
    a, b = alone[0], together[1]
    assert (a["rule"], a["seed"]) == (b["rule"], b["seed"])
    for key in ("exact_final", "err_mask", "learned_rule", "first_exact_step"):
        assert a[key] == b[key]
    assert a["loss_final"] == pytest.approx(b["loss_final"], abs=1e-6)


def test_dead_origin_has_zero_gradient_in_the_2024_setup():
    """H1: {0,1} inputs and zero biases leave neighbourhood 000 without any gradient."""
    odd = [(r, 0) for r in range(1, 256, 2)]
    base = E.Config(name="t")
    g, g1 = E.origin_gradient_norms(_init(base, odd), base, E.rule_tables([r for r, _ in odd]))
    assert np.all(g == 0) and np.all(g1 == 0)
    # {-1,+1} inputs revive 000 in the detectors, but the 2024 output ReLU (H2) still
    # blocks it whenever the initial output pre-activation at 000 is negative (about half)
    pm1 = E.Config(name="t", enc="pm1")
    g, g1 = E.origin_gradient_norms(_init(pm1, odd), pm1, E.rule_tables([r for r, _ in odd]))
    assert 0.2 < np.mean(g == 0) < 0.8
    # without the output ReLU every network receives a gradient at 000
    fixed = E.Config(name="t", enc="pm1", head="sigmoid_bce")
    g, g1 = E.origin_gradient_norms(_init(fixed, odd), fixed, E.rule_tables([r for r, _ in odd]))
    assert np.all(g > 0) and np.mean(g1 > 0) > 0.9


def test_reference_trajectories_match_the_package():
    from ca_emulators.reference import evolve

    x0 = E.closed_loop_configs(3, 16, seed=5)
    ref = E.reference_trajectories(x0, 12)
    for rule in (0, 1, 30, 54, 110, 150, 255):
        np.testing.assert_array_equal(ref[rule], np.moveaxis(evolve(x0, rule, 12), 1, 0))


def test_data_helpers():
    freq = E.pattern_frequencies(ca_rules.de_bruijn_configuration(16)[None, None])
    np.testing.assert_allclose(freq[0, 0], np.full(8, 1 / 8))
    pool = E.frequency_pool(4, 8, size=100, seed=1)
    np.testing.assert_allclose(pool.sum(axis=1), 1, atol=1e-6)
    x = E.closed_loop_configs(4, 16)
    assert np.allclose(E.pattern_frequencies(x[:1][None])[0, 0], 1 / 8)
    stages = E.curriculum_schedule(E.Config(name="t", mode="configs", T=8, curriculum=True,
                                            steps=10))
    assert [t for t, _ in stages] == [1, 2, 4, 8] and sum(n for _, n in stages) == 10


def test_pretraining_keeps_first_candidate_below_threshold():
    cfg = E.Config(name="t", pretrain=True, pretrain_candidates=3, pretrain_steps=4, steps=0,
                   cl_steps=5, pretrain_threshold=10.0)          # every candidate passes
    rows, _ = E.run_members(cfg, [(54, 0), (1, 0)], log=lambda *_: None)
    assert all(r["pretrain_restarts"] == 1 and r["pretrain_pass"] == 3 for r in rows)
    cfg = E.Config(name="t", pretrain=True, pretrain_candidates=3, pretrain_steps=4, steps=0,
                   cl_steps=5, pretrain_threshold=-1.0)          # none passes: keep the best
    rows, _ = E.run_members(cfg, [(54, 0)], log=lambda *_: None)
    assert rows[0]["pretrain_restarts"] == 3 and rows[0]["pretrain_pass"] == 0
