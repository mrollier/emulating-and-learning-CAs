"""Fig. 2 (thesis Fig. 7.2): one update of rule 54 inside the CNN, step by step.

The one-hot panel is the actual activation of the detector layer and the
bottom row the actual network output, for the published input configuration.
"""
import numpy as np
import tensorflow as tf

import _style
from ca_emulators import EcaEmulator
from ca_emulators.plotting import plot_decomposition

STEM = "eca-N32-rule54-decomposition"


def main(argv=None):
    args = _style.parser(__doc__.splitlines()[0]).parse_args(argv)
    _style.apply(args)
    fig2 = _style.load_npz(_style.DATA / "published_inputs" / "fig2_decomposition_rule54.npz")
    x, rule = fig2["x"], int(fig2["rule"])

    model = EcaEmulator(len(x), rule).model()
    inputs = x[np.newaxis, :, np.newaxis].astype(np.float32)
    detectors = tf.keras.Model(model.input, model.get_layer("detectors").output)(inputs).numpy()[0]
    output = model(inputs).numpy()[0, :, 0]
    assert np.array_equal(output, fig2["y"]), "the emulator does not reproduce Fig. 2"

    fig, _ = plot_decomposition(x, rule, detector_output=detectors, output=output)
    _style.save(fig, STEM, args)


if __name__ == "__main__":
    main()
