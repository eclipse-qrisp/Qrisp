# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests the qml preprocessing functions minmax_scale, normalize and binarize."""

import numpy as np
import pytest

from qrisp import QuantumFloat
from qrisp.qml import Encoding, binarize, minmax_scale, normalize


@pytest.mark.parametrize("feature_range", [None, (0.0, 2 * np.pi), (-np.pi, np.pi), (0.0, 1.0)])
def test_minmax_scale_spans_the_requested_range(feature_range):
    """
    The extremes of each training feature land on the boundaries of the
    requested range, which is (0, pi) when none is given.
    """
    X = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])

    if feature_range is None:
        scaled = minmax_scale(X)
        low, high = 0.0, np.pi
    else:
        scaled = minmax_scale(X, feature_range=feature_range)
        low, high = feature_range

    assert np.allclose(scaled.min(axis=0), low)
    assert np.allclose(scaled.max(axis=0), high)
    # Evenly spaced inputs stay evenly spaced.
    assert np.allclose(scaled[1], (low + high) / 2)


def test_minmax_scale_applies_the_training_range_to_the_test_set():
    """
    The test set is scaled with the training minimum and maximum, not its own.
    Scaling it separately would place its own extremes on the boundaries and
    leak its distribution into the model.
    """
    X_train = np.array([[0.0], [10.0]])
    X_test = np.array([[2.5], [5.0]])

    _, scaled_test = minmax_scale(X_train, X_test)

    # 2.5 and 5.0 are a quarter and a half of the way through [0, 10].
    assert np.allclose(scaled_test.ravel(), [np.pi / 4, np.pi / 2])


def test_minmax_scale_clips_test_features_outside_the_training_range():
    """
    A test feature beyond the training range would leave the interval the
    encodings expect. It is therefore clipped to the boundary.
    """
    X_train = np.array([[0.0], [1.0]])
    X_test = np.array([[-5.0], [7.0]])

    _, scaled_test = minmax_scale(X_train, X_test)

    assert np.allclose(scaled_test.ravel(), [0.0, np.pi])


def test_minmax_scale_maps_a_constant_feature_to_the_lower_bound():
    """A feature that never varies has no range to scale onto the interval."""
    X = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])

    scaled = minmax_scale(X)

    assert np.allclose(scaled[:, 0], 0.0)
    assert np.allclose(scaled[:, 1], [0.0, np.pi / 2, np.pi])


@pytest.mark.parametrize(
    "feature_range, periodic, message",
    [
        ((0.0, 7.0), True, "longer than 2\\*pi"),
        ((1.0, 1.0), True, "empty feature range"),
        ((1.0, 1.0), False, "empty feature range"),
        ((1.0, np.inf), False, "not finite"),
        ((np.nan, 1.0), True, "not finite"),
    ],
)
def test_minmax_scale_rejects_an_unusable_range(feature_range, periodic, message):
    """
    A range needs a positive, finite width to scale onto. A range longer than
    2*pi is not injective when the features are read as rotation angles.
    """
    with pytest.raises(ValueError, match=message):
        minmax_scale(np.array([[0.0], [1.0]]), feature_range=feature_range, periodic=periodic)


def test_minmax_scale_matches_the_scaling_of_amplitude_encoding():
    """
    Without the period limit, minmax_scale reproduces the default scaling of
    amplitude encoding. Scaling in advance and encoding without a scaling
    therefore prepares the same state as letting the encoder scale.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5, 3))
    X_scaled = minmax_scale(X, feature_range=(1.0, 10.0), periodic=False)

    for sample, scaled in zip(X, X_scaled):
        qf_encoder = QuantumFloat(2)
        Encoding.amplitude(n_features=3, data=X).apply(qf_encoder, sample)

        qf_scaled = QuantumFloat(2)
        Encoding.amplitude(n_features=3).apply(qf_scaled, scaled)

        assert np.allclose(qf_encoder.qs.statevector("array"), qf_scaled.qs.statevector("array"))


def test_minmax_scale_rejects_a_test_set_with_a_different_feature_count():
    """The training range only defines a scaling for the features it covers."""
    with pytest.raises(ValueError, match="Tried to scale 3 features"):
        minmax_scale(np.array([[0.0, 1.0], [1.0, 2.0]]), np.array([[0.0, 1.0, 2.0]]))


def test_normalize_scales_each_sample_separately():
    """
    Each sample becomes its own quantum state. Each row is therefore divided by
    its own norm, not by a norm over the whole array. The 3-4-5 triangle gives
    hand-derived values.
    """
    normalized = normalize(np.array([[3.0, 4.0], [1.0, 1.0]]))

    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


@pytest.mark.parametrize(
    "features",
    [np.array([1.0, 1j, -2.0, 0.5 - 1j]), [1.0, 1j, -2.0, 0.5 - 1j]],
    ids=["complex_array", "complex_list"],
)
def test_normalize_preserves_the_encoded_state(features):
    """
    Amplitude encoding normalizes internally. Encoding the normalized features
    therefore prepares the same state as encoding the raw ones. This is the
    property that makes the magnitude unrecoverable from the circuit. Complex
    features keep their phases on both paths. A float conversion in normalize
    dropped them.
    """
    encoding = Encoding.amplitude(n_features=4)

    raw = QuantumFloat(2)
    encoding.apply(raw, features)

    scaled = QuantumFloat(2)
    encoding.apply(scaled, normalize(features))

    assert np.allclose(
        np.array(raw.qs.statevector("array"), dtype=complex),
        np.array(scaled.qs.statevector("array"), dtype=complex),
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda X: minmax_scale(X),
        lambda X: minmax_scale(np.ones((2, 2)), X),
        lambda X: binarize(X),
        lambda X: Encoding.angle(n_features=2, data=X),
        lambda X: Encoding.angle(n_features=2, data_range=(X[0], X[1])),
    ],
    ids=["minmax_train", "minmax_test", "binarize", "encoder_data", "encoder_data_range"],
)
def test_real_valued_preprocessing_rejects_complex_features(operation):
    """
    Scaling and thresholding order real values. A float conversion discarded
    the imaginary parts with no more than a warning.
    """
    with pytest.raises(ValueError, match="complex features"):
        operation(np.array([[1.0, 1j], [2.0, -1j]]))


def test_normalize_rejects_a_zero_norm_sample():
    """A sample of zero norm has no direction and no state to prepare."""
    with pytest.raises(ValueError, match="zero norm"):
        normalize(np.array([[1.0, 2.0], [0.0, 0.0]]))


@pytest.mark.parametrize(
    "features, threshold, expected",
    [
        ([0.8, -0.2], 0.0, [1, 0]),
        ([0.2, 0.7], 0.5, [0, 1]),
        # The threshold itself is not exceeded, and it maps to 0.
        ([0.5, 0.5001], 0.5, [0, 1]),
    ],
)
def test_binarize_splits_at_the_threshold(features, threshold, expected):
    """Features above the threshold become 1, the rest become 0."""
    assert np.array_equal(binarize(np.array([features]), threshold).ravel(), expected)


def test_minmax_scale_rejects_a_single_sample():
    """
    normalize and binarize both document that a single sample shaped
    (n_features,) is accepted. minmax_scale derives its range along axis 0,
    which means a 1-D input is read as N samples of one feature instead, the
    opposite convention, silently producing plausible-looking output.
    """
    with pytest.raises(ValueError, match="two-dimensional"):
        minmax_scale([1.0, 2.0, 3.0])


def test_minmax_scale_with_a_known_range_matches_the_derived_one():
    """
    A saved model scales new data with the training range given as data_min and
    data_max. That has to reproduce the scaling derived from the training data,
    including the clipping of test values outside the range.
    """
    X_train = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    X_test = np.array([[0.5, 15.0], [5.0, 0.0]])

    derived = minmax_scale(X_train, X_test)
    known = minmax_scale(X_train, X_test, data_min=X_train.min(axis=0), data_max=X_train.max(axis=0))

    for a, b in zip(derived, known):
        assert np.allclose(a, b)


@pytest.mark.parametrize(
    "x, data_min, data_max, expected",
    [
        # A single sample, which is the usual case after loading a model.
        ([0.5, 15.0], [0.0, 10.0], [2.0, 30.0], [np.pi / 4, np.pi / 4]),
        # Scalar bounds apply to every feature.
        ([[1.0, 1.0]], 0.0, 2.0, [[np.pi / 2, np.pi / 2]]),
        # Equal bounds describe a constant feature, which maps to the lower bound.
        ([[5.0, 3.0]], [5.0, 0.0], [5.0, 6.0], [[0.0, np.pi / 2]]),
    ],
    ids=["single_sample", "scalar_bounds", "equal_bounds"],
)
def test_minmax_scale_with_a_known_range(x, data_min, data_max, expected):
    """The given bounds are used as they are, without looking at the data."""
    assert np.allclose(minmax_scale(np.array(x), data_min=data_min, data_max=data_max), expected)


@pytest.mark.parametrize(
    "bounds, message",
    [
        ({"data_min": [0.0, 10.0]}, "only one of data_min and data_max"),
        ({"data_max": [2.0, 30.0]}, "only one of data_min and data_max"),
        ({"data_min": [0.0, np.nan], "data_max": [2.0, 30.0]}, "not finite"),
        ({"data_min": [2.0, 10.0], "data_max": [0.0, 30.0]}, "data_max is smaller than data_min"),
        ({"data_min": [0.0], "data_max": [2.0]}, "data range for 1 features"),
        ({"data_min": [0.0, 10.0, 0.0], "data_max": [2.0, 30.0, 1.0]}, "data range for 3 features"),
    ],
    ids=["only_min", "only_max", "not_finite", "reversed", "one_value", "too_many_values"],
)
def test_minmax_scale_rejects_an_unusable_data_range(bounds, message):
    """
    A range that cannot scale the data is rejected. A single value shaped (1,)
    would otherwise broadcast onto every feature without an error.
    """
    with pytest.raises(ValueError, match=message):
        minmax_scale(np.array([[0.0, 10.0], [2.0, 30.0]]), **bounds)


def test_normalize_handles_a_vector_whose_norm_underflows():
    """
    A vector of tiny components has a well defined direction, but its norm
    underflows to exactly zero in double precision. Rescaling before
    taking the norm recovers the direction instead of reporting a zero-norm sample.
    """
    normalized = normalize(np.array([1e-300, 1e-300]))

    assert np.allclose(normalized, [2**-0.5, 2**-0.5])


@pytest.mark.parametrize(
    "function, X",
    [
        (minmax_scale, np.array([[1.0, np.nan], [3.0, 4.0]])),
        (normalize, np.array([[1.0, np.nan]])),
        (binarize, np.array([[1.0, np.nan]])),
    ],
)
def test_preprocessing_rejects_non_finite_values(function, X):
    """
    A NaN propagates through the scalers into every encoded angle, and binarize
    maps it to 0 because a NaN comparison is False. Neither problem is
    recoverable downstream, which is why the input is rejected where it can
    still be diagnosed.
    """
    with pytest.raises(ValueError, match="finite"):
        function(X)
