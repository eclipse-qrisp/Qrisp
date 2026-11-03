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

"""Provides minmax_scale, normalize and binarize, which prepare features for the encodings."""

import numpy as np

__all__ = ["binarize", "minmax_scale", "normalize"]


def _check_finite(X, name):
    """
    Rejects features that are not finite.

    A NaN spreads through the scalers into every encoded angle, and because
    every comparison against it is False, it would be encoded as a zero bit
    instead of being reported. Neither problem can be undone once the features
    reach a circuit, which is why they are rejected here, where the bad value
    can still be seen.
    """
    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"Tried to {name} features that are not all finite, meaning a feature holds a NaN or an infinity."
        )


def _as_real(X, name):
    """
    Converts features to floats, rejecting complex ones.

    A float conversion drops imaginary parts with only a warning, after which a
    complex feature would continue as its real part alone.
    """
    X = np.asarray(X)

    if np.iscomplexobj(X):
        raise ValueError(
            f"Tried to {name} complex features. Only amplitude encoding accepts them. Prepare them with normalize."
        )

    return X.astype(float)


def minmax_scale(  # noqa: PLR0913
    X_train, X_test=None, feature_range=(0.0, np.pi), periodic=True, *, data_min=None, data_max=None
):
    r"""
    Scales features into an interval suitable for an encoding.

    Each feature is mapped linearly from its range on the training data onto
    ``feature_range``

    .. math::

        x' = a + (b - a) \frac{x - \min(x)}{\max(x) - \min(x)}

    Information about the test data must not reach the model during training,
    because it would make the measured accuracy look better than the model
    really is. The minimum and maximum therefore come from ``X_train`` alone
    and are applied unchanged to ``X_test``.

    Since the test data is scaled with the training range, a test feature
    outside that range would leave ``feature_range``. Such values are clipped
    to its bounds, which keeps every encoded angle inside the interval that
    the encoding expects.

    **Reusing the range after saving a model**

    A model trained on scaled features needs the same scaling for every new
    sample, also after :func:`load_model`. Pass ``data_min`` and ``data_max``
    to scale with a known range instead of deriving it. Compute them once from
    the training data, save them as parts of the model, and pass the loaded
    values to scale new data. The training data itself is then no longer
    needed. With a known range, a single sample shaped ``(n_features,)`` can
    be scaled too. An encoder configured with ``data`` stores the range itself
    and is an alternative to saving it separately.

    **Why the interval matters**

    :meth:`Encoding.angle` and :meth:`Encoding.dense_angle` read features as
    rotation angles. Rotations are $2\pi$-periodic, and features spanning more
    than $2\pi$ wrap around. The values $x$ and $x + 2\pi$ prepare the same
    state, which the model cannot separate. The default range $[0, \pi]$ stays
    well inside one period and, for a $Y$ rotation, spans $\ket{0}$ to
    $\ket{1}$.

    A range of exactly $2\pi$ is a full turn, and its two endpoints encode to
    the same state. The smallest and largest values of a feature then look the
    same. Only the endpoints collide, and the range is accepted for callers who
    want the full period. The default $[0,\pi]$ avoids the collision
    entirely.

    **Amplitude encoding**

    :meth:`Encoding.amplitude` normalizes the features rather than reading
    them as angles, which means no period limits the width of the range. Pass
    ``periodic=False`` to lift the limit. Only the ratio of the bounds then
    reaches the state. A range starting at $0$ maps the sample at the minimum
    of every feature to the zero vector, which has no direction to encode, and
    the lower bound should therefore be positive. The range ``(1, 10)`` matches
    the default scaling of :meth:`Encoding.amplitude`.

    A feature that is constant across the training data has a zero range and
    carries no information. Dividing by that range would fail, and the feature
    is mapped to the lower bound of ``feature_range`` instead.

    Parameters
    ----------
    X_train : numpy.ndarray
        The training features, shaped ``(N, n_features)``. The scaling is
        derived from this array. If ``data_min`` and ``data_max`` are given,
        this is simply the data to scale, and a single sample shaped
        ``(n_features,)`` is accepted as well.
    X_test : numpy.ndarray, optional
        Further features to scale with the training range, shaped
        ``(M, n_features)``. The default is ``None``, which returns only the
        scaled training features.
    feature_range : tuple[float, float], optional
        The interval to scale into. The default is ``(0, numpy.pi)``. A range
        longer than $2\pi$ raises a ``ValueError`` unless ``periodic`` is
        ``False``, because the rotation based encodings cannot tell its
        endpoints apart.
    periodic : bool, optional
        Whether the features will be read as rotation angles, which limits
        ``feature_range`` to a width of $2\pi$. The default is ``True``. Pass
        ``False`` for amplitude encoding.
    data_min : float or numpy.ndarray, optional
        The smallest value of each feature in the training data, as a scalar
        or one value per feature. Must be given together with ``data_max``.
        The default is ``None``, which derives the range from ``X_train``.
    data_max : float or numpy.ndarray, optional
        The largest value of each feature in the training data, as a scalar
        or one value per feature. Must be given together with ``data_min``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        The scaled training features, and the scaled test features if
        ``X_test`` was given.

    Raises
    ------
    ValueError
        If ``feature_range`` is empty or not finite, or spans more than $2\pi$
        while ``periodic`` is ``True``.
    ValueError
        If ``X_test`` does not have the same number of features as
        ``X_train``.
    ValueError
        If only one of ``data_min`` and ``data_max`` is given, if they are not
        finite, if ``data_max`` is smaller than ``data_min``, or if they do not
        hold one value per feature.

    Examples
    --------
    We scale a training and a test set together.

    ::

        import numpy as np
        from qrisp.qml import minmax_scale

        X_train = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        X_test = np.array([[0.5, 15.0]])

        X_train_scaled, X_test_scaled = minmax_scale(X_train, X_test)

    The training features now span the full interval.

    >>> np.round(X_train_scaled, 4)
    array([[0.    , 0.    ],
           [1.5708, 1.5708],
           [3.1416, 3.1416]])

    The test features are placed on the same scale rather than on their own.

    >>> np.round(X_test_scaled, 4)
    array([[0.7854, 0.7854]])

    The scaled features are ready to be encoded.

    ::

        from qrisp import QuantumFloat
        from qrisp.qml import Encoding

        qf = QuantumFloat(2)
        Encoding.angle(n_features=2).apply(qf, X_train_scaled[0])

    For a model that will be saved, we keep the training range and scale with
    it. After loading, the saved values scale a new sample in the same way.

    ::

        from qrisp.qml import load_model, save_model

        data_min, data_max = X_train.min(axis=0), X_train.max(axis=0)
        save_model("model.pkl", data_min=data_min, data_max=data_max)

        saved = load_model("model.pkl")
        x_new = minmax_scale(
            np.array([0.5, 15.0]), data_min=saved["data_min"], data_max=saved["data_max"]
        )

    The new sample lands where the test sample above did.

    >>> np.round(x_new, 4)
    array([0.7854, 0.7854])

    """
    low, high = feature_range

    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"Tried to scale into the feature range ({low}, {high}), which is not finite.")

    if not high > low:
        raise ValueError(f"Tried to scale into an empty feature range ({low}, {high}).")

    if periodic and high - low > 2 * np.pi:
        raise ValueError(
            f"Tried to scale into the feature range ({low}, {high}), which is "
            f"longer than 2*pi. Rotation angles are 2*pi-periodic. Its "
            "endpoints would encode to the same state. Pass periodic=False if "
            "the features are amplitude encoded."
        )

    if (data_min is None) != (data_max is None):
        raise ValueError(
            "Tried to scale with only one of data_min and data_max. A known range "
            "needs both bounds. Pass neither to derive the range from X_train."
        )

    X_train = _as_real(X_train, "scale")
    _check_finite(X_train, "scale")

    if data_min is None:
        if X_train.ndim != 2:  # noqa: PLR2004
            raise ValueError(
                f"Tried to scale features shaped {X_train.shape}. minmax_scale "
                "derives its range across samples. It requires a two-dimensional "
                "array shaped (n_samples, n_features). A single sample has no range "
                "to scale onto the feature range. Pass data_min and data_max to "
                "scale it with a known range."
            )

        minimum = X_train.min(axis=0)
        maximum = X_train.max(axis=0)
    else:
        minimum = _as_real(data_min, "scale with")
        maximum = _as_real(data_max, "scale with")

        if not (np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum))):
            raise ValueError(f"Tried to scale with the data range ({minimum}, {maximum}), which is not finite.")

        if not np.all(maximum >= minimum):
            raise ValueError(
                f"Tried to scale with the data range ({minimum}, {maximum}), where data_max is smaller than data_min."
            )

        # A bound shaped (1,) would broadcast silently onto every feature, and
        # a bound with any other wrong count would fail only once data arrives.
        for bound in (minimum, maximum):
            if bound.shape not in ((), (X_train.shape[-1],)):
                raise ValueError(
                    f"Tried to scale {X_train.shape[-1]} features with a data range for {bound.size} features."
                )

    n_features = X_train.shape[-1]
    spread = maximum - minimum

    # A constant feature spans nothing to scale onto the range, and dividing
    # by its zero spread would fail. It is mapped to the lower bound instead.
    constant = spread == 0
    spread = np.where(constant, 1.0, spread)

    def scale(X):
        X = _as_real(X, "scale")
        _check_finite(X, "scale")

        if X.shape[-1] != n_features:
            raise ValueError(f"Tried to scale {X.shape[-1]} features with a range for {n_features} features.")

        scaled = low + (high - low) * (X - minimum) / spread
        scaled = np.where(constant, low, scaled)

        # Test features outside the training range would land outside
        # feature_range, where the encodings do not behave as documented.
        return np.clip(scaled, low, high)

    if X_test is None:
        return scale(X_train)

    return scale(X_train), scale(X_test)


def normalize(X):
    r"""
    Scales each sample to unit norm for amplitude encoding.

    Every row is divided by its Euclidean norm, and complex features keep their
    phases

    .. math::

        x' = \frac{x}{\lVert x \rVert}

    A quantum state has unit norm, and :meth:`Encoding.amplitude`, which writes
    the features into the amplitudes of a register, therefore normalizes
    internally. The length of a feature vector never reaches the circuit, only
    its direction, and two samples differing by a scalar factor prepare the
    same state.

    Applying this function first makes that visible in the data instead of
    leaving it to happen silently inside the encoding. Where the magnitude
    carries information, keep it outside the amplitudes or append a constant
    feature before encoding, as described for :meth:`Encoding.amplitude`.
    Appending the norm itself does not keep the magnitude.

    Parameters
    ----------
    X : numpy.ndarray
        The features, shaped ``(N, n_features)``. A single sample shaped
        ``(n_features,)`` is also accepted.

    Returns
    -------
    numpy.ndarray
        The features, each sample scaled to unit norm.

    Raises
    ------
    ValueError
        If any sample has zero norm, which leaves it no direction to
        encode.

    Examples
    --------
    ::

        import numpy as np
        from qrisp.qml import normalize

        X = np.array([[3.0, 4.0], [1.0, 0.0]])
        X_normalized = normalize(X)

    The first sample had norm $5$ and is divided by it.

    >>> np.round(X_normalized, 4)
    array([[0.6, 0.8],
           [1. , 0. ]])

    Every sample now has unit norm, similar to the encoded state.

    >>> np.round(np.linalg.norm(X_normalized, axis=1), 4)
    array([1., 1.])

    """
    X = np.asarray(X)
    X = X.astype(np.promote_types(X.dtype, np.float64))
    _check_finite(X, "normalize")

    # The norm of a sample of very small components can underflow to zero,
    # which would make it look as if it had no direction. Dividing by the
    # largest component first prevents this.
    largest = np.max(np.abs(X), axis=-1, keepdims=True)

    if np.any(largest == 0):
        raise ValueError("Tried to normalize a sample of zero norm, which has no direction to encode into amplitudes.")

    rescaled = X / largest
    norms = np.linalg.norm(rescaled, axis=-1, keepdims=True)

    return rescaled / norms


def binarize(X, threshold=0.0):
    r"""
    Turns features into the bit values basis encoding requires.

    Each feature above ``threshold`` becomes ``1``, and each feature at or
    below the ``threshold`` becomes ``0``

    .. math::

        x' = \begin{cases} 1 & x > t \\ 0 & x \leq t \end{cases}

    :meth:`Encoding.basis` maps a bit vector onto a computational basis state
    and accepts only the values 0 and 1. Continuous features therefore have to
    be reduced to bits before basis encoding.

    This keeps only the side of the threshold that a feature falls on and
    discards everything else. It suits data that is already close to binary,
    while it loses most of the information in other data.

    The default threshold of $0$ splits data centred on zero at its centre.
    Features scaled with :func:`minmax_scale` lie in its ``feature_range``,
    whose lower bound is $0$ by default, and this threshold then turns every
    feature above its training minimum into ``1``. Split such features at the
    middle of the range instead, which is ``threshold=numpy.pi / 2`` for the
    default range.

    Parameters
    ----------
    X : numpy.ndarray
        The features, shaped ``(N, n_features)``. A single sample shaped
        ``(n_features,)`` is also accepted.
    threshold : float, optional
        The value a feature has to exceed to become ``1``. The default is
        ``0.0``.

    Returns
    -------
    numpy.ndarray
        The features as integers, holding only the values 0 and 1.

    Examples
    --------
    ::

        import numpy as np
        from qrisp.qml import binarize

        X = np.array([[0.8, -0.2], [-1.0, 2.0]])
        bits = binarize(X)

    >>> bits
    array([[1, 0],
           [0, 1]])

    The result is directly encodable.

    ::

        from qrisp import QuantumFloat
        from qrisp.qml import Encoding

        qf = QuantumFloat(2)
        Encoding.basis(n_features=2).apply(qf, bits[0])

    A threshold splits continuous features at a chosen value instead.

    >>> binarize(np.array([[0.2, 0.7]]), threshold=0.5)
    array([[0, 1]])

    """
    X = _as_real(X, "binarize")
    _check_finite(X, "binarize")

    return (X > threshold).astype(int)
