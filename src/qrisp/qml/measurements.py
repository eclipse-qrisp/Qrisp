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

"""Defines expectation and parameter_shift, which read out expectation values and their derivatives."""

import jax.numpy as jnp

__all__ = ["expectation", "parameter_shift"]


def expectation(state_prep, observables, precision=0.01):
    r"""
    Estimates the expectation values of one or more observables.

    A single observable gives a scalar, and a sequence of observables gives one
    entry per observable, each estimated from its own sampling run. A model
    with several outputs is read out this way, while an operator such as
    ``Z(0) + Z(1)`` gives a single number that the contributions of the qubits
    cannot be recovered from. The cost grows with the number of observables.

    Parameters
    ----------
    state_prep : callable
        A function returning a QuantumVariable. The expectation values are
        estimated for the state that this function prepares. It can only take
        classical values as arguments, because a quantum value would have to be
        copied for each sampling iteration.
    observables : QubitOperator or iterable[QubitOperator]
        The observable, or a sequence of observables to estimate separately.
        Any iterable is accepted, including a numpy array of operators.
    precision : float, optional
        How precisely each expectation value is estimated. The number of shots
        grows with the square of the inverse precision. The default is
        ``0.01``.

    Returns
    -------
    callable
        A function that takes the arguments of ``state_prep`` and returns the
        expectation value, or an array with one expectation value per
        observable.

    Raises
    ------
    ValueError
        If ``observables`` is an empty sequence.

    Examples
    --------
    We estimate a single observable for an encoded feature vector.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.operators import Z
        from qrisp.qml import Encoding, expectation

        encoding = Encoding.angle(n_features=2)

        def state_prep(x):
            qf = QuantumFloat(2)
            encoding.apply(qf, x)
            return qf

        ev = expectation(state_prep, Z(0))

    The features are all zero, which leaves the register in $\ket{00}$. The
    first qubit is measured as $+1$.

    >>> ev([0, 0])
    1.0

    A sequence returns one entry per observable, which is how a model with one
    output per class is read out.

    ::

        ev = expectation(state_prep, [Z(0), Z(1)])

    >>> ev([np.pi, 0])
    array([-1.,  1.])

    **Hybrid models**

    The returned function takes and returns Jax arrays, which lets it compose
    with classical layers written in ``jnp`` on either side of it. The circuit
    of a hybrid model takes the trainable angles of an ansatz as a second
    argument.

    ::

        import jax.numpy as jnp
        from qrisp.qml import Ansatz

        ansatz = Ansatz.real_amplitudes(num_qubits=2, layers=1)

        def circuit(x, theta):
            qf = QuantumFloat(2)
            encoding.apply(qf, x)
            ansatz.apply(qf, theta)
            return qf

        quantum_block = expectation(circuit, [Z(0), Z(1)])

        def model(x, p):
            h = jnp.tanh(x * p["w_in"])                     # classical
            q = quantum_block(h, p["theta"])                # quantum
            return jnp.tanh(jnp.dot(q, p["w_out"]))         # classical

    Such a model trains with the gradient-free optimizers of
    :func:`qrisp.jasp.minimize`, or with gradients from
    :func:`parameter_shift`.

    """
    # A single operator is identified by what it provides rather than by what
    # it is. Any iterable of observables is then read as a sequence.
    if hasattr(observables, "expectation_value"):
        return observables.expectation_value(state_prep, precision=precision)

    observables = list(observables)

    if not observables:
        raise ValueError("Tried to estimate the expectation values of an empty sequence of observables.")

    ev_functions = [observable.expectation_value(state_prep, precision=precision) for observable in observables]

    def expectation_eval_function(*args):
        return jnp.array([ev_function(*args) for ev_function in ev_functions])

    return expectation_eval_function


def parameter_shift(state_prep, observables, precision=0.01, argnums=0):
    r"""
    Estimates expectation values together with their derivatives, using the parameter shift rule.

    Let an argument $\theta_k$ enter the circuit as the angle of a single
    rotation. Every expectation value is then a function of the form
    $\alpha + \beta \cos\theta_k + \gamma \sin\theta_k$ in that argument, and
    two further evaluations give its derivative exactly

    .. math::

        \frac{\partial \braket{O}}{\partial \theta_k}
        = \frac{\braket{O}(\theta_k + \pi/2) - \braket{O}(\theta_k - \pi/2)}{2}

    A vector argument with $n$ entries costs $2n$ evaluations on top of the
    one at the unshifted point.

    The rule holds when every entry of a differentiated argument is the angle
    of exactly one $R_X$, $R_Y$, $R_Z$ or phase gate, reaches that gate
    unchanged, and appears nowhere else in the circuit. The parameters of the
    built-in ansatzes meet this condition, as do the features of
    :meth:`Encoding.angle` and :meth:`Encoding.dense_angle` without ``data``
    or ``data_range``. Scaled, amplitude encoded, basis encoded and re-uploaded
    features do not, and differentiating them returns wrong numbers without an
    error. To differentiate scaled features, scale them beforehand with
    :func:`minmax_scale` and configure the encoder without a scaling.

    Parameters
    ----------
    state_prep : callable
        A function returning a QuantumVariable, as for :func:`expectation`.
    observables : QubitOperator or iterable[QubitOperator]
        The observable, or a sequence of observables to estimate separately.
    precision : float, optional
        How precisely each expectation value is estimated, shifted or not. The
        default is ``0.01``.
    argnums : int or tuple[int], optional
        The positions of the arguments of ``state_prep`` to differentiate. Each
        of them must be one-dimensional. The default is ``0``.

    Returns
    -------
    callable
        A function that takes the arguments of ``state_prep`` and returns a
        pair. The first entry holds the expectation values, as
        :func:`expectation` returns them. The second entry holds the
        derivatives, which for an argument with $n$ entries is an array with
        $n$ rows, row $k$ holding the derivatives of the expectation values
        with respect to entry $k$. If ``argnums`` is a tuple, the second entry
        is a tuple with one such array per position. The parameters are in rows
        and the observables in columns, which is the transpose of the usual
        Jacobian. For a loss $L$ of the expectation values $f$, the gradient
        with respect to the parameters is therefore ``D @ g``, where ``D`` is
        the array of derivatives and $g_o = \partial L / \partial f_o$.

    Raises
    ------
    ValueError
        If a differentiated argument is not one-dimensional.

    Examples
    --------
    One feature and one parameter, each the angle of an $R_Y$ rotation,
    together give $\braket{Z} = \cos(x_0 + w_0)$.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.operators import Z
        from qrisp.qml import Ansatz, Encoding, parameter_shift

        encoding = Encoding.angle(n_features=1)
        ansatz = Ansatz.real_amplitudes(num_qubits=1, layers=0)

        def state_prep(x, w):
            qf = QuantumFloat(1)
            encoding.apply(qf, x)
            ansatz.apply(qf, w)
            return qf

        value_and_derivatives = parameter_shift(state_prep, Z(0), argnums=(0, 1))

    At $x_0 + w_0 = \pi/2$ both derivatives are $-\sin(\pi/2) = -1$. The two
    shifted states are $\ket{1}$ and $\ket{0}$, which means these derivatives
    carry no sampling noise.

    >>> value, (dx, dw) = value_and_derivatives(np.array([np.pi / 2]), np.array([0.0]))
    >>> float(dx[0]), float(dw[0])
    (-1.0, -1.0)

    **Training with gradients**

    Every separate call of a jaspified function holds on to memory, and one
    call per shifted evaluation exhausts the memory of a long training run.
    Call the returned function inside a single ``@jaspify`` function instead,
    which traces all its evaluations together.

    ::

        from qrisp.jasp import jaspify

        @jaspify(terminal_sampling=True)
        def circuit_with_derivatives(x, w):
            return value_and_derivatives(x, w)

    References
    ----------
    .. [1] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, "Quantum circuit
       learning", Phys. Rev. A 98, 032309 (2018). arXiv:1803.00745
    .. [2] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran,
       "Evaluating analytic gradients on quantum hardware", Phys. Rev. A 99,
       032331 (2019). arXiv:1811.11184

    """
    estimate = expectation(state_prep, observables, precision=precision)
    single = isinstance(argnums, int)
    positions = (argnums,) if single else tuple(argnums)

    def value_and_derivatives(*args):
        values = estimate(*args)
        derivatives = []

        for position in positions:
            argument = jnp.asarray(args[position])

            if jnp.ndim(argument) != 1:
                raise ValueError(
                    f"Tried to differentiate argument {position} of shape {jnp.shape(argument)}. "
                    "The parameter shift rule takes one-dimensional arguments, one angle per entry."
                )

            # A shift in place keeps the dtype of the argument, which means an
            # integer argument would truncate the shift of pi/2 to 1. Promoting
            # to float64 gives the dtype that adding a float64 shift produced.
            argument = argument.astype(jnp.promote_types(argument.dtype, jnp.float64))
            shifted = list(args)
            rows = []

            for k in range(jnp.shape(argument)[0]):
                shifted[position] = argument.at[k].add(jnp.pi / 2)
                plus = estimate(*shifted)
                shifted[position] = argument.at[k].add(-jnp.pi / 2)
                minus = estimate(*shifted)
                rows.append((plus - minus) / 2)

            derivatives.append(jnp.stack(rows))

        return values, derivatives[0] if single else tuple(derivatives)

    return value_and_derivatives
