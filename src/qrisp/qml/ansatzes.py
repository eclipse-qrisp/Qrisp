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

"""Defines the Ansatz class, the parameterized circuits that a model trains."""

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class, tree_leaves

from qrisp.core.gate_application_functions import cx, cz, rx, ry, rz
from qrisp.jasp import check_for_tracing_mode, jrange

__all__ = [
    "Ansatz",
    "hardware_efficient_size",
    "real_amplitudes_size",
]

_ROTATION_GATES = {"x": rx, "y": ry, "z": rz}

_ENTANGLING_GATES = {"cx": cx, "cz": cz}

_ENTANGLEMENTS = ("linear", "ring")


def _hardware_efficient(qv, w, layers=1, rotations=("y",), entangler="cx"):
    r"""
    Applies a hardware efficient ansatz to a QuantumVariable.

    The ansatz repeats a parameterized rotation layer and an entangling layer
    ``layers`` times, and a final rotation layer closes it

    .. math::

        R(w_L)\, U_{\text{ent}} \cdots R(w_1)\, U_{\text{ent}}\, R(w_0)

    The operators act from right to left, with $R(w_0)$ applied first.

    Each rotation layer applies one rotation per qubit for every axis in
    ``rotations``, which means a register of $n$ qubits uses
    $(L+1) \cdot n \cdot \lvert\text{rotations}\rvert$ parameters. The final
    rotation layer makes sure that the output of the last entangling layer is
    rotated before it is measured.

    The entangling layer applies a two-qubit gate to each neighbouring pair of
    qubits in a linear chain, which keeps the ansatz shallow and runs on
    hardware with nearest-neighbour connectivity.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to apply the ansatz to.
    w : list[float] or numpy.ndarray
        The parameters. Either a flat vector of
        $(L+1) \cdot n \cdot \lvert\text{rotations}\rvert$ entries or an array
        shaped ``(layers + 1, len(rotations), n)``. Both are read in the same
        order, and a vector from an optimizer can be passed unchanged.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.
    rotations : tuple[str], optional
        The rotation axes applied to each qubit in a rotation layer. The
        default is ``("y",)``, which keeps the amplitudes real.
    entangler : {'cx', 'cz'}, optional
        The two-qubit gate of the entangling layer. The default is ``'cx'``.

    Raises
    ------
    ValueError
        If the number of parameters does not match the ansatz.
    ValueError
        If ``rotations`` holds an axis other than ``'x'``, ``'y'`` or ``'z'``.
    ValueError
        If ``entangler`` is not ``'cx'`` or ``'cz'``.

    References
    ----------
    .. [1] A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink,
       J. M. Chow, and J. M. Gambetta, "Hardware-efficient variational
       quantum eigensolver for small molecules and quantum magnets",
       Nature 549, 242-246 (2017). arXiv:1704.05018

    """
    for axis in rotations:
        if axis not in _ROTATION_GATES:
            raise ValueError(
                f"Unknown rotation axis '{axis}' for the hardware efficient ansatz. "
                f"Supported axes are {sorted(_ROTATION_GATES)}."
            )

    if entangler not in _ENTANGLING_GATES:
        raise ValueError(
            f"Unknown entangler '{entangler}' for the hardware efficient ansatz. "
            f"Supported entanglers are {sorted(_ENTANGLING_GATES)}."
        )

    if layers < 0:
        raise ValueError(
            f"Tried to apply a hardware efficient ansatz with a negative layer "
            f"count ({layers}). A negative count expects no parameters and would "
            "leave the QuantumVariable untouched, which the parameter count check "
            "does not catch."
        )

    num_rotation_layers = layers + 1

    if not check_for_tracing_mode():
        expected = hardware_efficient_size(qv.size, layers, rotations)
        if np.size(w) != expected:
            raise ValueError(
                f"Tried to apply a hardware efficient ansatz with {np.size(w)} "
                f"parameters to a QuantumVariable of size {qv.size}. Expected "
                f"{expected} parameters for {layers} layers and rotations "
                f"{tuple(rotations)}."
            )

    w = _flatten_parameters(w)

    entangling_gate = _ENTANGLING_GATES[entangler]
    num_rotations = len(rotations)

    for layer in range(num_rotation_layers):
        for axis_index, axis in enumerate(rotations):
            rotation = _ROTATION_GATES[axis]
            # The parameters of one rotation layer are contiguous, and the axes
            # of a layer follow each other before the next layer begins.
            offset = (layer * num_rotations + axis_index) * qv.size
            for i in jrange(qv.size):
                rotation(w[offset + i], qv[i])

        if layer < layers:
            _entangling_layer(qv, entangling_gate, "linear")

    return qv


def _real_amplitudes(qv, w, layers=1, entanglement="ring"):
    r"""
    Applies an ansatz of $R_Y$ rotations and CX entanglers to a QuantumVariable.

    The ansatz repeats an $R_Y$ rotation layer and a CX entangling layer
    ``layers`` times, and a final rotation layer closes it

    .. math::

        R_Y(w_L)\, U_{\text{ent}} \cdots R_Y(w_1)\, U_{\text{ent}}\, R_Y(w_0)

    The operators act from right to left, with $R_Y(w_0)$ applied first. A
    register of $n$ qubits uses $(L+1) \cdot n$ parameters, one per qubit and
    rotation layer.

    $R_Y$ and CX both have real matrix representations, which means the ansatz
    maps real amplitudes to real amplitudes. Starting from
    $\ket{0}^{\otimes n}$, or after an encoding with real amplitudes such as
    angle encoding about the Y axis, it explores only the real subspace of the
    Hilbert space. That subspace contains a state for every value that an
    expectation of a real observable can take.

    Some encodings introduce complex phases, such as dense angle encoding or
    angle encoding about the X or Z axis. After them, the ansatz rotates the
    real and imaginary parts of the state separately and never mixes them. The
    expectation value of a real observable is then a sum of one contribution
    from each part, which restricts the functions of the features that the
    model can represent. An ansatz with rotations about further axes lifts this
    restriction.

    With ``entanglement="linear"`` this ansatz is the same circuit as
    :meth:`Ansatz.hardware_efficient` with ``rotations=("y",)``. The default
    ``entanglement="ring"`` adds a closing entangler between the last and the
    first qubit on three or more qubits, which the hardware efficient ansatz
    does not offer. This ansatz is kept under its established name for
    familiarity.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to apply the ansatz to.
    w : list[float] or numpy.ndarray
        The parameters. Either a flat vector of $(L+1) \cdot n$ entries or an
        array shaped ``(layers + 1, n)``. Both are read in the same order, and a
        vector from an optimizer can be passed unchanged.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.
    entanglement : {'ring', 'linear'}, optional
        The connectivity of the entangling layer. ``'linear'`` applies a CX to
        each neighbouring pair of qubits, while ``'ring'`` also closes the
        chain between the last and the first qubit on three or more qubits. The
        default is ``'ring'``.

    Raises
    ------
    ValueError
        If the number of parameters does not match the ansatz.
    ValueError
        If ``entanglement`` is not ``'ring'`` or ``'linear'``.

    References
    ----------
    .. [1] A real-valued special case of the hardware-efficient ansatz,
       restricted to $R_Y$ rotations and CX entanglers. A. Kandala,
       A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, and
       J. M. Gambetta, "Hardware-efficient variational quantum eigensolver
       for small molecules and quantum magnets", Nature 549, 242-246 (2017).
       arXiv:1704.05018

    """
    if entanglement not in _ENTANGLEMENTS:
        raise ValueError(
            f"Unknown entanglement '{entanglement}' for the real amplitudes "
            f"ansatz. Supported entanglements are {sorted(_ENTANGLEMENTS)}."
        )

    if layers < 0:
        raise ValueError(
            f"Tried to apply a real amplitudes ansatz with a negative layer "
            f"count ({layers}). A negative count expects no parameters and would "
            "leave the QuantumVariable untouched, which the parameter count check "
            "does not catch."
        )

    if not check_for_tracing_mode():
        expected = real_amplitudes_size(qv.size, layers)
        if np.size(w) != expected:
            raise ValueError(
                f"Tried to apply a real amplitudes ansatz with {np.size(w)} "
                f"parameters to a QuantumVariable of size {qv.size}. Expected "
                f"{expected} parameters for {layers} layers."
            )

    w = _flatten_parameters(w)

    for layer in range(layers + 1):
        offset = layer * qv.size
        for i in jrange(qv.size):
            ry(w[offset + i], qv[i])

        if layer < layers:
            _entangling_layer(qv, cx, entanglement)

    return qv


def real_amplitudes_size(num_qubits, layers=1):
    """
    Returns the number of parameters of a :meth:`Ansatz.real_amplitudes` ansatz.

    Parameters
    ----------
    num_qubits : int
        The size of the QuantumVariable the ansatz is applied to.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.

    Returns
    -------
    int
        The number of parameters.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.qml import real_amplitudes_size

        w = np.random.uniform(0, 2 * np.pi, real_amplitudes_size(3, layers=2))

    """
    return (layers + 1) * num_qubits


def _data_reuploading(qv, x, w, ansatz_size, encoding, ansatz, layers=2):
    r"""
    Applies alternating encoding and ansatz layers to a QuantumVariable.

    The feature vector is encoded once per layer rather than only at the start
    of the circuit

    .. math::

        W(w_L)\, S(x) \cdots W(w_1)\, S(x)

    Here $S$ is the encoding and $W$ the parameterized ansatz, and the
    operators act from right to left.

    Re-encoding the same features between parameterized layers raises the
    frequency spectrum that the circuit can represent. A single encoding layer
    followed by an ansatz produces a function of the features whose Fourier
    spectrum is limited by the encoding alone, and no amount of added
    parameters widens it. With $L$ encoding layers the accessible spectrum
    grows with $L$, which lets a circuit on few qubits fit a rich function.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to apply the circuit to.
    x : list[float] or numpy.ndarray
        The feature vector, encoded once per layer.
    w : list[float] or numpy.ndarray
        The parameters of all ansatz layers, concatenated. A flat vector of
        ``layers * ansatz_size`` entries.
    ansatz_size : int
        The number of parameters that ``ansatz`` uses on ``qv``, as returned by
        the size function that accompanies the ansatz. It splits ``w`` into one
        block per layer.
    layers : int, optional
        The number of encoding and ansatz layer repetitions. The default is
        ``2``, the smallest number that re-uploads the features at all.
    encoding : Encoding
        Applied as ``encoding.apply(qv, x)`` at the start of each layer.
    ansatz : Ansatz
        Applied as ``ansatz.apply(qv, w_layer)`` after each encoding.

    Raises
    ------
    ValueError
        If the number of parameters does not match the layers.

    References
    ----------
    .. [1] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and
       J. I. Latorre, "Data re-uploading for a universal quantum
       classifier", Quantum 4, 226 (2020). arXiv:1907.02085

    """
    if np.size(w) != layers * ansatz_size:
        raise ValueError(
            f"Tried to apply data reuploading with {np.size(w)} parameters over "
            f"{layers} layers. Expected {layers * ansatz_size} parameters, "
            f"{ansatz_size} for each layer."
        )

    w = _flatten_parameters(w)

    for layer in range(layers):
        encoding.apply(qv, x)
        ansatz.apply(qv, w[layer * ansatz_size : (layer + 1) * ansatz_size])

    return qv


def _entangling_layer(qv, entangling_gate, entanglement):
    """
    Applies a two-qubit gate to each neighbouring pair of qubits.

    For ``"ring"`` an additional gate on the last and first qubit closes the
    chain. It is omitted on a register of two qubits, where it would duplicate
    the pair that the linear chain already covers.
    """
    for i in jrange(qv.size - 1):
        entangling_gate(qv[i], qv[i + 1])

    if entanglement == "ring":
        # The closing gate runs once on more than two qubits and not at all
        # otherwise, where it would repeat the pair that the chain already
        # covered. Since qv.size is a tracer, the count drives a jrange rather
        # than an if.
        for _ in jrange(jnp.minimum(jnp.maximum(qv.size - 2, 0), 1)):
            entangling_gate(qv[qv.size - 1], qv[0])


def _flatten_parameters(w):
    """
    Views the parameters as a flat vector.

    Accepts an array shaped ``(layers + 1, len(rotations), n)`` as well as a
    flat vector, which makes both an explicitly shaped array and a vector
    straight from an optimizer valid inputs.
    """
    if hasattr(w, "ravel"):
        return w.ravel()

    # A nested list describes the same shape as the equivalent array, and it
    # is flattened rather than kept as the two-dimensional array that asarray
    # builds.
    return jnp.asarray(w, dtype=float).ravel()


def hardware_efficient_size(num_qubits, layers=1, rotations=("y",)):
    """
    Returns the number of parameters of a :meth:`Ansatz.hardware_efficient` ansatz.

    Parameters
    ----------
    num_qubits : int
        The size of the QuantumVariable the ansatz is applied to.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.
    rotations : tuple[str], optional
        The rotation axes applied to each qubit in a rotation layer. The
        default is ``("y",)``.

    Returns
    -------
    int
        The number of parameters.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.qml import hardware_efficient_size

        w = np.random.uniform(0, 2 * np.pi, hardware_efficient_size(3, layers=2))

    Raises
    ------
    ValueError
        If an axis follows itself in ``rotations``.

    """
    for axis, next_axis in zip(rotations, rotations[1:]):
        if axis == next_axis:
            raise ValueError(
                f"Repeated rotation axis in {tuple(rotations)} for the hardware efficient ansatz. "
                f"Consecutive rotations about the same axis compose into a single one, "
                f"which means the repetition adds parameters the circuit cannot use. An axis may "
                f"recur once another axis separates the repeats, as in ('z', 'y', 'z')."
            )

    return (layers + 1) * len(rotations) * num_qubits


class Ansatz:
    r"""
    A configured parameterized circuit.

    An ansatz is built once for a given register size and applied once per
    evaluation. The classmethods :meth:`hardware_efficient`,
    :meth:`real_amplitudes` and :meth:`data_reuploading` each return a
    configured ansatz.

    ::

        from qrisp import QuantumFloat
        from qrisp.jasp import jaspify
        from qrisp.qml import Ansatz

        ansatz = Ansatz.real_amplitudes(num_qubits=3, layers=2)

        @jaspify
        def circuit(w):
            qf = QuantumFloat(3)
            ansatz.apply(qf, w)
            return qf

    **The qubit count is fixed at construction**

    An ansatz is configured for ``num_qubits`` qubits, which fixes the number of
    parameters that it expects, and checks each parameter vector against that
    number. Under :ref:`Jasp <jasp>` tracing the register size is a tracer
    rather than a number, which means the count cannot be checked against the
    register instead. Without this check a wrong number of parameters would be
    accepted silently, with a surplus ignored or a short vector read past its
    end by reusing its last value.

    **Register width is the caller's responsibility**

    As for :ref:`Encoding <Encoding>`, ``num_qubits`` is not checked against
    the register that the ansatz is applied to, because that register remains
    a tracer. An ansatz configured for three qubits and applied to a register
    of four is not reported, and keeping the two in step is left to the
    caller.

    **Parameters the readout cannot see**

    ``num_parameters`` counts gate angles, not directions the model can learn.
    A gate that commutes with the observable cannot change its expectation
    value. $R_Z$ rotations and CZ gates commute with a $Z$ readout, which means
    a final $R_Z$ layer before such a readout does not affect the result.
    Neither does a gate acting only on qubits that are not measured. The
    derivatives for such parameters are zero, although sampling noise can make
    them look small but nonzero.

    **Writing an ansatz**

    A subclass of this class gets the same parameter count check and error
    messages as the built-in ansatzes. Implement :meth:`apply_gates` to emit
    the circuit, and :meth:`parameter_count` to derive the number of
    parameters from the register size instead of restating it at every
    construction.

    ::

        from jax.tree_util import register_pytree_node_class
        from qrisp import ry
        from qrisp.jasp import jrange
        from qrisp.qml import Ansatz

        @register_pytree_node_class
        class RotationAnsatz(Ansatz):
            def parameter_count(self, num_qubits):
                return num_qubits

            def apply_gates(self, qv, w):
                for i in jrange(qv.size):
                    ry(w[i], qv[i])
                return qv

        ansatz = RotationAnsatz(num_qubits=3)

    ``num_parameters`` is then derived on construction, and ``apply`` rejects a
    parameter vector that does not match it. Registering the class as a PyTree
    lets it be passed as an argument to a traced function, rather than only
    captured from the enclosing scope. Return your own settings from
    ``_options`` to keep them across that roundtrip. Since the rebuilt ansatz
    does not pass through ``__init__``, a subclass may give ``__init__`` any
    signature.

    """

    def __init__(self, num_qubits, num_parameters=None):
        if not isinstance(num_qubits, (int, np.integer)) or isinstance(num_qubits, bool):
            raise ValueError(f"Tried to configure an ansatz for {num_qubits!r} qubits, which is not an integer.")

        if num_qubits < 1:
            raise ValueError(f"Tried to configure an ansatz for {num_qubits} qubits.")

        self.num_qubits = int(num_qubits)

        if num_parameters is None:
            num_parameters = self.parameter_count(self.num_qubits)

        self.num_parameters = int(num_parameters)

    @classmethod
    def hardware_efficient(cls, num_qubits, layers=1, rotations=("y",), entangler="cx"):
        r"""
        Returns an ansatz of rotation layers separated by entangling layers.

        Each rotation layer applies one parameterized rotation per qubit and
        rotation axis, and each entangling layer applies a two qubit gate to
        neighbouring qubits. There is one more rotation layer than entangling
        layers, which means the circuit both begins and ends with rotations.

        Parameters
        ----------
        num_qubits : int
            The size of the QuantumVariable the ansatz is applied to.
        layers : int, optional
            The number of rotation and entangling layer repetitions. The default
            is ``1``.
        rotations : tuple[str], optional
            The rotation axes applied to each qubit in a rotation layer, in
            order. The default is ``("y",)``. An axis may recur when another
            axis separates the repeats, as in ``("z", "y", "z")``, which
            parameterizes an arbitrary single qubit rotation up to a global
            phase.
        entangler : {'cx', 'cz'}, optional
            The two qubit gate of the entangling layers. The default is
            ``'cx'``.

        Returns
        -------
        Ansatz
            The configured ansatz.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Ansatz

            qf = QuantumFloat(3)
            ansatz = Ansatz.hardware_efficient(num_qubits=3, layers=2)
            w = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
            ansatz.apply(qf, w)

        """
        for axis in rotations:
            if axis not in _ROTATION_GATES:
                raise ValueError(
                    f"Unknown rotation axis '{axis}' for the hardware efficient ansatz. "
                    f"Supported axes are {sorted(_ROTATION_GATES)}."
                )

        if entangler not in _ENTANGLING_GATES:
            raise ValueError(
                f"Unknown entangler '{entangler}' for the hardware efficient ansatz. "
                f"Supported entanglers are {sorted(_ENTANGLING_GATES)}."
            )

        _check_layers(layers, "hardware efficient")

        ansatz = _HardwareEfficient(num_qubits, hardware_efficient_size(num_qubits, layers, rotations))
        ansatz.layers = layers
        ansatz.rotations = rotations
        ansatz.entangler = entangler

        return ansatz

    @classmethod
    def real_amplitudes(cls, num_qubits, layers=1, entanglement="ring"):
        r"""
        Returns an ansatz preparing states with real amplitudes.

        Each rotation layer applies one $R_Y$ rotation per qubit, and entangling
        layers of CX gates separate the rotation layers. Since both gates have
        real matrix entries, a state with real amplitudes, such as the initial
        state of a register, keeps real amplitudes.

        With a real observable, such as a product of $Z$ and $X$ operators, a
        real ansatz gives the same expectation value for a state and its
        complex conjugate. A phase encoded as $\phi$, for example by
        :meth:`Encoding.dense_angle`, then looks the same as $-\phi$. Where the
        sign of a phase carries information, use an ansatz with rotations about
        other axes, such as :meth:`hardware_efficient` with
        ``rotations=("x", "y")``, or an observable involving $Y$.

        Parameters
        ----------
        num_qubits : int
            The size of the QuantumVariable the ansatz is applied to.
        layers : int, optional
            The number of rotation and entangling layer repetitions. The default
            is ``1``.
        entanglement : {'ring', 'linear'}, optional
            The shape of the entangling layers. The default is ``'ring'``,
            which closes the chain by entangling the last qubit with the first
            on three or more qubits.

        Returns
        -------
        Ansatz
            The configured ansatz.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Ansatz

            qf = QuantumFloat(3)
            ansatz = Ansatz.real_amplitudes(num_qubits=3, layers=2)
            w = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
            ansatz.apply(qf, w)

        Every amplitude of the resulting state is real.

        >>> np.allclose(np.array(qf.qs.statevector("array")).imag, 0)
        True

        """
        if entanglement not in _ENTANGLEMENTS:
            raise ValueError(
                f"Unknown entanglement '{entanglement}' for the real amplitudes "
                f"ansatz. Supported entanglements are {sorted(_ENTANGLEMENTS)}."
            )

        _check_layers(layers, "real amplitudes")

        ansatz = _RealAmplitudes(num_qubits, real_amplitudes_size(num_qubits, layers))
        ansatz.layers = layers
        ansatz.entanglement = entanglement

        return ansatz

    @classmethod
    def data_reuploading(cls, num_qubits, encoding, ansatz, layers=2):
        r"""
        Returns an ansatz alternating encoding and ansatz layers.

        The feature vector is encoded once per layer rather than only at the
        start of the circuit

        .. math::

            W(w_L)\, S(x) \cdots W(w_1)\, S(x)

        Here $S$ is the encoding and $W$ the parameterized ansatz, and the
        operators act from right to left.

        Re-encoding the same features between parameterized layers raises the
        frequency spectrum that the circuit can represent. A single encoding
        layer followed by an ansatz produces a function of the features whose
        Fourier spectrum is limited by the encoding alone, and no amount of
        added parameters widens it. With $L$ encoding layers the accessible
        spectrum grows with $L$, which lets a circuit on few qubits fit a rich
        function.

        The growth requires encoding and ansatz layers that do not commute. A
        basis encoding applies $X$ gates, which commute with $R_X$ rotations,
        and with an ansatz of $R_X$ rotations alone the encodings cancel in
        pairs and the output does not depend on the features. With $R_Y$ in
        both encoding and ansatz on one qubit, the layers combine into one
        rotation by $Lx + \sum_j w_j$. The frequency rises, but the parameters
        enter only through their sum.

        Unlike the other ansatzes, this one also takes the features and is
        applied as ``apply(qv, w, x)``.

        Parameters
        ----------
        num_qubits : int
            The size of the QuantumVariable the ansatz is applied to.
        encoding : Encoding
            Applied at the start of each layer. It acts on the state the
            previous layers left. An encoding defined only on a register in
            $\ket{0}$, such as :meth:`Encoding.amplitude`, is rejected.
        ansatz : Ansatz
            Applied after each encoding.
        layers : int, optional
            The number of encoding and ansatz layer repetitions. The default is
            ``2``, the smallest number that re-uploads the features at all.

        Returns
        -------
        Ansatz
            The configured ansatz.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Ansatz, Encoding

            qf = QuantumFloat(2)
            reuploading = Ansatz.data_reuploading(
                num_qubits=2,
                encoding=Encoding.angle(n_features=2),
                ansatz=Ansatz.real_amplitudes(num_qubits=2),
                layers=3,
            )
            w = np.random.uniform(0, 2 * np.pi, reuploading.num_parameters)
            reuploading.apply(qf, w, [0.4, 1.2])

        """
        _check_layers(layers, "data reuploading")

        if getattr(encoding, "_from_zero_state", False):
            raise ValueError(
                "Tried to re-upload an encoding that prepares a state from |0>, such "
                "as Encoding.amplitude. Only the first layer finds the register in "
                "|0>, and the later layers would apply it to a state it is not "
                "defined on. Use an encoding built from rotations, such as "
                "Encoding.angle or Encoding.dense_angle."
            )

        # A reuploading ansatz takes the features as a third argument, while
        # the inner ansatz is applied as apply(qv, w_layer). Nesting one inside
        # another would raise a TypeError once applied, which is why it is
        # rejected here rather than at the first evaluation.
        if isinstance(ansatz, _DataReuploading):
            raise ValueError(
                "Tried to use a data reuploading ansatz as the inner ansatz of "
                "another one. The inner ansatz is applied without the features, "
                "which a reuploading ansatz requires. Pass an ansatz such as "
                "Ansatz.real_amplitudes. Raise layers to re-upload more often."
            )

        if ansatz.num_qubits != num_qubits:
            raise ValueError(
                f"Tried to configure a data reuploading ansatz for {num_qubits} qubits "
                f"around an inner ansatz configured for {ansatz.num_qubits}. Both act "
                "on the same register and must be configured for the same size."
            )

        reuploading = _DataReuploading(num_qubits, layers * ansatz.num_parameters)
        reuploading.layers = layers
        reuploading.encoding = encoding
        reuploading.ansatz = ansatz

        return reuploading

    def apply(self, qv, w):
        """
        Applies the ansatz to ``qv``.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to apply the ansatz to. Its size is not
            validated. See the note on register width in the class docstring.
        w : list[float] or numpy.ndarray
            The parameters, either flat or shaped.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the ansatz was applied to.

        Raises
        ------
        ValueError
            If ``w`` does not hold exactly ``num_parameters`` parameters.

        """
        self._check_parameters(w)

        return self.apply_gates(qv, w)

    def apply_gates(self, qv, w):
        """
        Emits the gates of the ansatz.

        Subclasses implement this to define their circuit. It is called by
        :meth:`apply`, which has already rejected a parameter vector of the
        wrong size, and an implementation can rely on ``w`` holding exactly
        ``num_parameters`` parameters.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to apply the gates to. Under tracing its size
            is a tracer, which can neither be compared against ``num_qubits``
            nor size an allocation. Loop over it with
            :ref:`jrange <jrange>` rather than ``range(len(qv))``.
        w : list[float] or numpy.ndarray
            The parameters, either flat or shaped.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the gates were applied to.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement apply_gates, which an ansatz needs to emit its gates."
        )

    def parameter_count(self, num_qubits):
        """
        Returns the number of parameters the ansatz expects.

        Subclasses override this to derive the count from the register size
        rather than have the caller restate it. It is called once, during
        construction, and the result is available as ``num_parameters``.

        Parameters
        ----------
        num_qubits : int
            The size the ansatz is configured for, as an ordinary integer.

        Returns
        -------
        int
            The number of parameters.

        """
        raise NotImplementedError(
            f"{type(self).__name__} neither implements parameter_count nor was "
            f"given a num_parameters. The number of parameters it expects is unknown."
        )

    def _check_parameters(self, w):
        """
        Rejects a parameter vector of the wrong size.

        np.size reads a static shape, and num_parameters was computed outside
        the traced region. Unlike a count derived from qv.size, this comparison
        therefore also runs while tracing.
        """
        if np.size(w) != self.num_parameters:
            raise ValueError(
                f"Tried to apply an ansatz configured for {self.num_qubits} qubits "
                f"with {np.size(w)} parameters. Expected {self.num_parameters}."
            )

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        The configuration is auxiliary data rather than a child, which keeps
        ``num_qubits`` and ``num_parameters`` the plain integers that the trace
        time check compares against.
        """
        return (), (self.num_qubits, self.num_parameters, self._options())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        PyTree unflatten for JAX.
        """
        num_qubits, num_parameters, options = aux_data

        # Built without __init__, whose signature a subclass may change. The
        # configuration was validated when the flattened ansatz was built.
        ansatz = cls.__new__(cls)
        ansatz.num_qubits = num_qubits
        ansatz.num_parameters = num_parameters

        for name, value in options.items():
            setattr(ansatz, name, value)

        return ansatz

    def __setstate__(self, state):
        # A subclass defined in a script is rebuilt from its code when loaded
        # in another process, without running the decorator that registered it
        # as a PyTree. Unregistered, it is its own single leaf, which
        # treedef_is_leaf cannot tell, because it also holds for a registered
        # node without children.
        self.__dict__.update(state)
        leaves = tree_leaves(self)

        if len(leaves) == 1 and leaves[0] is self:
            register_pytree_node_class(type(self))

    def _options(self):
        """
        Returns the per-ansatz settings, which subclasses set after building.
        """
        return {}


def _check_layers(layers, name):
    """
    Rejects a layer count that is not a non-negative integer.

    A negative count expects no parameters and would leave the QuantumVariable
    untouched, which the parameter count check does not catch. A fractional
    count would give a fractional parameter count that construction
    truncates.
    """
    if not isinstance(layers, (int, np.integer)) or isinstance(layers, bool):
        raise ValueError(f"Tried to configure a {name} ansatz with {layers!r} layers, which is not an integer.")

    if layers < 0:
        raise ValueError(f"Tried to configure a {name} ansatz with a negative layer count ({layers}).")


@register_pytree_node_class
class _HardwareEfficient(Ansatz):
    def _options(self):
        return {"layers": self.layers, "rotations": self.rotations, "entangler": self.entangler}

    def apply_gates(self, qv, w):
        return _hardware_efficient(qv, w, layers=self.layers, rotations=self.rotations, entangler=self.entangler)


@register_pytree_node_class
class _RealAmplitudes(Ansatz):
    def _options(self):
        return {"layers": self.layers, "entanglement": self.entanglement}

    def apply_gates(self, qv, w):
        return _real_amplitudes(qv, w, layers=self.layers, entanglement=self.entanglement)


@register_pytree_node_class
class _DataReuploading(Ansatz):
    def _options(self):
        return {"layers": self.layers, "encoding": self.encoding, "ansatz": self.ansatz}

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        The encoding and the inner ansatz are children rather than auxiliary
        data, because auxiliary data is a constant of every compiled function,
        while the scaling of the encoding has to be an input to it.
        """
        return (self.encoding, self.ansatz), (self.num_qubits, self.num_parameters, self.layers)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        PyTree unflatten for JAX.
        """
        reuploading = cls.__new__(cls)
        reuploading.num_qubits, reuploading.num_parameters, reuploading.layers = aux_data
        reuploading.encoding, reuploading.ansatz = children

        return reuploading

    def apply(self, qv, w, x):
        """
        Applies alternating encoding and ansatz layers to ``qv``.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to apply the circuit to.
        w : list[float] or numpy.ndarray
            The parameters of all ansatz layers, concatenated.
        x : list[float] or numpy.ndarray
            The feature vector, encoded once per layer.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the circuit was applied to.

        Raises
        ------
        ValueError
            If ``w`` does not hold exactly ``num_parameters`` parameters.

        """
        self._check_parameters(w)

        return _data_reuploading(
            qv,
            x,
            w,
            self.ansatz.num_parameters,
            self.encoding,
            self.ansatz,
            layers=self.layers,
        )
