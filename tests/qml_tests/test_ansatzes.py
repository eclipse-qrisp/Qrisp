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

"""Tests the qml ansatzes, their parameter counts, subclassing and tracing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_leaves, tree_unflatten

from qrisp import QuantumFloat, h, measure, ry
from qrisp.jasp import jaspify, jrange, terminal_sampling
from qrisp.qml import (
    Ansatz,
    Encoding,
    hardware_efficient_size,
    real_amplitudes_size,
)
from qrisp.qml.ansatzes import _hardware_efficient


def statevector_array(qv):
    return np.array(qv.qs.statevector("array"), dtype=complex)


@pytest.mark.parametrize(
    "num_qubits, layers, rotations, expected",
    [(3, 2, ("y",), 9), (3, 2, ("y", "z"), 18), (4, 1, ("y",), 8), (2, 0, ("y",), 2), (2, 1, ("z", "y", "z"), 12)],
)
def test_hardware_efficient_size(num_qubits, layers, rotations, expected):
    """
    The ansatz uses one parameter per qubit, rotation axis and rotation layer,
    with one more rotation layer than entangling layers.
    """
    assert hardware_efficient_size(num_qubits, layers, rotations) == expected


@pytest.mark.parametrize("constructor", [Ansatz.real_amplitudes, Ansatz.hardware_efficient])
@pytest.mark.parametrize("as_nested_list", [False, True], ids=["array", "nested_list"])
def test_ansatzes_read_shaped_parameters_like_the_flat_vector(constructor, as_nested_list):
    """
    Parameters may be passed as a flat vector, as the optimizers require, or
    shaped with one row per rotation layer. The docstrings promise that a shaped
    array and the nested list describing it are read in the order of the flat
    vector.
    """
    ansatz = constructor(num_qubits=3, layers=2)
    flat = np.linspace(0.1, 2.9, ansatz.num_parameters)
    shaped = flat.reshape(3, 3)
    if as_nested_list:
        shaped = shaped.tolist()

    def run(w):
        qf = QuantumFloat(3)
        ansatz.apply(qf, w)
        return statevector_array(qf)

    assert np.allclose(run(shaped), run(flat))


def test_hardware_efficient_single_qubit_matches_rotation_product():
    """
    On one qubit there is nothing to entangle. The layers therefore reduce to a
    single RY rotation by the sum of the parameters.
    """
    w = np.array([0.3, 0.7, 1.1])

    qf = QuantumFloat(1)
    Ansatz.hardware_efficient(num_qubits=1, layers=2).apply(qf, w)

    total = w.sum()
    expected = np.array([np.cos(total / 2), np.sin(total / 2)])

    assert np.allclose(statevector_array(qf), expected)


_SINGLE_QUBIT_GATES = {
    "x": lambda t: np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]]),
    "y": lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]]),
    "z": lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)]),
}


def _apply_single(state, qubit, matrix):
    """Applies a 2x2 matrix to one qubit of a state indexed with qubit 0 as the least significant bit."""
    state = state.copy()
    for i in range(len(state)):
        if not i >> qubit & 1:
            j = i | 1 << qubit
            state[i], state[j] = matrix @ np.array([state[i], state[j]])
    return state


def _apply_controlled(state, control, target, gate):
    """Applies CX or CZ to a state indexed with qubit 0 as the least significant bit."""
    state = state.copy()
    for i in range(len(state)):
        if i >> control & 1 and not i >> target & 1:
            j = i | 1 << target
            if gate == "cx":
                state[i], state[j] = state[j], state[i]
            else:
                state[j] = -state[j]
    return state


@pytest.mark.parametrize("entangler", ["cx", "cz"])
def test_hardware_efficient_matches_a_reference_simulation(entangler):
    """
    Three rotation axes, two layers and generic angles on three qubits make
    every gate visible in the state, including the X rotations and the CZ
    entangler. The reference is a separate numpy simulation of the documented
    gate order, rotation layers of each axis in turn and a linear chain of
    entanglers between them. Global phases are ignored by comparing the
    overlap.
    """
    num_qubits, layers, rotations = 3, 2, ("x", "y", "z")
    ansatz = Ansatz.hardware_efficient(num_qubits=num_qubits, layers=layers, rotations=rotations, entangler=entangler)
    w = np.linspace(0.2, 2.8, ansatz.num_parameters)

    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    for layer in range(layers + 1):
        for axis_index, axis in enumerate(rotations):
            offset = (layer * len(rotations) + axis_index) * num_qubits
            for i in range(num_qubits):
                state = _apply_single(state, i, _SINGLE_QUBIT_GATES[axis](w[offset + i]))
        if layer < layers:
            for i in range(num_qubits - 1):
                state = _apply_controlled(state, i, i + 1, entangler)

    qf = QuantumFloat(num_qubits)
    ansatz.apply(qf, w)

    # statevector("array") puts qubit 0 in the most significant bit, and
    # reversing the bits of each index lines it up with the reference.
    order = [int(format(i, f"0{num_qubits}b")[::-1], 2) for i in range(2**num_qubits)]
    actual = statevector_array(qf)[order]

    assert np.isclose(abs(np.vdot(state, actual)), 1.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rotations": ("w",)}, "Unknown rotation axis"),
        ({"entangler": "swap"}, "Unknown entangler"),
    ],
)
def test_hardware_efficient_rejects_invalid_arguments(kwargs, match):
    """The rotation axes and entanglers are restricted to the supported gates."""
    with pytest.raises(ValueError, match=match):
        Ansatz.hardware_efficient(num_qubits=3, **kwargs)


@pytest.mark.parametrize("rotations", [(), ""])
def test_hardware_efficient_rejects_an_empty_set_of_rotations(rotations):
    """
    Without a rotation axis the ansatz has no parameters, and training it would
    change nothing. The size function, the constructor and the gate sequence
    under tracing all reject it.
    """
    with pytest.raises(ValueError, match="without rotation axes"):
        hardware_efficient_size(3, rotations=rotations)

    with pytest.raises(ValueError, match="without rotation axes"):
        Ansatz.hardware_efficient(num_qubits=3, rotations=rotations)

    @jaspify
    def main():
        qf = QuantumFloat(3)
        _hardware_efficient(qf, jnp.zeros(0), rotations=rotations)
        return measure(qf)

    with pytest.raises(ValueError, match="without rotation axes"):
        main()


def test_hardware_efficient_reads_a_string_of_axes_per_character():
    """A string of axes is the documented shorthand for the tuple of its characters."""
    w = np.linspace(0.3, 2.9, hardware_efficient_size(2, rotations=("y", "z")))

    qf_string = QuantumFloat(2)
    Ansatz.hardware_efficient(num_qubits=2, rotations="yz").apply(qf_string, w)

    qf_tuple = QuantumFloat(2)
    Ansatz.hardware_efficient(num_qubits=2, rotations=("y", "z")).apply(qf_tuple, w)

    assert hardware_efficient_size(2, rotations="yz") == len(w)
    assert np.allclose(statevector_array(qf_string), statevector_array(qf_tuple))


@pytest.mark.parametrize("run", ["direct", "traced"])
@pytest.mark.parametrize(
    "make_ansatz, expected",
    [
        (lambda: Ansatz.hardware_efficient(num_qubits=3, layers=2), 9),
        (lambda: Ansatz.real_amplitudes(num_qubits=3, layers=2), 9),
        (lambda: RotationAnsatz(num_qubits=3), 3),
    ],
    ids=["hardware_efficient", "real_amplitudes", "subclass"],
)
def test_ansatzes_reject_the_wrong_parameter_count(make_ansatz, expected, run):
    """
    The parameter count is checked on the base class before any gate is
    emitted, for the built-in ansatzes and for a subclass. The subclass derives
    its count from parameter_count, and the fixed expected numbers also pin
    that derivation. The check runs while a program is being traced as well.
    A mismatch would otherwise pass silently there, because too few parameters
    are read past the end of the array and Jax clamps the read rather than
    raising.
    """
    ansatz = make_ansatz()
    w = np.zeros(expected - 1)

    def main(w):
        qf = QuantumFloat(3)
        ansatz.apply(qf, w)
        return measure(qf)

    with pytest.raises(ValueError, match=f"Expected {expected}"):
        if run == "traced":
            jaspify(main)(jnp.asarray(w))
        else:
            main(w)


def test_real_amplitudes_keeps_amplitudes_real():
    """
    RY and CX have real matrix representations, and the prepared state stays in
    the real subspace. This is the property the ansatz is named for.
    """
    w = np.linspace(0.1, 2.9, real_amplitudes_size(3, layers=2))

    qf = QuantumFloat(3)
    Ansatz.real_amplitudes(num_qubits=3, layers=2).apply(qf, w)

    assert np.allclose(statevector_array(qf).imag, 0.0)


def test_real_amplitudes_rejects_an_unknown_entanglement():
    """The entanglement is restricted to the supported connectivities."""
    with pytest.raises(ValueError, match="Unknown entanglement"):
        Ansatz.real_amplitudes(num_qubits=3, entanglement="full")


@pytest.mark.parametrize(
    "num_qubits, entanglement", [(3, "linear"), (3, "ring"), (2, "ring")], ids=["3_linear", "3_ring", "2_ring"]
)
def test_real_amplitudes_matches_a_reference_simulation_under_tracing(num_qubits, entanglement):
    """
    Two layers and generic angles keep every amplitude away from zero and one.
    On three qubits the ring adds CX(2, 0) after every rotation layer but the
    last, while on two qubits the chain already covers that pair and nothing is
    added. A wrong parameter offset, a missing or duplicated closing entangler
    or a swapped control therefore changes the distribution. The reference is a
    separate numpy simulation, which means it shares no code with the ansatz.
    """
    layers = 2
    ansatz = Ansatz.real_amplitudes(num_qubits=num_qubits, layers=layers, entanglement=entanglement)
    w = np.linspace(0.3, 2.7, ansatz.num_parameters)

    @terminal_sampling
    def main(w):
        qf = QuantumFloat(num_qubits)
        ansatz.apply(qf, w)
        return qf

    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    for layer in range(layers + 1):
        for i in range(num_qubits):
            state = _apply_single(state, i, _SINGLE_QUBIT_GATES["y"](w[layer * num_qubits + i]))
        if layer < layers:
            for i in range(num_qubits - 1):
                state = _apply_controlled(state, i, i + 1, "cx")
            if entanglement == "ring" and num_qubits >= 3:
                state = _apply_controlled(state, num_qubits - 1, 0, "cx")

    distribution = main(jnp.array(w))
    observed = np.array([distribution.get(i, 0.0) for i in range(2**num_qubits)])

    assert np.allclose(observed, np.abs(state) ** 2)


def _untraced_and_traced_distributions(prepare, *args):
    """Returns the outcome probabilities of a preparation run directly and run traced."""

    def direct(*args):
        qf = prepare(*args)
        return np.abs(statevector_array(qf)) ** 2

    traced = terminal_sampling(prepare)(*[jnp.array(a) for a in args])
    num_states = len(direct(*args))
    observed = np.array([traced.get(i, 0.0) for i in range(num_states)])

    # statevector("array") puts qubit 0 in the most significant bit, while the
    # traced outcomes put it in the least significant bit. Reversing the bits
    # of each index lines the two up.
    num_qubits = num_states.bit_length() - 1
    order = [int(format(i, f"0{num_qubits}b")[::-1], 2) for i in range(num_states)]
    return direct(*args)[order], observed


def test_hardware_efficient_with_generic_parameters_traces_to_the_untraced_state():
    """
    Generic angles over three rotation axes and two layers leave no outcome
    with probability zero or one. The traced and the untraced circuit therefore
    agree only if every parameter reaches the same gate on both paths.
    """
    ansatz = Ansatz.hardware_efficient(num_qubits=3, layers=2, rotations=("x", "y", "z"))
    w = np.linspace(0.2, 2.8, ansatz.num_parameters)

    def prepare(w):
        qf = QuantumFloat(3)
        ansatz.apply(qf, w)
        return qf

    expected, observed = _untraced_and_traced_distributions(prepare, w)

    assert np.allclose(observed, expected)


def test_data_reuploading_with_generic_inputs_traces_to_the_untraced_state():
    """
    Generic features and parameters on two qubits and two layers make every
    upload and every parameter block visible in the distribution. A block read
    from the wrong offset under tracing changes it.
    """
    reuploading = Ansatz.data_reuploading(
        num_qubits=2,
        encoding=Encoding.angle(n_features=2),
        ansatz=Ansatz.real_amplitudes(num_qubits=2, layers=1),
        layers=2,
    )
    x = np.array([0.7, 2.1])
    w = np.linspace(0.4, 2.6, reuploading.num_parameters)

    def prepare(x, w):
        qf = QuantumFloat(2)
        reuploading.apply(qf, w, x)
        return qf

    expected, observed = _untraced_and_traced_distributions(prepare, x, w)

    assert np.allclose(observed, expected)


def test_data_reuploading_encodes_once_per_layer():
    """
    The features are re-encoded in every layer. On an ansatz that applies no
    rotation, three layers of RY(x/3) therefore add up to a single RY(x).
    """
    reuploading = Ansatz.data_reuploading(
        num_qubits=1,
        encoding=Encoding.angle(n_features=1),
        ansatz=Ansatz.real_amplitudes(num_qubits=1),
        layers=3,
    )

    qf = QuantumFloat(1)
    reuploading.apply(qf, np.zeros(reuploading.num_parameters), [np.pi / 3])

    # Three RY(pi/3) rotations compose to RY(pi), which flips the qubit.
    assert np.allclose(np.abs(statevector_array(qf)), [0.0, 1.0], atol=1e-9)


def test_data_reuploading_uses_a_distinct_parameter_block_per_layer():
    """
    Each layer consumes its own block of w, and permuting the blocks changes the
    circuit. A shared block would make the two states identical.
    """
    per_layer = real_amplitudes_size(2)
    first = np.concatenate([np.full(per_layer, 0.3), np.full(per_layer, 1.1)])
    second = np.concatenate([np.full(per_layer, 1.1), np.full(per_layer, 0.3)])

    reuploading = Ansatz.data_reuploading(
        num_qubits=2,
        encoding=Encoding.angle(n_features=2),
        ansatz=Ansatz.real_amplitudes(num_qubits=2),
    )

    qf_first = QuantumFloat(2)
    reuploading.apply(qf_first, first, [0.4, 0.8])

    qf_second = QuantumFloat(2)
    reuploading.apply(qf_second, second, [0.4, 0.8])

    assert not np.allclose(statevector_array(qf_first), statevector_array(qf_second))


@pytest.mark.parametrize("run", ["direct", "traced"])
def test_data_reuploading_rejects_parameter_count_mismatch(run):
    """
    The parameters are split evenly over the layers, and the split is checked
    while tracing as well. A mismatch therefore cannot reach the gates through
    either path.
    """
    reuploading = Ansatz.data_reuploading(
        num_qubits=2,
        encoding=Encoding.angle(n_features=2),
        ansatz=Ansatz.real_amplitudes(num_qubits=2),
        layers=3,
    )

    def main(w, x):
        qf = QuantumFloat(2)
        reuploading.apply(qf, w, x)
        return measure(qf)

    with pytest.raises(ValueError, match="Expected 12"):
        if run == "traced":
            jaspify(main)(jnp.zeros(5), jnp.array([0.1, 0.2]))
        else:
            main(np.zeros(5), [0.1, 0.2])


@pytest.mark.parametrize("constructor", [Ansatz.real_amplitudes, Ansatz.hardware_efficient])
@pytest.mark.parametrize("layers, match", [(-1, "negative"), (1.5, "not an integer"), (True, "not an integer")])
def test_ansatzes_reject_a_layer_count_that_is_not_a_non_negative_integer(constructor, layers, match):
    """
    A negative layer count leaves the rotation loop empty, applying no gates at
    all and silently returning an untouched QuantumVariable. The parameter
    count check does not catch it, because a negative count expects no
    parameters and an empty vector satisfies that. A fractional count yields a fractional
    parameter count, which construction truncates.
    """
    with pytest.raises(ValueError, match=match):
        constructor(num_qubits=3, layers=layers)


def test_data_reuploading_rejects_an_encoding_defined_only_from_zero():
    """
    Amplitude encoding prepares a state from |0>, which the register holds
    only before the first layer. Applied to the state that later layers leave,
    its result is not the encoded vector and differs between the traced and the
    untraced path.
    """
    with pytest.raises(ValueError, match="prepares a state from"):
        Ansatz.data_reuploading(
            num_qubits=2,
            encoding=Encoding.amplitude(n_features=4),
            ansatz=Ansatz.real_amplitudes(num_qubits=2),
        )


def test_data_reuploading_rejects_an_inner_ansatz_of_another_size():
    """
    The inner ansatz acts on the register the reuploading ansatz is applied to.
    An inner ansatz configured for another size therefore cannot fit it.
    """
    with pytest.raises(ValueError, match="same size"):
        Ansatz.data_reuploading(
            num_qubits=3,
            encoding=Encoding.angle(n_features=3),
            ansatz=Ansatz.real_amplitudes(num_qubits=2),
        )


@register_pytree_node_class
class RotationAnsatz(Ansatz):
    """
    A minimal user written ansatz, built only from the documented hooks.
    """

    def parameter_count(self, num_qubits):
        return num_qubits

    def apply_gates(self, qv, w):
        for i in jrange(qv.size):
            ry(w[i], qv[i])
        return qv


@pytest.mark.parametrize(
    "ansatz",
    [
        Ansatz.real_amplitudes(num_qubits=2, layers=0),
        RotationAnsatz(num_qubits=2),
    ],
    ids=["builtin", "subclass"],
)
def test_ansatzes_can_be_passed_as_traced_arguments(ansatz):
    """
    An unregistered object cannot cross into a traced function as an argument at
    all, only as a closure. Registering the PyTree keeps num_qubits and
    num_parameters plain integers. The trace time check then still has
    something to compare against.
    """

    @jaspify
    def main(a, w):
        qf = QuantumFloat(2)
        a.apply(qf, w)
        return measure(qf)

    assert main(ansatz, jnp.full(2, jnp.pi)) == 3


@register_pytree_node_class
class CountOnlyAnsatz(Ansatz):
    """A subclass whose constructor takes the qubit count alone."""

    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def parameter_count(self, num_qubits):
        return num_qubits

    def apply_gates(self, qv, w):
        for i in jrange(qv.size):
            ry(w[i], qv[i])
        return qv


@register_pytree_node_class
class FactorFirstAnsatz(Ansatz):
    """A subclass whose constructor takes a setting of its own first."""

    def __init__(self, factor, num_qubits=1):
        self.factor = factor
        super().__init__(num_qubits)

    def _options(self):
        return {"factor": self.factor}

    def parameter_count(self, num_qubits):
        return num_qubits

    def apply_gates(self, qv, w):
        for i in jrange(qv.size):
            ry(self.factor * w[i], qv[i])
        return qv


@pytest.mark.parametrize(
    "ansatz, parameters",
    [(CountOnlyAnsatz(1), [np.pi]), (FactorFirstAnsatz(2.0), [np.pi / 2])],
    ids=["count_only", "factor_first"],
)
def test_subclass_with_its_own_constructor_can_be_a_traced_argument(ansatz, parameters):
    """
    Jax rebuilds an argument from its flattened form. Rebuilding through the
    constructor passed it the base class arguments. A subclass with a signature
    of its own rejects or misreads them.
    """

    @jaspify
    def main(a, w):
        qf = QuantumFloat(1)
        a.apply(qf, w)
        return measure(qf)

    assert main(ansatz, jnp.array(parameters)) == 1


def test_ansatz_without_hooks_reports_what_is_missing():
    """
    A subclass missing both parameter_count and num_parameters cannot know what
    to check against. A default count would accept any parameter vector.
    Failing at construction names the missing hook instead.
    """

    class Incomplete(Ansatz):
        pass

    with pytest.raises(NotImplementedError, match="parameter_count"):
        Incomplete(num_qubits=2)


@pytest.mark.parametrize("rotations", [("y", "y"), ("z", "z", "y")])
def test_hardware_efficient_rejects_repeated_rotation_axes(rotations):
    """
    Two consecutive rotations about the same axis compose into a single
    rotation with the summed angle. The repetition therefore inflates the
    parameter count with parameters the circuit cannot use independently.
    """
    with pytest.raises(ValueError, match="Repeated rotation axis"):
        Ansatz.hardware_efficient(num_qubits=3, rotations=rotations)

    with pytest.raises(ValueError, match="Repeated rotation axis"):
        hardware_efficient_size(3, rotations=rotations)


def test_hardware_efficient_applies_an_axis_separated_by_another():
    """
    A rotation about another axis in between keeps two rotations about the same
    axis from composing, which makes Z-Y-Z a full single qubit rotation with
    three independent angles. The register starts in |+>, where the first Z
    rotation is observable, and each angle differs, which means a dropped or
    reordered gate changes the state.
    """
    a, b, c = 0.3, 1.1, 2.0

    qf = QuantumFloat(1)
    h(qf[0])
    Ansatz.hardware_efficient(num_qubits=1, layers=0, rotations=("z", "y", "z")).apply(qf, [a, b, c])

    def rz(t):
        return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

    def ry_matrix(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])

    expected = rz(c) @ ry_matrix(b) @ rz(a) @ (np.array([1.0, 1.0]) / np.sqrt(2))

    assert np.isclose(abs(np.vdot(expected, statevector_array(qf))), 1.0)


def test_data_reuploading_rejects_a_reuploading_inner_ansatz():
    """
    The inner ansatz is applied as apply(qv, w_layer), without the features a
    reuploading ansatz needs. Nesting one inside another would otherwise build
    a configuration with a parameter count it can never be applied with. It
    would fail with a TypeError at the first evaluation rather than here.
    """
    inner = Ansatz.data_reuploading(
        num_qubits=2,
        encoding=Encoding.angle(n_features=2),
        ansatz=Ansatz.real_amplitudes(num_qubits=2),
        layers=2,
    )

    with pytest.raises(ValueError, match="inner ansatz"):
        Ansatz.data_reuploading(
            num_qubits=2,
            encoding=Encoding.angle(n_features=2),
            ansatz=inner,
            layers=2,
        )


def reuploading_with_range(data_range):
    return Ansatz.data_reuploading(
        num_qubits=1,
        encoding=Encoding.angle(n_features=1, data_range=data_range),
        ansatz=Ansatz.real_amplitudes(num_qubits=1, layers=0),
        layers=2,
    )


def test_data_reuploading_exposes_the_scaling_of_its_encoding():
    """
    A standalone encoding passes its scaling to Jax as leaves. Nested in a
    reuploading ansatz, it was auxiliary data, which left the ansatz without
    any leaves. The rebuilt ansatz keeps the configuration it was built with.
    """
    ansatz = reuploading_with_range(([0.0], [2.0]))

    assert len(tree_leaves(ansatz)) == len(tree_leaves(ansatz.encoding)) == 2

    rebuilt = tree_unflatten(*reversed(tree_flatten(ansatz)))

    assert (rebuilt.num_qubits, rebuilt.num_parameters, rebuilt.layers) == (1, 2, 2)
    assert np.allclose(rebuilt.encoding._prepare(jnp.array([1.0])), np.pi / 2)


def test_data_reuploading_scaling_is_an_input_of_a_compiled_function():
    """
    Two ansatzes differing only in the scaling of their encoding share one
    compiled function, and each gets its own scaling. As auxiliary data the
    scaling was compiled in, which made each ansatz retrace and hid a scaling
    changed in place after compilation.
    """
    traces = []

    @jax.jit
    def scaled(ansatz, x):
        traces.append(None)
        return ansatz.encoding._prepare(x)

    # The feature 1 lies at the middle of [0, 2] and at the top of [0, 1].
    assert np.allclose(scaled(reuploading_with_range(([0.0], [2.0])), jnp.array([1.0])), np.pi / 2)
    assert np.allclose(scaled(reuploading_with_range(([0.0], [1.0])), jnp.array([1.0])), np.pi)
    assert len(traces) == 1
