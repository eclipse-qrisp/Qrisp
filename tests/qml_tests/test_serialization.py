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

"""Tests saving and loading qml models, including custom classes loaded in a new process."""

import subprocess
import sys
import textwrap
import warnings

import dill
import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import QuantumFloat, measure
from qrisp.jasp import jaspify, terminal_sampling
from qrisp.operators import X, Z
from qrisp.qml import Ansatz, Encoding, load_model, save_model


def statevector_array(qv):
    return np.array(qv.qs.statevector("array"), dtype=complex)


def round_trip(tmp_path, **parts):
    filename = tmp_path / "model.pkl"
    save_model(filename, **parts)
    return load_model(filename)


@pytest.mark.parametrize(
    "make_encoding, num_qubits, features",
    [
        (lambda: Encoding.angle(n_features=2, data_range=([3.0, 1.0], [6.9, 2.5])), 2, [5.1, 1.6]),
        (
            lambda: Encoding.dense_angle(n_features=4, data=np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])),
            2,
            [1.0, 2.0, 5.0, 6.0],
        ),
        (lambda: Encoding.amplitude(n_features=3), 2, [1.0, 2.0, 3.0]),
        (lambda: Encoding.basis(n_features=3), 3, [1, 1, 0]),
    ],
)
def test_loaded_encoding_prepares_the_same_state(tmp_path, make_encoding, num_qubits, features):
    original = make_encoding()
    loaded = round_trip(tmp_path, encoding=original)["encoding"]

    expected = QuantumFloat(num_qubits)
    original.apply(expected, features)
    actual = QuantumFloat(num_qubits)
    loaded.apply(actual, features)

    assert np.allclose(statevector_array(actual), statevector_array(expected))


@pytest.mark.parametrize(
    "make_ansatz, uses_features",
    [
        (lambda: Ansatz.hardware_efficient(num_qubits=2, layers=1, rotations=("y", "z")), False),
        (lambda: Ansatz.real_amplitudes(num_qubits=2, layers=2), False),
        (
            lambda: Ansatz.data_reuploading(
                num_qubits=2,
                encoding=Encoding.angle(n_features=2),
                ansatz=Ansatz.real_amplitudes(num_qubits=2, layers=1),
                layers=2,
            ),
            True,
        ),
    ],
)
def test_loaded_ansatz_prepares_the_same_state(tmp_path, make_ansatz, uses_features):
    original = make_ansatz()
    loaded = round_trip(tmp_path, ansatz=original)["ansatz"]
    weights = np.linspace(0.3, 2.9, original.num_parameters)
    features = [0.4, 1.2]

    def prepared(ansatz):
        qf = QuantumFloat(2)
        if uses_features:
            ansatz.apply(qf, weights, features)
        else:
            ansatz.apply(qf, weights)
        return statevector_array(qf)

    assert loaded.num_parameters == original.num_parameters
    assert np.allclose(prepared(loaded), prepared(original))


@pytest.mark.parametrize(
    "make_ansatz",
    [
        lambda: Ansatz.hardware_efficient(num_qubits=2, layers=1, rotations=("y", "z")),
        lambda: Ansatz.real_amplitudes(num_qubits=2, layers=2),
    ],
    ids=["hardware_efficient", "real_amplitudes"],
)
def test_loaded_ansatz_is_a_traced_argument(tmp_path, make_ansatz):
    """
    A loaded built-in ansatz can be passed into a traced function as an
    argument and gives the same distribution as the original in static mode.
    Generic weights give a distribution with no zero entries, which makes it
    depend on every gate of the ansatz.
    """
    original = make_ansatz()
    ansatz = round_trip(tmp_path, ansatz=original)["ansatz"]
    weights = np.linspace(0.3, 2.9, original.num_parameters)

    @terminal_sampling
    def run(a, w):
        qf = QuantumFloat(2)
        a.apply(qf, w)
        return qf

    expected = QuantumFloat(2)
    original.apply(expected, weights)
    # get_measurement rounds each probability to five decimals.
    expected = expected.get_measurement()

    distribution = run(ansatz, jnp.array(weights))

    assert np.allclose(
        [distribution.get(i, 0.0) for i in range(4)], [expected.get(i, 0.0) for i in range(4)], atol=1e-4
    )


@pytest.mark.parametrize("observable", [Z(0), Z(0) * Z(1) + 0.5 * X(1), [Z(0), X(1)]])
def test_observables_keep_their_terms(tmp_path, observable):
    loaded = round_trip(tmp_path, observable=observable)["observable"]

    if isinstance(observable, list):
        assert [str(term) for term in loaded] == [str(term) for term in observable]
    else:
        assert str(loaded) == str(observable)


@pytest.mark.parametrize("parameters", [np.array([0.1, -0.2, 3.5]), jnp.array([0.1, -0.2, 3.5])])
def test_parameters_keep_their_values_and_array_type(tmp_path, parameters):
    loaded = round_trip(tmp_path, parameters=parameters)["parameters"]

    assert type(loaded) is type(parameters)
    assert np.array_equal(np.asarray(loaded), np.asarray(parameters))


def test_all_parts_come_back_under_their_names(tmp_path):
    parts = {
        "encoding": Encoding.angle(n_features=2),
        "support_vectors": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "bias": 0.25,
    }

    loaded = round_trip(tmp_path, **parts)

    assert sorted(loaded) == sorted(parts)
    assert loaded["bias"] == 0.25


def test_loaded_encoding_scales_raw_features_under_tracing(tmp_path):
    # The upper end of the saved data range must land on pi, which flips both
    # qubits and makes the traced measurement deterministic.
    original = Encoding.angle(n_features=2, data_range=([3.0, 1.0], [6.9, 2.5]))
    encoding = round_trip(tmp_path, encoding=original)["encoding"]

    @jaspify
    def run():
        qf = QuantumFloat(2)
        encoding.apply(qf, jnp.array([6.9, 2.5]))
        return measure(qf)

    assert run() == 3


def test_loaded_scaled_encoding_is_a_traced_argument(tmp_path):
    """
    A loaded built-in encoding can be passed into a traced function as an
    argument, not only captured in a closure, and its scaling range travels
    with it. Generic raw features land on generic angles, which makes the
    distribution depend on the exact scaling. With RY(a_i) on qubit i, the
    qubit reads 1 with probability sin^2(a_i/2).
    """
    low, high = np.array([3.0, 1.0]), np.array([6.9, 2.5])
    encoding = round_trip(tmp_path, encoding=Encoding.angle(n_features=2, data_range=(low, high)))["encoding"]
    features = np.array([5.1, 1.6])

    @terminal_sampling
    def run(e, x):
        qf = QuantumFloat(2)
        e.apply(qf, x)
        return qf

    angles = (features - low) / (high - low) * np.pi
    p = np.sin(angles / 2) ** 2
    # Qubit 0 is the least significant bit of each outcome.
    expected = [(1 - p[0]) * (1 - p[1]), p[0] * (1 - p[1]), (1 - p[0]) * p[1], p[0] * p[1]]

    distribution = run(encoding, jnp.array(features))

    assert np.allclose([distribution.get(i, 0.0) for i in range(4)], expected)


@pytest.mark.parametrize("prepare_basis, warns", [(None, True), (False, False)], ids=["unspecified", "false"])
def test_loaded_z_axis_encoding_keeps_its_warning_behaviour(tmp_path, prepare_basis, warns):
    """
    The z-axis warning depends on whether prepare_basis was given, and that
    distinction is stored in the encoding. A loaded encoding still warns when
    it was left unspecified and stays silent when it was set to False.
    """
    options = {} if prepare_basis is None else {"prepare_basis": prepare_basis}
    encoding = round_trip(tmp_path, encoding=Encoding.angle(n_features=2, axis="z", **options))["encoding"]

    if warns:
        with pytest.warns(UserWarning, match="applies a Hadamard gate"):
            encoding.apply(QuantumFloat(2), [0.3, 0.4])
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*applies a Hadamard gate")
            encoding.apply(QuantumFloat(2), [0.3, 0.4])


def test_custom_classes_defined_in_a_script_load_in_a_new_process(tmp_path):
    """
    A class defined in a script is rebuilt from its code when loaded in another
    process. A plain subclass then encodes as before. The decorator that
    registered a class as a PyTree does not run again, and unregistered, the
    loaded object would be a single opaque leaf that cannot be passed into a
    traced function. Loading the file a second time finds the classes already
    registered. One save and one load process cover all three classes.
    """
    filename = tmp_path / "custom.pkl"
    save_script = textwrap.dedent(
        f"""
        from jax.tree_util import register_pytree_node_class
        from qrisp import rx, ry
        from qrisp.jasp import jrange
        from qrisp.qml import Ansatz, Encoding, save_model

        class FlipEncoding(Encoding):
            def encode(self, qv, x):
                for i in range(len(qv)):
                    rx(x[i], qv[i])

        @register_pytree_node_class
        class DoubleAngleEncoding(Encoding):
            def encode(self, qv, x):
                for i in jrange(qv.size):
                    ry(2 * x[i], qv[i])
                return qv

        @register_pytree_node_class
        class RotationAnsatz(Ansatz):
            def parameter_count(self, num_qubits):
                return num_qubits

            def apply_gates(self, qv, w):
                for i in jrange(qv.size):
                    ry(w[i], qv[i])
                return qv

        save_model(
            {str(filename)!r},
            flip=FlipEncoding(n_features=2),
            encoding=DoubleAngleEncoding(n_features=2),
            ansatz=RotationAnsatz(num_qubits=2),
        )
        """
    )
    load_script = textwrap.dedent(
        f"""
        import jax.numpy as jnp
        import numpy as np
        from jax.tree_util import tree_flatten, tree_unflatten
        from qrisp import QuantumFloat, measure
        from qrisp.jasp import jaspify
        from qrisp.qml import load_model

        load_model({str(filename)!r})
        model = load_model({str(filename)!r})
        flip, encoding, ansatz = model["flip"], model["encoding"], model["ansatz"]

        qf = QuantumFloat(2)
        flip.apply(qf, [np.pi, 0.0])
        print("RESULT", type(flip).__name__, abs(qf.qs.statevector("array")).argmax())

        for part in (encoding, ansatz):
            leaves, treedef = tree_flatten(part)
            rebuilt = tree_unflatten(treedef, leaves)
            print("RESULT", len(leaves) == 1 and leaves[0] is part, type(rebuilt).__name__)

        @jaspify
        def main(e, a, x, w):
            qf = QuantumFloat(2)
            e.apply(qf, x)
            a.apply(qf, w)
            return measure(qf)

        print("RESULT", main(encoding, ansatz, jnp.array([jnp.pi / 2, 0.0]), jnp.array([0.0, jnp.pi])))
        """
    )

    subprocess.run([sys.executable, "-c", save_script], check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-c", load_script], check=True, capture_output=True, text=True)

    # The simulator's progress bar shares stdout, and only the marked part of
    # each line is read.
    lines = [line[line.index("RESULT") :].split()[1:] for line in result.stdout.splitlines() if "RESULT" in line]

    # rx(pi) acts on qubit 0 alone, and statevector("array") puts qubit 0 in
    # the most significant bit, which makes the populated index 2. In the
    # traced run ry(2 * pi/2) flips qubit 0 and ry(pi) flips qubit 1, giving |3>.
    assert lines == [
        ["FlipEncoding", "2"],
        ["False", "DoubleAngleEncoding"],
        ["False", "RotationAnsatz"],
        ["3.0"],
    ]


def test_saving_nothing_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no parts"):
        save_model(tmp_path / "model.pkl")


@pytest.mark.parametrize(
    "record",
    [[1, 2, 3], {"format_version": 1}, {"format_version": 1, "parts": [1, 2]}],
    ids=["not_a_record", "no_parts", "parts_not_named"],
)
def test_a_file_not_written_by_save_model_is_rejected(tmp_path, record):
    """A record without named parts fails with the same message as any other foreign file."""
    filename = tmp_path / "other.pkl"
    with open(filename, "wb") as file:
        dill.dump(record, file)

    with pytest.raises(ValueError, match="not written by save_model"):
        load_model(filename)


def test_a_newer_file_format_is_rejected(tmp_path):
    filename = tmp_path / "future.pkl"
    with open(filename, "wb") as file:
        dill.dump({"format_version": 99, "qrisp_version": "99.0", "parts": {"bias": 0.0}}, file)

    with pytest.raises(ValueError, match="format 99"):
        load_model(filename)
