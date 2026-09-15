"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

"""
Tests verifying that Qrisp's statevector simulator and CUDA-Q agree on the
*exact* resulting statevector (not just measurement statistics) for every
gate Qrisp lowers to Quake.

Coverage
--------
- All 0-param and 1-param single-qubit gates in ``GATE_MAP``, plus ``u3``.
- Native 2-qubit controlled gates in ``GATE_MAP`` (cx, cy, cz).
- Controlled versions of all single-qubit ``GATE_MAP`` gates, built via
  ``with control(...):`` (this also exercises ``cgphase``, since ``gphase``
  turns into a phase gate when controlled).
- Composite/decomposed 2-qubit gates that are not literal ``GATE_MAP``
  entries: ``cp``, ``swap``, ``rxx``, ``rzz``, ``xxyy``.

Unlike ``test_qiskit_converter.py`` (which compares unitaries), a CUDA-Q
kernel has no standalone unitary to extract, so gates here are compared via
full statevectors instead, using random (but fixed, for reproducibility)
parameters and a generic (non-basis) input state so phase errors are caught.

Qrisp's ``statevector_array`` uses big-endian qubit ordering while CUDA-Q
uses little-endian; every comparison goes through the shared
``_to_little_endian`` helper below.
"""

import cudaq
import numpy as np
import pytest

from qrisp import (
    QuantumVariable,
    control,
    cp,
    cx,
    cy,
    cz,
    gphase,
    h,
    p,
    rx,
    rxx,
    ry,
    rz,
    rzz,
    s,
    s_dg,
    swap,
    sx,
    sx_dg,
    t,
    t_dg,
    u3,
    x,
    xxyy,
    y,
    z,
)
from qrisp.jasp.cudaq_interface import cudaq_kernel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Fixed seed: parameters/prep angles are "random" but reproducible across runs.
_rng = np.random.default_rng(20240804)


def _rand_angle() -> float:
    return float(_rng.uniform(-np.pi, np.pi))


def _to_little_endian(sv_big_endian: np.ndarray) -> np.ndarray:
    """Convert Qrisp's big-endian statevector ordering to CUDA-Q's little-endian ordering."""
    n = int(np.log2(len(sv_big_endian)))
    return sv_big_endian.reshape([2] * n).transpose().flatten()


def qrisp_statevector(circuit_fn) -> np.ndarray:
    """Run *circuit_fn* as a plain (eager) Qrisp circuit; return its statevector, little-endian."""
    qv = circuit_fn()
    return _to_little_endian(qv.qs.statevector_array())


def cudaq_statevector(circuit_fn) -> np.ndarray:
    """Trace *circuit_fn* through ``@cudaq_kernel`` and return its CUDA-Q statevector."""

    def _kernel_body():
        circuit_fn()

    kernel = cudaq_kernel(_kernel_body)
    state = cudaq.get_state(kernel)
    return np.array(state, copy=False)


def assert_statevectors_close(circuit_fn, atol=1e-5):
    sv_qrisp = qrisp_statevector(circuit_fn)
    sv_cudaq = cudaq_statevector(circuit_fn)
    assert np.allclose(sv_qrisp, sv_cudaq, atol=atol), f"statevectors differ\nqrisp: {sv_qrisp}\ncudaq: {sv_cudaq}"


def _prep_angles(num_qubits):
    """Fixed per-qubit u3 angles, sampled once so the same state is prepared
    identically in both the qrisp and the cudaq execution of a circuit."""
    return [(_rand_angle(), _rand_angle(), _rand_angle()) for _ in range(num_qubits)]


def _apply_prep(qv, angles):
    """Rotate every qubit into a generic non-basis state so gate tests are phase-sensitive."""
    for i, (theta, phi, lam) in enumerate(angles):
        u3(theta, phi, lam, qv[i])


def _single_qubit_circuit(gate_fn):
    """Build a 1-qubit circuit: generic prep state, then *gate_fn(qv)*."""
    angles = _prep_angles(1)

    def circuit():
        qv = QuantumVariable(1)
        _apply_prep(qv, angles)
        gate_fn(qv)
        return qv

    return circuit


def _two_qubit_circuit(gate_fn):
    """Build a 2-qubit circuit: generic prep state, then *gate_fn(qv)*."""
    angles = _prep_angles(2)

    def circuit():
        qv = QuantumVariable(2)
        _apply_prep(qv, angles)
        gate_fn(qv)
        return qv

    return circuit


def _with_control(qv, gate_fn):
    """Apply *gate_fn* to ``qv[1]``, controlled on ``qv[0]``."""
    with control(qv[0]):
        gate_fn(qv[1])


# ---------------------------------------------------------------------------
# Single-qubit GATE_MAP gates (0 params)
# ---------------------------------------------------------------------------

_NO_PARAM_SINGLE_QUBIT_GATES = [
    ("h", h),
    ("x", x),
    ("y", y),
    ("z", z),
    ("s", s),
    ("t", t),
    ("s_dg", s_dg),
    ("t_dg", t_dg),
    ("sx", sx),
    ("sx_dg", sx_dg),
]


@pytest.mark.parametrize("gate_name,gate", _NO_PARAM_SINGLE_QUBIT_GATES)
def test_gate_map_single_qubit_gates(gate_name, gate):
    circuit = _single_qubit_circuit(lambda qv: gate(qv[0]))
    assert_statevectors_close(circuit)


@pytest.mark.parametrize("gate_name,gate", _NO_PARAM_SINGLE_QUBIT_GATES)
def test_controlled_single_qubit_gates(gate_name, gate):
    circuit = _two_qubit_circuit(lambda qv: _with_control(qv, lambda q: gate(q)))
    assert_statevectors_close(circuit)


# ---------------------------------------------------------------------------
# Single-qubit GATE_MAP gates (1 param)
# ---------------------------------------------------------------------------

_PARAM_SINGLE_QUBIT_GATES = [
    ("rx", rx),
    ("ry", ry),
    ("rz", rz),
    ("p", p),
    ("gphase", gphase),
]


@pytest.mark.parametrize("gate_name,gate", _PARAM_SINGLE_QUBIT_GATES)
def test_gate_map_single_qubit_parameterized_gates(gate_name, gate):
    phi = _rand_angle()
    circuit = _single_qubit_circuit(lambda qv: gate(phi, qv[0]))
    assert_statevectors_close(circuit)


@pytest.mark.parametrize("gate_name,gate", _PARAM_SINGLE_QUBIT_GATES)
def test_controlled_single_qubit_parameterized_gates(gate_name, gate):
    """Also covers ``cgphase``: gphase turns into a phase gate when controlled."""
    phi = _rand_angle()
    circuit = _two_qubit_circuit(lambda qv: _with_control(qv, lambda q: gate(phi, q)))
    assert_statevectors_close(circuit)


# ---------------------------------------------------------------------------
# u3 (3-param GATE_MAP gate)
# ---------------------------------------------------------------------------


def test_u3_gate():
    theta, phi, lam = _rand_angle(), _rand_angle(), _rand_angle()
    circuit = _single_qubit_circuit(lambda qv: u3(theta, phi, lam, qv[0]))
    assert_statevectors_close(circuit)


def test_controlled_u3_gate():
    theta, phi, lam = _rand_angle(), _rand_angle(), _rand_angle()
    circuit = _two_qubit_circuit(lambda qv: _with_control(qv, lambda q: u3(theta, phi, lam, q)))
    assert_statevectors_close(circuit)


# ---------------------------------------------------------------------------
# Native 2-qubit controlled GATE_MAP gates
# ---------------------------------------------------------------------------

_NATIVE_CONTROLLED_GATES = [
    ("cx", cx),
    ("cy", cy),
    ("cz", cz),
]


@pytest.mark.parametrize("gate_name,gate", _NATIVE_CONTROLLED_GATES)
def test_gate_map_native_controlled_gates(gate_name, gate):
    circuit = _two_qubit_circuit(lambda qv: gate(qv[0], qv[1]))
    assert_statevectors_close(circuit)


# ---------------------------------------------------------------------------
# Composite / decomposed 2-qubit gates (not literal GATE_MAP entries)
# ---------------------------------------------------------------------------


def test_cp_gate():
    phi = _rand_angle()
    circuit = _two_qubit_circuit(lambda qv: cp(phi, qv[0], qv[1]))
    assert_statevectors_close(circuit)


def test_swap_gate():
    circuit = _two_qubit_circuit(lambda qv: swap(qv[0], qv[1]))
    assert_statevectors_close(circuit)


@pytest.mark.parametrize("gate_name,gate", [("rxx", rxx), ("rzz", rzz)])
def test_composite_two_qubit_rotation_gates(gate_name, gate):
    phi = _rand_angle()
    circuit = _two_qubit_circuit(lambda qv: gate(phi, qv[0], qv[1]))
    assert_statevectors_close(circuit)


def test_xxyy_gate():
    phi, beta = _rand_angle(), _rand_angle()
    circuit = _two_qubit_circuit(lambda qv: xxyy(phi, beta, qv[0], qv[1]))
    assert_statevectors_close(circuit)
