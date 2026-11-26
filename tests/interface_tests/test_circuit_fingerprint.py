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

"""Tests for ``circuit_fingerprint``, which identifies circuits in a recording."""

import os
import subprocess
import sys

from qrisp import QuantumCircuit
from qrisp.circuit import Clbit, Qubit
from qrisp.interface.recording import circuit_fingerprint


def _circuit(gate="x", angle=0.5):
    """Small measured circuit."""
    qc = QuantumCircuit(1, 1)
    if gate == "x":
        qc.x(0)
    elif gate == "rx":
        qc.rx(angle, 0)
    qc.measure(0, 0)
    return qc


def _named_circuit(qubit_name, clbit_name):
    """Return an H-then-measure circuit on one qubit and one clbit with the given names."""
    qc = QuantumCircuit()
    qc.add_qubit(Qubit(qubit_name))
    qc.add_clbit(Clbit(clbit_name))
    qc.h(0)
    qc.measure(0, 0)
    return qc


_FINGERPRINT_SNIPPET = """
from qrisp import QuantumCircuit
from qrisp.interface.recording import circuit_fingerprint
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.rx(0.25, 1)
qc.measure(0, 1)
qc.measure(1, 0)
print(circuit_fingerprint(qc))
"""


class TestCircuitFingerprint:
    """Test :func:`circuit_fingerprint`."""

    def test_fingerprint_format_and_determinism(self):
        """The fingerprint is a sha256 string and equal for equal circuits."""
        fingerprint = circuit_fingerprint(_circuit())
        assert fingerprint.startswith("sha256:")
        # Fingerprint generation is deterministic.
        assert fingerprint == circuit_fingerprint(_circuit())

    def test_fingerprint_ignores_bit_names(self):
        """Renaming qubits and clbits does not change the fingerprint."""
        assert circuit_fingerprint(_named_circuit("a", "c")) == circuit_fingerprint(_named_circuit("b", "d"))

    def test_fingerprint_sees_clbit_indices(self):
        """Measuring into a different clbit changes the fingerprint."""
        a = QuantumCircuit(1, 2)
        a.measure(0, 0)
        b = QuantumCircuit(1, 2)
        b.measure(0, 1)
        assert circuit_fingerprint(a) != circuit_fingerprint(b)

    def test_fingerprint_sees_qubit_indices(self):
        """Applying a gate to a different qubit changes the fingerprint."""
        a = QuantumCircuit(2, 1)
        a.x(0)
        a.measure(0, 0)
        b = QuantumCircuit(2, 1)
        b.x(1)
        b.measure(0, 0)
        assert circuit_fingerprint(a) != circuit_fingerprint(b)

    def test_fingerprint_sees_params(self):
        """A different gate parameter changes the fingerprint."""
        assert circuit_fingerprint(_circuit(gate="rx", angle=0.5)) != circuit_fingerprint(
            _circuit(gate="rx", angle=0.6)
        )

    def test_fingerprint_recurses_into_definitions(self):
        """Two composite gates with the same name but different bodies differ."""

        def wrap(body_gate):
            inner = QuantumCircuit(1)
            getattr(inner, body_gate)(0)
            qc = QuantumCircuit(1, 1)
            qc.append(inner.to_gate(name="blob"), [0])
            qc.measure(0, 0)
            return qc

        assert circuit_fingerprint(wrap("x")) != circuit_fingerprint(wrap("z"))

    def test_fingerprint_stable_across_processes(self):
        """The fingerprint is equal across processes with different PYTHONHASHSEED values."""
        outputs = set()
        for seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            proc = subprocess.run(
                [sys.executable, "-c", _FINGERPRINT_SNIPPET],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            outputs.add(proc.stdout.strip().splitlines()[-1])
        assert len(outputs) == 1
