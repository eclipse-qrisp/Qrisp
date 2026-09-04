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


import pytest

from qrisp.jasp.evaluation_tools.buffered_quantum_state import BufferedQuantumState

# BufferedQuantumState is otherwise only exercised indirectly, through the full
# jaspify/stimulate/sample pipeline (see test_jaspify.py, test_stim_simulation.py,
# test_sampling.py). These tests target its own constructor/copy() behavior
# directly, since neither is reachable from that higher-level API.


def test_buffered_quantum_state_invalid_simulator_raises_with_name():
    """The error message must actually name the invalid simulator.

    Regression test: the message used to be a plain string missing the f-prefix
    ("Don't know simulator {simulator}"), so it never actually interpolated the
    invalid value.
    """
    with pytest.raises(Exception, match="not_a_real_simulator"):
        BufferedQuantumState("not_a_real_simulator")


def test_buffered_quantum_state_copy_preserves_simulator_and_gate_counts():
    """copy() must carry over the source's simulator backend and gate_counts.

    Regression test: copy() used to always construct a plain
    BufferedQuantumState() (defaulting to the "qrisp" backend) regardless of
    the source's actual simulator, and never copied gate_counts at all -- so
    copying a "stim"-backed state used to silently produce an inconsistent
    object: simulator == "qrisp" but quantum_state a real stim.TableauSimulator.
    """
    bqs = BufferedQuantumState("stim")
    bqs.gate_counts["h"] = 3
    bqs.gate_counts["cx"] = 2

    bqs_copy = bqs.copy()

    assert bqs_copy.simulator == "stim"
    assert bqs_copy.gate_counts == {"h": 3, "cx": 2}

    # Must be an independent copy, not a shared reference.
    bqs_copy.gate_counts["h"] = 99
    assert bqs.gate_counts["h"] == 3

    import stim

    assert isinstance(bqs_copy.quantum_state, stim.TableauSimulator)


def test_buffered_quantum_state_copy_qrisp_backend():
    """copy() on the default "qrisp" backend preserves the backend and gate_counts."""
    from qrisp.simulator import QuantumState

    bqs = BufferedQuantumState("qrisp")
    bqs.gate_counts["x"] = 1

    bqs_copy = bqs.copy()

    assert bqs_copy.simulator == "qrisp"
    assert bqs_copy.gate_counts == {"x": 1}
    assert isinstance(bqs_copy.quantum_state, QuantumState)
