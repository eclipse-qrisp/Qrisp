"""
********************************************************************************
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

from itertools import product
from collections.abc import Mapping

import numpy as np
import pytest
from qrisp import (
    QuantumVariable,
    QuantumArray,
    OutcomeArray,
    QuantumBool,
    QuantumFloat,
    auto_uncompute,
    h,
    invert,
    mcx,
    multi_measurement,
    p,
)
from qrisp.grover import tag_state, grovers_alg


def assert_valid_measurement(mes_res):
    assert isinstance(mes_res, Mapping), "Measurement result is not a mapping"
    assert all(isinstance(k, tuple) for k in mes_res.keys()), "Keys must be tuples"
    assert all((isinstance(v, float) and 0 <= v <= 1) for v in mes_res.values()), (
        "Values must be probabilities between 0 and 1"
    )


def test_grovers_basic_oracle():
    """Tests Grover's algorithm with a simple oracle that tags a specific state."""

    qf_list = [QuantumFloat(2), QuantumFloat(2)]

    def test_oracle(qf_list):
        tag_state({qf_list[0]: 0, qf_list[1]: 1})

    grovers_alg(qf_list, test_oracle)
    mes_res = multi_measurement(qf_list)

    assert_valid_measurement(mes_res)

    winner_state = (0, 1)
    assert list(mes_res.keys())[0] == winner_state

    valid_states = list(product(range(4), repeat=2))
    assert all(item in valid_states for item in mes_res.keys())


def test_grovers_equation_oracle():
    """Tests Grover's algorithm with an oracle that tags states based on a simple equation (multiplication)."""

    qf_list = [QuantumFloat(2, -1, signed=True), QuantumFloat(2, -1, signed=True)]

    def equation_oracle(qf_list):
        factor_0, factor_1 = qf_list[0], qf_list[1]
        ancilla = factor_0 * factor_1
        tag_state({ancilla: -0.25})

        inj_mul = ancilla << (lambda x, y: x * y)
        with invert():
            inj_mul(factor_0, factor_1)
        ancilla.delete()

    grovers_alg(qf_list, equation_oracle, winner_state_amount=2)
    mes_res = multi_measurement(qf_list)

    assert_valid_measurement(mes_res)

    winner_states = {(0.5, -0.5), (-0.5, 0.5)}
    assert set(list(mes_res.keys())[:2]) == winner_states

    valid_states = list(
        product([0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0], repeat=2)
    )
    assert all(item in valid_states for item in mes_res.keys())


def test_grovers_exact_mode():
    """Tests Grover's algorithm in exact mode using a custom phase oracle."""

    def exact_oracle(qv, phase=np.pi):
        temp_qbl = QuantumBool()
        mcx(qv[1:], temp_qbl)
        p(phase, temp_qbl)
        temp_qbl.uncompute()

    qv = QuantumVariable(6)
    grovers_alg(qv, exact_oracle, exact=True, winner_state_amount=2)

    mes_res = qv.get_measurement()
    winner_states = {"011111", "111111"}
    assert set(mes_res.keys()) == winner_states
    assert all(np.isclose(prob, 0.5, atol=1e-5) for prob in mes_res.values())


def test_grovers_quantum_array():
    """Tests Grover's algorithm when the input is a QuantumArray."""

    def array_oracle(qa):
        tag_state({qa[0]: 0, qa[1]: 0, qa[2]: 0})

    qa = QuantumArray(QuantumFloat(2), shape=(3,))
    grovers_alg(qa, array_oracle)

    mes_res = qa.get_measurement()
    winner_state = OutcomeArray([0, 0, 0])
    assert winner_state in mes_res
    assert mes_res[winner_state] > 0.99


def test_grovers_list_quantum_array():
    """Tests Grover's algorithm when the input is a list of QuantumArrays."""

    def array_oracle(qa_list):
        tag_state({qa_list[0][0]: 0, qa_list[0][1]: 0, qa_list[0][2]: 0})

    qa = QuantumArray(QuantumFloat(2), shape=(3,))
    qa_list = [qa]
    grovers_alg(qa_list, array_oracle)

    mes_res = qa.get_measurement()
    winner_state = OutcomeArray([0, 0, 0])
    assert winner_state in mes_res
    assert mes_res[winner_state] > 0.99


def test_grovers_tuple():
    """Tests Grover's algorithm when the input is a tuple of QuantumFloats."""

    def array_oracle(qf_tuple):
        tag_state({qf_tuple[0]: 0, qf_tuple[1]: 0, qf_tuple[2]: 0})

    qf_tuple = (QuantumFloat(2), QuantumFloat(2), QuantumFloat(2))
    grovers_alg(qf_tuple, array_oracle)

    mes_res = multi_measurement(qf_tuple)
    winner_state = (0, 0, 0)
    assert winner_state in mes_res
    assert mes_res[winner_state] > 0.99


@pytest.mark.parametrize("exact", [True, False])
def test_grovers_all_states_tagged(exact):
    """Tests Grover's algorithm when all states are tagged by the oracle."""

    def all_tagged_oracle(qv):
        for i in range(2**qv.size):
            tag_state({qv: i})

    qv = QuantumFloat(3)
    grovers_alg(qv, all_tagged_oracle, winner_state_amount=2**qv.size, exact=exact)

    mes_res = qv.get_measurement()
    assert len(mes_res) == 8
    assert all(np.isclose(prob, 0.125, atol=1e-5) for prob in mes_res.values())


def test_grovers_exact_missing_winner_amount():
    """Tests that Grover's algorithm raises a ValueError when exact mode is enabled but winner_state_amount is not provided."""
    qv = QuantumVariable(2)

    def dummy_oracle(qv):
        pass

    # Assert that the correct ValueError is raised with the expected message
    with pytest.raises(
        ValueError, match="Exact Grover's algorithm requires 'winner_state_amount'"
    ):
        grovers_alg(qv, dummy_oracle, exact=True, winner_state_amount=None)


# https://github.com/eclipse-qrisp/Qrisp/issues/586
def test_grovers_oracle_with_side_effects():
    """Tests Grover's algorithm with an oracle that has side effects."""

    def prepare_oracle(qv, forbidden_value):

        @auto_uncompute
        def oracle(qv, phase=np.pi):

            flag = qv != forbidden_value
            p(phase, flag)

        grovers_alg(qv, oracle, winner_state_amount=2**qv.size - 1, exact=True)

    a = QuantumFloat(2)
    h(a)
    b = QuantumFloat(2)
    prepare_oracle(b, forbidden_value=a)

    mes_res = multi_measurement([a, b])
    assert_valid_measurement(mes_res)
    forbidden_states = {(0, 0), (1, 1), (2, 2), (3, 3)}
    assert all(state not in mes_res for state in forbidden_states)
