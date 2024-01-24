"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

# Created by ann81984 at 06.05.2022
import numpy as np

from qrisp.misc import multi_measurement
from qrisp.core import QuantumSession
from qrisp.arithmetic import QuantumFloat, sbp_mult
from qrisp.grover import tag_state, grovers_alg
from qrisp.environments import invert
from itertools import product
from qrisp import p, QuantumVariable, h, mcx, QuantumBool
import time

def test_grovers_algorithm():
    qf1 = QuantumFloat(2)
    qf2 = QuantumFloat(2)

    # Put into list
    qf_list = [qf1, qf2]

    # Create test oracle (simply tags the states where qf1 = -3 and qf2 = 2)
    def test_oracle(qf_list):
        tag_dic = {qf_list[0]: 0, qf_list[1]: 1}
        tag_state(tag_dic)

    # Apply grovers algorithm
    grovers_alg(qf_list, test_oracle)

    print(qf_list[0].qs)
    # Perform measurement
    mes_res = multi_measurement(qf_list)
    print(mes_res)

    # check if it is dictionary
    assert isinstance(mes_res, dict)

    # check if the key value pairs are of the type tuple:float
    # Example: (0, 1) :  0.9592
    assert all(isinstance(item, tuple) for item in mes_res.keys()) and all(
        (isinstance(item, float) and item < 1) for item in mes_res.values()
    )

    # Check measurement outcome
    assert list(mes_res.keys())[0] == (0, 1)

    # check if the first tuple in the tuple contains expected values. Style example of the whole tuple: ((0, 1), 0.9592)
    assert all(
        (item in [ele for ele in product(range(0, 4), repeat=2)])
        for item in mes_res.keys()
    )

    ###################
    # Create quantum floats
    qf1 = QuantumFloat(2, -1, signed=True)
    qf2 = QuantumFloat(2, -1, signed=True)

    # Put into list
    qf_list = [qf1, qf2]

    # Create test oracle (tags the states where the multiplication results in -0.25)
    def equation_oracle(qf_list):
        # Set aliases
        factor_0 = qf_list[0]
        factor_1 = qf_list[1]

        # Calculate ancilla value
        ancilla = factor_0 * factor_1

        # Tag state
        tag_dic = {ancilla: -0.25}
        tag_state(tag_dic)

        # Uncompute ancilla
        with invert():
            sbp_mult(factor_0, factor_1, ancilla)

        ancilla.delete()

    # Execute grovers algorithm
    grovers_alg(qf_list, equation_oracle, winner_state_amount=2)

    # Perform measurement

    mes_res = multi_measurement(qf_list)
    print(mes_res)

    # check if it is a list of dictionary
    assert isinstance(mes_res, dict)

    # check if the key value pairs are of the type tuple:float
    # Example: (0, 1) :  0.9592
    assert all(isinstance(item, tuple) for item in mes_res.keys()) and all(
        (isinstance(item, float) and item < 1) for item in mes_res.values()
    )

    assert list(mes_res.keys())[0] == (0.5, -0.5)
    assert list(mes_res.keys())[1] == (-0.5, 0.5)

    # check if the first tuple in the tuple contains expected values. Style example of the whole tuple: ((0, 1), 0.9592)
    assert all(
        (
            item
            in [
                ele
                for ele in product(
                    [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0], repeat=2
                )
            ]
        )
        for item in mes_res.keys()
    )

    # Test exact Grovers alg

    def oracle(qv, phase=np.pi):
        temp_qbl = QuantumBool()
        mcx(qv[1:], temp_qbl)

        p(phase, temp_qbl)

        temp_qbl.uncompute()

    qv = QuantumVariable(6)
    grovers_alg(qv, oracle, exact=True, winner_state_amount=2)

    assert qv.get_measurement() == {"011111": 0.5, "111111": 0.5}