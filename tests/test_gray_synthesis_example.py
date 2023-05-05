"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""

# Created by ann81984 at 27.04.2022
import pytest

from qrisp.logic_synthesis import TruthTable, gray_logic_synth
from qrisp.misc import int_encoder
from qrisp.core import QuantumVariable, QuantumSession


@pytest.mark.parametrize("num", ["001", "011", "000", "100"])
def test_gray_synthesis_example(num):
    # split the input string into a list of characters
    tmp = list(num)

    # Create a testing truth table
    tt = TruthTable(tmp)

    # Create new quantum session
    qs = QuantumSession()

    # Create input variable
    input_var = QuantumVariable(tt.bit_amount, qs)

    # Encode index

    int_encoder(input_var, 0)

    # Create output variable
    output_var = QuantumVariable(tt.shape[1], qs)

    # Perform logic synthesis
    gray_logic_synth(input_var, output_var, tt, phase_tolerant=False)

    # Expected bitstring comes from tt in format array [0 0 1]
    # create string for assert comparison
    ttstr = "".join([str(item) for item in tt.n_rep[0, :]])
    mes_res = output_var.get_measurement()

    assert ttstr == str(list(mes_res.keys())[0])
    assert ttstr in ["001", "011", "000", "100"] and str(list(mes_res.keys())[0]) in [
        "001",
        "011",
        "000",
        "100",
    ]
