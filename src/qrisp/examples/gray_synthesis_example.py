"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

# This example displays the logic synthesis capabilities of Qrisp
# The user-interface for this feautre is the QuantumDictionary

from qrisp.core import QuantumVariable
from qrisp.alg_primitives.logic_synthesis import TruthTable, gray_logic_synth  # , pprm
from qrisp.misc import int_encoder
from qrisp import h

# Create a testing truth table
tt = TruthTable(["00010101", "01001101", "01011100"])


for index in range(tt.shape[0]):
    # Create new quantum session
    # qs = QuantumSession()

    # Create input variable
    # input_var = QuantumVariable(tt.bit_amount, qs)
    input_var = QuantumVariable(tt.bit_amount)

    # Encode index

    int_encoder(input_var, index)

    # Create output variable
    # output_var = QuantumVariable(tt.shape[1], qs)
    output_var = QuantumVariable(tt.shape[1])

    # Perform logic synthesis
    gray_logic_synth(input_var, output_var, tt, phase_tolerant=False)
    # pprm(input_var, output_var, tt)

    print(input_var.qs.compile().depth())
    print(input_var.qs.compile().cnot_count())

    # Print results
    print("---")
    print("Expected bitstring", *tt.n_rep[index, :])
    print("Measured bitstring", list(output_var.get_measurement().keys())[0])


input_var = QuantumVariable(tt.bit_amount)

# Encode index
h(input_var)
# int_encoder(input_var, index)


# Create output variable
# output_var = QuantumVariable(tt.shape[1], qs)
output_var = QuantumVariable(tt.shape[1])


# Perform logic synthesis
gray_logic_synth(input_var, output_var, tt, phase_tolerant=False)
# pprm(input_var, output_var, tt, phase_tolerant = False)

# Check phases are correct
output_var.qs.statevector(plot=True)

output_var.qs.compile().depth()
