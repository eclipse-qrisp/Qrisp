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

import sympy as sp
from qrisp.logic_synthesis.truth_tables import TruthTable, synth_poly


def pprm_synth(input_var, output_var, tt, qb_nr, phase_tolerant=False):
    from qrisp import mcx
    from qrisp.arithmetic.poly_tools import (
        expr_to_list,
    )

    qs = input_var.qs
    output_qubit = output_var.reg[qb_nr]
    expr = synth_poly(tt, column=qb_nr)
    # print(expr)
    args = expr_to_list(expr)
    args_len = [len(i) for i in args]

    for element in range(len(args_len)):
        if args_len[element] > 1:
            from sympy.core.numbers import One

            if isinstance(args[element][0], One):
                args[element].pop(0)

            # input_qubits = [(-int(symb.name[1:])-1)%input_var.size for symb in args[element]]  # noqa
            input_qubits = [int(symb.name[1:]) for symb in args[element]]
            input_qubits = [input_var.reg[i] for i in input_qubits]

            product = 1
            for i in range(len(args[element])):
                product = product * (args[element][i])

            mul_tt = TruthTable(product)

            if len(product.args) == 0:
                if isinstance(product, sp.Symbol):
                    qs.cx(input_qubits[0], output_qubit)
                else:
                    qs.x(output_qubit)
            else:
                if phase_tolerant:
                    qs.append(
                        mul_tt.gate_synth(method="gray_pt"),
                        input_qubits + [output_qubit],
                    )
                else:
                    mcx(input_qubits, output_qubit, method="gray")
                    # qs.append(mul_tt.gate_synth(method = "gray"),input_qubits+[output_qubit])  # noqa
        else:
            if isinstance(args[element][0], sp.core.symbol.Symbol):
                arg_qubit = (-int(args[element][0].name[1:]) - 1) % input_var.size

                arg_qubit = input_var.reg[arg_qubit]
                qs.cx(arg_qubit, output_qubit)
            if isinstance(args[element][0], sp.core.numbers.One):
                qs.x(output_qubit)


def pprm(input_var, output_var, tt, phase_tolerant=False):
    for column in range(tt.shape[1]):
        pprm_synth(input_var, output_var, tt, column, phase_tolerant)
