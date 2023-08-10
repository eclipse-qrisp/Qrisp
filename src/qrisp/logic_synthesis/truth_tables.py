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


import numpy as np
import sympy as sp

from qrisp.misc import int_as_array
from qrisp.circuit import Operation
#from qrisp import int_encoder, get_ordered_symbol_list


# from qiskit import *


# Class to describe truth tables
# Can be intialized with a list of bitstrings, or a numpy array with 1 and 0s or a
# sympy expression
class TruthTable:
    def __init__(self, init_object):
        # This handles the case that the initializing object is a sympy expression
        # the variables will be sorted alphabetically regarding their name
        if isinstance(init_object, sp.Expr):
            expr = init_object

            init_str = ""

            self.expr = expr
            from qrisp.arithmetic.poly_tools import get_ordered_symbol_list

            # Retrieve a list of symbols in a given expression
            # Reverse to get the correct endian convention for qiskit
            symbol_list = get_ordered_symbol_list(expr)[::-1]

            # Replace the variables in the expression with 1s and 0s and add them to the
            # intialization string
            from sympy import lambdify

            expr_func = lambdify([symbol_list], expr, "numpy")

            for i in range(2 ** len(symbol_list)):
                if len(symbol_list):
                    symbol_const = int_as_array(i, len(symbol_list))
                else:
                    symbol_const = []

                res = expr_func(symbol_const) % 2
                if not set({int(res)}) <= set([0, 1]):
                    raise Exception("Sympy expression returned non-boolean value")

                init_str += str(int(res))
            # this new object will now contain the string list which will be further
            # prosecuted in this function
            init_object = [init_str]

        # This handles the case that the truth table is instantiated as a list of
        # strings
        if isinstance(init_object, list):
            # Check if the list contains only bit strings of the same length
            for i in range(1, len(init_object)):
                if len(init_object[0]) != len(init_object[i]):
                    raise Exception(
                        "Tried to initialize truth table with varying lengths"
                    )

            # Save string list into object structure
            self.str_l = init_object

            # Generate numpy array representation
            self.n_rep = [[int(c) for c in singular_tt] for singular_tt in init_object]
            self.n_rep = np.array(self.n_rep).transpose()

        # This handles the case that the truth table is instantiated as a numpy array
        if isinstance(init_object, np.ndarray):
            # Save array into object structure
            self.n_rep = init_object

            # Generate list of bitstrings
            self.str_l = [
                "".join(
                    [str(int(init_object[i, j])) for i in range(init_object.shape[0])]
                )
                for j in range(init_object.shape[1])
            ]

        # Check some conditions
        if not (set(self.n_rep.flatten()) <= set([0, 1])):
            raise Exception("Tried to initialize truth table with a non boolean values")

        # Set shortcuts to some prevalent parameters of the truth table
        self.shape = self.n_rep.shape

        if not (int(np.log2(self.shape[0])) == np.log2(self.shape[0])):
            raise Exception(
                "Tried to initialize truth table with a boolean function with a "
                "length which is not an integer power of 2"
            )

        self.bit_amount = int(np.log2(self.shape[0]))

    # Returns the "left side" of the truth table, ie. the variable values
    # (starting with 00..00, 00..01, 00..10,...)
    def variable_array(self):
        variable_table_array = [
            int_as_array(i, int(np.log2(self.shape[0]))) for i in range(self.shape[0])
        ]
        variable_table_array = np.array(variable_table_array)

        return TruthTable(variable_table_array)

    # Returns the cofactors of the truth table
    # If the truth table describes a boolean function f(x0,x1,x2...)
    # Then the cofactors for index 2 is f(x0,x1,0,...) and f(x0,x1,1,...)
    def cofactors(self, i):
        # Set up lists for the values of the cofactor tables
        cofactor_tables = [[], []]

        # Iterate over all value constellations and put them either
        # into the "first" or the "second" cofactor
        for k in range(self.shape[0]):
            if int_as_array(k, self.bit_amount)[i]:
                cofactor_tables[1].append(self.n_rep[k, :])
            else:
                cofactor_tables[0].append(self.n_rep[k, :])

        return [
            TruthTable(np.array(cofactor_tables[0])),
            TruthTable(np.array(cofactor_tables[1])),
        ]

    # Returns a single column as a truth tables
    def sub_table(self, i):
        return TruthTable(np.array([self.n_rep[:, i]]).transpose())

    # Method to print truth tables
    def __str__(self):
        return str(np.array(self.n_rep))

    # Swaps to columns of the truth table
    def swap_col(self, i, j):
        temp_array = np.array(self.n_rep)
        temp_col = self.n_rep[:, i]
        temp_array[:, i] = temp_array[:, j]
        temp_array[:, j] = temp_col
        return TruthTable(temp_array)

    # Calculates the complexity of the truth table (more to that in the definition
    # of "D")
    def calc_complexity(self, as_array=False):
        if as_array:
            return np.array([D(self.n_rep[:, i]) for i in range(len(self.str_l))])

        c = 0

        for i in range(len(self.str_l)):
            c += D(self.n_rep[:, i])

        return -c

    # Synthesizes a ciruict which represents the truth table
    def q_synth(self, input_var, output_var, method="gray"):
        if output_var.size != self.shape[1]:
            raise Exception(
                "Given output variable doesn't include the required amount of qubits"
            )

        if len(input_var) != self.bit_amount:
            raise Exception(
                "Given input variable doesn't include the required amount of qubits"
            )

        # Use gray synthesis to synthesize truth table
        if method == "gray":
            from qrisp.logic_synthesis.gray_synthesis import gray_logic_synth

            gray_logic_synth(input_var, output_var, self, phase_tolerant=False)

        # Use phase tolerant gray synthesis to synthesize truth table
        elif method == "gray_pt":
            from qrisp.logic_synthesis.gray_synthesis import gray_logic_synth

            gray_logic_synth(input_var, output_var, self, phase_tolerant=True)

        elif method == "gray_pt_inv":
            from qrisp.logic_synthesis.gray_synthesis import gray_logic_synth
            from qrisp.misc import quantum_invert

            quantum_invert(
                gray_logic_synth, [input_var, output_var, self, True], input_var.qs
            )

        elif method == "pprm_pt":
            from qrisp.logic_synthesis.pprm_synthesis import pprm

            pprm(input_var, output_var, self, phase_tolerant=True)

        elif method == "pprm":
            from qrisp.logic_synthesis.pprm_synthesis import pprm

            pprm(input_var, output_var, self, phase_tolerant=False)

        elif method == "best":
            input_var.qs.append(
                self.gate_synth(), list(input_var.reg) + list(output_var.reg)
            )

        else:
            raise Exception("Given synthesis method unknown")

        # Append the truth table to the generated operation, so that it can be
        # resynthesized phase tolerantly during uncomputation
        # input_var.qs.data[-1].op.tt = self
        # input_var.qs.data[-1].op.logic_synth_method = method

        input_var.qs.data[-1].op = LogicSynthGate(
            input_var.qs.data[-1].op, self, phase_tolerant=method
        )

    def gate_synth(self, method="best", inv=False):
        from qrisp.core import QuantumSession, QuantumVariable

        if method == "best":
            qs_list = []
            methods = ["gray_pt", "td", "td_pk", "pprm_pt"]

            if self.bit_amount != 1:
                methods = ["gray_pt", "pprm_pt"]
            else:
                methods = ["pprm_pt"]

            for m in methods:
                qs = QuantumSession()
                input_var = QuantumVariable(self.bit_amount, qs)
                output_var = QuantumVariable(self.shape[1], qs)

                self.q_synth(input_var, output_var, method=m)

                qs_list.append(qs)

            cnot_list = [qs_list[i].cnot_count() for i in range(len(methods))]

            qs = qs_list[cnot_list.index(min(cnot_list))]

        else:
            qs = QuantumSession()

            input_var = QuantumVariable(self.bit_amount, qs)
            output_var = QuantumVariable(self.shape[1], qs)

            self.q_synth(input_var, output_var, method=method)

        if not inv:
            return qs.to_gate()
        else:
            return qs.to_gate().inverse()

    def __or__(self, other):
        if self.shape != other.shape:
            raise Exception("Tried to or two truth tables of different shape")

        new_tt = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                new_tt[i, j] = self.n_rep[i, j] or other.n_rep[i, j]

        return TruthTable(new_tt)


T_bib = {}


# Implementation of the truth table complexity measure described in Miller, D. M.,
# “Spectral and Two-Place Decomposition Techniques in Reversible Logic,”
# Proc. Midwest Symposium on Circuits and Systems, on CD-ROM, August 2
def T(n):
    global T_bib

    if n in T_bib.keys():
        return T_bib[n]

    if n == 0:
        return np.array([[1]])

    result = np.zeros((2**n, 2**n))

    temp = T(n - 1)

    result[: 2 ** (n - 1), : 2 ** (n - 1)] = temp
    result[2 ** (n - 1) :, : 2 ** (n - 1)] = temp
    result[: 2 ** (n - 1), 2 ** (n - 1) :] = temp
    result[2 ** (n - 1) :, 2 ** (n - 1) :] = -temp

    T_bib[n] = result
    return result


def fwht(a):
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def rw_spectrum(f):
    if isinstance(f, str):
        f = [int(c) for c in f]
        f = np.array(f)

    a = f.copy()
    fwht(a)
    return a

    size = len(f)

    if np.log2(size) != int(np.log2(size)):
        raise Exception(
            "The given function does not have the length to properly represent "
            "a truth table"
        )

    n = int(np.log2(size))

    return np.dot(T(n), f)


def C(f):
    size = len(f)

    if np.log2(size) != int(np.log2(size)):
        raise Exception(
            "The given function does not have the length to properly represent "
            "a truth table"
        )

    n = int(np.log2(size))

    rw_spec = rw_spectrum(f)
    sum_ = 0
    for i in range(size):
        sum_ += sum(int_as_array(i, n)) * rw_spec[i] ** 2

    sum_ = sum_ / 2 ** (n - 2)

    return 1 / 2 * (n * size - sum_)


def NZ(f):
    R = rw_spectrum(f)

    sum_ = 0
    for i in R:
        if i == 0:
            sum_ += 1
    return sum_


def D(f):
    size = len(f)
    if np.log2(size) != int(np.log2(size)):
        raise Exception(
            "The given function does not have the length to properly represent "
            "a truth table"
        )

    n = int(np.log2(size))

    return int(n * 2 ** (n - 3) * NZ(f) + C(f))


def synth_poly(truth_table, column=0, coeff=None):
    if coeff is None:
        coeff = sp.symbols(
            "".join([" x" + str(i) for i in range(truth_table.bit_amount)])
        )
        if truth_table.bit_amount == 1:
            coeff = [coeff]

    try:
        expr = truth_table.expr
        symbols = get_ordered_symbol_list(expr)
        coeff_temp = sp.symbols(
            "".join(
                [" abcdefghijkllmn" + str(i) for i in range(truth_table.bit_amount)]
            )
        )
        subs_dic = {symbols[i]: coeff_temp[i] for i in range(len(coeff))}
        expr = expr.subs(subs_dic)
        subs_dic = {coeff_temp[i]: coeff[i] for i in range(len(coeff))}
        expr = expr.subs(subs_dic).expand()
        # print(subs_dic)
        # print(sp.Poly(expr.subs(subs_dic).expand(), domain = sp.GF(2)))
        return sp.Poly(expr.subs(subs_dic).expand(), domain=sp.GF(2)).expr
    except:
        pass

    poly = sp.sympify(0)
    for i in range(truth_table.shape[0]):
        if truth_table.n_rep[i, column]:
            temp = sp.sympify(1)
            array = int_as_array(i, truth_table.bit_amount)
            for j in range(len(array)):
                if array[j]:
                    temp *= coeff[-j - 1]
                else:
                    temp *= coeff[-j - 1] + 1
            poly += temp

    if poly == 0:
        return poly
    return sp.Poly(poly.expand(), domain=sp.GF(2)).expr
    # return filter_pow(sp.Poly(poly.expand(), domain = sp.GF(2)).expr)


class LogicSynthGate(Operation):
    def __init__(self, init_op, tt, phase_tolerant=False):
        self.tt = tt

        self.logic_synth_method = phase_tolerant

        for var in init_op.__dict__.keys():
            if var != "inverse":
                self.__dict__[var] = init_op.__dict__[var]

    def inverse(self):
        return LogicSynthGate(Operation.inverse(self), self.tt, self.logic_synth_method)


def check_synthesis(tt, gate, log_output=False):
    synth_correct = True
    from qrisp.core import QuantumSession, QuantumVariable

    for i in range(tt.shape[0]):
        qs = QuantumSession()
        input_var = QuantumVariable(tt.bit_amount, qs)
        output_var = QuantumVariable(gate.num_qubits - tt.bit_amount, qs)
        int_encoder(input_var, i)
        qs.append(gate, range(gate.num_qubits))

        tt_value = "".join([str(x) for x in tt.n_rep[i]])
        synth_res = list(output_var.get_measurement().keys())[0]

        if log_output:
            print("TruthTable:", tt_value)
            print("Synthesis:", synth_res)
            print("---")

        synth_correct = synth_correct and tt_value == synth_res

    return synth_correct
