"""
********************************************************************************
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
********************************************************************************
"""

import numpy as np
import sympy as sp
from sympy.polys.polytools import degree

from qrisp.alg_primitives.arithmetic.poly_tools import (
    expr_to_list,
    filter_pow,
    get_ordered_symbol_list,
)
from qrisp.core import (
    QuantumArray,
    QuantumVariable,
    cp,
    cx,
    cz,
    h,
    mcx,
    p,
    z,
    rz,
    rzz,
    crz,
    mcp,
    gphase,
)
from qrisp.misc import gate_wrap, lifted
from qrisp.circuit import XGate

# Threshold of rounding used in detecting integer multiples of pi
pi_mult_round_threshold = 11

depth_tracker = {}


# Check if the given expression is a polynomial
def check_for_polynomial(expr):
    for x in sp.preorder_traversal(expr):
        if not isinstance(
            x,
            (
                sp.core.mul.Mul,
                sp.core.add.Add,
                sp.core.symbol.Symbol,
                sp.core.numbers.Integer,
                sp.core.numbers.Float,
                sp.core.power.Pow,
            ),
        ):
            return False

        if isinstance(x, sp.core.power.Pow):
            if not isinstance(x.args[1], sp.core.numbers.Integer):
                return False
    return True


# Efficient implementation of the multicontrolled U_g gate
def multi_controlled_U_g(
    output_qf, control_qb_list, y, phase_tolerant=False, use_gms=False
):
    # Set alias for quantum session
    # qs = output_qf.qs
    qs = control_qb_list[0].qs()

    # For one control qubit, there is no advantage in calculating an ancilla qubit
    if len(control_qb_list) == 1:
        ancilla = control_qb_list
    # Otherwise request an ancilla qubit
    else:
        ancilla = QuantumVariable(1, qs=output_qf.qs, name="sbp_anc*")

    # If an ancilla qb is available
    # we synthesize the result of the boolean multiplication in it

    if len(control_qb_list) != 1:
        if use_gms:
            toffoli_method = "gms"
        else:
            toffoli_method = "gray_pt"

        # Apply multi-controlled x gate
        mcx(control_qb_list, ancilla[0], method=toffoli_method)

    # Now we execute the phase gates
    # For this there is 2 possibilies
    # Either use regular phase gates or use GMS gates
    # GMS gates are ion-trap native gates, that allow the entanglement
    # of multiple qubits within a single laser pulse

    from qrisp.environments import (
        QuantumEnvironment,
        control,
        GMSEnvironment,
        custom_control,
    )

    if use_gms:
        # GMSEnvironment is an environment that allows programming
        # with regular phase gates but converts everything programmed
        # into a GMS gate upon leaving the environment

        env = GMSEnvironment(qs)

    else:
        # This is an environment that simply does nothing
        env = QuantumEnvironment()

    # Enter environment
    env.manual_allocation_management = True

    @custom_control
    def crz_helper(rot_angle, a, b, ctrl=None, use_gms=False):
        if ctrl is None and not use_gms:
            cx(a, b)
            p(-rot_angle / 2, b)
            cx(a, b)
            p(rot_angle / 2, b)
        elif use_gms:
            crz(rot_angle, ancilla[0], output_qf[i])
        else:
            cx(a, b)
            cp(-rot_angle / 2, ctrl, b)
            cx(a, b)
            cp(rot_angle / 2, ctrl, b)

    with env:
        phase_accumulator = 0
        cz_counter = 0

        # Execute controlled rotations
        for i in range(len(output_qf)):
            # The -i (instead of i) reverses the gate order implying that
            # we can leave out the swaps of the QFT

            rot_angle = 2 * np.pi * 2 ** (-i - 1) * y
            rot_angle = rot_angle % (2 * np.pi)

            if (
                np.round(rot_angle, pi_mult_round_threshold) != 0
                and np.round((-rot_angle) % (2 * np.pi), pi_mult_round_threshold) != 0
            ):
                # If phase is equal to pi, execute cz gate (costs one CNOT less than CP)
                if False:
                    # if np.round(abs(rot_angle) - np.pi, pi_mult_round_threshold) == 0:
                    # cz(ancilla[0], output_qf.reg[i])

                    z(output_qf.reg[i])
                    cz_counter += 1

                    phase_accumulator += rot_angle / 2
                # Otherwise execute cp gate
                else:
                    crz_helper(rot_angle, ancilla[0], output_qf[i], use_gms=use_gms)
                    phase_accumulator += rot_angle / 2

        p(phase_accumulator - np.pi * cz_counter / 2, ancilla[0])

    # Uncompute boolean multiplication
    if len(control_qb_list) != 1:
        if not use_gms:
            toffoli_method += "_inv"

        mcx(control_qb_list, ancilla[0], method=toffoli_method)
        ancilla.delete()

    return phase_accumulator


# This function returns the U_g gate (up to qiskit endian conventions)
# This is used for applying U_g gates which have no control-knobs
def U_g(y, qv):
    for i in range(len(qv)):
        rot_angle = np.pi * 2 ** (-i) * y
        rot_angle = rot_angle % (2 * np.pi)
        if (
            np.round(rot_angle, pi_mult_round_threshold) != 0
            and np.round((-rot_angle) % (2 * np.pi), pi_mult_round_threshold) != 0
        ):
            p(rot_angle, qv[i])


# This function encodes a semi-boolean polynomial into a circuit i.e.
# For the polynomial p(x_0, x_1, x_2) = 4*x_0*x_1 + 2*x_0*x_2
# the effect of this function is:
# U_f |x_0, x_1, x_2>|0> = |x_0, x_1, x_2>|p(x_1,x_2,x_3)>
def sb_polynomial_encoder(
    input_qf_list, output_qf, poly, inplace_mult=1, use_gms=False, init_op="auto"
):
    # As the polynomial has only boolean variables,
    # powers can be ignored since x**k = x for x in GF(2)
    poly = filter_pow(poly.expand()).expand() / 2.0**output_qf.exponent

    # Acquire list of symbols present in the polynomial
    symbol_list = []

    for qf in input_qf_list:
        temp_var_list = list(sp.symbols(str(hash(qf)) + "_" + "0:" + str(qf.size)))
        symbol_list += temp_var_list

    n = len(symbol_list)

    if n != sum([var.size for var in input_qf_list]):
        raise Exception(
            "Input variables do not the required amount of qubits to encode polynomial"
        )

    # Acquire monomials in list form
    monomial_list = expr_to_list(poly)

    from qrisp import QFT, conjugate, QuantumEnvironment

    if inplace_mult == 1:
        env = conjugate(QFT)(
            output_qf,
            inv=False,
            exec_swap=False,
            inplace_mult=inplace_mult,
            use_gms=use_gms,
        )
    else:
        QFT(
            output_qf,
            inv=False,
            exec_swap=False,
            inplace_mult=inplace_mult,
            use_gms=use_gms,
        )
        env = QuantumEnvironment()

    with env:
        # The list of qubits contained in the variables of input_var_list
        input_qubits = sum([list(var.reg) for var in input_qf_list], [])

        control_qubit_list = []
        y_list = []

        # Iterate through the monomials
        for monom in monomial_list:
            # Prepare the two variables coeff (which is the coefficient of the monomial)
            # And the list of variables which appear in the monomial

            # For this, go through the cases which can appear

            # This describes the case where there is only a single term in the monomial
            # Either a constant or a variable
            if len(monom) == 1:
                if isinstance(monom[0], sp.core.symbol.Symbol):
                    coeff = 1
                    variables = list(monom)
                else:
                    coeff = float(monom[0])
                    variables = []

            # This describes the case where there is multiple terms in the monomial
            elif not isinstance(monom[0], sp.core.symbol.Symbol):
                coeff = monom[0]
                variables = list(monom[1:])
            else:
                coeff = 1
                variables = list(monom)

            # Check if the coefficient is an integer (up to float errors)
            if abs(int(np.round(float(coeff))) - coeff) > 1e-14:
                pass
                # raise Exception("Tried to encode sb-polynomial
                # with non-integer coefficient")

            # Append coefficient to y_list
            y_list.append(int(np.round(float(coeff))))

            # Prepare the qubits on which the U_g should be controlled
            control_qubit_numbers = [symbol_list.index(var) for var in variables]

            control_qubits = [input_qubits[nr] for nr in control_qubit_numbers]

            control_qubits = list(set(control_qubits))

            control_qubits.sort(key=lambda x: x.identifier)

            control_qubit_list.append(control_qubits)

        # Now we apply the multi controlled U_g gate
        # Here the order in which they are applied makes a huge difference
        # In order to determine, which U_g to apply next, we evaluate a cost function
        # (which we determined through trial and error)
        # and choose the U_g with the lowest cost

        # Note that for this feature to yield an improvement, the quantum session requires
        # multiple free ancilla qubits to work on
        def delay_cost(depth_array):
            return max(depth_array)

        def find_best_monomial(control_qubit_list, depth_dic):
            delay_cost_list = []

            for i in range(len(control_qubit_list)):
                depth_array = np.array([depth_dic[qb] for qb in control_qubit_list[i]])

                delay_cost_list.append(delay_cost(depth_array))

            return np.argmin(delay_cost_list)

        # Iterate through the list of U_g gates
        while control_qubit_list:
            # TO-DO fix depth calculation inside environment
            # Update depth_dic (contains the depth of each qubit)
            # depth_dic = output_qf.qs.get_depth_dic()

            # Determine best U_g
            # monomial_index = find_best_monomial(control_qubit_list, depth_dic)
            monomial_index = 0
            # Find control qubits and their coefficient
            control_qubits = control_qubit_list.pop(monomial_index)
            y = y_list.pop(monomial_index)

            # Apply (controlled) U_g
            if len(control_qubits):
                multi_controlled_U_g(output_qf, control_qubits, y, use_gms=use_gms)
            else:
                U_g(y, output_qf)

    if inplace_mult != 1:
        # Apply QFT
        QFT(output_qf, inv=True, exec_swap=False, use_gms=use_gms)


# Multiplies two integers
# The general idea is to represent the integers as values of two polynomials
# p(x) = x_0 + 2*x_1 + 4*x_2 ...
# p(y) = y_0 + 2*y_1 + 4*y_2 ...
# these polynomials get multiplied and which forms a bigger polynomial
# which can be encoded using the polynomial encoder
def sbp_mult(factor_1_qf, factor_2_qf, output_qf=None):
    """
    Performs multiplication based on the evaluation of
    `semi-boolean polynomials <https://ieeexplore.ieee.org/document/9815035>`_.

    Parameters
    ----------
    factor_1_qf : QuantumFloat
        The first factor to multiply.
    factor_2_qf : QuantumFloat
        The second factor to multiply.
    output_qf : QuantumFloat, optional
        The QuantumFloat to store the result in.
        By default, a suited new QuantumFloat is created.

    Returns
    -------
    output_qf : QuantumFloat
        A QuantumFloat containing the result of the multiplication.

    Examples
    --------

    We multiply two QuantumFloats:

    ::

        from qrisp import QuantumFloat, sbp_mult
        qf_0 = QuantumFloat(3)
        qf_1 = QuantumFloat(3)
        qf_0[:] = 3
        qf_1[:] = 4
        qf_res = sbp_mult(qf_0, qf_1)
        print(qf_res)


    ::

        #Yields: {12: 1.0}


    """

    if output_qf is None:
        from qrisp.qtypes.quantum_float import create_output_qf

        output_qf = create_output_qf([factor_1_qf, factor_2_qf], op="mul")
    # Multiply the polynmials
    mult_poly = factor_1_qf.sb_poly(output_qf.msize) * factor_2_qf.sb_poly(
        output_qf.msize
    )

    # Apply sb encoder
    sb_polynomial_encoder([factor_1_qf, factor_2_qf], output_qf, mult_poly)

    return output_qf


def sbp_add(summand_1_qf, summand_2_qf, output_qf=None):
    """
    Performs addition based on the evaluation of
    `semi-boolean polynomials <https://ieeexplore.ieee.org/document/9815035>`_.

    Parameters
    ----------
    summand_1_qf : QuantumFloat
        The first summand to add.
    summand_2_qf : QuantumFloat
        The second summand to add.
    output_qf : QuantumFloat, optional
        The QuantumFloat to store the result in.
        By default, a suited new QuantumFloat is created.

    Returns
    -------
    output_qf : QuantumFloat
        A QuantumFloat containing the result of the addition.

    Examples
    --------

    We add two QuantumFloats:

    ::

        from qrisp import QuantumFloat, sbp_add
        qf_0 = QuantumFloat(3)
        qf_1 = QuantumFloat(3)
        qf_0[:] = 3
        qf_1[:] = 4
        qf_res = sbp_add(qf_0, qf_1)
        print(qf_res)


     ::

        # Yields: {7: 1.0}


    """

    if output_qf is None:
        from qrisp.qtypes.quantum_float import create_output_qf

        output_qf = create_output_qf([summand_1_qf, summand_2_qf], op="add")

    sum_poly = summand_1_qf.sb_poly(output_qf.msize) + summand_2_qf.sb_poly(
        output_qf.msize
    )

    # Apply sb encoder
    sb_polynomial_encoder([summand_1_qf, summand_2_qf], output_qf, sum_poly)

    return output_qf


def sbp_sub(summand_1_qf, summand_2_qf, output_qf=None):
    """
    Performs subtraction based on the evaluation of
    `semi-boolean polynomials <https://ieeexplore.ieee.org/document/9815035>`_.

    Parameters
    ----------
    summand_1_qf : QuantumFloat
        The QuantumFloat to subtract from.
    summand_2_qf : QuantumFloat
        The QuantumFloat to subtract.
    output_qf : QuantumFloat, optional
        The QuantumFloat to store the result in.
        By default, a suited new QuantumFloat is created.

    Returns
    -------
    output_qf : QuantumFloat
        A QuantumFloat containing the result of the subtraction.

    Examples
    --------

    We add two QuantumFloats:

    ::

        from qrisp import QuantumFloat, sbp_sub
        qf_0 = QuantumFloat(3)
        qf_1 = QuantumFloat(3)
        qf_0[:] = 3
        qf_1[:] = 4
        qf_res = sbp_sub(qf_0, qf_1)
        print(qf_res)


    ::

        # Yields: {-1: 1.0}


    """

    if output_qf is None:
        from qrisp.qtypes.quantum_float import create_output_qf

        output_qf = create_output_qf([summand_1_qf, summand_2_qf], op="sub")

    dif_poly = summand_1_qf.sb_poly(output_qf.msize) - summand_2_qf.sb_poly(
        output_qf.msize
    )

    # Apply sb encoder
    sb_polynomial_encoder([summand_1_qf, summand_2_qf], output_qf, dif_poly)

    return output_qf


# Encode the polynomial given in poly in output var
# depending on the input variables qv_list
@gate_wrap(is_qfree=True, permeability=[0])
def polynomial_encoder(qf_list, output_qf, poly, encoding_dic=None, inplace_mult=1):
    """
    Evaluates a (multivariate) sympy polynomial on a list of QuantumFloats using
    `semi-boolean polynomials <https://ieeexplore.ieee.org/document/9815035>`_.

    Parameters
    ----------
    qf_list : list[QuantumFloat]
        The list of QuantumFloats to evaluate the polynomial on.
    output_qf : QuantumFloat
        The QuantumFloat to evaluate into.
    poly : sympy expression
        The polynomial to evaluate.
    encoding_dic : dict, optional
        A dictionary which has the QuantumFloats of qf_list as keys and the associated
        sympy symbols as values. By default, the symbols of the polynomial
        will be ordered alphabetically and then matched to the order in qf_list.
    inplace_mult : int, optional
        This integer allow to perform an inplace multiplication on output_qf,
        before the polynomial is evaluated.
        Note that due to reversibility only odd numbers are supported.

    Raises
    ------
    Exception
        Provided QuantumFloat list does not include the appropriate amount of elements
        to encode given polynomial".

    Returns
    -------
    None.

    Examples
    --------

    We evaluate the polynomial $x^2 + 2y^2$ on two QuantumFloats:


    ::

        from sympy import Symbol
        x = Symbol("x")
        y = Symbol("y")
        poly = x**2 + 2*y**2
        from qrisp import QuantumFloat, polynomial_encoder
        x_qf = QuantumFloat(3)
        y_qf = QuantumFloat(3)
        x_qf[:] = 3
        y_qf[:] = 2
        res_qf = QuantumFloat(7)
        encoding_dic = {x_qf : x, y_qf : y}
        polynomial_encoder([x_qf, y_qf], res_qf, poly, encoding_dic)
        print(res_qf)


    ::

        # Yields: {17.0: 1.0}


    """

    if isinstance(qf_list, QuantumArray):
        qf_list = list(qf_list.flatten().qv_array)

    if encoding_dic is not None:
        symbol_list = [encoding_dic[qv.name] for qv in qf_list]

    else:
        symbol_list = get_ordered_symbol_list(poly)

    if len(symbol_list) != len(qf_list):
        raise Exception(
            "Provided QuantumFloat list does not include the appropriate amount"
            "of elements to encode given polynomial"
        )

    if not output_qf.signed:
        for qf in qf_list:
            if qf.signed:
                raise Exception(
                    "When encoding into an unsigned quantum float"
                    "provide only unsigned inputs"
                )

    sb_poly_list = [qf.sb_poly(output_qf.size) for qf in qf_list]

    repl_dic = {symbol_list[i]: sb_poly_list[i] for i in range(len(qf_list))}
    # Substitute SB-polynomials
    sb_polynomial = poly.subs(repl_dic).expand()

    # Apply sb encoder
    sb_polynomial_encoder(qf_list, output_qf, sb_polynomial, inplace_mult=inplace_mult)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


# Performs inplace addition on a QFT-ed variable
# by applying successive U_g gates
def U_g_inpl_adder(modified_var, summand, mult_factor=1):
    applied_phases = []
    # Extract the coefficients of the SB-polynomial of x
    from sympy import Poly

    summand_coeffs = list(Poly(summand.sb_poly(modified_var.msize)).as_dict().values())
    summand_coeffs = [float(coeff) for coeff in summand_coeffs][::-1]
    # summand_coeffs.sort()

    summand_coeffs = summand_coeffs + (summand.size - len(summand_coeffs)) * [0]

    for j in range(summand.size):
        if summand_coeffs[j] == 0:
            continue
        applied_phases.append(
            multi_controlled_U_g(
                modified_var,
                [summand[j]],
                mult_factor * summand_coeffs[j] * 2**-modified_var.exponent,
            )
        )

    return applied_phases


# This algorithm is based on the equation X*U_g(y)*X = U_g(-y)*G with G as global phase
# This is used to perform conditional addition/subtraction required for multiplication
# using algorithm 3 from https://arxiv.org/abs/2112.10537
# s = x << (n + 1)
# s− = x
# for i in range(y.size):
#   if y[i]:
#       s+= (x << i)
#   else:
#       s−= (x << i)
# return s >> 1
# We will use the function U_g_inply_adder for the inplace additions
# while performing conditional additions using CNOT gates.
# The approach is therefore a hybrid of SBP-method and traditional logic approaches
def hybrid_mult(
    x,
    y,
    output_qf=None,
    init_op="h",
    terminal_op="qft",
    phase_tolerant=False,
    cl_factor=1,
):
    """
    An advanced algorithm for multiplication which has better depth, gate-count
    and compile time than :meth:`sbp_mult <qrisp.sbp_mult>`.
    It does not support squaring a single QuantumFloat though.

    This algorithm also operates on the Fourier transform. Because of this,
    between successive multiplications targeting the same QuantumFloat
    it is not neccessary to Fourier-Transform.
    This advantage is expressed in the parameters init_op and terminal_op.
    These can be set to either 'h', 'qft' or None
    to leave out self canceling Fourier-transforms.

    Parameters
    ----------
    x : QuantumFloat
        The first factor to multiply.
    y : QuantumFloat
        The second factor to multiply.
    output_qf : QuantumFloat, optional
        The QuantumFloat to store the result in.
        By default a suited QuantumFloat is created.
    init_op : str, optional
        The operation to bring output_qf into it's Fourier-transform.
        The default is 'h'.
    terminal_op : str, optional
        The operation to bring output_qf back from it's Fourier-transform.
        The default is "qft".
    phase_tolerant : bool, optional
        If set to True, differing results introduce differing extra phases.
        This can be usefull to save resources incase this functions will get uncomputed.
        The default is False.
    cl_factor : float, optional
        Allows to multiply the result by a classical factor without any extra gates.
        The default is 1.

    Returns
    -------
    output_qf : QuantumFloat
        The QuantumFloat containing the result.

    Examples
    --------

    We multiply two QuantumFloat with eachother and an additional classical factor

    ::

        from qrisp import QuantumFloat, hybrid_mult
        qf_0 = QuantumFloat(3)
        qf_1 = QuantumFloat(3)
        qf_0[:] = 3
        qf_1[:] = 4
        qf_res = hybrid_mult(qf_0, qf_1, cl_factor = 2)
        print(qf_res)


    ::

        # Yields: {24: 1.0}


    """

    from qrisp import QFT, cx, h, merge, z

    # The two factors take asymetrical roles in this algorithm
    # This implies that there is most likely a prefered choice
    # for the roles depending on the size of the factors
    # Several trials showed that these roles work best
    if not x.size > y.size:
        x, y = y, x

    # We shift the exponent of both factors such that they are integers
    # and shift the result in the end, by the sum of both exponents
    # This allows convenient treatment of non-integer inputs
    # while only constructing the algorithm for integers
    x_exp = int(x.exponent)
    y_exp = int(y.exponent)

    x.exp_shift(-x_exp)
    y.exp_shift(-y_exp)

    # If no output_qf is given, create one
    from qrisp.qtypes.quantum_float import create_output_qf

    if isinstance(output_qf, type(None)):
        output_qf = create_output_qf([y, x], op="mul")

    else:
        output_qf.exp_shift(-(x_exp + y_exp))

    output_qf.extend(1, position=0)

    # output_qf.exp_shift(-1)
    # Since the result is right shifted at the end, this implies that the zero-th qubit
    # of the output will not contain any information (otherwise this would imply,
    # that integer multiplications can yield non-integer results).
    # However the result still needs to be able to
    # display every possible result. This is why it needs "extra working-space"
    # We therefore extend it by one qubit

    # Merge the sessions of all involved variables
    merge([output_qf, x, y])

    # Perform initial operation
    # (check the general documentation of SBP arithmetic for more details)
    if init_op == "h":
        h(output_qf)
    elif init_op == "qft":
        QFT(output_qf, exec_swap=False)
    else:
        h(output_qf[0])

    # We treat the case that y is signed by applying a negation on output_qf that is
    # conditioned on the sign qubit of y (see command [2]). This will be reversed
    # at the end of the algorithm. This implies that every addition that happens
    # in between is actually a subtraction.
    # So this acts as a sign reversal of the result.
    # We then need to make sure we multiply with the absolute of y
    # We do this by applying a bitwise negation on the mantissa of y conditioned on
    # the sign qubit (command [4]). This bitwise negation acts as
    # NOT y = -y + 1
    # This implies we need to correct the additional +1.
    # This additional +1 is equivalent to an extra +x in the result
    # Therefore we need to subtract this +x from output_qf.
    # Note that this subtraction also has to be performed
    # conditioned on the sign qubit of y.
    # Instead of adding an additional control qubit, we again follow the strategy
    # of turning a conditional subtraction into a conditional subtraction/addition
    if y.signed:
        # This performs the first part of the conditional subtraction/addition
        # for j in range(x.size):
        # multi_controlled_U_g(output_qf, [x[j]], -x_coeffs[j])
        U_g_inpl_adder(output_qf, x, -1 * cl_factor)  # command [1]

        # Negate output_qf conditioned on the sign qubit of y
        cx(y[-1], output_qf)  # command [2]

        # This would perform the second part of the subtraction/addition
        # However this command is merged into command [5] for more performance gains.
        # If commented in, this command reverses command [1]
        # if output_qf has not been negated IF the sign qubit
        # has not negated output_qf. Otherwise the subtraction of x is completed.
        # for j in range(x.size):
        # multi_controlled_U_g(output_qf, [x[j]], x_coeffs[j]) #command [3]

        # Negate mantissa of y
        cx(y[-1], y[:-1])  # command [4]

    # This performs the initial two lines of the initially mentioned
    # multiplication algorithm:
    # s = x << (n + 1)
    # s− = x

    # Note that the boolean y.signed adds the phase that would have been added
    # in command [3]
    # however not requiring another round of U_g gates
    applied_phases = U_g_inpl_adder(
        output_qf, x, cl_factor * (2 ** (y.msize) - 1 + y.signed)
    )

    # We now come to the loop of the multiplication algorithm

    # for i in range(y.size):
    #   if y[i]:
    #       s+= (x << i)
    #   else:
    #       s−= (x << i)

    # Negate output_qf in order to perform the conditional addition/subtraction

    if not phase_tolerant and terminal_op != "qft":
        hybrid_mult_anc = QuantumVariable(1)
        if y.signed:
            cx(y[-1], hybrid_mult_anc[0])

            for k in range(len(applied_phases)):
                # cp(-applied_phases[k] * 2, hybrid_mult_anc[0], x[k])
                cp(-applied_phases[k] * 2, x[k], hybrid_mult_anc[0])

    cx(y[0], output_qf)
    for i in range(y.msize):
        # Perform in-place addition
        # (note that if y[i] is active an addition has to be performed).
        # Therefore, we need to perform a subtraction here,
        # because if y[i] is active output_qf has been negated
        applied_phases = U_g_inpl_adder(output_qf, x, -(2**i) * cl_factor)

        if not phase_tolerant and terminal_op != "qft":
            if i != 0:
                cx(y[i - 1], hybrid_mult_anc[0])
            cx(y[i], hybrid_mult_anc[0])

            for k in range(len(applied_phases)):
                # cp(-applied_phases[k] * 2, hybrid_mult_anc[0], x[k])
                cp(-applied_phases[k] * 2, x[k], hybrid_mult_anc[0])
                # sbp+= -2*applied_phases[k]*Symbol("y_" + str(i))*Symbol("x_" + str(k))

        # This command is equivalent to
        # cx(y[i], output_qf)
        # cx(y[i+1], output_qf)
        # ie. performs the output_qf negation but requires less cnot gates
        if i != y.msize - 1:
            cx(y[i], y[i + 1])
            cx(y[i + 1], output_qf)
            cx(y[i], y[i + 1])
        else:
            if not y.signed:
                # This command performs the final negation
                # if y is signed, we can perform the negation
                # together with the negation conditioned on the sign qubit
                # of y (command [6])
                cx(y[i], output_qf)  # command [5]

    if not phase_tolerant and terminal_op != "qft":
        cx(y[i], hybrid_mult_anc[0])
        if y.signed:
            cx(y[-1], hybrid_mult_anc[0])

        # from sympy import Symbol, simplify
        # temp_x = sum([2**i*Symbol("x_" + str(i)) for i in range(3)], 0)
        # temp_y = sum([2**i*Symbol("y_" + str(i)) for i in range(3)], 0)

        # sbp = sbp/(2*np.pi)

        # sbp = simplify((sbp + (temp_x*temp_y).expand()/2**7))
        # print(sbp)
        hybrid_mult_anc.delete()

    if y.signed:
        # This makes sure the negation from command [5] is performed
        cx(y[i], y[-1])

        cx(y[-1], output_qf)  # command[6]

        cx(y[i], y[-1])

        # Reverse command [4]
        cx(y[-1], y[:-1])

    # Free up the qubit which we identified as containing no information

    h(output_qf[0])
    output_qf.reduce(output_qf[0])

    # Perform terminal qft
    if terminal_op == "qft":
        QFT(output_qf, inv=True, exec_swap=False)

    if not phase_tolerant and terminal_op == "qft":
        # This is a phase tolerant correction using only single qubit gates.
        # We found it by looking at the sbp of the correction using cp gates and
        # identified the sbp of x*y in it.
        for i in range(output_qf.size):
            p(-(2 * np.pi) * 2**i / 2 ** (output_qf.size + 1), output_qf[i])

        if output_qf.signed:
            z(output_qf[-1])

    output_qf.exp_shift((x_exp + y_exp))

    # Perform exponent shifts

    x.exp_shift(x_exp)
    y.exp_shift(y_exp)

    return output_qf


# Wrapper for choosing the best multiplication algorithm
def q_mult(factor_1, factor_2, target=None, method="auto"):
    if method == "auto":
        if factor_1.reg == factor_2.reg:
            return q_mult(factor_1, factor_2, target, method="sbp")
        else:
            return q_mult(factor_1, factor_2, target, method="hybrid")

    elif method == "sbp":
        from qrisp.qtypes.quantum_float import create_output_qf

        if target is None:
            target = create_output_qf([factor_1, factor_2], op="mul")

        sbp_mult(factor_1, factor_2, target)
        return target

    elif method == "hybrid":
        from qrisp.alg_primitives.arithmetic import hybrid_mult

        return hybrid_mult(factor_1, factor_2)


def QFT_inpl_mult(qv, inplace_mult=1):
    from qrisp.misc import is_inv

    qv = list(qv)
    qv = qv[::-1]
    n = len(qv)

    if not is_inv(inplace_mult, n):
        raise Exception(
            "Tried to perform non-invertible inplace multiplication during Fourier-Transform"
        )

    # Perform QFT with inplace multiplication
    for i in range(n):
        if i != n - 1:
            h(qv[i])

        if i == n - 1:
            break

        for k in range(n - i - 1):
            if k + i + 1 != n - 1:
                cp(inplace_mult * 2 * np.pi / 2 ** (k + 2), qv[k + i + 1], qv[i])
            else:
                # The -1 here cancels some cp gates of the inverse qft
                cp((inplace_mult - 1) * 2 * np.pi / 2 ** (k + 2), qv[k + i + 1], qv[i])

    # Perform reversed QFT without inplace multiplication and without canceled steps
    for i in range(n)[::-1]:
        for k in range(n - i - 1)[::-1]:
            if k + i + 1 != n - 1:
                cp(-2 * np.pi / 2 ** (k + 2), qv[k + i + 1], qv[i])

        if i != n - 1:
            h(qv[i])

    return qv


def inpl_mult(qf, mult_int, treat_overflow=True):
    """
    Performs inplace multiplication of a :ref:`QuantumFloat` with a classical integer.
    To prevent overflow errors, this function automatically adjusts the mantissa size.
    If you want to prevent this behavior, set ``treat_overflow = False``.

    Parameters
    ----------
    qf : QuantumFloat
        The QuantumFloat to inplace-multiply.
    mult_int : int
        The integer to perform the multiplication with.
    treat_overflow : bool
        If set to ``False``, the mantissa will not be extended to prevent overflow errors. The default is ``True``.

    Examples
    --------

    We create a QuantumFloat, bring it to superposition and perform an inplace multiplication.

    ::

        from qrisp import QuantumFloat, h, inpl_mult
        a = QuantumFloat(5, signed = True)
        h(a[0])
        h(a[-1])
        print(a)


    ::

        # Yields: {0: 0.25, 1: 0.25, -32: 0.25, -31: 0.25}


    ::

        inpl_mult(a, -5)
        print(a)


    ::

        # Yields: {0: 0.25, 155: 0.25, 160: 0.25, -5: 0.25}


    """

    if not isinstance(mult_int, (int, float)):
        raise Exception("Quantum inplace multiplication is restricted to classical values due to reversibility constraints")

    if mult_int < 0 and not qf.signed:
        raise Exception(
            "Tried to inplace-multiply unsigned QuantumFloat with negative factor"
        )

    bit_shift = 0

    if int(mult_int) != mult_int:

        c = abs(mult_int)

        for i in range(32):
            if int(2**i * c) == 2**i * c:
                break
        else:
            raise Exception(
                "Tried to inplace multiply with number of to much precision"
            )

        bit_shift = -i
        mult_int = 2**i * mult_int

    else:
        while not mult_int % 2:
            bit_shift += 1
            mult_int = mult_int // 2

    if mult_int != 1:
        if treat_overflow:
            extension_size = int(np.ceil(np.log2(abs(mult_int))))
            qf.extend(extension_size, position=qf.size - 1)
            if qf.signed:
                cx(qf[-1], qf[-1 - extension_size : -1])

        from qrisp.alg_primitives.arithmetic.SBP_arithmetic import QFT_inpl_mult

        QFT_inpl_mult(qf, inplace_mult=mult_int)

    quantum_bit_shift(qf, bit_shift, treat_overflow)

    if treat_overflow and bit_shift < 0 and qf.signed:
        cx(qf[-1], qf[bit_shift - 1 : -1])


def quantum_bit_shift(qf, bit_shift, treat_overflow=True):

    from qrisp import cyclic_shift, control, QuantumFloat

    if isinstance(bit_shift, QuantumFloat):

        if bit_shift.signed or qf.signed:
            raise Exception(
                "Quantum-quantum bitshifting is currently only supported for unsigned arguments"
            )

        for i in range(*bit_shift.mshape):
            with control(bit_shift.significant(i)):
                quantum_bit_shift(qf, 2**i)

        return

    if treat_overflow:

        if bit_shift > 0:

            if qf.signed:
                qf.extend(bit_shift, position=qf.size - 1)
            else:
                qf.extend(bit_shift, position=qf.size)

        else:
            qf.extend(abs(bit_shift), position=0)
            qf.exp_shift(bit_shift)

    if qf.signed:
        cyclic_shift(qf[:-1], bit_shift)
    else:
        cyclic_shift(qf, bit_shift)


# @lifted
def app_sb_phase_polynomial(qv_list, poly, symbol_list=None, t=1):
    """
    Applies a phase function specified by a `semi-Boolean polynomial <https://ieeexplore.ieee.org/document/9815035>`_ acting on a list of QuantumVariables.
    That is, this method implements the transformation

    .. math::

        \ket{y_1}\dotsb\ket{y_n}\\rightarrow e^{itP(y_1,\dotsc,y_n)}\ket{y_1}\dotsb\ket{y_n}

    where :math:`\ket{y_1},\dotsc,\ket{y_n}` are QuantumVariables and :math:`P(y_1,\dotsc,y_n)=P(y_{1,1},\dotsc,y_{1,m_1},\dotsc,y_{n,1}\dotsc,y_{n,m_n})` is a semi-Boolean polynomial in variables
    :math:`y_{1,1},\dotsc,y_{1,m_1},\dotsc,y_{n,1}\dotsc,y_{n,m_n}`. Here, $m_i$ is the size of the $i$ th variable.

    Parameters
    ----------
    qv_list : list[QuantumVariable] or QuantumArray
        The list of QuantumVariables to evaluate the semi-Boolean polynomial on.
    poly : SymPy expression
        The semi-Boolean polynomial to evaluate.
    symbol_list : list, optional
        An ordered list of SymPy symbols associated to the qubits of the QuantumVariables of ``qv_list``.
        For each QuantumVariable in ``qv_list`` a number of symbols according to its size is required.
        By default, the symbols of the polynomial
        will be ordered alphabetically and then matched to the order in ``qv_list``.
    t : Float or SymPy expression, optional
        The argument ``t`` in the expression $\exp(itP)$. The default is 1.

    Raises
    ------
    Exception
        Provided QuantumVariable list does not include the appropriate amount
        of elements to evaluate the given polynomial.

    Examples
    --------

    We apply the phase function specified by the polynomial :math:`P(x,y,z) = \pi xyz` on a QuantumVariable:

    ::

        import sympy as sp
        import numpy as np
        from qrisp import QuantumVariable, app_sb_phase_polynomial

        x, y, z = sp.symbols('x y z')
        P = np.pi*x*y*z

        qv = QuantumVariable(3)
        qv.init_state({'000': 0.5, '111': 0.5})

        app_sb_phase_polynomial([qv], P)

    We print the ``statevector``:

    >>> print(qv.qs.statevector())
    sqrt(2)*(|000> - |111>)/2

    """

    if isinstance(qv_list, QuantumArray):
        qv_list = list(qv_list.flatten())

    # As the polynomial has only boolean variables,
    # powers can be ignored since x**k = x for x in GF(2)
    poly = filter_pow(poly.expand()).expand()

    if symbol_list is None:
        symbol_list = get_ordered_symbol_list(poly)

    if len(symbol_list) != sum([var.size for var in qv_list]):
        raise Exception(
            "Provided QuantumVariable list does not include the appropriate amount "
            "of elements to evaluate the given polynomial"
        )

    # The list of qubits contained in the variables of input_var_list
    input_qubits = sum([list(var.reg) for var in qv_list], [])

    # Monomials in list form
    monomial_list = expr_to_list(poly)

    control_qubit_list = []
    y_list = []

    # Iterate through the monomials
    for monom in monomial_list:
        # Prepare coeff (coefficient of the monomial) and variables (list of variables from symbol_list in the monomial)
        # Note: coeff may also contain symbolic variables
        coeff = float(1)
        variables = []
        for term in monom:
            if isinstance(term, sp.core.symbol.Symbol) and term in symbol_list:
                variables.append(term)
            elif isinstance(term, sp.core.symbol.Symbol):
                coeff = coeff * term
            else:
                coeff = coeff * float(term)

        # Append coefficient to y_list
        y_list.append(coeff)

        # Prepare the qubits on which the phase gate should be controlled
        control_qubit_numbers = [symbol_list.index(var) for var in variables]

        control_qubits = [input_qubits[nr] for nr in control_qubit_numbers]

        control_qubits = list(set(control_qubits))

        control_qubits.sort(key=lambda x: x.identifier)

        control_qubit_list.append(control_qubits)

    # Now we apply the multi controlled phase gates
    # Iterate through the list of phase gates
    while control_qubit_list:
        monomial_index = 0
        # Find control qubits and their coefficient
        control_qubits = control_qubit_list.pop(monomial_index)
        y = y_list.pop(monomial_index)

        # Apply (controlled) phase gate
        if len(control_qubits):
            mcp(y * t, control_qubits)
        else:
            gphase(y * t, input_qubits[0])


# @lifted
def app_phase_polynomial(qf_list, poly, symbol_list=None, t=1):
    """
    Applies a phase function specified by a polynomial acting on a list of QuantumFloats.
    That is, this method implements the transformation

    .. math::

        \ket{y_1}\dotsb\ket{y_n}\\rightarrow e^{itP(y_1,\dotsc,y_n)}\ket{y_1}\dotsb\ket{y_n}

    where :math:`\ket{y_1},\dotsc,\ket{y_n}` are QuantumFloats and :math:`P(y_1,\dotsc,y_n)` is a polynomial in variables
    :math:`y_1,\dotsc,y_n`.

    Parameters
    ----------
    qf_list : list[QuantumFloat] or QuantumArray[QuantumFloat]
        The list of QuantumFloats to evaluate the polynomial on.
    poly : SymPy expression
        The polynomial to evaluate.
    symbol_list : list, optional
        An ordered list of SymPy symbols associated to the QuantumFloats of ``qf_list``.
        By default, the symbols of the polynomial
        will be ordered alphabetically and then matched to the order in ``qf_list``.
    t : Float or SymPy expression, optional
        The argument ``t`` in the expression $\exp(itP)$. The default is 1.

    Raises
    ------
    Exception
        Provided QuantumFloat list does not include the appropriate amount of elements
        to encode given polynomial.

    Examples
    --------

    We apply the phase function specified by the polynomial :math:`P(x,y) = \pi x + \pi xy` on two QuantumFloats:

    ::

        import sympy as sp
        import numpy as np
        from qrisp import QuantumFloat, h, app_phase_polynomial

        x, y = sp.symbols('x y')
        P = np.pi*x + np.pi*x*y

        qf1 = QuantumFloat(3, signed = False)
        qf2 = QuantumFloat(3,-1, signed = False)
        h(qf1[0])
        qf2[:]=0.5

        app_phase_polynomial([qf1,qf2], P)

    We print the ``statevector``:

    >>> print(qf1.qs.statevector())
    sqrt(2)*(|0>*|0.5> - I*|1>*|0.5>)/2


    We apply the phase function specified by the polynomial :math:`P(x) = 1 - 0.9x^2 + x^3` on a QuantumFloat:

    ::

        import numpy as np
        import sympy as sp
        import matplotlib.pyplot as plt
        from qrisp import QuantumFloat, h, app_phase_polynomial

        x = sp.symbols('x')
        P = 1-0.9*x**2+x**3

        qf = QuantumFloat(3,-3)
        h(qf)

        app_phase_polynomial([qf],P)

    To visualize the results we retrieve the ``statevector`` as a function and determine the phase of each entry.

    ::

        sv_function = qf.qs.statevector("function")

    This function receives a dictionary of QuantumVariables specifiying the desired label constellation and returns its complex amplitude.
    We calculate the phases corresponding to the complex amplitudes, and compare the results with the values of the function $P(x)$.

    ::

        qf_values = np.array([qf.decoder(i) for i in range(2 ** qf.size)])
        sv_phase_array = np.angle([sv_function({qf : i}) for i in qf_values])

        P_func = sp.lambdify(x, P, 'numpy')
        x_values = np.linspace(0, 1, 100)
        y_values = P_func(x_values)

    Finally, we plot the results.

    ::

        plt.plot(x_values, y_values, label = "P(x)")
        plt.plot(qf_values , sv_phase_array%(2*np.pi), "o", label = "Simulated phases")
        plt.ylabel("Phase [radian]")
        plt.xlabel("QuantumFloat outcome labels")
        plt.grid()
        plt.legend()
        plt.show()

    .. figure:: /_static/PhasePolynomialApplication.png
        :alt: PhasePolynomialApplication
        :scale: 80%
        :align: center

    """

    if isinstance(qf_list, QuantumArray):
        qf_list = list(qf_list.flatten())

    if symbol_list is None:
        symbol_list = get_ordered_symbol_list(poly)

    if len(symbol_list) != len(qf_list):
        raise Exception(
            "Provided QuantumFloat list does not include the appropriate amount "
            "of elements to evaluate the given polynomial"
        )

    sb_poly_list = []
    new_symbol_list = []
    for qf in qf_list:
        if qf.signed:
            # We do not use modular arithmetic.
            sb_poly_list.append(
                qf.sb_poly()
                - 2 ** (qf.msize + 2 + qf.exponent)
                * sp.symbols(str(hash(qf)) + "_" + str(qf.msize))
            )
        else:
            sb_poly_list.append(qf.sb_poly())

        temp_var_list = list(sp.symbols(str(hash(qf)) + "_" + "0:" + str(qf.size)))
        new_symbol_list += temp_var_list

    repl_dic = {symbol_list[i]: sb_poly_list[i] for i in range(len(qf_list))}
    # Substitute semi-Boolean polynomials
    sb_polynomial = poly.subs(repl_dic).expand()
    # As the polynomial has only boolean variables,
    # powers can be ignored since x**k = x for x in GF(2)
    sb_polynomial = filter_pow(sb_polynomial.expand()).expand()

    # Apply sb phase polynomial
    app_sb_phase_polynomial(qf_list, sb_polynomial, symbol_list=new_symbol_list, t=t)


# Workaround to keep the docstring but still gatewrap

temp = app_sb_phase_polynomial.__doc__

app_sb_phase_polynomial = gate_wrap(permeability="args", is_qfree=True)(
    app_sb_phase_polynomial
)

app_sb_phase_polynomial.__doc__ = temp


temp = app_phase_polynomial.__doc__

app_phase_polynomial = gate_wrap(permeability="args", is_qfree=True)(
    app_phase_polynomial
)

app_phase_polynomial.__doc__ = temp
