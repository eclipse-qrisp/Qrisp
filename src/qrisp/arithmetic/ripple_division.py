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

from qrisp import gate_wrap, x


@gate_wrap(is_qfree=True, permeability="args")
def q_int_div(numerator, divisor, adder="cuccaro", n=None, log_output=True):
    # This function performs integer division based on the following algorithm
    # Which is a reformulation of the division algorithm presented in
    # https://en.wikipedia.org/wiki/Division_algorithm#Non-restoring_division

    # Q = 2**(n)-1
    # D *= 2**(n-1)
    # R = int(N)

    # import numpy as np
    # from qrisp.misc import int_as_array

    # for i in range(n-1, -1, -1):

    #     D = D/2
    #     if not np.sign(D)*np.sign(R) != -1:
    #         Q -= 2**(i)

    #     if int_as_array(Q, n)[::-1][i]:
    #         R = R-D
    #     else:
    #         R = R+D

    # Q -= 2**(n-1)
    # Q += 0.5

    # if D < 0:
    #     Q += -0.5
    #     R = R - D
    # else:
    #     Q += -0.5
    #     R = R + D

    # D *= 2

    # The D multiplications/divisions are realized by bitshifts
    # The conditioned Q -= 2**i operation is performed by NOTing the appropriate
    # qubit controlled on the condition

    # The R = R+-D operation is realized by wrapping applying NOT gates on the R
    # qubits before and after the addition controlled on the corresponding Q-qubit
    # This yields the desired behavior because (x' + y)' = x - y
    # where the prime denotes negation

    # The Q -= 2**(n-1) is simply performed by flipping the significance n-1 bit
    # The Q = Q + 0.5 is performed by adding an aditional qubit at the -1 significance
    # of Q and turning it on

    # The final conditional is performed similar to the previous condition
    # Where the variables which are being operated on are wrapped in CNOTs

    qs = numerator.qs

    if numerator.exponent < 0 or divisor.exponent < 0:
        raise Exception(
            "Tried to call integer division with non integer quantum floats"
        )

    # n is the bitsize of the quotient
    # The largest possible number the quotient can take is
    # 2**(numerator_max_sig - divisor_min_sig)
    # We need one additional bit (why?)

    if n is None:
        n = int(numerator.mshape[1] - divisor.exponent) + 1

    # The quotient bits will successively be calculated/
    # added starting at the most significant bit.
    # As the algorithm produces empty bits in the remainder at the same rate as
    # quotient bits are required, we initialize the quotient as a 1 bit variable
    # Since the highest significance ist calculated fist, this bit needs significance n.

    # We initialize this float with a single Qubit which will be freed up
    # once the algorithm produced the first bit
    # We can't initialize variables with 0 qubits
    from qrisp.qtypes.quantum_float import QuantumFloat

    quotient = QuantumFloat(1, n, signed=True)

    # In order to determine the datashape of the remainder, we note that
    # within the algorithm, there are steps where the remainder variable needs to hold
    # larger+finer numbers than the final result for the remainder

    # The crucial step is the addition
    # remainder += divisor

    # Regarding the exponent of the remainder, we know that it has to be
    # less than the exponent of the numerator (so we can properly initialize)
    # But also less than the divisor exponent - 1 in order to properly execute
    # the last iteration which is a D/2 addition
    remainder_exponent = min(numerator.exponent, divisor.exponent - 1)

    # For the maximum significance,
    # we note that the divisor is initially bit shifted by n
    divisor.exp_shift(n - 1)

    # Therefore, the required maximum significance is acquired
    # with the formula for addition max_sig_add = max(max_sig_a, max_sig_b) + 1
    remainder_max_sig = max(numerator.mshape[1], divisor.mshape[1]) + 1

    # Create remainder float
    remainder_size = remainder_max_sig - remainder_exponent
    remainder = QuantumFloat(remainder_size, remainder_exponent, signed=True)

    # We now can initialize the remainder as the value of the numerator
    remainder.init_from(numerator)

    # If the divisor is signed we need to flip the quotient sign bit (why?)
    if divisor.signed:
        qs.cx(divisor[-1], quotient[-1])

    from qrisp.arithmetic import inpl_add

    # Perform iterations
    for i in range(n - 1, -1, -1):
        if log_output:
            R = list(remainder.get_measurement().keys())[0]
            print("R: ", R)

            D = list(divisor.get_measurement().keys())[0]
            print("D: ", D)

            Q = list(quotient.get_measurement().keys())[0]
            print("Q: ", Q)

            # print("Q: " + bin_rep(int(Q/2.**quotient.exponent),
            # quotient.size-1) + "(" + str(Q) + ")")
            print("---")

        # The sign bit of the remainder is the newly calculated bit
        # of the quotient of this iteration
        # Therefore we need to move this bit from the remainder to the quotient
        # We do this in a very "hacky" fashion

        # Remove from remainder
        remainder_sign_bit = remainder.reg.pop(-1)
        remainder.size -= 1
        remainder.mshape[1] -= 1

        # Add to quotient at position 0
        quotient.reg.insert(0, remainder_sign_bit)
        quotient.size += 1
        quotient.mshape[1] += 1

        # This next instruction is a bit involved to understand

        # If we didn't use the technique of transfering the bits from the remainder
        # to the quotient, we would have to call quotient.x() in order to realize
        # the initial Q = 2**n - 1 statement. In this case we would then afterwards
        # CNOT from remainder sign bit it the quotient sign bit from this iteration
        # Since x gate on the quotient qubit and the CNOT commute,
        # we can also perform the x gate after the CNOT

        # Translating this to our case we replaced the CNOT with the bit transfer,
        # we arive at the neccessity of an x at this point
        x(quotient[0])

        # Since the new bit was added at the least significant end,
        # we need to lower the exponent
        quotient.exp_shift(-1)

        # This is to handle the D part of the statement "np.sign(D)*np.sign(R) != 1"
        if divisor.signed:
            qs.cx(divisor[-1], quotient[0])

        # We constructed Q with one initial qubit, which will be freed up after
        # the first iteration. This does not produce a qubit overhead,
        # because the upcomming add function needs more than one ancilla anyway
        if i == n - 1:
            quotient.reduce(quotient[1])

        # Perform the D = D/2 instruction
        divisor.exp_shift(-1)

        # We wrap the remainder in CNOT gates to make use of
        # (R' + D)' = (R - D)
        # where the prime denotes the negation
        for j in range(remainder.size):
            qs.cx(quotient[0], remainder[j])

        # This perform the addition R += D or R -= D
        inpl_add(
            remainder,
            divisor,
            ignore_rounding_error=True,
            ignore_overflow_error=True,
            adder=adder,
        )

        # Instead of executing this layer of CNOT gates,
        # we use the layer from the next iteration and only
        # CNOT the control qubit from this layer
        # for j in range(remainder.size):
        # qs.cx(quotient[0], remainder[j])

        # We still need to make up for the fact that the next layer of CNOT
        # gates doesnt include remainder[-1]

        # qs.cx(quotient[0], remainder[-1])

        # However remainder[-1] is also the control qubit of the next iteration
        # implying our plan of CNOTting the next control qubit cancels the
        # previous line

        # qs.cx(quotient[0], remainder[-1])

        # In any iteration apart from the first, quotient[0] does not contain
        # the desired value because of the sheenanigans of the previous lines.
        # In detail: It contains the desired value (+) the quotient[0] qubit
        # of the previous iteration. We cancel with this command
        if i != n - 1:
            qs.cx(quotient[1], quotient[0])

    # Since there is no more upcoming iteration, we simply perform the CNOT layer
    # without any further tricks.
    for j in range(remainder.size):
        qs.cx(quotient[0], remainder[j])

    # This performs the Q -= 2**(n-1) instruction (the -1 qubit is the sign bit =>
    # the -2 qubit is the significance n-1 bit)
    x(quotient[-2])

    # Handle the case that the divisor is signed
    if divisor.signed:
        # Create Qubit to account for the fact that we will add 0.5 to the quotient,
        # even though it only supports integers
        quotient.extend(1, position=0)
        quotient.exp_shift(-1)
        x(quotient[0])

        # Add/Subtract 0.5 executed via (Q' + 0.5)' = Q - 0.5
        qs.cx(divisor[-1], quotient.reg)
        quotient.incr(-(2**quotient.exponent))
        qs.cx(divisor[-1], quotient.reg)

        # Remove 0.5 significance qubit (contains 0 now)
        quotient.reduce(quotient[0])
        quotient.exp_shift(1)

        # Incase the divisor is not only signed but actually negative, we CNOT the
        # remainder such that the upcoming addition circuit results in a subtraction
        qs.cx(divisor[-1], remainder.reg)

    # Perform R = R +- D
    inpl_add(
        remainder,
        divisor,
        ignore_rounding_error=True,
        ignore_overflow_error=True,
        adder=adder,
    )

    if divisor.signed:
        # Undo the CNOT
        pass
        qs.cx(divisor[-1], remainder.reg)

    # Perform D = 2*D
    divisor.exp_shift(1)

    # Finally, if the numerator was signed, we flip the quotient sign bit
    if numerator.signed:
        qs.cx(numerator[-1], quotient[-1])

    return quotient, remainder


# Wrapper for the integer division
# Supports division for quantum float of arbitrary exponent.
# And allows to give a precision threshold (prec)
# The resulting quotient variable Q_res has the property |Q_res - Q_real| < 2**prec
def q_divmod(numerator, divisor, adder="thapliyal", prec=0):
    """
    Performs division up to arbitrary precision. Returns the quotient and the remainder.

    Parameters
    ----------
    numerator : QuantumFloat
        The QuantumFloat to divide.
    divisor : QuantumFloat
        The QuantumFloat to divide by.
    adder : str, optional
        The type of adder to use. Available are "thapliyal" and "cuccarro".
        The default is "thapliyal".
    prec : int, optional
        The precision of the division. If the precision is set to $k$,
        the approximated quotient $q_{apr}$ and the true quotient $q_{true}$
        satisfy $|q_{apr} - q_{true}|<2^{-k}$. The default is 0.

    Returns
    -------
    quotient : QuantumFloat
        The approximated quotient.
    remainder : QuantumFloat
        The remainder which satisfying $q*d + r = n$.

    Examples
    --------

    We calculate 10/8 with varying precision:

    >>> from qrisp import QuantumFloat, q_divmod, multi_measurement
    >>> num = QuantumFloat(4)
    >>> div = QuantumFloat(4)
    >>> num[:] = 10
    >>> div[:] = 8
    >>> quotient, remainder = q_divmod(num, div, prec = 1)
    >>> multi_measurement([quotient, remainder])
    {(1.0, 2.0): 1.0}

    Now with higher precision

    >>> num = QuantumFloat(4)
    >>> div = QuantumFloat(4)
    >>> num[:] = 10
    >>> div[:] = 8
    >>> quotient, remainder = q_divmod(num, div, prec = 3)
    >>> multi_measurement([quotient, remainder])
    {(1.25, 0.0): 1.0}

    """

    # The idea is to bit shift numerator and divisor by s such that,
    # they are both integers. To increase the precision we shift N by -prec
    # (we'll see shortly why this increases the precision).
    # We write

    # N_tilde = 2**(s-prec) * N
    # D_tilde = 2**s * D

    # We then perform integer division giving Q_tilde, R_tilde such that

    # N_tilde/D_tilde = Q_tilde + R_tilde/D_tilde
    # where |R_tilde| < |D_tilde|

    # We now set

    # Q = Q_tilde 2**(prec)
    # and insert, which yields

    # N_tilde/D_tilde = 2**(-prec) * N/D  = 2**(-prec) * Q + R_tilde/D_tilde

    # =>  N/D = Q + 2**prec*R_tilde/D_tilde

    # So the error is
    # |2**prec*R_tilde/D_tilde| < 2**prec

    # Because R_tilde < D_tilde

    prec = -prec
    s = -min(numerator.exponent, divisor.exponent)

    num_exp_shift = s - prec
    div_exp_shift = s

    numerator.exp_shift(num_exp_shift)
    divisor.exp_shift(div_exp_shift)

    # print("N_tilde: ", numerator.get_measurement())
    # print("D_tilde: ", divisor.get_measurement())

    quotient, remainder = q_int_div(numerator, divisor, adder=adder, log_output=False)

    # print("Q_tilde: ", quotient.get_measurement())
    # print("R_tilde: ", remainder.get_measurement())

    quotient.exp_shift(prec)
    remainder.exp_shift(-num_exp_shift)

    divisor.exp_shift(-div_exp_shift)
    numerator.exp_shift(-num_exp_shift)

    return quotient, remainder


def q_div(numerator, divisor, prec=None):
    """
    Performs division up to arbitrary precision and uncomputes the remainder.

    Parameters
    ----------
    numerator : QuantumFloat
        The QuantumFloat to divide.
    divisor : QuantumFloat
        The QuantumFloat to divide by.
    prec : int, optional
        The precision of the division. If the precision is set to $k$,
        the approximated quotient $q_{apr}$ and the true quotient $q_{true}$
        satisfy $|q_{apr} - q_{true}|<2^{-k}$.
        By default, a suited precision will be determined from the other inputs.

    Returns
    -------
    quotient : QuantumFloat
        The result of the division.

    Examples
    --------

    We calculate 10/8:

    >>> from qrisp import QuantumFloat, q_div
    >>> num = QuantumFloat(4)
    >>> div = QuantumFloat(4)
    >>> num[:] = 10
    >>> div[:] = 8
    >>> quotient = q_div(num, div, prec = 2)
    >>> print(quotient)
    {1.25: 1.0}

    """

    from qrisp import U_g_inpl_adder, h, hybrid_mult

    if prec is None:
        prec = divisor.size - numerator.exponent

    quotient, remainder = q_divmod(numerator, divisor, prec=prec)

    hybrid_mult(quotient, divisor, remainder, init_op="qft", terminal_op=None)

    U_g_inpl_adder(remainder, numerator, mult_factor=-1)

    h(remainder)
    # QFT(remainder, inv = True, exec_swap = False)
    # remainder += -1

    remainder.delete()

    return quotient


def qf_inversion(qf, prec=None):
    """
    Calculates the multiplicative inverse of a QuantumFloat.

    Parameters
    ----------
    qf : QuantumFloat
        The QuantumFloat to invert.
    prec : int, optional
        The precision of the inversion. If the precision is set to $k$,
        the approximated inverse $q_{apr}$ and the true inverse $q_{true}$ satisfy
        $|q_{res} - q_{true}|<2^{-k}$.
        By default, a suited precision will be determined from the other input.

    Returns
    -------
    result : QuantumFloat
        A QuantumFloat containing the inverse.

    Examples
    --------

    We calculate the inverse of 0.75

    >>> from qrisp import QuantumFloat, qf_inversion
    >>> qf = QuantumFloat(2, -2)
    >>> qf[:] = 0.75
    >>> qf_inv = qf_inversion(qf, prec = 4)
    >>> print(qf_inv)
    {1.3125: 1.0}
    >>> 0.75**-1
    1.3333333333333333

    """

    if prec is None:
        prec = qf.size

    from qrisp import QuantumFloat, x

    numerator = QuantumFloat(1, signed=False)
    numerator.encode(1)

    result = q_div(numerator, qf, prec)

    x(numerator)
    numerator.delete()

    return result
