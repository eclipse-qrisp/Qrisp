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

from qrisp import QuantumArray, p, z
from qrisp.alg_primitives.arithmetic.SBP_arithmetic import hybrid_mult, polynomial_encoder


def q_matmul(
    q_array_0, q_array_1, output_array=None, res_bit_shape="eq", phase_tolerant=False
):
    """
    Matrix multiplication for QuantumArrays.

    Parameters
    ----------
    q_array_0 : QuantumArray
        The first factor of the matrix multiplication.
    q_array_1 : QuantumArray
        The second factor of the matrix multiplication.
    output_array : QuantumArray, optional
        The QuantumArray to store the results in.
        By default, a new QuanumArray is created.
    res_bit_shape : str or QuantumFloat, optional
        Specification of the dimension of the output bitshape
        of the output QuantumArray.
        Possible are "eq", which will take the bitshape equal two the first factor,
        "safe" which automatically determines the bitshape such that there can be
        no overflow, or a QuantumFloat which has the desired bitshape.
        The default is "eq".
    phase_tolerant : bool, optional
        If set to True, the required gate count is reduced but each constellation of
        computational basis states of the inputs will introduce a different phase.
        This is helpful when it's clear that this function will be at some point
        uncomputed, resulting in the cancelation of these phases. The default is False.

    Raises
    ------
    Exception
        Tried to perform matrix multiplication with differing contraction index size.

    Returns
    -------
    res : QuantumArray
        The result of the matrix multiplication.

    Examples
    --------

    We multiply a QuantumArray with a multiply of the identity matrix (np.eye):

    >>> import numpy as np
    >>> from qrisp import QuantumFloat, QuantumArray, q_matmul
    >>> qf = QuantumFloat(4,0, signed = False)
    >>> q_arr_0 = QuantumArray(qf, shape = (2,2))
    >>> q_arr_1 = QuantumArray(qf, shape = (2,2))
    >>> q_arr_0[:] = [[1,2],[3,4]]
    >>> q_arr_1[:] = 2*np.eye(2)
    >>> res = q_matmul(q_arr_0, q_arr_1)
    >>> print(res)
    {OutcomeArray([[2, 4],
                   [6, 8]]): 1.0}

    """

    if q_array_0.shape[1] != q_array_1.shape[0]:
        raise Exception(
            "Tried to perform matrix multiplication"
            "with differing contraction index size"
        )

    L = q_array_0.shape[0]
    K = q_array_0.shape[1]
    J = q_array_1.shape[1]

    if output_array is not None:
        res = output_array

    else:
        from qrisp import QuantumFloat

        if isinstance(res_bit_shape, QuantumFloat):
            qtype = res_bit_shape.duplicate()

        if res_bit_shape == "safe":
            from sympy import Symbol

            q_array_0_symbols = []
            q_array_1_symbols = []
            poly = 0
            for i in range(K):
                q_array_0_symbols.append(Symbol(q_array_0[0, i].name))
                q_array_1_symbols.append(Symbol(q_array_1[i, 0].name))

                poly += Symbol(q_array_0[0, i].name) * Symbol(q_array_1[i, 0].name)
                # poly += q_array_0_symbols[-1]*q_array_1_symbols[-1]

            from qrisp.alg_primitives.arithmetic import create_output_qf

            qtype = create_output_qf(
                list(q_array_0.flatten()) + list(q_array_1.flatten()), poly
            )

        if res_bit_shape == "eq":
            qtype = q_array_0[0, 0]

        res = QuantumArray(shape=(L, J), qtype=qtype, qs=q_array_0.qs)

    for i in range(L):
        for j in range(J):
            for k in range(K):
                if K == 1:
                    if output_array is not None:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op="qft",
                            terminal_op="qft",
                            phase_tolerant=phase_tolerant,
                        )
                    else:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op="h",
                            terminal_op="qft",
                            phase_tolerant=phase_tolerant,
                        )
                    continue

                # If the res_bit_shape is safe,we can use the strategy
                # of correcting phases on the result as displayed in hybrid mult
                if res_bit_shape == "safe" and not phase_tolerant:
                    if k == 0:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op="h",
                            terminal_op=None,
                            phase_tolerant=True,
                        )

                    elif k == K - 1:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op=None,
                            terminal_op="qft",
                            phase_tolerant=True,
                        )

                    elif k != q_array_0.shape[1] - 1:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op=None,
                            terminal_op=None,
                            phase_tolerant=True,
                        )

                        for m in range(res[i, j].size):
                            p(
                                -(2 * np.pi) * 2**l / 2 ** (res[i, j].size + 1),
                                res[i, j][m],
                            )

                        if res[i, j].signed and K % 2:
                            z(res[i, j][-1])

                else:
                    if k == 0:
                        if output_array is not None:
                            hybrid_mult(
                                q_array_0[i, k],
                                q_array_1[k, j],
                                output_qf=res[i, j],
                                init_op="qft",
                                terminal_op=None,
                                phase_tolerant=phase_tolerant,
                            )
                        else:
                            hybrid_mult(
                                q_array_0[i, k],
                                q_array_1[k, j],
                                output_qf=res[i, j],
                                init_op="h",
                                terminal_op=None,
                                phase_tolerant=phase_tolerant,
                            )
                    elif k == K - 1:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op=None,
                            terminal_op="qft",
                            phase_tolerant=phase_tolerant,
                        )
                    else:
                        hybrid_mult(
                            q_array_0[i, k],
                            q_array_1[k, j],
                            output_qf=res[i, j],
                            init_op=None,
                            terminal_op=None,
                            phase_tolerant=phase_tolerant,
                        )

    return res


def semi_classic_matmul(q_matrix, cl_matrix, output_array=None, res_bit_shape="eq"):
    """
    Performs matrix multiplication between a classical numpy array and a QuantumArray

    Parameters
    ----------
    q_matrix : QuantumArray
        The QuantumArray to multiply.
    cl_matrix : numpy.ndarray
        The numpy array to multiply.
    output_array : QuantumArray, optional
        The QuantumArray to store the result in. The default is None.
    res_bit_shape : str or QuantumFloat, optional
        Specification of the dimension of the output bitshape
        of the output QuantumArray.
        Possible are "eq", which will take the bitshape equal two the first factor,
        "safe" which automatically determines the bitshape such that there can be no
        overflow, or a QuantumFloat which has the desired bitshape. The default is "eq".

    Raises
    ------
    Exception
        Tried to apply matrix multiplication with unfitting dimensions.

    Returns
    -------
    output_array : QuantumArray
        The QuantumArray containing the result.

    Examples
    --------

    We multiply a QuantumArray with a scalar multiple of the identity matrix (np.eye):

    >>> import numpy as np
    >>> from qrisp import QuantumFloat, QuantumArray, semi_classic_matmul
    >>> qf = QuantumFloat(4,0, signed = False)
    >>> q_arr_0 = QuantumArray(qf, shape = (2,2))
    >>> q_arr_1 = QuantumArray(qf, shape = (2,2))
    >>> q_arr_0[:] = [[1,2],[3,4]]
    >>> res = semi_classic_matmul(q_arr_0, 2*np.eye(2))
    >>> print(res)
    {OutcomeArray([[2, 4],
                   [6, 8]]): 1.0}


    """
    if q_matrix.shape[1] != cl_matrix.shape[0]:
        raise Exception(
            "Tried to apply matrix multiplication with unfitting dimensions"
        )

    L = q_matrix.shape[0]
    K = q_matrix.shape[1]
    J = cl_matrix.shape[1]

    from sympy import Symbol
    from qrisp import QuantumFloat

    if output_array is None:
        if isinstance(res_bit_shape, QuantumFloat):
            qtype = res_bit_shape.duplicate()

        if res_bit_shape == "safe":
            q_array_0_symbols = []
            poly = 0
            for i in range(K):
                q_array_0_symbols.append(Symbol(q_matrix[0, i].name))

                poly += Symbol(q_matrix[0, i].name)

            from qrisp.alg_primitives.arithmetic import create_output_qf

            qtype = create_output_qf(list(q_matrix.flatten()), poly)

        if res_bit_shape == "eq":
            qtype = q_matrix[0, 0]

        output_array = QuantumArray(shape=(L, J), qtype=qtype, qs=q_matrix.qs)

    if output_array.shape != (L, J):
        raise Exception(
            "Tried to to encode matrix multiplication"
            "into Quantum Array of unfitting size"
        )

    from sympy.matrices import zeros

    symb_matrix = zeros(q_matrix.shape[0], q_matrix.shape[1])

    symbol_dic = {}
    for i in range(L):
        for k in range(K):
            temp_symb = Symbol("x_" + str(i) + "_" + str(k))
            symb_matrix[i, k] = temp_symb
            symbol_dic[q_matrix[i, k].name] = temp_symb

    res_symb_matrix = symb_matrix @ cl_matrix

    qv_list = list(q_matrix.flatten())
    for i in range(L):
        for j in range(J):
            if res_symb_matrix[i, j] != 0:
                polynomial_encoder(
                    qv_list,
                    output_array[i, j],
                    res_symb_matrix[i, j],
                    encoding_dic=symbol_dic,
                )

    return output_array


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception("modular inverse does not exist")
    else:
        return x % m


# Function for inplace multiplication with a quantum array
# [qv1, qv2,.. qvn] with a n x n matrix  A
# ie. the array contains [a11*qv1 +a12*qv2+ ... , a21*qv1..] after
def inplace_matrix_app(vector, matrix):
    r"""
    Performs inplace matrix application to a vector-valued QuantumArray.
    Note that due to reversibility reasons, the matrix can only contain integers
    and has to be invertible over $\text{GL}(2^n)$.
    This is equal to the condition that the determinant is odd.

    Parameters
    ----------
    vector : QuantumArray
        The QuantumArray to apply the matrix to.
    matrix : numpy.ndarray
        The matrix to apply.

    Raises
    ------
    Exception
        Tried to multiply matrix with Quantum Array with unfitting shape.

    Examples
    --------

    We perform an inplace matrix multiplication between a randomly chosen matrix
    and a randomly chosen vector

    >>> from qrisp import QuantumFloat, QuantumArray, inplace_matrix_app
    >>> qtype = QuantumFloat(4)
    >>> q_vector = QuantumArray(qtype, 3)
    >>> q_vector[:] = [1,0,2]
    >>> matrix = np.array([[2., 2., 1.],
                           [2., 3., 1.],
                           [3., 1., 2.]])
    >>> inplace_matrix_app(q_vector, matrix)
    >>> print(q_vector)
    {OutcomeArray([4, 4, 7]): 1.0}
    """

    if len(vector.shape) != 1:
        raise Exception(
            "Tried to multiply matrix with Quantum Array with unfitting shape"
        )

    n = vector.shape[0]

    if n != matrix.shape[0] or n != matrix.shape[1]:
        raise Exception(
            "Tried to multiply matrix with Quantum Array with unfitting shape"
        )

    bit = vector[0].size

    # The general idea is to calculate a row of the matrix into the variable x_j which
    # can be reconstructed from the result of the calculation and thus discarded.

    # Calling the result of this evaluation x_j_new we have

    # x_j_new = a*x_j + b*x_0 ... c*x_n

    # For this equation to be invertible a has to be invertible. We then have

    # x_j = 1/a*(x_j_new - b*x_0 + .. c*x_n)

    # we then replace x_j in the remaining evaluation equations with this expression

    # We will perform this process in two steps:
    # 1. Prepare a list of equations that contains what should be evaluated
    # 2. Evaluate these equations in the quantum session

    # Step 1:

    # Prepare symbols
    ancilla_symbol = sp.Symbol("a")
    x = [sp.Symbol("x" + str(i)) for i in range(n)]

    # This list contains the equations that need to be calculated
    target_values = matrix * sp.Matrix(x)

    # This list contains the variables that have been eliminated in favor
    # of the completed equations
    eliminated_variables = []

    # This list contains the equations
    # that will be evaluated later by the circuit generator
    eval_eq = []

    from qrisp.misc import is_inv

    for i in range(n):
        # Save evaluation
        eval_eq.append(target_values[i])

        # Find a column with invertible coeffiecient
        j = 0
        while True:
            # Determine that coefficient(can be done by differentiating the evaluating
            # eq. after the column variable)
            coeff = int(sp.diff(target_values[i], x[j]))

            # Check if the coefficient is invertible and has not been eliminated yet
            if is_inv(coeff, bit) and (j not in eliminated_variables):
                break
            j += 1

            if j >= n:
                raise Exception("Could not find invertible element")

        # Generate inverse equation
        eval_inverse = modinv(coeff, 2**bit) * (
            -target_values[i].subs({x[j]: 0}) + ancilla_symbol
        )

        from qrisp.qtypes.quantum_float import trunc_poly

        # Truncate coefficients in order to stay inside Z/ 2^n Z
        eval_inverse = trunc_poly(eval_inverse, (0, bit))

        # Rewrite the remaining evaluation equations in terms of the new variable
        subs_dic = {x[j]: eval_inverse}

        for k in range(i + 1, n):
            # Substitue in the following equations
            target_values[k] = (
                target_values[k].subs(subs_dic).subs({ancilla_symbol: x[j]})
            )

            # Truncate coefficients
            target_values[k] = trunc_poly(target_values[k], (0, bit))

        eliminated_variables.append(j)

    # Step 2:

    # Create symbol dic in order for the polynomial encoder
    # to know which symbol to use as which quantum float
    symbol_dic = {vector[j].name: x[j] for j in range(vector.shape[0])}

    for i in range(n):
        # Evaluate the value of the entry in eval_eq into the x_j variable

        elim_var = eliminated_variables[i]

        # Create list of quantum variables which are not being written on
        non_elim_var_list = list(vector)
        non_elim_var_list.pop(elim_var)

        # Perform evaluation. The first step is to determine the coefficient for the
        # inplace multiplication (ie. in the language of the initial description "a")
        inplace_mult_coeff = int(sp.diff(eval_eq[i], x[elim_var]))

        # Now determine the remaining polynomial
        eval_poly = eval_eq[i].subs({x[elim_var]: 0})
        print(inplace_mult_coeff)
        polynomial_encoder(
            non_elim_var_list,
            vector[elim_var],
            eval_poly,
            encoding_dic=symbol_dic,
            inplace_mult=inplace_mult_coeff,
        )

    # Reorder quantum array
    qv_reordering_array = np.zeros(n, dtype="object")

    for i in range(n):
        qv_reordering_array[i] = vector[eliminated_variables[i]]

    # In order to enter the reorder values in the QuantumArray, we need to use the
    # setitem method of ndarray because setitem for QuantumArray
    # is overloaded with initialization
    np.ndarray.__setitem__(vector, slice(None, None, None), qv_reordering_array)


def auto_matmul_wrapper(a, b, out=None):
    if isinstance(a, QuantumArray) and isinstance(b, QuantumArray):
        return q_matmul(a, b, out)

    elif isinstance(a, QuantumArray):
        return semi_classic_matmul(a, b, out)

    elif isinstance(b, QuantumArray):
        return semi_classic_matmul(b.transpose(), a.transpose(), out).transpose()

    else:
        raise Exception(
            "Could not proccess input constellation "
            + str(type(a))
            + " and "
            + str(type(b))
        )


def dot(a, b, out=None):
    """
    Port of the popular `numpy function
    <https://numpy.org/doc/stable/reference/generated/numpy.dot.html>`_
    with similar semantics.

    Parameters
    ----------
    a : QuantumArray or QuantumFloat or numpy.ndarray
        The first operand.
    b : QuantumArray or QuantumFloat or numpy.ndarray
        The second operand.
    out : QuantumArray, optional
        The QuantumArray to store the output in. The default is None.

    Returns
    -------
    QuantumArray
        The result as described in the numpy documentation.

    Examples
    --------

    We create two QuantumArrays and apply dot as a function
    performing matrix-vector multiplication.

    >>> import numpy as np
    >>> from qrisp import QuantumFloat, QuantumArray, dot
    >>> qf = QuantumFloat(5,0, signed = False)
    >>> q_arr_0 = QuantumArray(qf)
    >>> q_arr_1 = QuantumArray(qf)
    >>> q_arr_0[:] = [2,3]
    >>> q_arr_1[:] = 2*np.eye(2)
    >>> res = dot(q_arr_0, q_arr_1)
    >>> print(res)
    {OutcomeArray([[4, 6]]): 1.0}

    Scalar-product:

    >>> q_arr_0 = QuantumArray(qf)
    >>> q_arr_1 = QuantumArray(qf)
    >>> q_arr_0[:] = [3,4,5]
    >>> q_arr_1[:] = [1,1,1]
    >>> res = dot(q_arr_0, q_arr_1)
    >>> print(res)
    {12: 1.0}

    Matrix-matrix multiplication

    >>> qf = QuantumFloat(3,0, signed = True)
    >>> q_arr_0 = QuantumArray(qf)
    >>> q_arr_1 = QuantumArray(qf)
    >>> q_arr_0[:] = [[0,1],[1,0]]
    >>> q_arr_1[:] = [[1,0],[0,-1]]
    >>> res = dot(q_arr_0, q_arr_1)
    >>> print(res)
    {OutcomeArray([[ 0, -1],
                   [ 1,  0]]): 1.0}

    """
    from qrisp import QuantumFloat

    if isinstance(a, QuantumFloat) or isinstance(b, QuantumFloat):
        return np.dot(a, b, out)

    else:
        if len(a.shape) == 1 and len(b.shape) == 1:
            temp_0 = a.reshape((1, a.shape[0]))
            temp_1 = b.reshape((a.shape[0], 1))
            res = (auto_matmul_wrapper(temp_0, temp_1, out))[0, 0]
            return res

        elif len(a.shape) == 2 and len(b.shape) == 2:
            return auto_matmul_wrapper(a, b, out)

        elif len(b.shape) == 1:
            temp_0 = a.reshape((a.size // b.shape[-1], b.shape[-1]))
            temp_1 = b.reshape((b.shape[0], 1))

            new_shape = list(b.shape)
            new_shape[-1] = b.shape[0]

            res = (auto_matmul_wrapper(temp_0, temp_1, out)).reshape(new_shape)

            return res

        else:
            temp_0 = a.reshape((a.size // b.shape[-1], b.shape[-1]))

            temp_1 = np.swapaxes(b, 0, -2)
            temp_1 = np.reshape(
                temp_1, (temp_1.shape[0], temp_1.size // temp_1.shape[0])
            )

            res = auto_matmul_wrapper(temp_0, temp_1, out)

            res_shape = list(temp_1.shape)
            res_shape.pop(-2)

            res_shape = list(temp_0.shape)[:-1] + res_shape

            return np.reshape(res, res_shape)


def tensordot(a, b, axes):
    r"""
    Port of `numpy tensordot
    <https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html>`_
    with similar semantics.

    Parameters
    ----------
    a : QuantumArray
        The first operand.
    b : QuantumArray
        The second operand.
    axes : tuple
        The axes to contract.

    Returns
    -------
    QuantumArray
        The QuantumArray containing the result of tensordot.

    Examples
    --------

    Using ``tensordot`` we can perform the arithmetic for simulating a quantum computer
    *on a quantum computer*.

    >>> import numpy as np
    >>> from qrisp import QuantumFloat, QuantumArray, tensordot

    Initiate the QuantumArray holding the statevector. We initate the state of uniform
    superposition

    .. math::

        \ket{+} = \frac{1}{\sqrt{2^n}} \sum_{i = 0}^{2^n - 1} \ket{i}

    >>> qfloat_type = QuantumFloat(3, -2, signed = True)
    >>> num_qubits = 4
    >>> statevector = QuantumArray(shape = 2**num_qubits, qtype = qfloat_type)
    >>> statevector[:] = [1/(2**num_qubits)**0.5]*2**num_qubits
    >>> print(statevector)
    {OutcomeArray([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                   0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]): 1.0}

    Initiate the QuantumArray holding the unitary of a Z-gate

    >>> z_gate = QuantumArray(shape = (2,2), qtype = qfloat_type)
    >>> z_gate[:] = [[1,0], [0,-1]]
    >>> print(z_gate)
    {OutcomeArray([[ 1.,  0.],
                  [ 0., -1.]]): 1.0}

    Perform the contraction

    >>> statevector = statevector.reshape(num_qubits*[2])
    >>> target_qubit = 3
    >>> new_statevector = tensordot(z_gate, statevector, (1, target_qubit))
    >>> new_statevector = new_statevector.reshape(2**num_qubits)
    >>> print(new_statevector)
    {OutcomeArray([ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
                   -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]): 1.0}

    We perform a similar contraction using numpy arrays

    >>> from numpy import tensordot
    >>> statevector = 0.25*np.ones(2**num_qubits)
    >>> print(statevector)
    [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
     0.25 0.25]
    >>> statevector = statevector.reshape(num_qubits*[2])
    >>> z_gate = np.zeros((2,2))
    >>> z_gate[:] = [[1,0], [0,-1]]
    >>> new_statevector = tensordot(z_gate, statevector, (1, target_qubit))
    >>> print(new_statevector.reshape(2**num_qubits))
    [ 0.25  0.25  0.25  0.25  0.25  0.25  0.25  0.25 -0.25 -0.25 -0.25 -0.25
     -0.25 -0.25 -0.25 -0.25]

    """

    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = np.asanyarray(a), np.asanyarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True

    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)

    res = dot(at, bt)
    return res.reshape(olda + oldb)
