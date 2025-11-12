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



from qrisp import control, cx , swap, QuantumFloat, QuantumArray, h, z, invert,ry, gate_wrap


from scipy.sparse.linalg import eigsh
from qrisp.alg_primitives.dicke_state_prep import dicke_state
from scipy.sparse import diags

import numpy as np 

def gauss_jordan_operations_general(matrix):
    """
    Extracts the actions of a Gauss-Jordan-elimination, which affect the relevants (last) column of an augmented matrix (the augmenting rows are not included in the input matrix) and returns the operations, 
    encoded as a list of strings (gates) and integers (qubits).

    Parameters
    ----------
    matrix : np.ndarray
        Definition of the matrix for extraction of the operations.

    Returns
    -------
    operations : List
        The operations on the relevant rows, encoded as a list of strings (gates) and integers (qubits).
    mat : np.ndarray
        The matrix after the operations.
     
    """
    # quick exit on empty or invalid matrix
    if not matrix or not matrix[0]:
        return [], []
    m = len(matrix)
    num_cols = len(matrix[0])
    n = num_cols - 1  # last column is RHS
    # work on a copy
    mat = [row.copy() for row in matrix]
    operations = []
    pivot_row = 0
    pivot_col = 0
    while pivot_row < m and pivot_col < n:
        # find pivot in this column at or below pivot_row
        pivot_idx = None
        for r in range(pivot_row, m):
            if mat[r][pivot_col] == 1:
                pivot_idx = r
                break
        if pivot_idx is None:
            pivot_col += 1
            continue

        # swap into pivot position if needed
        if pivot_idx != pivot_row:
            operations.append(('swap', pivot_row, pivot_idx))
            mat[pivot_row], mat[pivot_idx] = mat[pivot_idx], mat[pivot_row]

        # eliminate other rows
        pivot_data = mat[pivot_row]
        for r in range(m):
            if r != pivot_row and mat[r][pivot_col] == 1:
                operations.append(('xor', pivot_row, r))
                # XOR rows from pivot_col onward
                row_r = mat[r]
                row_r[pivot_col:] = [a ^ b for a, b in zip(row_r[pivot_col:], pivot_data[pivot_col:])]

        pivot_row += 1
        pivot_col += 1

    return operations, mat

def syndrome_decoding(matrix, qv_syndrome):
    """
    Performs the decoding step of the DQI algorithm on a binary ``QuantumArray``, by extracting the operations in a classical way. 
    Only the actions, which affect the relevants (last) column of an augmented matrix are considered. The row, by which the matrix is augmented is given by ``qv_syn``.

    Parameters
    ----------
    matrix : np.ndarray
        Definition of the matrix on which the the decoding is based on.
    qv_syndrome : QuantumArray
        ``QuantumArray`` to be decoded. 
    
    Returns
    -------
    mat : np.ndarray
        The matrix after the operations, according to function ``gauss_jordan_operations_general```.
    qv_syn : QuantumArray
        ``QuantumArray`` after decoding. 
     
    """

    qv_syn = qv_syndrome
    if isinstance(matrix, np.ndarray):
        mat_list = matrix.tolist()
    ops, mat = gauss_jordan_operations_general(mat_list)
    num_qubits = len(mat_list)
    for op_type, src, tgt in ops:
        if op_type == "swap":
            swap(qv_syn[src][0], qv_syn[tgt][0])
        elif op_type == "xor":
            cx(qv_syn[src][0], qv_syn[tgt][0])
    
    return mat , qv_syn






# m is number of constraints, p is the that defines F_p (=2 in case of MAX-XORSAT), r=1 in case of MAX-XORSAT
def get_optimal_w(m: int, l: int, p: int, r: int) -> np.ndarray:
    r"""
    Solve for the eigenvector of the symmetric tridiagonal matrix, which emerges from the expected number of solved constraints.
    Used for a fan-out Uniary Amplitude Encoding state-preparation. 
    (See Lemma 9.2 in the `DQI paper <https://arxiv.org/pdf/2408.08292>`_)

    Parameters
    ----------
    m : int
        The number of constraints, i.e. rows of the input matrix.
    l : int
        Degree for polynomial to encode objective function in. Can be chosen in an optimized way.
    p : int
        Defines the modulus of the field, i.e. $\mathcal{F}_p$.
    r : int
        Number of inputs yielding $+1$.    

    Returns
    -------
    weights : np.ndarray
        Eigenvector of the symmetric tridiagonal matrix.

    """
    d    = (p - 2*r) / np.sqrt(r * (p - r))
    diag = np.arange(l + 1) * d
    off  = np.sqrt(np.arange(1, l + 1) * (m - np.arange(1, l + 1) + 1))

    A = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")

    _, vecs = eigsh(A, k=1, which="LA")
    w = vecs.flatten()
    
    orig = len(w)
    pad_len = 2**int(np.ceil(np.log2(l+1)))
    weights = list(w)
    weights += [0 for i in range((pad_len -orig)) ]
    return weights


#@gate_wrap
def uae_encoding(qa_error, num_constraints, weights):
    r"""
    Fan-out Uniary Amplitude Encoding state-preparation, based on the weights received from solving for the eigenvector of the symmetric tridiagonal matrix.
    
    Parameters
    ----------
    qa_error : QuantumArray
        The ``QuantumArray`` (in the form of single qubit ``QuantumVariables``) to encode.
    num_constraints : int
        The number of constraints, i.e. rows of the input matrix.
    weights : np.ndarray
        Eigenvector of the symmetric tridiagonal matrix.

    """
    
    weights = weights/sum(weights)
    w2 = weights  
    cum = np.concatenate(([0.0], np.cumsum(w2[:-1])))  
    denom = 1.0 - cum
    # Avoid division by zero, clip ratios
    ratio = np.where(denom > 0, w2 / denom, 0.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    betas = 2.0 * np.arccos(np.sqrt(ratio))

    # Apply first RY if nonzero
    if betas[0] != 0.0 and not np.isnan(betas[0]):
       ry(betas[0], qa_error[0][0])

    # Controlled rotations
    for i in range(1, num_constraints):
        if i < betas.size:
            b = betas[i]
            if b != 0.0 and not np.isnan(b):
                with control( qa_error[i-1][0]):
                    ry(b,  qa_error[i][0])


def specific_phase_encoding(qa_error, v):
    r"""
    Specific phase encoding for DQI problem instance.

    Parameters
    ----------
    qa_error : QuantumArray
        The ``QuantumArray`` (in the form of single qubit ``QuantumVariables``) to encode.
    v_j : List
        Phase information corresponding to the problem constraints

    """
    for index in range(len(v)):
        if v[index] == 1:
            z(qa_error[index])



def constraint_encoding(qa_error, qv_syndrome, B):
    r"""
    Constraint encoding for DQI problem instance.

    Parameters
    ----------
    qa_error : QuantumArray
        The ``QuantumArray`` (in the form of single qubit ``QuantumVariables``) which represents the error register.
    qa_error : QuantumArray
        The ``QuantumArray`` (in the form of single qubit ``QuantumVariables``) which represents the syndrome register.
    B : np.ndarray
        Problem matrix to encode.

    """
    # reverse indices due to circuit definition
    qa_error, qv_syndrome = qa_error[::-1], qv_syndrome[::-1]

    # create B^T
    B_t = np.transpose(B)

    # encode constraints
    i_ind, j_ind = B_t.shape
    for i in range(i_ind):
        for j in range(j_ind):
            if  B_t[i][j] == 1: 
                cx(qa_error[j], qv_syndrome[i])

    return qa_error, qv_syndrome






def DQI(B, v, l=2, p=2, r=1):
    r"""
    Implementation of `Optimization by Decoded Quantum Interferometry <https://arxiv.org/pdf/2408.08292>`_, with inspiration from `Quantum Circuit Design for Decoded Quantum
    Interferometry <https://arxiv.org/pdf/2504.18334>`_.
    This algorithm is used to solve optimization problems, by using the quantum
    Fourier transform (or more so properties) to reduce optimization problems to decoding problems.
    For an indepth explanation, please refer to the referenced papers.

    Parameters
    ----------
    B : np.ndarray
        Definition of the problem instance.
    v : List
        Constraints enforced on the solution.
    l : int
        Degree for polynomial to encode objective function in. Can be chosen in an optimized way.
    p : int
        Defines the modulus of the field, i.e. $\mathcal{F}_p$.
    r : int
        Number of inputs yielding $+1$.    

    Returns
    -------
    qa_error : QuantumArray
        The ``QuantumArray`` (in the form of single qubit ``QuantumVariables``) which represents the final error register.
    qa_error : QuantumArray
        The ``QuantumArray`` (in the form of single qubit ``QuantumVariables``) which represents the final syndrome register.
    
    Examples
    --------   

    We create an example case for a maxCut instance. 
    
    ::

        B  = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1]
        ])

        v = np.array(
            [1, 1, 1, 1, 1, 1],
            ) 



        qvin, resqf = dqi_optimization(B, v)
        res_dict = multi_measurement([qvin,  resqf])

    """
    # L =2 for MaxCut 
    
    m = B.shape[0]
    n = B.shape[1] 
    q_type = QuantumFloat(1, 0)
    qa_error = QuantumArray(qtype= q_type , shape=(m,))

    # create syndrome quantumArray
    qa_syndrome = QuantumArray(qtype= q_type , shape=(m,))

    # l can be chosen to be equal to m, or as rank(B) as an optimized version (or even more optimized)
    # --> it defines the uae_eigenvector consideration and size qv_inner,
    princ_eigenvec = get_optimal_w(m,l,p,r)
    
    # UAE encoding encoding
    uae_encoding(qa_error, m, princ_eigenvec)

    # need to reverse due to dicke_state function definition
    dicke_state(qa_error[::-1], len(qa_error)) 

    # phase encoding
    specific_phase_encoding(qa_error, v)

    # constraint encoding
    qa_error, qa_syndrome = constraint_encoding(qa_error, qa_syndrome, B)

    # syndrome decoding step, this will in the future be replaced by a quantum version
    syndrome_decoding(B.T, qa_syndrome)

    # then perform further application onto the error register
    for i in range(m):
        cx(qa_syndrome[i], qa_error[i]) 
    
    with invert():
        #quantum_GJE(Bt_full) 
        syndrome_decoding(B.T, qa_syndrome)

    # H-transform for readout
    h(qa_syndrome)   


    return qa_error, qa_syndrome


