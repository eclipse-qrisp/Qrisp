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
from sympy import Symbol, Quaternion, I
import numpy as np

class Spin(Symbol):

    __slots__ = ("axes", "index")

    def __new__(cls, axes, index):
        if axes not in ["X", "Y", "Z"]:
            raise IndexError("Invalid Pauli spin")
        obj = Symbol.__new__(cls, "%s%d" %(axes,index), commutative=False, hermitian=True)
        obj.axes = axes
        obj.index = index
        return obj

    def get_quaternion(self):
        if self.axes == "X":
            return Quaternion(0,I,0,0)
        elif self.axes == "Y":
            return Quaternion(0,0,I,0)
        else:
            return Quaternion(0,0,0,I)


def set_bit(n,k):
    return n | (1 << k)        

def evaluate_observable(observable: int, x: int):
    """
    This method evaluates an observable that is a tensor product of Pauli-:math:`Z` operators
    with respect to a measurement outcome. 
        
    A Pauli operator of the form :math:`\prod_{i\in I}Z_i`, for some finite set of indices :math:`I\subset \mathbb N`, 
    is identified with an integer:
    we identify the Pauli operator with the binary string that has ones at positions :math:`i\in I`
    and zeros otherwise, and then convert this binary string to an integer.
        
    Parameters
    ----------
        
    observable : int
        The observable represented as integer.
     x : int 
         The measurement outcome represented as integer.
        
    Returns
    -------
    int
        The value of the observable with respect to the measurement outcome.
        
    """
        
    if bin(observable & x).count('1') % 2 == 0:
        return 1
    else:
        return -1    


def convert_to_spin(quaternion, index):
    return quaternion.a-I*quaternion.b*Spin("X",index)-I*quaternion.c*Spin("Y",index)-I*quaternion.d*Spin("Z",index)


def simplify_spin(expr):
    simplified_expr = 0

    for monomial in expr.expand().as_ordered_terms():
        factors = monomial.as_ordered_factors()

        simplified_factor = 1
        pauli_indices = []
        pauli_dict = {}

        for arg in factors:
            if isinstance(arg, Spin):
                if arg.index in pauli_indices:
                    pauli_dict[arg.index] *= arg.get_quaternion()
                else:
                    pauli_dict[arg.index] = arg.get_quaternion()   
                    pauli_indices.append(arg.index) 

            elif isinstance(arg, sp.core.power.Pow,) and isinstance(arg.args[0], Spin):
                if arg.args[1]%2!=0:
                    if arg.args[0].index in pauli_indices:
                        pauli_dict[arg.args[0].index] *= arg.args[0].get_quaternion()
                    else:
                        pauli_dict[arg.args[0].index] = arg.args[0].get_quaternion()  
                        pauli_indices.append(arg.args[0].index)

            else:
                simplified_factor *= arg

        sorted_pauli_dict = dict(sorted(pauli_dict.items()))

        for index,quaternion in sorted_pauli_dict.items():
            simplified_factor *= convert_to_spin(quaternion, index)
        
        simplified_expr += simplified_factor

    return simplified_expr


def ground_state_energy(H):

    from sympy import I as i
    from sympy.physics.quantum import TensorProduct as TP
    import numpy as np

    I = sp.Matrix([[1,0],[0,1]])
    X = sp.Matrix([[0,1],[1,0]])
    Y = sp.Matrix([[0,-i],[i,0]])
    Z = sp.Matrix([[1,0],[0,-1]])

    def spin_matrix(str):
        if str=="X":
            return X
        if str=="Y":
            return Y
        else:
            return Z
        
    def recursive_TP(keys,spin_dict):
        if len(keys)==1:
            return spin_dict.get(keys[0],I)
        return TP(spin_dict.get(keys.pop(0),I),recursive_TP(keys,spin_dict))

    coeffs = []
    spin_dicts = []

    expr = simplify_spin(H)

    for monomial in expr.expand().as_ordered_terms():
        factors = monomial.as_ordered_factors()

        spin_dict = {}
        coeff = 1

        for arg in factors:
            if isinstance(arg, Spin):
                spin_dict[arg.index] = spin_matrix(arg.axes)
            else:
                coeff *= arg

        coeffs.append(coeff)
        spin_dicts.append(spin_dict)

    keys = set()
    for item in spin_dicts:
        keys.update(set(item.keys()))
    keys = sorted(keys)
    dim = len(keys)

    m = len(coeffs)
    M = sp.zeros(2**dim)
    for k in range(m):
        M += coeffs[k]*recursive_TP(keys.copy(),spin_dicts[k])

    eigenvalues = M.eigenvals()
    return min(eigenvalues)


def spin_operator_to_matrix(H):

    from numpy import kron as TP

    I = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])

    def spin_matrix(str):
        if str=="X":
            return X
        if str=="Y":
            return Y
        else:
            return Z
        
    def recursive_TP(keys,spin_dict):
        if len(keys)==1:
            return spin_dict.get(keys[0],I)
        return TP(spin_dict.get(keys.pop(0),I),recursive_TP(keys,spin_dict))

    coeffs = []
    spin_dicts = []

    expr = simplify_spin(H)

    for monomial in expr.expand().as_ordered_terms():
        factors = monomial.as_ordered_factors()

        spin_dict = {}
        coeff = 1

        for arg in factors:
            if isinstance(arg, Spin):
                spin_dict[arg.index] = spin_matrix(arg.axes)
            else:
                coeff *= arg

        coeffs.append(coeff)
        spin_dicts.append(spin_dict)

    keys = set()
    for item in spin_dicts:
        keys.update(set(item.keys()))
    keys = sorted(keys)
    dim = len(keys)

    m = len(coeffs)
    M = np.zeros((2**dim, 2**dim)).astype(np.complex128)
    for k in range(m):
        M += complex(coeffs[k])*recursive_TP(keys.copy(),spin_dicts[k])

    return M
        

def ground_state_energy(H):
    """
    Calculates the ground state energy of a spin operator classically.

    Parameters
    ----------
    H : SymPy expression
        The spin operator.
    
    Returns
    -------
    E : Float
        The ground state energy. 

    """

    M = spin_operator_to_matrix(H)
    eigenvalues = np.linalg.eigvals(M) 
    E = min(eigenvalues)
    return E


def get_measurement_settings(qarg, spin_op, method=None):
    """
    todo 

    Parameters
    ----------
    qarg : QuantumVariable or QuantumArray
        The argument the spin operator is evaluated on.
    spin_op : SymPy expr
        The quantum Hamiltonian.
    method : string, optional
        The default is None.

    Returns
    -------

    """
    
    from qrisp import QuantumVariable, QuantumArray, QuantumCircuit

    if isinstance(qarg, QuantumArray):
        num_qubits = sum(qv.size for qv in list(qarg.flatten()))
    else:
        num_qubits = qarg.size
        
    measurement_circuits = []
    measurement_coeffs = []
    measurement_ops = []

    expr = simplify_spin(spin_op)

    for monomial in expr.as_ordered_terms():
        factors = monomial.as_ordered_factors()

        qc = QuantumCircuit(num_qubits)
        meas_op = 0
        coeff = 1

        for arg in factors:
            if isinstance(arg, Spin):
                if arg.index >= num_qubits:
                    raise Exception("Insufficient number of qubits")

                if arg.axes=="X":
                    qc.ry(-np.pi/2,arg.index)
                if arg.axes=="Y":
                    qc.rx(np.pi/2,arg.index)

                meas_op = set_bit(meas_op, arg.index)
            else:
                coeff *= arg
        
        measurement_circuits.append(qc)
        measurement_ops.append(meas_op)
        measurement_coeffs.append(coeff)
        
    return measurement_circuits, measurement_ops, measurement_coeffs



