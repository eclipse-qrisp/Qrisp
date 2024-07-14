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

from qrisp import *
import sympy as sp
from sympy import Symbol, Quaternion, I
import numpy as np

#
# helper functions
#

def delta(i, j):
    if i==j:
        return 1
    else:
        return 0
    
def epsilon(i, j, k):
    if (i, j, k) in (("X", "Y", "Z"), ("Y", "Z", "X"), ("Z", "X", "Y")):
        return 1
    if (i, j, k) in (("Z", "Y", "X"), ("Y", "X", "Z"), ("X", "Z", "Y")):
        return -1
    return 0

def set_bit(n,k):
    return n | (1 << k)        

#
# Pauli symbols
#

#class Spin(Symbol):
#    __slots__ = ("axis","index")

#    def __new__(cls, axis, index):
#        obj = Symbol.__new__(cls, "%s%d" %(axis,index), commutative=False, hermitian=True)
#        obj.index = index
#        obj.axis = axis
#        return obj       

class X(Symbol):

    __slots__ = ("axis","index")

    def __new__(cls, index):
        obj = Symbol.__new__(cls, "%s%d" %("X",index), commutative=False, hermitian=True)
        obj.index = index
        obj.axis = "X"
        return obj

    def get_quaternion(self):
        return Quaternion(0,I,0,0)
    
    def get_matrix(self):
        return np.array([[0,1],[1,0]])
    
    def get_axis(self):
        return "X" 
    
    def _eval_power(b, e):
        if e.is_Integer and e.is_positive:
            return super().__pow__(int(e) % 2)
        
    def __mul__(self, other):
        if isinstance(other, (X,Y,Z)):
            if self.index == other.index:
                i = self.axis
                j = other.axis
                return delta(i, j) \
                    + I*epsilon(i, j, "X")*X(self.index) \
                    + I*epsilon(i, j, "Y")*Y(self.index) \
                    + I*epsilon(i, j, "Z")*Z(self.index)
        return super().__mul__(other)

class Y(Symbol):

    __slots__ = ("axis","index")

    def __new__(cls, index):
        obj = Symbol.__new__(cls, "%s%d" %("Y",index), commutative=False, hermitian=True)
        obj.index = index
        obj.axis = "Y"
        return obj

    def get_quaternion(self):
        return Quaternion(0,0,I,0)
    
    def get_matrix(self):
        return np.array([[0,-1j],[1j,0]])
    
    def get_axis(self):
        return "Y"
    
    def _eval_power(b, e):
        if e.is_Integer and e.is_positive:
            return super().__pow__(int(e) % 2)

    def __mul__(self, other):
        if isinstance(other, (X,Y,Z)):
            if self.index == other.index:
                i = self.axis
                j = other.axis
                return delta(i, j) \
                    + I*epsilon(i, j, "X")*X(self.index) \
                    + I*epsilon(i, j, "Y")*Y(self.index) \
                    + I*epsilon(i, j, "Z")*Z(self.index)
        return super().__mul__(other)       
       
class Z(Symbol):

    __slots__ = ("axis","index")

    def __new__(cls, index):
        obj = Symbol.__new__(cls, "%s%d" %("Z",index), commutative=False, hermitian=True)
        obj.index = index
        obj.axis = "Z"
        return obj

    def get_quaternion(self):
        return Quaternion(0,0,0,I)
    
    def get_matrix(self):
        return np.array([[1,0],[0,-1]])
    
    def get_axis(self):
        return "Z"
    
    def _eval_power(b, e):
        if e.is_Integer and e.is_positive:
            return super().__pow__(int(e) % 2)
        
    def __mul__(self, other):
        if isinstance(other, (X,Y,Z)):
            if self.index == other.index:
                i = self.axis
                j = other.axis
                return delta(i, j) \
                    + I*epsilon(i, j, "X")*X(self.index) \
                    + I*epsilon(i, j, "Y")*Y(self.index) \
                    + I*epsilon(i, j, "Z")*Z(self.index)
        return super().__mul__(other)
    
#
#
#

def evaluate_observable(observable: int, x: int):
    """
    This method evaluates an observable that is a tensor product of Pauli-:math:`Z` operators
    with respect to a measurement outcome. 
        
    A Pauli operator of the form :math:`\prod_{i\in I}Z_i`, for some finite set of indices :math:`I\subset \mathbb N`, 
    is identified with an integer:
    We identify the Pauli operator with the binary string that has ones at positions :math:`i\in I`
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
    return quaternion.a-I*quaternion.b*X(index)-I*quaternion.c*Y(index)-I*quaternion.d*Z(index)


def simplify_spin(expr):
    """
    Simplifies a quantum Hamiltonian

    Parameters
    ----------
    H : sympy.expr

    Returns
    -------


    """

    #threshold = 1e-9
    threshold = 1e-12

    simplified_expr = 0

    for monomial in expr.expand().as_ordered_terms():
        factors = monomial.as_ordered_factors()

        simplified_factor = 1
        pauli_indices = []
        pauli_dict = {}

        for arg in factors:
            if isinstance(arg, (X,Y,Z)):
                if arg.index in pauli_indices:
                    pauli_dict[arg.index] *= arg.get_quaternion()
                else:
                    pauli_dict[arg.index] = arg.get_quaternion()   
                    pauli_indices.append(arg.index) 
            else:
                simplified_factor *= arg

        sorted_pauli_dict = dict(sorted(pauli_dict.items()))

        for index,quaternion in sorted_pauli_dict.items():
            simplified_factor *= convert_to_spin(quaternion, index)
        
        simplified_expr += simplified_factor

    # filter terms with small coefficient
    filtered_expr = sp.Add(*[term for term in simplified_expr.as_ordered_terms() if abs(term.as_coeff_Mul()[0]) >= threshold])

    return filtered_expr


def spin_operator_to_list(H):
    """

    Parameters
    ----------
    H : SymPy expr
        The quantum Hamiltonian.
    
    Returns
    -------
    terms : list
        The quantum Hamiltonian as list. 
        Each term is represetend by a tuple of a dictionary specifiying the Pauli product and a complex coefficient.

    Examples
    --------

    """

    terms = []
    expr = simplify_spin(H)

    for monomial in expr.expand().as_ordered_terms():
        factors = monomial.as_ordered_factors()

        pauli_dict = {}
        coeff = 1

        for arg in factors:
            if isinstance(arg, (X,Y,Z)):
                pauli_dict[arg.index] = arg.axis
            else:
                coeff *= arg

        terms.append((pauli_dict,coeff))

    return terms


def spin_operator_to_matrix(H):
    """

    Parameters
    ----------
    H : SymPy expr
        The quantum Hamiltonian.
    
    Returns
    -------
    M : numpy.array
        A matrix representation of the quantum Hamiltonian.

    Examples
    --------

    """

    from numpy import kron as TP

    I = np.array([[1,0],[0,1]])

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
            if isinstance(arg, (X,Y,Z)):
                spin_dict[arg.index] = arg.get_matrix()
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
        The method for evaluating the expected value of the Hamiltonian.
        Available is ``QWC``: Pauli terms are grouped based on qubit-wise commutativity.
        The default is None: The expected value of each Pauli term is computed independently.

    Returns
    -------
    measurement_circuits : list[QuantumCircuit]
    
    measurement_ops : list[list[int]]
    
    measurement_coeffs : list[list[float]]

    constant_term : float
        The constant term in the quantum Hamiltonian.

    """

    if method=='QWC':
        return qubit_wise_commutativity(qarg,spin_op)

    # no grouping (default):

    from qrisp import QuantumVariable, QuantumArray, QuantumCircuit

    if isinstance(qarg, QuantumArray):
        num_qubits = sum(qv.size for qv in list(qarg.flatten()))
    else:
        num_qubits = qarg.size
        
    measurement_circuits = []
    measurement_coeffs = []
    measurement_ops = []
    constant_term = 0

    expr = simplify_spin(spin_op)

    for monomial in expr.as_ordered_terms():
        factors = monomial.as_ordered_factors()

        qc = QuantumCircuit(num_qubits)
        meas_op = 0
        coeff = 1

        for arg in factors:
            if isinstance(arg, (X,Y,Z)):
                if arg.index >= num_qubits:
                    raise Exception("Insufficient number of qubits")

                if isinstance(arg, X):
                    qc.ry(-np.pi/2,arg.index)
                if isinstance(arg, Y):
                    qc.rx(np.pi/2,arg.index)

                meas_op = set_bit(meas_op, arg.index)
            else:
                coeff *= arg
        
        # exclude constant terms
        if meas_op==0:
            constant_term += coeff
        else:
            measurement_circuits.append(qc)
            measurement_ops.append([meas_op])
            measurement_coeffs.append([coeff])
        
    return measurement_circuits, measurement_ops, measurement_coeffs, constant_term


def qubit_wise_commutativity(qarg,spin_op):
    """


    """

    from qrisp import QuantumVariable, QuantumArray, QuantumCircuit

    if isinstance(qarg, QuantumArray):
        num_qubits = sum(qv.size for qv in list(qarg.flatten()))
    else:
        num_qubits = qarg.size

    pauli_dicts = []
    measurement_circuits = []
    measurement_coeffs = []
    measurement_ops = []
    constant_term = 0

    expr = simplify_spin(spin_op)

    for monomial in expr.as_ordered_terms():
        factors = monomial.as_ordered_factors()

        meas_op = 0
        coeff = 1

        curr_dict = {}
        for arg in factors:
            if isinstance(arg, (X,Y,Z)):
                if arg.index >= num_qubits:
                    raise Exception("Insufficient number of qubits")    
                
                curr_dict[arg.index]=arg.axis
                meas_op = set_bit(meas_op, arg.index)
            else:
                coeff *= arg

        # exclude constant terms
        if meas_op==0:
            constant_term += coeff
        else:
            # number of distict meaurement settings
            settings = len(pauli_dicts)
            commute_bool = False

            if settings > 0:   
                for k in range(settings):
                    # check if Pauli terms commute qubit-wise 
                    commute_bool = commute_qw(pauli_dicts[k],curr_dict)
                    if commute_bool:
                        pauli_dicts[k].update(curr_dict)
                        measurement_ops[k].append(meas_op)
                        measurement_coeffs[k].append(coeff)
                        break
            if settings==0 or not commute_bool: 
                pauli_dicts.append(curr_dict)
                measurement_ops.append([meas_op])
                measurement_coeffs.append([coeff])

    # construct change of basis circuits
    for dict in pauli_dicts:
        qc = QuantumCircuit(num_qubits)
        for index,axis in dict.items():
            if axis=="X":
                qc.ry(-np.pi/2,index)
            if axis=="Y":
                qc.rx(np.pi/2,index)  
        measurement_circuits.append(qc)      

    return measurement_circuits, measurement_ops, measurement_coeffs, constant_term


#
# commutativity checks
#


def commute_qw(a,b):
    """
    Check if two Pauli products commute qubit-wise.

    Parameters
    ----------
    a : dict
        A dictionary encoding a Pauli product.
    b : dict
        A dictionary encoding a Pauli product.

    """

    keys = set()
    keys.update(set(a.keys()))
    keys.update(set(b.keys()))

    for key in keys:
        if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
            return False
    return True


def commute(a,b):
    """
    Check if two Pauli products commute.

    Parameters
    ----------
    a : dict
        A dictionary encoding a Pauli product.
    b : dict
        A dictionary encoding a Pauli product.

    """

    keys = set()
    keys.update(set(a.keys()))
    keys.update(set(b.keys()))

    # Count non-commuting Pauli operators
    commute = True

    for key in keys:
        if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
            commute = not commute
    return commute

#
# helper functions for Hamiltonian simulation
#

def change_of_basis(qv, pauli_dict):
    for index, axis in pauli_dict:
        if axis=="X":
            qv.ry(-np.pi/2,index)
        if axis=="Y":
            qv.rx(np.pi/2,index)


def parity(qv, indices):
    n = len(indices)
    for i in range(n-1):
        cx(qv[indices[i]],qv[indices[i+1]])


def apply_Pauli(qv, pauli_dict, theta):

    indices = pauli_dict.keys()

    change_of_basis(qv, pauli_dict)
    parity(qv, indices)

    rz(theta,qv[indices[-1]])

    with invert():
        change_of_basis(qv, pauli_dict)
        parity(qv, indices)        



    


