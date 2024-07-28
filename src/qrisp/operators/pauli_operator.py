"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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
from qrisp.operators.spin import X_,Y_,Z_, to_pauli_dict
import numpy as np
import sympy as sp
from sympy import init_printing

# Initialize automatic LaTeX rendering
init_printing()

threshold = 1e-9

#
# Helper functions
#

def set_bit(n,k):
    return n | (1 << k)   

def mul_helper(P1,P2):
    pauli_table = {("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",-1j),
            ("Y","X"):("Z",-1j),("Y","Y"):("I",1),("Y","Z"):("X",1j),
            ("Z","X"):("Y",1j),("Z","Y"):("X",-1j),("Z","Z"):("I",1)}
    
    if P1=="I":
        return (P2,1)
    if P2=="I":
        return (P1,1)
    return pauli_table[(P1,P2)]

def mul_paulis(pauli1,pauli2):
    result_list = []
    result_coeff = 1
    a = dict(pauli1)
    b = dict(pauli2)
    keys = set()
    keys.update(set(a.keys()))
    keys.update(set(b.keys()))
    for key in sorted(keys):
        pauli, coeff = mul_helper(a.get(key,"I"),b.get(key,"I"))
        if pauli!="I":
            result_list.append((key,pauli))
        result_coeff *= coeff
    return tuple(result_list), result_coeff

#
# Commutativity checks
#

def commute_qw(a,b):
    """
    Checks if two Pauli products commute qubit-wise.
 
    Parameters
    ----------
    a : dict
        A dictionary encoding a Pauli product.
    b : dict
        A dictionary encoding a Pauli product.

    Returns
    -------

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
    Checks if two Pauli products commute.

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
# Trotterization
#

def change_of_basis(qarg, pauli_dict):
    for index, axis in pauli_dict.items():
        if axis=="X":
            ry(-np.pi/2,qarg[index])
        if axis=="Y":
            rx(np.pi/2,qarg[index])

def parity(qarg, indices):
    n = len(indices)
    for i in range(n-1):
        cx(qarg[indices[i]],qarg[indices[i+1]])

#
# Evaluate observable
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

#
# PauliOperator
#

class PauliOperator:
    r"""
    This class provides an efficient implementation of Pauli operators i.e.,
    operators of the form

    .. math::
        
        H=\sum\limits_{j}\alpha_jP_j 
            
    where $P_j=\prod_i\sigma_i^j$ is a Pauli product, 
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.

    Pauli operators are implemented by Python dictionaries where:

    * The key is an (ordered) tuple encoding a Pauli product.
      Each Pauli operator is represented by a tuple ``(i,"P")`` for $P=X,Y,Z$.
      For example: the Pauli product $X_0Y_1Z_2$ is represented as 
      ``((0,"X"),(1,"Y"),(2,"Z"))``
    * The value is the coefficent of the Pauli product.

    For example, the operator 

    .. math::

        1+2X_0+3X_0Y_1

    is represented as 

    ::
    
        {():1, ((0,"X"),):2, ((0,"X"),(1,"Y")):3}

    Parameters
    ----------
    arg : dict, optional
        A dictionary representing a Pauli operator.

    Examples
    --------

    An operator can be specified by a dictionary, or more conveniently expressed in terms of ``X``, ``Y``, ``Z``operators:

    ::
        
        from qrisp.operators import PauliOperator, X,Y,Z

        P1 = 1+2*X(0)+3*X(0)*Y(1)
        P2 = PauliOperator({():1,((0,'X'),):2,((0,'X'),(1,'Y')):3})
        P1+P2

    yields:

    .. math::

        2+4X_0+6X_0Y_1

    """

    def __init__(self, arg=None):

        if arg is None:
            self.pauli_dict = {():0}
        elif isinstance(arg, dict):
            self.pauli_dict = arg
        elif isinstance(arg, sympy.Basic):
            self.pauli_dict = to_pauli_dict(arg)
        else:
            raise TypeError("Cannot initialize from "+str(type(arg)))

    #
    # Arithmetic
    #

    def __add__(self,other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, commplex or PauliOperator
            A scalar or a PauliOperator to add to the operator self.

        Returns
        -------
        result : PauliOperator
            The sum of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot add PauliOperator and "+str(type(other)))

        res_pauli_dict = {}

        for pauli,coeff in self.pauli_dict.items():
            res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0)+coeff
            if abs(res_pauli_dict[pauli])<threshold:
                del res_pauli_dict[pauli]
    
        for pauli,coeff in other.pauli_dict.items():
            res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0)+coeff
            if abs(res_pauli_dict[pauli])<threshold:
                del res_pauli_dict[pauli]
        
        result = PauliOperator(res_pauli_dict)
        return result

    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, commplex or PauliOperator
            A scalar or a PauliOperator to substract from the operator self.

        Returns
        -------
        result : PauliOperator
            The difference of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot substract PauliOperator and "+str(type(other)))

        res_pauli_dict = {}

        for pauli,coeff in self.pauli_dict.items():
            res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0)+coeff
            if abs(res_pauli_dict[pauli])<threshold:
                del res_pauli_dict[pauli]
    
        for pauli,coeff in other.pauli_dict.items():
            res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0)-coeff
            if abs(res_pauli_dict[pauli])<threshold:
                del res_pauli_dict[pauli]
        
        result = PauliOperator(res_pauli_dict)
        return result
    
    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, commplex or PauliOperator
            A scalar or a PauliOperator to multiply with the operator self.

        Returns
        -------
        result : PauliOperator
            The product of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot multipliy PauliOperator and "+str(type(other)))

        res_pauli_dict = {}

        for pauli1, coeff1 in self.pauli_dict.items():
            for pauli2, coeff2 in other.pauli_dict.items():
                curr_tuple, curr_coeff = mul_paulis(pauli1,pauli2)
                res_pauli_dict[curr_tuple] = res_pauli_dict.get(curr_tuple,0) + curr_coeff*coeff1*coeff2

        result = PauliOperator(res_pauli_dict)
        return result

    __radd__ = __add__
    __rmul__ = __mul__

    #
    # Inplace operations
    #

    def __iadd__(self,other):
        """
        Adds other to the operator self.

        Parameters
        ----------
        other : int, float, commplex or PauliOperator
            A scalar or a PauliOperator to add to the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.pauli_dict[()] = self.pauli_dict.get((),0)+other
            return self
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot add PauliOperator and "+str(type(other)))

        for pauli,coeff in other.pauli_dict.items():
            self.pauli_dict[pauli] = self.pauli_dict.get(pauli,0)+coeff
            if abs(self.pauli_dict[pauli])<threshold:
                del self.pauli_dict[pauli]       
        return self           

    def __isub__(self,other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, commplex or PauliOperator
            A scalar or a PauliOperator to substract from the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.pauli_dict[()] = self.pauli_dict.get((),0)-other
            return self
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot add PauliOperator and "+str(type(other)))

        for pauli,coeff in other.pauli_dict.items():
            self.pauli_dict[pauli] = self.pauli_dict.get(pauli,0)-coeff
            if abs(self.pauli_dict[pauli])<threshold:
                del self.pauli_dict[pauli]  
        return self
    
    #
    # Miscellaneous
    #
    
    def apply_threshold(self,threshold):
        """
        Removes all Pauli terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the Pauli terms.

        """

        delete_list = []
        for pauli,coeff in self.pauli_dict.items():
            if abs(coeff)<threshold:
                delete_list.append(pauli)
        for pauli in delete_list:
            del self.pauli_dict[pauli]
    
    def _repr_latex_(self):
        # Convert the sympy expression to LaTeX and return it
        expr = self.to_expr()
        return f"${sp.latex(expr)}$"

    #
    # Measurement settings
    #

    def get_measurement_settings(self, qarg, method=None):
        """
        Returns the measurement settings to evaluate the operator. 

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument the spin operator is evaluated on.
        method : string, optional
            The method for evaluating the expected value of the Hamiltonian.
            Available is ``QWC``: Pauli terms are grouped based on qubit-wise commutativity.
            The default is ``None`: The expected value of each Pauli term is computed independently.

        Returns
        -------
        measurement_circuits : list[QuantumCircuit]
            The change of basis circuits.
        measurement_ops : list[list[int]]
            The Pauli products (after change of basis) to be measured represented as an integer.
        measurement_coeffs : list[list[float]]
            The coefficents of the Pauli products to be measured.
        constant_term : float
            The constant term.

        """

        if method=='QWC':
            return self.get_measurment_settings_qwc(qarg)

        # no grouping (default):

        from qrisp import QuantumVariable, QuantumArray, QuantumCircuit

        if isinstance(qarg, QuantumArray):
            num_qubits = sum(qv.size for qv in list(qarg.flatten()))
        else:
            num_qubits = qarg.size
        
        measurement_circuits = []
        measurement_coeffs = []
        measurement_ops = []
        constant_term = float(self.pauli_dict.get((),0).real)

        for pauli,coeff in self.pauli_dict.items():
            if pauli!=():
                qc = QuantumCircuit(num_qubits)
                meas_op = 0
                for item in pauli:
                    if item[0] >= num_qubits:
                        raise Exception("Insufficient number of qubits")
                    if item[1]=="X":
                        qc.ry(-np.pi/2,item[0])
                    if item[1]=="Y":
                        qc.rx(np.pi/2,item[0])
                
                    meas_op = set_bit(meas_op, item[0])

                measurement_circuits.append(qc)
                measurement_ops.append([meas_op])
                measurement_coeffs.append([float(coeff.real)])

        return measurement_circuits, measurement_ops, measurement_coeffs, constant_term    
    
    # Measurement settings for 'QWC' method
    def get_measurment_settings_qwc(self, qarg):

        from qrisp import QuantumVariable, QuantumArray, QuantumCircuit

        if isinstance(qarg, QuantumArray):
            num_qubits = sum(qv.size for qv in list(qarg.flatten()))
        else:
            num_qubits = qarg.size
        
        pauli_dicts, measurement_ops, index_ops, measurement_coeffs, constant_term = self.qubit_wise_commutativity()
        measurement_circuits = []

        # construct change of basis circuits
        for pauli in pauli_dicts:
            qc = QuantumCircuit(num_qubits)
            for item in pauli.items():
                if item[0] >= num_qubits:
                    raise Exception("Insufficient number of qubits")
                if item[1]=="X":
                    qc.ry(-np.pi/2,item[0])
                if item[1]=="Y":
                    qc.rx(np.pi/2,item[0])  
            measurement_circuits.append(qc)    

        return measurement_circuits, measurement_ops, measurement_coeffs, constant_term
    
    # partitions the operator in qubit-wise commuting groups
    def qubit_wise_commutativity(self):

        pauli_dicts = [] # list of Pauli products represented as dictionaries (change of basis)
        measurement_ops = [] # operators as integer
        index_ops = [] # operators as list of indices
        measurement_coeffs = []
        constant_term = float(self.pauli_dict.get((),0).real)

        for pauli,coeff in self.pauli_dict.items():
            if pauli!=():
                meas_op = 0
                curr_indices = []
                for item in pauli:
                    meas_op = set_bit(meas_op, item[0])
                    curr_indices.append(item[0])

                # number of distict meaurement settings
                settings = len(pauli_dicts)
                commute_bool = False
                curr_dict = dict(pauli)

                if settings > 0:   
                    for k in range(settings):
                        # check if Pauli terms commute qubit-wise 
                        commute_bool = commute_qw(pauli_dicts[k],curr_dict)
                        if commute_bool:
                            pauli_dicts[k].update(curr_dict)
                            measurement_ops[k].append(meas_op)
                            measurement_coeffs[k].append(float(coeff.real))
                            index_ops[k].append(curr_indices)
                            break
                if settings==0 or not commute_bool: 
                    pauli_dicts.append(curr_dict)
                    measurement_ops.append([meas_op])
                    measurement_coeffs.append([float(coeff.real)]) 
                    index_ops.append([curr_indices])

        return pauli_dicts, measurement_ops, index_ops, measurement_coeffs, constant_term

    #
    # Tools
    #

    def to_expr(self):
        """
        Returns a SymPy expression representing the operator.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the operator.

        """
        
        expr = 0

        def to_spin(P, index):
            if P=="I":
                return 1
            if P=="X":
                return X_(index)
            if P=="Y":
                return Y_(index)
            else:
                return Z_(index)
        
        for pauli,coeff in self.pauli_dict.items():
            curr_expr = coeff
            for item in pauli:
                curr_expr *= to_spin(item[1],item[0])
            expr += curr_expr

        return expr

    def to_sparse_matrix(self):
        """
        Returns a matrix representing the operator.
    
        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.

        """

        import scipy.sparse as sp
        from scipy.sparse import kron as TP, csr_matrix

        I = csr_matrix([[1,0],[0,1]])

        def get_matrix(P):
            if P=="I":
                return csr_matrix([[1,0],[0,1]])
            if P=="X":
                return csr_matrix([[0,1],[1,0]])
            if P=="Y":
                return csr_matrix([[0,-1j],[1j,0]])
            else:
                return csr_matrix([[1,0],[0,-1]])

        def recursive_TP(keys,pauli_dict):
            if len(keys)==1:
                return get_matrix(pauli_dict.get(keys[0],"I"))
            return TP(get_matrix(pauli_dict.get(keys.pop(0),"I")),recursive_TP(keys,pauli_dict))

        pauli_dicts = []
        coeffs = []

        keys = set()
        for pauli,coeff in self.pauli_dict.items():
            curr_dict = dict(pauli)
            keys.update(set(curr_dict.keys()))
            pauli_dicts.append(curr_dict)    
            coeffs.append(coeff)

        keys = set()
        for item in pauli_dicts:
            keys.update(set(item.keys()))
        keys = sorted(keys)
        dim = len(keys)

        m = len(coeffs)
        M = sp.csr_matrix((2**dim, 2**dim))
        for k in range(m):
            M += complex(coeffs[k])*recursive_TP(keys.copy(),pauli_dicts[k])

        return M

    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.
    
        Returns
        -------
        E : float
            The ground state energy. 

        """

        from scipy.sparse.linalg import eigsh

        M = self.to_sparse_matrix()
        # Compute the smallest eigenvalue
        eigenvalues, _ = eigsh(M, k=1, which='SA')  # 'SA' stands for smallest algebraic
        E = eigenvalues[0]

        return E

    #
    # Trotterization
    #

    def trotterization(self):
        r"""
        Returns a function for appling the operator $e^{itH}$ via trotterization.

        Returns
        -------
        trotterize : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            This function recieves the following arguments:

            * qarg : QuantumVariable or QuantumArray
                The quantum argument.
            * t : float, optional
                The evolution time. The default is 1.
            * steps : int, optional
                The number of Trotter steps. The default is 1.
        
        """

        bases, ops, indices, coeffs, constant = self.qubit_wise_commutativity()

        def trotter_step(qarg, t, steps):
            N = len(bases)
            for k in range(N):
                basis = bases[k]
                with conjugate(change_of_basis)(qarg, basis):
                    M = len(ops[k])
                    for l in range(M):
                        with conjugate(parity)(qarg, indices[k][l]):
                            rz(t/steps,qarg[indices[k][l][-1]])
        
        def trotterize(qarg, t=1, steps=1):

            for n in range(steps):
                trotter_step(qarg, t, steps)

        return trotterize
    

