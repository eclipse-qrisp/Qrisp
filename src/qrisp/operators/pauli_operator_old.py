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

#from qrisp import *
from qrisp import QuantumVariable, QuantumArray, rx, ry, rz, h, cx
from qrisp.operators.hamiltonian import Hamiltonian
from qrisp.operators.pauli.spin import X_,Y_,Z_
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
# PauliOperator
#

class PauliOperator(Hamiltonian):
    r"""
    This class provides an efficient implementation of Pauli operators, i.e.,
    operators of the form

    .. math::
        
        H=\sum\limits_{j}\alpha_jP_j 
            
    where $P_j=\prod_i\sigma_i^j$ is a Pauli product, 
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.

    Parameters
    ----------
    arg : dict, optional
        A dictionary representing a Pauli operator.

    Examples
    --------

    A Pauli operator can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::
        
        from qrisp.operators import PauliOperator, X,Y,Z

        P1 = 1+2*X(0)+3*X(0)*Y(1)

    """

    def __init__(self, arg=None):

        if arg is None:
            self.pauli_dict = {():0}
        elif isinstance(arg, dict):
            self.pauli_dict = arg
        else:
            raise TypeError("Cannot initialize from "+str(type(arg)))

    def _repr_latex_(self):
        # Convert the sympy expression to LaTeX and return it
        expr = self.to_expr()
        return f"${sp.latex(expr)}$"
    
    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)
    
    def len(self):
        return len(self.pauli_dict)
    
    #
    # Arithmetic
    #

    def __add__(self,other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or PauliOperator
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
        other : int, float, complex or PauliOperator
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
        other : int, float, complex or PauliOperator
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
        other : int, float, complex or PauliOperator
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
        other : int, float, complex or PauliOperator
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
    
    def __imul__(self,other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to multiply with the operator self.

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

        self.pauli_dict = res_pauli_dict
    
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
            The default is ``None``: The expected value of each Pauli term is computed independently.

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


    # partitions the operator in commuting groups 
    def commuting_groups2(self):

        groups = [] #groups
        coeffs = [] 
        constant_term = float(self.pauli_dict.get((),0).real)

        for pauli,coeff in self.pauli_dict.items():
            settings = len(groups)

            commute_bool = False
            if settings > 0:
                for k in range(settings):
                    for pauli_ in groups[k]:
                        commute_bool = commute(dict(pauli_),dict(pauli))
                        if not commute_bool:
                            break
                    if commute_bool:
                        groups[k].append(pauli)
                        coeffs[k].append(coeff)
                        break
            if settings==0 or not commute_bool: 
                groups.append([pauli])
                coeffs.append([coeff])

        return groups, coeffs, constant_term
    
    # partitions the Pauli operator in groups of commuting Pauli product operators
    def commuting_groups(self):

        groups = [] # groups of commuting Pauli product operators

        for pauli,coeff in self.pauli_dict.items():

            commute_bool = False
            if len(groups) > 0:
                for group in groups:
                    for pauli_,coeff_ in group.pauli_dict.items():
                        commute_bool = commute(dict(pauli_),dict(pauli))
                        if not commute_bool:
                            break
                    if commute_bool:
                        group+=PauliOperator({pauli:coeff})
                        break
            if len(groups)==0 or not commute_bool: 
                groups.append(PauliOperator({pauli:coeff}))

        return groups

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
        Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $e^{itH}$ via Trotterization.

        Returns
        -------
        U : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            Given a Hamiltonian $H=H_1+\dotsb +H_m$ the unitary evolution $e^{itH}$ is 
            approximated by 
            
            .. math::

                e^{itH}\approx U_1(t,N)=\left(e^{iH_1t/N}\dotsb e^{iH_mt/N}\right)^N

            This function recieves the following arguments:

            * qarg : QuantumVariable or QuantumArray
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.
        
        """

        from qrisp import conjugate, rx, ry, rz, cx, h, IterationEnvironment, gphase

        bases, ops, indices, coeffs, constant = self.qubit_wise_commutativity()

        def trotter_step(qarg, t, steps):

            if constant != 0:
                gphase(t/steps*constant,qarg[0])

            N = len(bases)
            for k in range(N):
                basis = bases[k]
                with conjugate(change_of_basis)(qarg, basis):
                    M = len(ops[k])
                    for l in range(M):
                        with conjugate(parity)(qarg, indices[k][l]):
                            rz(-2*coeffs[k][l]*t/steps,qarg[indices[k][l][-1]])

        def U(qarg, t=1, steps=1, iter=1):
            with IterationEnvironment(qarg.qs, iter):
                for i in range(steps):
                    trotter_step(qarg, t, steps)

        return U

#
# Define X,Y,Z operators
#

class X(PauliOperator):

    def __init__(self, index):
        super().__init__({((index,'X'),):1})

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliOperator({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

class Y(PauliOperator):

    def __init__(self, index):
        super().__init__({((index,'Y'),):1})

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliOperator({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

class Z(PauliOperator):

    def __init__(self, index):
        super().__init__({((index,'Z'),):1})

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliOperator({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))



    

