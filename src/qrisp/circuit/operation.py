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


import copy

import numpy as np
from sympy.core.expr import Expr
from sympy import lambdify

from qrisp.jisp.primitives import QuantumPrimitive, AbstractQuantumCircuit

# Class that describes an operation which can be performed on a quantum computer
# Example would be an X gate or a measurement
class Operation(QuantumPrimitive):
    """
    This class describes operations like quantum gates, measurements or classical logic
    gates. Operation objects do not carry information about which Qubit/Clbits they are
    applied to. This can be found in the Instruction class, which is a combination of an
    Operation object together with its operands.

    Operation objects consist of five basic attributes:

    * ``.name`` : A string identifying the operation
    * ``.num_qubits`` : An integer specifying the amount qubits on which to operate
    * ``.num_clbits`` : An integer specifying the amount of classical bits on which to
      operate.
    * ``.params`` : A list of floats specifying the parameters of the Operation
    * ``.definition`` : A :ref:`QuantumCircuit`. For synthesized (i.e. non-elementary)
      operations, this QuantumCircuit specifies the operation.

    Operation objects together with their Operands can be appended to QuantumCircuits by
    using the :meth:`append <qrisp.QuantumCircuit.append>` method.

    QuantumCircuits can be turned into Operations by using the
    :meth:`to_gate <qrisp.QuantumCircuit.to_gate>` method.

    Examples
    --------

    We create a QuantumCircuit and append a couple of operations

    >>> from qrisp import QuantumCircuit, XGate, CXGate, PGate
    >>> qc = QuantumCircuit(2)
    >>> qc.append(XGate(), 0)
    >>> qc.append(CXGate(), [0,1])
    >>> qc.append(PGate(0.5), 1)
    >>> synthed_op = qc.to_op()
    >>> qc.append(synthed_op, qc.qubits)

    """

    # If only given the operation init_op (which can be portable) a copied instance of
    # this operation will be returned
    def __init__(
        self,
        name=None,
        num_qubits=0,
        num_clbits=0,
        definition=None,
        params=[],
        init_op=None,
    ):
        if init_op is not None:
            name = init_op.name
            num_qubits = init_op.num_qubits
            num_clbits = init_op.num_clbits
            params = init_op.params
            definition = init_op.definition

        elif not isinstance(name, str):
            raise Exception("Tried to create a Operation with name of type({type(name)} (required is str)")

        # Name of the operation - this is how the backend behind the interface will
        # identify the operation
        self.name = name

        # Amount of qubits
        self.num_qubits = num_qubits

        # Amount of classical bits
        self.num_clbits = num_clbits

        # List of parameters (also available behind the interface)
        self.params = []
        
        QuantumPrimitive.__init__(self, name)
        
        @self.def_abstract_eval
        def abstract_eval(qc, *args):
            """Abstract evaluation of the primitive.
            
            This function does not need to be JAX traceable. It will be invoked with
            abstractions of the actual arguments. 
            """
            if not isinstance(qc, AbstractQuantumCircuit):
                raise Exception(f"Tried to execute Operation.bind with the first argument of tpye {type(qc)} instead of AbstractQuantumCircuit")
            
            return AbstractQuantumCircuit()
        
        @self.def_impl
        def append_impl(qc, *args):
            """Concrete evaluation of the primitive.
            
            This function does not need to be JAX traceable. It will be invoked with
            actual instances. 
            """
            qc.append(self, args)
            return qc
        

        # If a definition circuit is given, this means we are supposed to create a
        # non-elementary operation
        if definition is not None:
            # Copy circuit in order to prevent modification
            # self.definition = QuantumCircuit(init_qc = definition)
            self.definition = definition

            self.abstract_params = set(definition.abstract_params)
        else:
            self.definition = None
            self.abstract_params = set()


        # Find abstract parameters (ie. sympy expressions and log them)
        for par in params:
            if isinstance(par, np.number):
                par = par.item()
            elif isinstance(par, Expr):
                if len(par.free_symbols):
                    self.abstract_params = self.abstract_params.union(par.free_symbols)
                else:
                    par = float(par)
            elif not isinstance(par, (float, int, complex)):
                raise Exception(
                    f"Tried to create operation with parameters of type {type(par)}"
                )

            self.params.append(par)

        # These attributes store some information for the uncomputation algorithm
        # Qfree basically means that the unitary is a permutation matrix
        # (up to local phase shifts). Permeability means that this gate commutes with
        # the z operator on a given qubit
        self.is_qfree = None
        self.permeability = {i: None for i in range(self.num_qubits)}

    def copy(self):
        """
        Returns a copy of the Operation object.

        Returns
        -------
        Operation
            The copied operation.

        """

        res = copy.copy(self)
        if self.definition:
            copied_definition = self.definition.copy()
        else:
            copied_definition = None

        res.definition = copied_definition

        return res

    # Method to get the unitary matrix of the operation

    # The parameter decimals has no influence on what is calculated
    # Rounding is usefull here because the floating point errors
    # sometimes make it hard to read the unitary
    def get_unitary(self, decimals=-1):
        """
        Returns the unitary matrix (if applicable) of the Operation as a numpy array.

        Parameters
        ----------
        decimals : int, optional
            Amount of decimals to return. By default, the full precision is returned.

        Raises
        ------
        Exception
            Could not calculate the unitary.

        Returns
        -------
        numpy.ndarray
            The unitary matrix of the Operation.


        Examples
        --------

        >>> from qrisp import CPGate
        >>> import numpy as np
        >>> CPGate(np.pi/2).get_unitary(decimals = 3)
        array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j]], dtype=complex64)
        """

        if self.name == "barrier":
            from qrisp.simulator.unitary_management import np_dtype
            return np.eye(2**self.num_qubits, dtype = np_dtype)

        # Check if the unitary is already available
        if hasattr(self, "unitary"):
            if decimals == -1:
                return self.unitary
            else:
                return np.round(self.unitary, decimals)

        # If we are dealing with a non-elementary gate, calculate the unitary from
        # the definition circuit
        else:
            if not isinstance(self.definition, type(None)):
                self.unitary = self.definition.get_unitary()
                return self.get_unitary()

            # If no definition circuit is known, raise an error.
            # Note that the get_unitary methods of more specific gate families specified
            # by the inheritors of this class
            else:
                raise Exception("Unitary of operation " + self.name + " not defined.")

    # Method to return the inverse of the given operation. Again, the methods of more
    # specific gate families are specified by the inheritors of this class
    def inverse(self):
        """
        Returns the inverse of this Operation (if applicable).

        Raises
        ------
        Exception
            Tried to invert non-unitary operation.

        Returns
        -------
        Operation
            The daggered operation.

        Examples
        --------

        We invert a phase gate and inspect it's parameters

        >>> from qrisp import PGate
        >>> phase_gate = PGate(0.8)
        >>> phase_gate.inverse().params
        [-0.8]

        """

        # Check if the instruction contains classical bits => operation is not
        # invertible
        if self.num_clbits:
            raise Exception("Tried to invert non-unitary operation")

        # Check if a definition is available and invert it, if so
        if self.definition is not None:
            inverse_circ = self.definition.inverse()
            
            if self.name[-3:] == "_dg":
                res = inverse_circ.to_op(name=self.name[:-3])
            else:
                res = inverse_circ.to_op(name=self.name + "_dg")
                

        elif self.name == "qb_alloc":
            from qrisp.circuit import QubitDealloc

            res = QubitDealloc()
        elif self.name == "qb_dealloc":
            from qrisp.circuit import QubitAlloc

            res = QubitAlloc()
        elif self.name == "barrier":

            res = self.copy()
        # Otherwise raise an error
        else:
            raise Exception("Don't know how to invert Operation " + self.name)

        res.is_qfree = self.is_qfree
        res.permeability = dict(self.permeability)

        return res

    # Method to create a controlled gate
    def control(self, num_ctrl_qubits=1, ctrl_state=-1, method=None):
        """
        Returns the controlled version of this Operation (if applicable).

        Parameters
        ----------
        num_ctrl_qubits : int, optional
            The amount of control qubits. The default is 1.
        ctrl_state : int or str, optional
            The state on which to activate the basis gate. The default is "1111...".
        method : str, optional
            The method for synthesizing the required multi-controlled X gates.
            Available are ``gray`` and ``gray_pt`` and ``auto``.
            Note that "gray_pt" introduces an extra phase (which needs to be uncomputed)
            but is more resource efficient. ``auto`` will be transformed into a more
            efficient ancilla supported version at compile time if used in a
            QuantumSession. The default is ``gray``.

        Raises
        ------
        AttributeError
            Tried to control non-unitary operation.

        Returns
        -------
        Operation
            The controlled operation.

        Examples
        --------

        We control a parametrized X Rotation.

        >>> from qrisp import QuantumCircuit, RXGate
        >>> mcrx_gate = RXGate(0.5).control(3)
        >>> qc = QuantumCircuit(4)
        >>> qc.append(mcrx_gate, qc.qubits)
        >>> print(qc)
        
        ::
        
            qb_4: ─────■─────
                       │
            qb_5: ─────■─────
                       │
            qb_6: ─────■─────
                  ┌────┴────┐
            qb_7: ┤ Rx(0.5) ├
                  └─────────┘


        """

        if method is None:
            method = "auto"

        if self.num_clbits != 0:
            raise AttributeError("Tried to control non-unitary operation")
            
        res_num_ctrl_qubits = num_ctrl_qubits
        
        if isinstance(self, PTControlledOperation):
            res_num_ctrl_qubits += len(self.controls)
        
        # Check if the method is phase tolerant
        if (method.find("pt") != -1 or method.find("gidney") != -1) and res_num_ctrl_qubits != 1:
            
            return PTControlledOperation(
                self, num_ctrl_qubits, ctrl_state=ctrl_state, method=method
            )
        else:
            return ControlledOperation(
                self, num_ctrl_qubits, ctrl_state=ctrl_state, method=method
            )

    # TO-DO implement more robust hashing method
    def __hash__(self):
        return hash(hash(self.name) + hash(tuple(self.params)))

    def is_permeable(self, indices):
        from qrisp.permeability import is_permeable

        return is_permeable(self, indices)

    def bind_parameters(self, subs_dic):
        """
        Binds abstract parameters to specified values.

        Parameters
        ----------
        subs_dic : dict
            A dictionary containing the parameters as keys.

        Returns
        -------
        res : Operation
            The Operation with bound parameters.

        Examples
        --------

        We create a phase gate with an abstract parameter and bind it to a specified
        value.

        >>> from qrisp import PGate
        >>> from sympy import Symbol
        >>> phi = Symbol("phi")
        >>> abstract_p_gate = PGate(phi)
        >>> abstract_p_gate.params
        [phi]
        >>> bound_p_gate = abstract_p_gate.bind_parameters({phi : 1.5})
        >>> bound_p_gate.params
        [1.5]
        """
        
        
        new_params = []
        repl_args = [subs_dic[symb] for symb in self.abstract_params]
        
        if not hasattr(self, "lambdified_params"):
            self.lambdified_params = []
            args = list(self.abstract_params)
            for par in self.params:
                self.lambdified_params.append(lambdify(args, par, modules = "numpy"))
                
            
        for l_par in self.lambdified_params:
            
            new_params.append(l_par(*repl_args))
        
        res = self.copy()
        res.params = new_params

        if res.definition is not None:
            res.definition = res.definition.bind_parameters(subs_dic)

        return res
    
    def c_if(self, num_control=1, ctrl_state=-1):
        if ctrl_state == -1:
            ctrl_state = 2**num_control - 1

        return ClControlledOperation(self, num_control, ctrl_state)

# Class to describe 1-Qubit gates as unitaries of the form U(theta, phi, lam) =
# = exp(-1j*phi/2*sigma_z) exp(-1j*theta/2*sigma_y) exp(-1j*lam/2*sigma_z)
# See https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html
# for more information
class U3Gate(Operation):
    def __init__(self, theta, phi, lam, name="u3", global_phase=0):
        
        # Initialize Operation instance
        super().__init__(
            name=name,
            num_qubits=1,
            num_clbits=0,
            definition=None,
            params=[theta, phi, lam],
        )
        
        if isinstance(global_phase, Expr):
            if len(global_phase.free_symbols):
                self.abstract_params = self.abstract_params.union(global_phase.free_symbols)
            else:
                global_phase = float(global_phase)
        self.global_phase = global_phase

        # Set parameters
        self.theta = self.params[0]
        self.phi = self.params[1]
        self.lam = self.params[2]

        if self.name in ["rx", "ry", "rz", "p"]:
            self.params = [sum(self.params)]

            if self.name in ["rz", "p"]:
                self.permeability[0] = True
                self.is_qfree = True
            else:
                self.permeability[0] = False
                self.is_qfree = False

        elif self.name in ["h"]:
            self.params = []

            self.permeability[0] = False
            self.is_qfree = False

    # Specify inversion method
    def inverse(self):
        # The inverse of a product of matrices if the reverted product of the inverses,
        # i.e. (A*B*C)^(-1) = C^-1 * B^-1 * A^-1

        if self.name[-3:] == "_dg":
            new_name = self.name[:-3]
        else:
            new_name = self.name + "_dg"

        # For exponentials of hermitian matrices, the inverse is the hermitean
        # conjugate, which implies that we simply have to negate the parameters
        # theta, phi, lambda
        res = U3Gate(
            -self.theta,
            -self.lam,
            -self.phi,
            name=new_name,
            global_phase=-self.global_phase,
        )

        if self.name == "u3":
            res.name = "u3"

        # These are special gates that require only a single parameter
        if self.name in ["rx", "ry", "rz", "p", "h", "gphase"]:
            res.name = self.name
            res.params = [-par for par in self.params]

        if self.name in ["s", "t", "s_dg", "t_dg"]:
            res.params = []

        if res.is_qfree is not None:
            res.is_qfree = bool(self.is_qfree)

        res.permeability = dict(self.permeability)
        res.is_qfree = bool(self.is_qfree)

        return res

    # Method to calculate the unitary matrix
    def get_unitary(self, decimals=-1):
        if hasattr(self, "unitary"):
            if decimals != -1:
                return np.around(self.unitary, decimals)
            else:
                return self.unitary
        else:
            from qrisp.simulator.unitary_management import u3matrix

            # Generate unitary

            self.unitary = u3matrix(self.theta, self.phi, self.lam, self.global_phase, use_sympy = bool(len(self.abstract_params)))

            return self.get_unitary(decimals)

    def bind_parameters(self, subs_dic):
        
        new_params = []
        repl_args = [subs_dic[symb] for symb in self.abstract_params]
        
        if not hasattr(self, "lambdified_params"):
            self.lambdified_params = []
            args = list(self.abstract_params)
            for par in [self.theta, self.phi, self.lam, self.global_phase]:
                self.lambdified_params.append(lambdify(args, par, modules = "numpy"))
                
            
        for l_par in self.lambdified_params:
            
            new_params.append(l_par(*repl_args))
        
        return U3Gate(new_params[0], new_params[1], new_params[2], self.name, new_params[3])
        



pi = float(np.pi)

# This class is special for pauli gates. In principle, we could also use the U3Gate
# class, but this could lead to unneccessary floating point errors
class PauliGate(U3Gate):
    def __init__(self, name):
        from qrisp.simulator.unitary_management import pauli_x, pauli_y, pauli_z
        
        if name == "x":
            super().__init__(pi, 0, pi)
            self.unitary = pauli_x

            self.is_qfree = True
            self.permeability[0] = False

        elif name == "y":
            super().__init__(pi, pi / 2, pi / 2)
            self.unitary = pauli_y

            self.is_qfree = True
            self.permeability[0] = False

        elif name == "z":
            super().__init__(0, 0, pi)
            self.unitary = pauli_z

            self.is_qfree = True
            self.permeability[0] = True

        else:
            raise Exception("Gate " + name + " is not a Pauli gate")

        self.name = name
        self.params = []

    def inverse(self):
        return PauliGate(self.name)

    def __repr__(self):
        return self.name


# This class describes phase tolerant controlled operations
# Phase tolerant means that the unitary can take the form
# [D_0, 0, 0, 0]
# [0, D_1,0, 0]
# [0, 0, D_2, 0]
# [0, 0, 0, U]
#Where U is the operation to be controlled and D_i are diagonal operators

#For a regular controlled gate, all D_i have to be identity matrices.

#Phase tolerant controlled operations are usally more efficient than their controlled equivalent.

#In many cases, they can replace the controlled version without changing the semantics,
#because the phases introduced by the D_i are uncomputed at some later point.
class PTControlledOperation(Operation):
    def __init__(self, base_operation, num_ctrl_qubits=1, ctrl_state=-1, method="auto"):
        # Object which describes the method. Can be a string lr a callable function
        self.method = method

        # QuantumOperation object which describes which action is controlled,
        # i.e. for a CX gate, this would be an X gate
        self.base_operation = base_operation

        # List of control qubits
        self.controls = list(range(num_ctrl_qubits))

        # The control state - can be specified either as a string or as an int
        if ctrl_state == -1:
            self.ctrl_state = num_ctrl_qubits * "1"
        elif isinstance(ctrl_state, int):
            from qrisp.misc import bin_rep

            self.ctrl_state = bin_rep(ctrl_state, num_ctrl_qubits)[::-1]
        else:
            self.ctrl_state = str(ctrl_state)

        # Check if control state specification matches control qubit amount
        if len(self.ctrl_state) != num_ctrl_qubits:
            raise Exception(
                "Specified control state incompatible with given control qubit amount"
            )

        # Now we generate the definition circuit. Note that most of the generation
        # process also applies to the ControlledOperation class, however this class has
        # more specific method (.inverse() for instance),
        # which is why we make this distinction

        # If the base operation is a ControlledOperation, the result
        # PTControlledOperation can be generated by applying the phase tolerant control
        # algorithm to the base gate of the controlled operation in question
        elif isinstance(base_operation, PTControlledOperation):
            if method == "gray":
                method = self.method
            self.__init__(
                base_operation.base_operation,
                num_ctrl_qubits=num_ctrl_qubits + len(base_operation.controls),
                ctrl_state=self.ctrl_state + base_operation.ctrl_state,
                method=method,
            )
            return

        if base_operation.name == "gphase":
            from qrisp import PGate, QuantumCircuit, bin_rep

            definition_circ = QuantumCircuit(num_ctrl_qubits + 1)

            if num_ctrl_qubits > 1:
                temp_gate = PGate(base_operation.params[0]).control(
                    num_ctrl_qubits - 1, ctrl_state=self.ctrl_state[1:], method=method
                )
            else:
                temp_gate = PGate(base_operation.params[0])

            if self.ctrl_state[0] == "0":
                definition_circ.x(-2)

            definition_circ.append(temp_gate, definition_circ.qubits[:num_ctrl_qubits])

            if self.ctrl_state[0] == "0":
                definition_circ.x(-2)

        elif self.base_operation.name == "gray_phase_gate":
            raise
            from qrisp.circuit import multi_controlled_gray_circ

            definition_circ, target_phases = multi_controlled_gray_circ(
                self.base_operation, num_ctrl_qubits, self.ctrl_state
            )
            self.target_phases = target_phases
            self.phase_tolerant = False

        elif self.base_operation.name == "swap":
            from qrisp.circuit import fredkin_qc

            definition_circ = fredkin_qc(num_ctrl_qubits, ctrl_state, method)

        # For the case of a pauli gate with a single control, we insert an extra case
        # since here is no need for any advanced algorithm here and we do not need
        # to apply the phase tolerant naming convention
        elif (
            isinstance(base_operation, PauliGate)
            and num_ctrl_qubits == 1
            and self.ctrl_state == "1"
        ):
            if base_operation.name == "x":
                super().__init__(
                    name="cx", num_qubits=2, num_clbits=0, params=[]
                )
                self.permeability = {0: True, 1: False}
            elif base_operation.name == "y":
                super().__init__(
                    name="cy", num_qubits=2, num_clbits=0, params=[]
                )
                self.permeability = {0: True, 1: False}
            elif base_operation.name == "z":
                super().__init__(
                    name="cz", num_qubits=2, num_clbits=0, params=[]
                )
                self.permeability = {0: True, 1: True}

            self.is_qfree = True
            return

        # In the case of an u3gate as a base operation, we first check, if the method
        # object is callable, we generate the definition circuit from this function.
        # Otherwise we call the algorithm for generating controlled u3 gates
        elif isinstance(base_operation, U3Gate):
            if callable(method):
                definition_circ = method(self, num_ctrl_qubits, self.ctrl_state)
            else:
                from qrisp.circuit.controlled_operations import multi_controlled_u3_circ
                from qrisp.circuit import fast_append
                with fast_append(3):
                    definition_circ = multi_controlled_u3_circ(
                        base_operation,
                        num_ctrl_qubits,
                        ctrl_state=self.ctrl_state,
                        method=method,
                    )

        # If the base operation has a definition, we can simply apply the phase tolerant
        # control algorithm to every gate contained in this defintion.
        # This is done in the function multi_controlled_circuit

        elif base_operation.definition:
            from qrisp.circuit import multi_controlled_circuit

            definition_circ = multi_controlled_circuit(
                base_operation.definition,
                num_ctrl_qubits,
                ctrl_state=self.ctrl_state,
                method=method,
            )

        # Raise exception if no possility of synthesizing a controlled game is known
        else:
            raise Exception(
                "Control method for gate " + base_operation.name + " not implemented"
            )

        # Generate gate name
        if num_ctrl_qubits == 1:
            name_prefix = "ptc"
        else:
            name_prefix = "pt" + str(num_ctrl_qubits) + "c"

        Operation.__init__(
            self,
            name=name_prefix + base_operation.name,
            num_qubits=base_operation.num_qubits + num_ctrl_qubits,
            num_clbits=0,
            definition=definition_circ,
            params=base_operation.params,
        )

        for i in range(self.num_qubits):
            if i < num_ctrl_qubits:
                self.permeability[i] = True
            else:
                self.permeability[i] = base_operation.permeability[i - num_ctrl_qubits]

        if base_operation.is_qfree is not None:
            self.is_qfree = bool(base_operation.is_qfree)

    def inverse(self):
        # Generate inverse operation by applying the constructor
        # to the inverse base_operation to get the meta-data right

        res = PTControlledOperation(
            self.base_operation.inverse(),
            len(self.controls),
            ctrl_state=self.ctrl_state,
            method=self.method,
        )

        # In order to make sure the definition circuit is also correct,
        # we invert the circuit if its invertible
        try:
            res.definition = self.definition.inverse()
        except AttributeError:
            pass

        if res.is_qfree is not None:
            res.is_qfree = bool(self.is_qfree)

        res.permeability = dict(self.permeability)

        return res

    def bind_parameters(self, subs_dic):
        from copy import copy
        res = copy(self)
        if not isinstance(self.definition, type(None)):
            res.definition = self.definition.bind_parameters(subs_dic)
        res.base_operation = self.base_operation.bind_parameters(subs_dic)
        res.params = res.base_operation.params
        res.abstract_params = set(self.base_operation.params) - set(subs_dic.keys())

        return res


# Class to describe controlled operation
# Very simlar to phase tolerant operations but with a more specifix naming
# convention, inversion algorithm and unitary generation algorithm
class ControlledOperation(PTControlledOperation):
    def __init__(self, base_operation, num_ctrl_qubits=1, ctrl_state=-1, method="gray"):
        super().__init__(
            base_operation,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
            method=method,
        )

        if num_ctrl_qubits == 1:
            name_prefix = "c"
        else:
            name_prefix = str(len(self.controls)) + "c"

        self.name = name_prefix + base_operation.name

    def inverse(self):
        return ControlledOperation(
            self.base_operation.inverse(),
            len(self.controls),
            ctrl_state=self.ctrl_state,
            method=self.method,
        )

    # For generating the unitary we don't have to generate the unitary by multiplying
    # the gates of the definition circuit.
    # Instead, we can use that the unitary of a controlled operation is
    # the identity matrix apart from the bottom right block matrix, where we find
    # the unitary of the base operation.

    def get_unitary(self, decimals=-1):
        if hasattr(self, "unitary"):
            if decimals != -1:
                return np.around(self.unitary, decimals)
            else:
                return self.unitary
        else:
            from qrisp.simulator.unitary_management import controlled_unitary

            self.unitary = controlled_unitary(self)
            return self.get_unitary(decimals)

class ClControlledOperation(Operation):
    
    def __init__(self, base_op, num_control = 1, ctrl_state = -1):
        
        if ctrl_state == -1:
            ctrl_state = num_control*"1"
        
        if isinstance(ctrl_state, int):
            from qrisp.misc import bin_rep
            ctrl_state = bin_rep(ctrl_state, num_control)[::-1]
        
        
        self.base_op = base_op
        self.num_control = num_control
        self.ctrl_state = ctrl_state
        
        Operation.__init__(self, 
                           name = "c_if_" + base_op.name, 
                           num_qubits = base_op.num_qubits,
                           num_clbits = base_op.num_clbits + num_control,
                           params = list(base_op.params))
        
        self.permeability = dict(base_op.permeability)
        
        
        
