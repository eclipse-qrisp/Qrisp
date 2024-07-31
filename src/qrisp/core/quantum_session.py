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

import weakref

import numpy as np

from qrisp.circuit import Clbit, QuantumCircuit, Qubit, QubitAlloc, QubitDealloc, Instruction, Operation
from qrisp.core.session_merging_tools import multi_session_merge
from qrisp.misc import get_depth_dic


class QuantumSession(QuantumCircuit):
    """
    The QuantumSession class manages the life cycle of QuantumVariables and enables
    features such as :ref:`QuantumEnvironments <QuantumEnvironment>` or
    :ref:`Uncomputation`. To create a QuantumSession, we call the constructor

    >>> from qrisp import QuantumSession
    >>> qs = QuantumSession()

    To create a QuantumVariable within that QuantumSession, we hand it over to the
    QuantumVariable constructor:

    >>> from qrisp import QuantumVariable
    >>> qv = QuantumVariable(3, qs = qs)

    As an inheritor of the :ref:`QuantumCircuit` class, QuantumSession objects can also
    be used for circuit construction.

    >>> qv.qs.cx(qv[0], qv[1])

    Nevertheless, users are encouraged to use the :ref:`designated gate application
    function <gate_application_functions>` in order to reduce code cluttering.

    >>> from qrisp import cx
    >>> cx(qv[0], qv[1])

    QuantumSessions can be visualized by calling ``print`` on them.

    >>> print(qv.qs)
    
    ::
    
        QuantumCircuit:
        --------------
        qv.0: ──■────■──
              ┌─┴─┐┌─┴─┐
        qv.1: ┤ X ├┤ X ├
              └───┘└───┘
        qv.2: ──────────
        
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv

    If not specified, QuantumVariables will create their own QuantumSession and register
    themselves in it.

    QuantumSessions can be manually merged using the merge function.

    >>> from qrisp import merge
    >>> qs_2 = QuantumSession()
    >>> qs_2 == qs
    False
    >>> merge(qs, qs_2)
    >>> qs == qs_2
    True

    Note that merge also works for QuantumVariables, lists of QuantumSession and lists
    of QuantumVariables.

    If an entangling operation between two QuantumVariables which are registered in
    different QuantumSessions is executed, these QuantumSessions are automatically
    merged. For more details on automatic QuantumSession merging check the
    :ref:`session merging documentation<SessionMerging>`.

    >>> qv_a = QuantumVariable(2)
    >>> qv_b = QuantumVariable(2)
    >>> qv_a.qs == qv_b.qs
    False
    >>> from qrisp import cx
    >>> cx(qv_a[0], qv_b[0])
    >>> qv_a.qs == qv_b.qs
    True

    QuantumSessions can be given a default backend on which to evaluate circuits:

    >>> from qrisp.interface import VirtualQiskitBackend
    >>> qiskit_backend = instantiate_qiskit_backend()
    >>> qs = QuantumSession(backend = VirtualQiskitBackend(qiskit_backend))

    In this piece of code, we assume that the function ``instantiate_qiskit_backend``
    creates a Qiskit backend instance (which could either be the QASM Simulator or a
    real backend). We then hand this to the :ref:`VirtualQiskitBackend` constructor to
    turn it into a Qrisp backend. Now, any measurements of variables that are registered
    in this session will be evaluated on that backend.

    If no backend is given, the backend specified in ``default_backend.py`` will be
    used.

    Note that it is not possible to merge two QuantumSessions with differing,
    non-trivial backends.
    """

    qs_tracker = []

    def __init__(self, backend=None):
        """
        Constructs a QuantumSession

        Parameters
        ----------
        backend : BackendClient, optional
            The backend on which to execute the circuits created by this QuantumSession.
            This choice can be overwritten by specifying a backend in the
            :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>` method of
            QuantumVariable.

        Returns
        -------
        None.

        Examples
        --------

        We create a QuantumSession with the QASM simulator as default backend and
        register a QuantumFloat in it:

        >>> from qiskit import Aer
        >>> qasm_sim = Aer.get_backend("qasm_simulator")
        >>> from qrisp.interface import VirtualQiskitBackend
        >>> vrtl_qasm_sim = VirtualQiskitBackend(qasm_sim)
        >>> from qrisp import QuantumSession, QuantumFloat
        >>> qs = QuantumSession(vrtl_qasm_sim)
        >>> qf = QuantumFloat(4, qs = qs)


        """

        if isinstance(backend, int):
            raise Exception

        super().__init__()

        self.backend = backend

        # Set up list of quantum variables appearing in this session

        self.qv_list = []
        self.deleted_qv_list = []

        # Set up environment stack
        # This list will be filled, once we enter an environment
        self.env_stack = []

        # This list will be filled with variables which are marked for uncomputation
        # Variables will be marked once there is no longer any reference to them apart
        # from the one in qv_list. This is for instance the case with local variables
        # inside a function after the function finished
        self.uncomp_stack = []

        self.qs_tracker.append(weakref.ref(self))
        
        self.will_be_uncomputed = False

        # This list will contain the QuantumSessions which have been merged into this
        # session. It needs to be tracked in order to also update the shadow sessions
        # when this session is merged into another session.
        self.shadow_sessions = []

    def register_qv(self, qv):
        """
        Method to register QuantumVariables

        Parameters
        ----------
        qv : QuantumVariable
            QuantumVariable to register.

        Raises
        ------
        RuntimeError
            Name of qv is already used in this QuantumSession.

        Returns
        -------
        None.

        """
        if qv.name in [temp_qv.name for temp_qv in self.qv_list + self.deleted_qv_list]:
            raise RuntimeError(
                "Variable name " + str(qv.name) + " already exists in quantum session"
            )

        # Determine amount of required qubits
        req_qubits = qv.size

        # Hand qubits to quantum variable
        qv.reg = self.request_qubits(req_qubits, name=qv.name)

        # Register in the list of active quantum variable
        self.qv_list.append(qv)

    def get_qv(self, key):
        for qv in self.qv_list:
            if qv.name == key:
                return qv
        raise Exception("Could not find QuantumVariable " + str(key))

    def __str__(self):
        temp_data = list(self.data)
        self.data = []
        i = 0

        from qrisp import QuantumEnvironment

        while temp_data:
            instr = temp_data.pop(i)
            if isinstance(instr, QuantumEnvironment):
                instr.compile()
            else:
                self.append(instr)

        res = "QuantumCircuit:\n---------------\n"

        qc_str = QuantumCircuit.__str__(self)

        # Remove blank line
        lines = qc_str.split("\n")
        if len(set(lines[0])) == 1:
            lines.pop(0)
        # if len(set(lines[-1])) == 1:
        # lines.pop(-1)

        for line in lines:
            res += line + "\n"

        if len(self.env_stack):
            res += "QuantumEnvironment Stack:\n-------------------------\n"
            for i in range(len(self.env_stack)):
                env = self.env_stack[i]
                res += (
                    "Level " + str(i) + ": " + str(type(env)).split(".")[-1][:-2] + "\n"
                )
            res += "\n"

        res += "Live QuantumVariables:\n----------------------"

        qv_name_list = []
        for qv in self.qv_list:
            qv_name_list.append(str(type(qv)).split(".")[-1][:-2] + " " + qv.name)

        for qv_name in qv_name_list:
            res += "\n" + qv_name

        return res

    def get_depth_dic(self):
        return get_depth_dic(self)

    def add_qubit(self, qubit=None):
        qb = super().add_qubit(qubit)
        qb.qs = weakref.ref(self)
        qb.perm_lock = False
        qb.lock = False
        return qb

    def __call__(self):
        return self

    def add_clbit(self, clbit=None):
        cb = super().add_clbit(clbit)
        cb.qs = weakref.ref(self)
        return cb

    def request_qubits(self, request_amount, name=None):
        # Create qubits and add to circuit
        return_qubits = []

        for i in range(request_amount):
            if name is None:
                qb = self.add_qubit()
            else:
                qb = self.add_qubit(Qubit(name + "." + str(i)))

            return_qubits.append(self.qubits[-1])

        for qb in return_qubits:
            self.append(QubitAlloc(), [qb])

        return return_qubits

    def clear_qubits(self, qubits, verify=False):
        # Apply initialization operation
        # The following is uncommented because the QASM simulator speed drastically
        # drops when having non unitary operations
        # Uncomment, when executing on real backends
        if not len(self.env_stack):
            pass
            # self.reset(qubits)
            
        if not set(qubits).issubset(set(self.qubits)):
            raise Exception(
                "Tried to free up qubits not registered in this quantum session"
            )
            

        if verify:
            
            
            if len(self.env_stack):
                verification_qc = self.copy()
            else:
                
                # In the case the qubits have been uncomputed automatically,
                # the uncomputation algorithm already appended the deallocation
                # instructions. In order to compile the QuantumSession without
                # deallocating the qubits in question, we temporarily remove the
                # deallocation instructions.
                qubits_copy = list(qubits)
                deallocation_instructions = []
                while qubits_copy:
                    for i in range(len(self.data))[::-1]:
                        instr = self.data[i]
                        if instr.op.name != "qb_dealloc":
                            continue
                        if instr.qubits[0] in qubits_copy:
                            qubits_copy.remove(instr.qubits[0])
                            deallocation_instructions.append(self.data.pop(i))
                            break
                    else:
                        break
                
                verification_qc = self.compile()
                
                for instr in deallocation_instructions:
                    self.data.append(instr)
            
            for qb in qubits:
                clbit = verification_qc.add_clbit()
                verification_qc.measure(qb, clbit)

            from qrisp.simulator import run

            res = run(verification_qc, 1000, insert_reset=False)
            for key in res.keys():
                if key[: len(qubits)] != len(qubits) * "0":
                    raise Exception("Tried to delete qubits not in |0> state")
                    
        for qb in qubits:
            self.append(QubitDealloc(), [qb])
            



    # Procedure to free up space for quantum variables not used anymore
    def delete_qv(self, qv, verify=False):
        # Check if quantum variable appears in this session
        if qv.name not in [qv.name for qv in self.qv_list]:
            raise Exception(
                "Tried to remove a non existent quantum variable from quantum session"
            )

        self.clear_qubits(qv.reg, verify)

        # Remove quantum variable from list
        for i in range(len(self.qv_list)):
            temp_qv = self.qv_list[i]

            if temp_qv.name == qv.name:
                self.qv_list.pop(i)
                break

        self.deleted_qv_list.append(qv)

    def cnot_count(self):
        """
        Method to determine the amount of CNOT gates used in this QuantumSession.

        Raises
        ------
        Exception
            Tried to compute the CNOT count with open environments.

        Returns
        -------
        int
            The amount of CNOT gates.

        """

        if len(self.env_stack) != 0:
            raise Exception("Tried to count CNOT gates with open if environments")
        from qrisp.misc import cnot_count

        return cnot_count(self)

    def get_local_qvs(self):
        import sys

        local_qvs = []
        for qv in object.__getattribute__(self, "qv_list"):
            if sys.getrefcount(qv) == 3:
                local_qvs.append(qv)

        return local_qvs

        # self.uncomp_stack = []

    def logic_synth(self, input_qubits, output_qubits, tt, method="best", inv=False):
        if len(input_qubits) != tt.bit_amount:
            raise Exception("Given truth table has unfitting amount of input variables")

        if len(output_qubits) != tt.shape[1]:
            raise Exception("Given truth table has unfitting amount of output columns")

        self.append(
            tt.gate_synth(method=method, inv=False), input_qubits + output_qubits
        )

    def __eq__(self, other):
        return id(self.data) == id(other.data)
    
    def append(self, operation_or_instruction, qubits=[], clbits=[]):
        # Check the type of the instruction/operation
        
        
        if issubclass(operation_or_instruction.__class__, Instruction):
            instruction = operation_or_instruction
            self.append(instruction.op, instruction.qubits, instruction.clbits)
            return
        
        
        elif issubclass(operation_or_instruction.__class__, Operation):
            operation = operation_or_instruction

        else:
            raise Exception(
                "Tried to append object type "
                + str(type(operation_or_instruction))
                + " which is neither Instruction nor Operation"
            )
        
        if operation.name == "qb_alloc":
            qubits[0].allocated = True
        
        if self.xla_mode >= 3:
            
            self.data.append(Instruction(operation, qubits, clbits))
            
            if operation.name == "qb_dealloc":
                qubits[0].allocated = False
            return
        
        # Convert arguments (possibly integers) to list
        # The logic here is that the list structure gets preserved ie.
        # [[0, 1] ,2] ==> [[qubit_0, qubit_1], qubit_2]
        # unless the input is a single qubit/integer.
        # In this case we have
        # qubit_0 ==> [qubit_0]

        from qrisp.circuit.quantum_circuit import convert_to_cb_list, convert_to_qb_list

        qubits = convert_to_qb_list(qubits, circuit=self)
        clbits = convert_to_cb_list(clbits, circuit=self)

        def check_alloc(input, res=None):
            if isinstance(input, list):
                for item in input:
                    check_alloc(item)
            else:
                if not input.allocated:
                    # pass
                    raise Exception(
                        f"Tried to perform operation {operation.name} on "
                        f"unallocated qubit {input}"
                    )

        if operation.name not in ["qb_alloc", "barrier"]:
            check_alloc(qubits)


        # We now need to merge the sessions and treat their differing environment
        # levels. The idea here is that if a quantum session A is not identical to the
        # environment session B, there have been no gates applied within that
        # environment so far (otherwise merging would have occured). Thus, all data of A
        # belongs into the original_data attribute of the environment with the highest
        # level environment, where the environment quantum session isn't identical to A.
        flattened_qubits = []
        for item in qubits:
            if isinstance(item, Qubit):
                flattened_qubits.append(item)
            else:
                flattened_qubits.extend(item)

        flattened_clbits = []
        for item in clbits:
            if isinstance(item, Clbit):
                flattened_qubits.append(item)
            else:
                flattened_qubits.extend(item)

        # Find the list of all quantum sessions that need to be treated
        qs_list = (
            [qb.qs() for qb in flattened_qubits]
            + [cb.qs() for cb in flattened_clbits]
            + [self]
        )

        # We now iterate through every quantum session and insert its data into the
        # correct original_data attribute
        # for qs in qs_list:

        #     #We need to find the environment where the env_qs quantum session is not merged into qs.
        #     #This implies that the instructions of this session have been appended in this environment's parent.
        #     #Therefore all the data needs to go into the original_data attribute of this environment.
        #     for env in qs.env_stack:
        #         if not env.env_qs == qs:
        #             env.original_data.extend(qs.data)
        #             qs.data = []
        #             merge([qs, env.env_qs])

        # We merge qs_list again since no merge happened incase there were no
        # environments.
        # if not operation.name == "qb_alloc":
        multi_session_merge(qs_list)

        # print([qb.identifier for qb in self.qubits])
        super().append(operation, qubits, clbits)

        
        if operation.name == "qb_dealloc":
            qubits[0].allocated = False

    def __getitem__(self, key):
        for qv in self.qv_list:
            if qv.name == key:
                return qv
        raise Exception(f"Could not find QuantumVariable {key}")

    # Instead of just resetting the list, we have to use this method.
    # This is because merging two quantum session works essentially by handing them
    # a pointer to the same data list (which contains the merged circuits)
    # If we clear the data list by setting it to an empty list, any session
    # that has been merged with self.qs doesnt point to the same data list anymore.
    # This method tackles this problem by keeping the pointer to the list alive,
    # but removing every single element
    def clear_data(self):
        self.data.clear()

    def statevector(self, return_type="sympy", plot=False, decimals=None):
        r"""
        Returns a representation of the statevector. Three options are available:

        * ``sympy`` returns a `Sympy quantum state
          <https://docs.sympy.org/latest/modules/physics/quantum/state.html>`_,
          which is great for visualization and symbolic investigation. The tensor factors
          are in the order of the creation of the QuantumVariables (or equivalently: as
          they appear, when listed in ``print(self)``).

        * ``latex`` returns the latex code for the Sympy quantum state.

        * ``function`` returns a statevector function, such that the amplitudes can be
          investigated by calling this function on a dictionary of this QuantumSession's
          QuantumVariables.

        If you need to retrieve the statevector as a numpy array, please use the
        corresponding
        :meth:`QuantumCircuit method <qrisp.QuantumCircuit.statevector_array>`.

        Parameters
        ----------
        return_type : str, optional
            String indicating how the statevector should be returned. Available are
            ``sympy``, ``array`` and ``function``. The default is ``sympy``.
        plot : bool, optional
            If the return type is set to ``array``, this boolean will trigger a plot of
            the statevector. The default is ``False``.
        decimals : int, optional
            The decimals to round in the statevector. The default is 5 for return type
            ``sympy`` and infinite otherwise.

        Returns
        -------
        sympy.Expression or LaTeX string or function
            An object representing the statevector.

        Examples
        --------

        We create some QuantumFloats and encode values in them:

        >>> from qrisp import QuantumFloat
        >>> qf_0 = QuantumFloat(3,-1)
        >>> qf_1 = QuantumFloat(3,-1)
        >>> qf_0[:] = 2
        >>> qf_1[:] = {0.5 : 1, 3.5: -1j}

        This encoded the state

        .. math::
            
            \ket{\psi} = \ket{\text{qf_0}} \ket{\text{qf_1}}
            = \frac{1}{\sqrt{2}}  \ket{2} (\ket{0.5} - i \ket{3.5})

        Now we add ``qf_0`` and ``qf_1``:

        >>> qf_res = qf_0 + qf_1

        This gives us the state

        .. math::

            \ket{\phi} = \frac{1}{\sqrt{2}}(\ket{2}\ket{0.5}\ket{2 + 0.5} -
            i \ket{2} \ket{3.5}\ket{2 + 3.5})



        We retrieve the statevector as a Sympy expression:

        >>> sv = qf_0.qs.statevector()
        >>> print(sv)
        sqrt(2)*(|2.0>*|0.5>*|2.5> - I*|2.0>*|3.5>*|5.5>)/4

        If you have Sympy's `pretty printing
        <https://docs.sympy.org/latest/tutorials/intro-tutorial/printing.html>`_ enabled
        in your IPython console, it will even give you a nice Latex rendering:

        >>> sv

        .. image:: ./statevector_print.png
            :width: 300
            :alt: Statevector print
            :align: left

        |
        |

        This feature also works with symbolic parameters:

        >>> from qrisp import QuantumVariable, ry, h, p
        >>> from sympy import Symbol
        >>> qv = QuantumVariable(1)
        >>> ry(Symbol("omega"), qv)
        >>> h(qv)
        >>> p(-Symbol("phi"), qv)
        >>> qv.qs.statevector()

        .. image:: ./symbolic_statevector_print.png
            :width: 350
            :alt: Statevector print
            :align: left

        |
        |

        .. note::

            Statevector simulation with symbolic parameters is significantly more
            demanding than simulation with numeric parameters.

        To retrieve the above expressions as latex code, we use
        ``return_type = "latex"``

        >>> print(qf_0.qs.statevector(return_type = "latex"))
        '\frac{\sqrt{2} \left({\left|2.0\right\rangle }
        {\left|0.5\right\rangle } {\left|2.5\right\rangle }
        - i {\left|2.0\right\rangle } {\left|3.5\right\rangle }
        {\left|5.5\right\rangle }\right)}{2}'


        We can also retrieve the statevector as a Python function:

        >>> sv_function = qf_0.qs.statevector("function")

        Specify the label constellations:

        >>> label_constellation_a = {qf_0 : 2, qf_1 : 0.5, qf_res : 2+0.5}
        >>> label_constellation_b = {qf_0 : 2, qf_1 : 3.5, qf_res : 2+3.5}
        >>> label_constellation_c = {qf_0 : 2, qf_1 : 3.5, qf_res : 4}

        And evaluate the function:

        >>> sv_function(label_constellation_a)
        (0.7071048-1.3411045e-07j)

        This is the expected amplitude up to floating point errors.

        To get a quicker understanding, we can tell the statevector function to round
        the amplitudes using the ``round`` keyword.

        >>> sv_function(label_constellation_b, round = 6)
        (-0-0.707105j)

        Finally, the last amplitude is 0 since the state of ``qf_res`` is not the sum of
        ``qf_0`` and ``qf_1``.

        >>> sv_function(label_constellation_c, round = 6)
        0j


        """

        if len(self.env_stack):
            raise Exception("Tried to evaluate statevector within open QuantumEnvironments")
            
        from qrisp import get_statevector_function, get_sympy_state

        if return_type == "array":
            from qrisp.simulator import statevector_sim

            # Simulate the statevector
            statevector_array = statevector_sim(self.compile())
            # statevector_array = statevector_sim(self)

            # Execute simulation

            # Plot results if required
            if plot:
                import matplotlib.pyplot as plt

                plt.plot(np.real(statevector_array), "o", label="Re(psi)")
                plt.plot(np.imag(statevector_array), "o", label="Im(psi)")
                plt.grid()
                plt.legend()
                plt.show()
            if decimals is None:
                return statevector_array
            else:
                return np.round(statevector_array, decimals)

        elif return_type == "sympy":
            return get_sympy_state(self, decimals)

        elif return_type == "latex":
            from sympy import latex

            return latex(self.statevector(return_type="sympy", decimals=decimals))

        elif return_type == "function":
            if decimals is None:
                decimals = 15

            return get_statevector_function(self, decimals)

        else:
            raise Exception(f"Don't know return type {return_type}")

    def compile(
        self,
        workspace=0,
        intended_measurements=[],
        cancel_qfts=True,
        disable_uncomputation=True,
        compile_mcm=False,
        gate_speed = None
    ):
        r"""
        Method to compile the QuantumSession into a :ref:`QuantumCircuit`. The compiler
        dynamically allocates the qubits of the QuantumSession on qubits that might have
        been used by priorly deleted :ref:`QuantumVariables <QuantumVariable>`.

        Using the ``workspace`` keyword, we can grant the compiler a number of extra
        qubits to use in order to reduce the circuit depth.

        Furthermore, the compiler recompiles any :meth:`mcx <qrisp.mcx>` instruction
        with ``method = auto`` using a dynamically generated mcx implementation that
        makes use of as much of the currently available clean and dirty ancillae.
        This feature will never allocate additional qubits on its own. If required,
        it can be supplied with additional space using the ``workspace`` keyword.
    
        Another important feature of this function is gate speed aware compilation.
        Gate speed here means the amount of time each basis gate requires in a
        physical execution of the QuantumCircuit.
        For NISQ era devices, CNOT gates are a bottleneck, whereas
        FT era devices are expected to be bottlenecked by T-gates. While these are
        two important examples, more backend specific gate-speed specifications
        are possible. The Qrisp compiler can leverage several non-trivial 
        commutation relations to reorder circuits such that the run-time is 
        optimal. To tell the compiler, the time that is required for each gate, 
        the ``gate_speed`` keyword argument exists. This argument should be a 
        function of :ref:`Operation` objects, that returns a float indicating
        the gate speed. For an example of such a function, check out
        :meth:`T-depth <qrisp.t_depth_indicator>`. For further details, check the examples.
        

        The .compile method is called by default, when executing the
        :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>` method of
        :ref:`QuantumVariable`. This method also allows specification of compilation
        option through the ``compilation_kwargs`` argument.



        Parameters
        ----------
        workspace : int, optional
            The amount of workspace qubits to be granted. The default is 0.
        intended_measurements : list[Qubit], optional
            A list of :ref:`Qubits <Qubit>` that are supposed to be measured. The
            compiler will remove any instructions that are not directly neccessary to
            perform the measurements. Note that the resulting :ref:`QuantumCircuit`
            contains no measurements, such that the user can still specify a classical
            bit for the measurement. The default ist [].
        cancel_qfts : bool, optional
            If set to True, any :meth:`QFT <qrisp.QFT>` instruction that is executed on
            a set of qubits that have just been allocated (ie. the $\ket{0}$ state) will
            be replaced by a set of H gates. The same goes for QFT instructions executed
            directly before deallocation. The default is ``True``.
        disable_uncomputation : bool, optional
            Experimental feature the allows fully automized uncomputation. If set to
            ``False`` any :ref:`QuantumVariable` that went out of scope will be
            uncomputed by the compiler. The default is ``True``.
        gate_speed : function, optional
            Enables the compiler to create circuits that are aware of differences
            in gate speed. For NISQ era devices, CNOT gates are a bottleneck, whereas
            FT era devices are expected to be bottlenecked by 
        compile_mcm : function, optional
            If set to ``True``, any instance of mcx gates with method either ``jones``
            or ``gidney`` will be compiled to use a mid-circuit measurement. If 
            set to ``False``, a functionally equivalent (but less efficient version)
            will be used without a mid-circuit measurement. For more information
            see :meth:`qrisp.mcx` The default is ``False``.

        Returns
        -------
        QuantumCircuit
            The compiled QuantumCircuit.

        Examples
        --------

        .. _workspace:
            
        **Workspace**

        We calculate a product of 2 :ref:`QuantumFloats <QuantumFloat>` using the
        :meth:`sbp_mult <qrisp.sbp_mult>` function which heavily profits from more
        workspace.

        >>> from qrisp import QuantumFloat, sbp_mult
        >>> qf_0 = QuantumFloat(5)
        >>> qf_0[:] = 3
        >>> qf_1 = QuantumFloat(5)
        >>> qf_1[:] = 5

        Calculate product:

        >>> qf_res = sbp_mult(qf_0, qf_1)
        >>> qf_res.qs.num_qubits()
        45

        Compile circuit with no workspace

        >>> qc_0 = qf_res.qs.compile(0)
        >>> qc_0.num_qubits()
        21
        >>> qc_0.depth()
        497

        Compile circuit with 4 workspace qubits

        >>> qc_1 = qf_res.qs.compile(4)
        >>> qc_1.num_qubits()
        25
        >>> qc_1.depth()
        258

        **mcx recompilation**

        To demonstrate the recompilation feature, we create two
        :ref:`QuantumVariables <QuantumVariable>`.

        >>> from qrisp import QuantumVariable, mcx, cx
        >>> ctrl = QuantumVariable(4)
        >>> target = QuantumVariable(1)
        >>> mcx(ctrl, target)
        >>> print(ctrl.qs)
        
        ::
        
            QuantumCircuit:
            --------------
              ctrl.0: ──■──
                        │
              ctrl.1: ──■──
                        │
              ctrl.2: ──■──
                        │
              ctrl.3: ──■──
                      ┌─┴─┐
            target.0: ┤ X ├
                      └───┘
            Live QuantumVariables:
            ---------------------
            QuantumVariable ctrl
            QuantumVariable target

        We can now call the ``.compile`` method

        >>> compiled_qc = ctrl.qs.compile()
        >>> compiled_qc.depth()
        50
        >>> print(compiled_qc)
        
        ::
        
                ctrl.0: ──■──
                          │
                ctrl.1: ──■──
                          │
                ctrl.2: ──■──
                          │
                ctrl.3: ──■──
                        ┌─┴─┐
              target.0: ┤ X ├
                        └───┘

        We see no change here, because there was no free space to execute a more optimal
        mcx implementation. We can grant additional space using the ``workspace``
        argument:

        >>> compiled_qc = ctrl.qs.compile(workspace = 2)
        >>> compiled_qc.depth()
        22
        >>> print(compiled_qc)
        
        ::
        
                         ┌────────┐               ┌────────┐
                 ctrl.0: ┤0       ├───────────────┤0       ├──────────
                         │        │               │        │
                 ctrl.1: ┤1       ├───────────────┤1       ├──────────
                         │        │┌────────┐     │        │┌────────┐
                 ctrl.2: ┤        ├┤0       ├─────┤        ├┤0       ├
                         │  pt2cx ││        │     │  pt2cx ││        │
                 ctrl.3: ┤        ├┤1       ├─────┤        ├┤1       ├
                         │        ││        │┌───┐│        ││        │
               target.0: ┤        ├┤  pt2cx ├┤ X ├┤        ├┤  pt2cx ├
                         │        ││        │└─┬─┘│        ││        │
            workspace_0: ┤2       ├┤        ├──■──┤2       ├┤        ├
                         └────────┘│        │  │  └────────┘│        │
            workspace_1: ──────────┤2       ├──■────────────┤2       ├
                                   └────────┘               └────────┘

        Granting extra qubits to use this feature is however not usually necessary. The
        compiler automatically detects and reuses qubit resources available at the
        corresponding stage of the compilation.
        To demonstrate this feature, we allocate a third QuantumVariable:

        >>> qv = QuantumVariable(2)
        >>> cx(target[0], qv)
        >>> print(ctrl.qs.compile())
        
        ::
        
                      ┌────────┐               ┌────────┐
              ctrl.0: ┤0       ├───────────────┤0       ├────────────────────
                      │        │               │        │
              ctrl.1: ┤1       ├───────────────┤1       ├────────────────────
                      │        │┌────────┐     │        │┌────────┐
              ctrl.2: ┤        ├┤0       ├─────┤        ├┤0       ├──────────
                      │  pt2cx ││        │     │  pt2cx ││        │
              ctrl.3: ┤        ├┤1       ├─────┤        ├┤1       ├──────────
                      │        ││        │┌───┐│        ││        │
            target.0: ┤        ├┤  pt2cx ├┤ X ├┤        ├┤  pt2cx ├──■────■──
                      │        ││        │└─┬─┘│        ││        │┌─┴─┐  │
                qv.0: ┤2       ├┤        ├──■──┤2       ├┤        ├┤ X ├──┼──
                      └────────┘│        │  │  └────────┘│        │└───┘┌─┴─┐
                qv.1: ──────────┤2       ├──■────────────┤2       ├─────┤ X ├
                                └────────┘               └────────┘     └───┘


        We see how the qubits that will later hold ``qv`` are used to efficiently
        compile the mcx gate.

        In situations of no free clean ancilla qubits, the Qrisp compiler even makes use
        of dirty ancillae. To demonstrate, we again create three QuantumVariables
        but this time we execute a :meth:`cx<qrisp.cx>`-gate before executing the
        :meth:`mcx<qrisp.mcx>`-gate. This way ``qv`` has to be allocated before the
        ``mcx`` gate.

        >>> ctrl = QuantumVariable(4)
        >>> target = QuantumVariable(1)
        >>> qv = QuantumVariable(2)
        >>> cx(target[0], qv)
        >>> mcx(ctrl, target)
        >>> print(ctrl.qs.compile())
        
        ::
        
              ctrl.0: ────────────────────────────────────■──────────────────────────»
                                     ┌─────────────────┐  │  ┌─────────────────┐     »
              ctrl.1: ───────────────┤1                ├──┼──┤1                ├─────»
                                     │                 │  │  │                 │     »
              ctrl.2: ───────────────┤2                ├──┼──┤2                ├─────»
                                     │                 │  │  │                 │     »
              ctrl.3: ────────────■──┤                 ├──┼──┤                 ├──■──»
                                ┌─┴─┐│  reduced_maslov │  │  │  reduced_maslov │┌─┴─┐»
            target.0: ──■────■──┤ X ├┤                 ├──┼──┤                 ├┤ X ├»
                      ┌─┴─┐  │  └─┬─┘│                 │┌─┴─┐│                 │└─┬─┘»
                qv.0: ┤ X ├──┼────┼──┤0                ├┤ X ├┤0                ├──┼──»
                      └───┘┌─┴─┐  │  │                 │└───┘│                 │  │  »
                qv.1: ─────┤ X ├──■──┤3                ├─────┤3                ├──■──»
                           └───┘     └─────────────────┘     └─────────────────┘     »
            «
            «  ctrl.0: ─────────────────────■─────────────────────
            «          ┌─────────────────┐  │  ┌─────────────────┐
            «  ctrl.1: ┤1                ├──┼──┤1                ├
            «          │                 │  │  │                 │
            «  ctrl.2: ┤2                ├──┼──┤2                ├
            «          │                 │  │  │                 │
            «  ctrl.3: ┤                 ├──┼──┤                 ├
            «          │  reduced_maslov │  │  │  reduced_maslov │
            «target.0: ┤                 ├──┼──┤                 ├
            «          │                 │┌─┴─┐│                 │
            «    qv.0: ┤0                ├┤ X ├┤0                ├
            «          │                 │└───┘│                 │
            «    qv.1: ┤3                ├─────┤3                ├
            «          └─────────────────┘     └─────────────────┘

        We see how the qubits of ``qv`` are utilized as dirty ancilla qubits in order
        to facilitate a more efficient ``mcx`` implementation compared to no ancillae
        at all.

        .. _gate_speed_aware_comp:

        **Gate speed aware compilation**
            
        Next to the mentioned features, the ``compile`` method performs a variety
        of techniques of reordering the gate sequence (without changing the semantics, of course)
        to reduce the overall depth. Some of these techniques allow for a consideration
        of the gate speed, which enables a unique compilation workflow for each backend.
        
        The gate speed of the backend can be specified as a function of :ref:`Operation`
        objects:
            
        ::
            
            def mock_gate_speed_0(op):
                
                if op.name == "x":
                    return 1
                if op.name == "y":
                    return 10
                else:
                    return 0
                
        This function describes a backend where the X-gate requires 1 time unit 
        (for instance nanoseconds), the Y-gate requires 10 time units
        and every other gate can be executed instantaneusly.
        
        We can now observe how this influences the compilation:
            
        >>> from qrisp import QuantumVariable, x, y, cx
        >>> qv = QuantumVariable(3)
        >>> y(qv[0])
        >>> x(qv[1])
        >>> cx(qv[2], qv[:2])
        >>> y(qv[1])
        >>> x(qv[0])
        >>> print(qv.qs)
        QuantumCircuit:
        ---------------
              ┌───┐┌───┐┌───┐     
        qv.0: ┤ Y ├┤ X ├┤ X ├─────
              ├───┤└─┬─┘├───┤┌───┐
        qv.1: ┤ X ├──┼──┤ X ├┤ Y ├
              └───┘  │  └─┬─┘└───┘
        qv.2: ───────■────■───────
        <BLANKLINE>                                                    
        Live QuantumVariables:
        ----------------------
        QuantumVariable qv
        
        Because the CNOT gate on ``qv.1`` has to wait for the other CNOT gate
        (which takes a lot of time because of the costly y gate), the second y gate
        can only be executed delayed, making the total runtime of this circuit 20
        time units.
        
        We can verify this using the ``depth_indicator`` keyword of the 
        :meth:`depth <qrisp.QuantumCircuit.depth>` method:
            
        >>> qv.qs.depth(depth_indicator = mock_gate_speed_0)
        20
        
        Call the compile method, which automatically fixes the problem
        
        >>> qc_fixed_0 = qv.qs.compile(gate_speed = mock_gate_speed_0)
        >>> print(qc_fixed_0)
              ┌───┐          ┌───┐┌───┐
        qv.0: ┤ Y ├──────────┤ X ├┤ X ├
              ├───┤┌───┐┌───┐└─┬─┘└───┘
        qv.1: ┤ X ├┤ X ├┤ Y ├──┼───────
              └───┘└─┬─┘└───┘  │       
        qv.2: ───────■─────────■───────
                                         
        We see that the order of the CNOT gates has been switched (which doesn't
        change the semantics) such that now ``qv.1`` no longer has to wait for
        the costly y gate.
        
        >>> qc_fixed_0.depth(depth_indicator = mock_gate_speed_0)
        11
        
        To see that the compilation function did not do this randomly, we can also
        create another ``gate_speed`` function.
        
        ::
            
            def mock_gate_speed_1(op):
                
                if op.name == "x":
                    return 10
                elif op.name == "y":
                    return 1
                else:
                    return 0
                
        
        >>> qc_fixed_1 = qv.qs.compile(gate_speed = mock_gate_speed_1)
        >>> print(qc_fixed_1)
              ┌───┐┌───┐┌───┐     
        qv.0: ┤ Y ├┤ X ├┤ X ├─────
              ├───┤└─┬─┘├───┤┌───┐
        qv.1: ┤ X ├──┼──┤ X ├┤ Y ├
              └───┘  │  └─┬─┘└───┘
        qv.2: ───────■────■───────
        
        Now the CNOT gate on ``qv.0`` is executed first, giving again a total depth
        of 11
        
        >>> qc_fixed_1.depth(depth_indicator = mock_gate_speed_1)

        Qrisp has the two most important depth indicators in-built: 
        :meth:`CNOT-depth <qrisp.cnot_depth_indicator>` (NISQ) and 
        :meth:`T-depth <qrisp.t_depth_indicator>` (FT).

        **Fully automized uncomputation**

        This feature is as of right now experimental. To demonstrate, we create a test
        function, creating a local :ref:`QuantumBool` ::

            from qrisp import QuantumBool, mcx

            def triple_AND(a, b, c):

                local = QuantumBool()
                result = QuantumBool()

                mcx([a,b], local)

                mcx([c, local], result)

                return result


        >>> a = QuantumBool()
        >>> b = QuantumBool()
        >>> c = QuantumBool()
        >>> res = triple_AND(a,b,c)
        >>> print(res.qs)
        
        ::
        
            QuantumCircuit:
            --------------
                 a.0: ──■───────
                        │
                 b.0: ──■───────
                        │
                 c.0: ──┼────■──
                      ┌─┴─┐  │
             local.0: ┤ X ├──■──
                      └───┘┌─┴─┐
            result.0: ─────┤ X ├
                           └───┘
            Live QuantumVariables:
            ---------------------
            QuantumBool a
            QuantumBool b
            QuantumBool c
            QuantumBool local
            QuantumBool result

        We now compile with the corresponding keyword argument:

        >>> print(a.qs.compile(disable_uncomputation = False))
        
        ::
        
                         ┌────────┐     ┌────────┐
                    a.0: ┤0       ├─────┤0       ├
                         │        │     │        │
                    b.0: ┤1       ├─────┤1       ├
                         │        │     │        │
                    c.0: ┤  pt2cx ├──■──┤  pt2cx ├
                         │        │┌─┴─┐│        │
               result.0: ┤        ├┤ X ├┤        ├
                         │        │└─┬─┘│        │
            workspace_0: ┤2       ├──■──┤2       ├
                         └────────┘     └────────┘

        We see that the ``local`` QuantumBool is no longer allocated but has been
        uncomputed and it's qubits are available as workspace.

        """
        from qrisp.core.compilation import qompiler

        return qompiler(
            self,
            workspace,
            disable_uncomputation=disable_uncomputation,
            intended_measurements=intended_measurements,
            cancel_qfts=cancel_qfts,
            compile_mcm=compile_mcm,
            gate_speed = gate_speed
        )

    def __del__(self):
        i = 0
        while i < len(self.qs_tracker):
            if self.qs_tracker[i]() is None or id(self) == id(self.qs_tracker[i]()):
                self.qs_tracker.pop(i)
                continue
            i += 1

    def __hash__(self):
        return id(self.data)

    # The .data attribute is used to identify QuantumSessions with each other even
    # though they are different object. If the .data attribute is set to a new list,
    # this identification is no longer possible because the two different
    # QuantumSessions no longer share the same data list. We overload setattr such that
    # setting a new list results in keeping the old one but with new content.
    def __setattr__(self, name, value):
        if name in ["data"]:
            attr = self.__dict__[name]
            attr.clear()
            attr.extend(value)
        else:
            QuantumCircuit.__setattr__(self, name, value)

    @classmethod
    def get_active_quantum_sessions(self):
        # Remove potential duplicates
        qs_list = list(
            set([qs() for qs in QuantumSession.qs_tracker if not qs() is None])
        )

        self.qs_tracker = [weakref.ref(qs) for qs in qs_list]

        return list(self.qs_tracker)
