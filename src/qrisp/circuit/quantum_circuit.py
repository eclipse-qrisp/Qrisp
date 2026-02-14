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
import sympy

import qrisp.circuit.standard_operations as ops
from qrisp.circuit import Clbit, Instruction, Operation, Qubit

# Class to describe quantum circuits
# The naming of the attributes is rather similar to the qiskit equivalent
# in order to allow compatibility of qiskit programs to qrisp
# The key attributes are

# The list of qubits (.qubits).
# the list of classical bits (.clbits)
# the list of instructions (.data)

TO_GATE_COUNTER = np.zeros(1)


class QuantumCircuit:
    """
    This class describes quantum circuits. Many of the attribute and method names are
    oriented at the `Qiskit QuantumCircuit
    <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html>`_ class
    in order to provide a high degree of compatibility.

    QuantumCircuits can be visualized by calling ``print`` on them.

    Qrisp QuantumCircuits can be quickly generated out of existing Qiskit
    QuantumCircuits with the :meth:`from_qiskit <qrisp.QuantumCircuit.from_qiskit>`
    method.


    Parameters
    ----------

    num_qubits : integer, optional
        The amount of qubits, this QuantumCircuit is initialized with. The default is 0.
    num_clbits : integer, optional
        The amount of classical bits. The default is 0.
    name : string, optional
        A name for the QuantumCircuit. The default will generated a generic name.


    Examples
    --------

    We create a QuantumCircuit containing a so-called fan-out gate:

    >>> from qrisp import QuantumCircuit
    >>> qc_0 = QuantumCircuit(4, name = "fan out")
    >>> qc_0.cx(0, range(1,4))
    >>> print(qc_0)


    .. code-block:: none

        qb_0: ──■────■────■──
              ┌─┴─┐  │    │
        qb_1: ┤ X ├──┼────┼──
              └───┘┌─┴─┐  │
        qb_2: ─────┤ X ├──┼──
                   └───┘┌─┴─┐
        qb_3: ──────────┤ X ├
                        └───┘

    Note that the :meth:`cx gate appending method <qrisp.QuantumCircuit.cx>` (like all
    other gate appending methods) can be called with integers, Qubit objects,
    lists of integers or lists of Qubit objects.

    We now turn this QuantumCircuit into a gate and append to another QuantumCircuit to
    generate a GHZ state:

    >>> qc_1 = QuantumCircuit(4)
    >>> qc_1.h(0)
    >>> qc_1.append(qc_0.to_gate(), qc_1.qubits)
    >>> print(qc_1)

    .. code-block:: none

              ┌───┐┌──────────┐
        qb_4: ┤ H ├┤0         ├
              └───┘│          │
        qb_5: ─────┤1         ├
                   │  fan out │
        qb_6: ─────┤2         ├
                   │          │
        qb_7: ─────┤3         ├
                   └──────────┘

    Finally, we add a measurement and evaluate the circuit:

    >>> qc_1.measure(qc_1.qubits)
    >>> print(qc_1)

    .. code-block:: none

              ┌───┐┌──────────┐┌─┐
        qb_4: ┤ H ├┤0         ├┤M├─────────
              └───┘│          │└╥┘┌─┐
        qb_5: ─────┤1         ├─╫─┤M├──────
                   │  fan out │ ║ └╥┘┌─┐
        qb_6: ─────┤2         ├─╫──╫─┤M├───
                   │          │ ║  ║ └╥┘┌─┐
        qb_7: ─────┤3         ├─╫──╫──╫─┤M├
                   └──────────┘ ║  ║  ║ └╥┘
        cb_0: ══════════════════╩══╬══╬══╬═
                                   ║  ║  ║
        cb_1: ═════════════════════╩══╬══╬═
                                      ║  ║
        cb_2: ════════════════════════╩══╬═
                                         ║
        cb_3: ═══════════════════════════╩═

    >>> qc_1.run(shots = 1000)
    {'0000': 500, '1111': 500}

    **Converting from Qiskit**

    We construct the very same fan out QuantumCircuit in Qiskit:

    >>> from qiskit import QuantumCircuit as QiskitQuantumCircuit
    >>> qc_2 = QiskitQuantumCircuit(4)
    >>> qc_2.cx(0, range(1,4))
    >>> print(qc_2)

    .. code-block:: none

        q_0: ──■────■────■──
             ┌─┴─┐  │    │
        q_1: ┤ X ├──┼────┼──
             └───┘┌─┴─┐  │
        q_2: ─────┤ X ├──┼──
                  └───┘┌─┴─┐
        q_3: ──────────┤ X ├
                       └───┘

    To acquire the Qrisp QuantumCircuit we call the
    :meth:`from_qiskit <qrisp.QuantumCircuit.from_qiskit>` method. Note that we don't
    need to create a QuantumCircuit object first as this is a class method.

    >>> qrisp_qc_2 = QuantumCircuit.from_qiskit(qc_2)
    >>> print(qrisp_qc_2)

    .. code-block:: none

         qb_8: ──■────■────■──
               ┌─┴─┐  │    │
         qb_9: ┤ X ├──┼────┼──
               └───┘┌─┴─┐  │
        qb_10: ─────┤ X ├──┼──
                    └───┘┌─┴─┐
        qb_11: ──────────┤ X ├
                         └───┘

    **Abstract Parameters**

    Abstract parameters are represented by `Sympy symbols
    <https://docs.sympy.org/latest/modules/core.html#module-sympy.core.symbol>`_
    in Qrisp.

    We create a QuantumCircuit with some abstract parameters and bind them subsequently.

    >>> from qrisp import QuantumCircuit
    >>> from sympy import symbols
    >>> qc = QuantumCircuit(3)

    Create some Sympy symbols and use them as abstract parameters for phase gates:

    >>> abstract_parameters = symbols("a b c")
    >>> for i in range(3): qc.p(abstract_parameters[i], i)

    Create the substitution dictionary and bind the parameters:

    >>> subs_dic = {abstract_parameters[i] : i for i in range(3)}
    >>> bound_qc = qc.bind_parameters(subs_dic)
    >>> print(bound_qc)

    .. code-block:: none

              ┌──────┐
        qb_0: ┤ P(0) ├
              ├──────┤
        qb_1: ┤ P(1) ├
              ├──────┤
        qb_2: ┤ P(2) ├
              └──────┘

    """

    qubit_index_counter = np.zeros(1, dtype=int)
    clbit_index_counter = np.zeros(1, dtype=int)
    xla_mode = 0

    def __init__(self, num_qubits=0, num_clbits=0):
        object.__setattr__(self, "data", [])
        object.__setattr__(self, "qubits", [])
        object.__setattr__(self, "clbits", [])

        self.abstract_params = set()

        if isinstance(num_qubits, int):
            for i in range(num_qubits):
                self.qubits.append(Qubit("qb_" + str(self.qubit_index_counter[0] + i)))
            self.qubit_index_counter[0] += num_qubits
        else:
            raise Exception(
                f"Tried to initialize QuantumCircuit with type {type(num_qubits)}"
            )

        if isinstance(num_clbits, int):
            for i in range(num_clbits):
                self.clbits.append(Clbit("cb_" + str(self.clbit_index_counter[0] + i)))
            self.clbit_index_counter[0] += num_clbits
        else:
            raise Exception(
                f"Tried to initialize QuantumCircuit with type {type(num_clbits)}"
            )

    # Method to add qubit objects to the circuit
    def add_qubit(self, qubit=None):
        """
        Adds a Qubit to the QuantumCircuit.

        Parameters
        ----------
        qubit : Qubit, optional
            The Qubit to be added. If given none, a new Qubit will be generated.

        Returns
        -------
        Qubit
            The added Qubit.

        Examples
        --------

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit()
        >>> qc.add_qubit()
        >>> qc.qubits
        [qb_0]

        """

        self.qubit_index_counter += 1

        if qubit is None:
            qubit = Qubit("qb_" + str(self.qubit_index_counter[0]))

        if self.xla_mode < 2:
            for qb in self.qubits:
                if qb.identifier == qubit.identifier:
                    raise Exception(f"Qubit name {qubit.identifier} already exists")

        if not isinstance(qubit, Qubit):
            raise Exception(f"Tried to add type {type(qubit)} as a qubit")

        self.qubits.append(qubit)

        return self.qubits[-1]

    # Method to add classical bit objects to the circuit
    def add_clbit(self, clbit=None):
        """
        Adds a classical bit to the QuantumCircuit.

        Parameters
        ----------
        clbit : Clbit, optional
            The classical bit to be added. If given none, a new Clbit will be generated.

        Returns
        -------
        Clbit
            The added Clbit.

        """

        if clbit is None:
            clbit = Clbit("cb_" + str(len(self.clbits)))

        if not isinstance(clbit, Clbit):
            raise Exception(f"Tried to add type {type(clbit)} as a classical bit")

        for cb in self.clbits:
            if cb.identifier == clbit.identifier:
                raise Exception(f"Clbit name {clbit.identifier} already exists")

        self.clbits.append(clbit)

        return self.clbits[-1]

    # Method to transform the given circuit into an operation object
    def to_op(self, name=None):
        """
        Method to return an Operation object generated out of this QuantumCircuit.

        Operation objects can be appended to other QuantumCircuits.

        An alias for Qiskit compatibility is the
        :meth:`to_gate<qrisp.QuantumCircuit.to_gate>` method.

        Parameters
        ----------
        name : string, optional
            The name of the gate. By default, the QuantumCircuit's name will be used.

        Returns
        -------
        Operation
            The Operation defined by this QuantumCircuit.

        Examples
        --------

        >>> from qrisp import QuantumCircuit
        >>> qc_0 = QuantumCircuit(4)
        >>> qc_0.x(qc.qubits)
        >>> operation = qc_0.to_gate()
        >>> qc_1 = QuantumCircuit(4)
        >>> qc_1.append(operation, qc_1.qubits)

        """

        if name is None:
            # name = "circuit" + str(id(self))[:5]
            name = "circuit" + str(int(TO_GATE_COUNTER[0]))[:7].zfill(7)

            TO_GATE_COUNTER[0] += 1

        definition = self.copy()
        i = 0

        while i < len(definition.data):
            if definition.data[i].op.name in ["qb_alloc", "qb_dealloc"]:
                definition.data.pop(i)
                continue
            i += 1

        return Operation(
            name=name,
            num_qubits=len(self.qubits),
            num_clbits=len(self.clbits),
            definition=definition,
            params=[],
        )

    # Wrapper to increase Qiskit compatibility
    def to_gate(self, name=None):
        """
        Similar to :meth:`to_op <qrisp.QuantumCircuit.to_op>` but raises an exception
        if self contains classical bits (like the
        `Qiskit equivalent
        <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.to_gate.html>`_).
        # noqa

        Parameters
        ----------
        name : str, optional
            A name for the resulting gate. The default is None.

        Raises
        ------
        Exception
            Tried to turn a circuit including classical bits into unitary gate

        Returns
        -------
        Operation
            The QuantumCircuit turned into an :ref:`Operation` instance.

        """

        if len(self.clbits) != 0:
            raise Exception(
                "Tried to turn a circuit including classical bits into unitary gate"
            )
        return self.to_op(name)

    # Method to extend the given circuit with another circuit
    # The dic translation dic encodes how the qubits should be plugged into each other
    def extend(self, other, translation_dic="id"):
        """
        Extends self in-place by another QuantumCircuit.

        Parameters
        ----------
        other : QuantumCircuit
            The QuantumCircuit to extend by.
        translation_dic : dict, optional
            The dictionary containing the information about which Qubits and Clbits
            should be plugged into each other. This dictionary should contain qubits of
            other as keys and qubits of self as values. If given none, it is assumed
            that both QuantumCircuits have matching Qubits.


        Examples
        --------

        We create two QuantumCircuits and extend the first with reversed qubit order by
        the other:

        >>> from qrisp import QuantumCircuit
        >>> extension_qc = QuantumCircuit(4)
        >>> qc_to_extend = QuantumCircuit(4)
        >>> extension_qc.cx(0, 1)
        >>> extension_qc.cy(0, 2)
        >>> extension_qc.cz(0, 3)
        >>> print(extension_qc)

        .. code-block:: none

            qb_0: ──■────■────■──
                  ┌─┴─┐  │    │
            qb_1: ┤ X ├──┼────┼──
                  └───┘┌─┴─┐  │
            qb_2: ─────┤ Y ├──┼──
                       └───┘┌─┴─┐
            qb_3: ──────────┤ Z ├
                            └───┘

        >>> translation_dic = {extension_qc.qubits[i] : qc_to_extend.qubits[-1-i]
        >>> for i in range(4)}
        >>> qc_to_extend.extend(extension_qc, translation_dic)
        >>> print(qc_to_extend)

        .. code-block:: none

                            ┌───┐
            qb_4: ──────────┤ Z ├
                       ┌───┐└─┬─┘
            qb_5: ─────┤ Y ├──┼──
                  ┌───┐└─┬─┘  │
            qb_6: ┤ X ├──┼────┼──
                  └─┬─┘  │    │
            qb_7: ──■────■────■──

        """

        if translation_dic == "id":
            translation_dic = {}
            for qb in other.qubits:
                translation_dic[qb] = qb

            for cb in other.clbits:
                translation_dic[cb] = cb

        # Copy in order to prevent modification
        translation_dic = dict(translation_dic)

        for key in list(translation_dic.keys()):
            if isinstance(key, (Qubit, Clbit)):
                translation_dic[key.identifier] = translation_dic[key]

        for i in range(len(other.data)):
            instruction_other = other.data[i]
            qubits = []
            for qb in instruction_other.qubits:
                qubits.append(translation_dic[qb.identifier])

            clbits = []

            for cb in instruction_other.clbits:
                clbits.append(translation_dic[cb.identifier])

            self.append(instruction_other.op, qubits, clbits)

    # Returns a copy of self
    def copy(self):
        """
        Returns a copy of the given QuantumCircuit.

        Returns
        -------
        QuantumCircuit
            The copied QuantumCircuit.

        """
        # If an inital circuit is given we construct a new instance

        res = QuantumCircuit()

        object.__setattr__(res, "data", list(self.data))
        object.__setattr__(res, "qubits", list(self.qubits))
        object.__setattr__(res, "clbits", list(self.clbits))

        try:
            res.abstract_params = set(self.abstract_params)
        except AttributeError:
            pass

        return res

    # Returns a copy of self but with no instructions
    def clearcopy(self):
        """
        Returns a copy of the given QuantumCircuit but without any data
        (i.e. just the Qubits and Clbits).

        Returns
        -------
        QuantumCircuit
            The empty, copied QuantumCircuit.

        """
        temp_data = list(self.data)
        self.data = []
        res = self.copy()
        self.data = temp_data
        return res

    # TO-DO write qiskit independent printer

    # Printing method
    def __str__(self):

        from qiskit.visualization.circuit_visualization import circuit_drawer

        from qrisp.interface import convert_to_qiskit

        try:
            res_str = str(
                circuit_drawer(
                    convert_to_qiskit(self, transpile=False),
                    output="text",
                    cregbundle=False,
                )
            )
        except AttributeError:
            raise Exception(
                "Tried to print QuantumSession with uncompiled QuantumEnvironments"
            )

        return res_str

    # Method which compares the unitary of two given circuits and returns
    # True if they are equivalent
    def compare_unitary(self, other, precision=4, ignore_gphase=False):
        """
        Compares the unitaries of two QuantumCircuits. This can be used to check if a
        QuantumCircuit transformation is valid.

        Parameters
        ----------
        other : QuantumCircuit
            The QuantumCircuit to compare to.
        precision : int, optional
            The precision of the comparison. This function will return True, if the norm
            of the difference of the unitaries is below the precision. The default is 4.
        ignore_gphase: bool, optional
            If set to True, this method returns True if the unitaries only differ in a
            global phase.

        Returns
        -------
        Bool
            The comparison outcome.

        Examples
        --------

        We create two QuantumCircuit with equivalent unitaries but differing by a
        non-trivial commutation:

        >>> from qrisp import QuantumCircuit
        >>> qc_0 = QuantumCircuit(2)
        >>> qc_1 = QuantumCircuit(2)
        >>> qc_0.z(0)
        >>> qc_0.cx(0,1)
        >>> print(qc_0)

        .. code-block:: none

                  ┌───┐
            qb_0: ┤ Z ├──■──
                  └───┘┌─┴─┐
            qb_1: ─────┤ X ├
                       └───┘

        >>> qc_1.cx(0,1)
        >>> qc_1.z(0)
        >>> print(qc_1)

        .. code-block:: none

                       ┌───┐
            qb_2: ──■──┤ Z ├
                  ┌─┴─┐└───┘
            qb_3: ┤ X ├─────
                  └───┘

        >>> qc_0.compare_unitary(qc_1)
        True


        """

        if len(self.qubits) != len(other.qubits):
            return False

        unitary_self = self.get_unitary()
        unitary_other = other.get_unitary()

        if ignore_gphase:
            arg_max = np.argmax(np.abs(unitary_self.flatten()))

            unitary_self = (
                unitary_self
                * unitary_other.flatten()[arg_max]
                / unitary_self.flatten()[arg_max]
            )

        from numpy.linalg import norm

        return bool(norm(unitary_self - unitary_other) < 10**-precision)

    # Converts several types of inputs to qubit lists.
    # Possible inputs are
    #
    # A qubit object
    # An integer
    # A list of integers
    # A list of qubits
    def convert_to_qubit_list(self, input, inner_recursion=False):
        if isinstance(input, Qubit):
            if inner_recursion:
                return input
            else:
                return [input]

        if isinstance(input, int):
            try:
                return self.convert_to_qubit_list(
                    self.qubits[input], inner_recursion=inner_recursion
                )
            except IndexError:
                raise Exception(
                    "Not enough qubits in circuit to access qubit " + str(input) + "."
                )

        if isinstance(input, list):
            return_list = []
            for qb in input:
                return_list.append(self.convert_to_qubit_list(qb, inner_recursion=True))
            return return_list

        raise Exception(
            "Could not convert input type " + type(input) + " to qubit list"
        )

    # Similar function as above but with classical bits
    def convert_to_clbit_list(self, input, inner_recursion=False):
        if isinstance(input, Clbit):
            if inner_recursion:
                return input
            else:
                return [input]

        if isinstance(input, int):
            try:
                return self.convert_to_clbit_list(
                    self.clbits[input], inner_recursion=inner_recursion
                )
            except IndexError:
                raise Exception(
                    "Not enough clbits in circuit to access clbit " + str(input) + "."
                )

        if isinstance(input, list):
            return_list = []
            for cb in input:
                return_list.append(self.convert_to_clbit_list(cb, inner_recursion=True))
            return return_list

        return self.convert_to_clbit_list(list(input))

    # Generates the inverse of self by applying the inverse gates in reversed order
    def inverse(self):
        """
        Returns the inverse/daggered QuantumCircuit.

        Returns
        -------
        inverted_circuit : QuantumCircuit
            The inverted QuantumCircuit.

        Examples
        --------

        Daggering a QuantumCircuit reverses the order and daggers each operation.

        >>> from qrisp import QuantumCircuit
        >>> import numpy as np
        >>> qc = QuantumCircuit(1)
        >>> qc.x(0)
        >>> qc.p(np.pi/2, 0)
        >>> qc.y(0)
        >>> print(qc.inverse())

        .. code-block:: none

                  ┌───┐┌─────────┐┌───┐
            qb_0: ┤ Y ├┤ P(-π/2) ├┤ X ├
                  └───┘└─────────┘└───┘

        For the phase gate, a daggering implies the reversal of the phase -
        Pauli gates however are invariant under daggering.

        """

        inverted_circuit = self.clearcopy()
        for instr in self.data[::-1]:
            inverted_circuit.append(instr.op.inverse(), instr.qubits, instr.clbits)

        return inverted_circuit

    # Generate the circuits unitary matrix
    def get_unitary(self, decimals=-1):
        """
        Acquires the unitary matrix of the given QuantumCircuit as a Numpy array.

        This method also works with abstract parameters. In this case a Numpy array
        with Sympy entries is returned.

        Parameters
        ----------
        decimals : integer, optional
            The amount of decimals to be rounded to. By default, the full precision is
            returned.

        Returns
        -------
        numpy.ndarray
            The unitary matrix as a numpy array.

        Examples
        --------

        We synthesize a controlled phase gate and inspect the unitary:

        >>> from qrisp import QuantumCircuit
        >>> import numpy as np
        >>> qc = QuantumCircuit(2)
        >>> phi = np.pi
        >>> qc.p(phi/2, 0)
        >>> qc.p(phi/2, 1)
        >>> qc.cx(0,1)
        >>> qc.p(-phi/2, 1)
        >>> qc.cx(0,1)
        >>> qc.get_unitary(decimals = 4)
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]], dtype=complex64)

        We now synthesize the exact same QuantumCircuit but this time ``phi`` is a Sympy
        symbol.

        >>> from sympy import Symbol
        >>> qc = QuantumCircuit(2)
        >>> phi = Symbol("phi")
        >>> qc.p(phi/2, 0)
        >>> qc.p(phi/2, 1)
        >>> qc.cx(0,1)
        >>> qc.p(-phi/2, 1)
        >>> qc.cx(0,1)
        >>> qc.get_unitary(decimals = 4)
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, exp(I*phi)]], dtype=object)

        """
        from qrisp.simulator import calc_circuit_unitary

        res = calc_circuit_unitary(self)

        if decimals != -1:
            if res.dtype == np.dtype("O"):
                raveled_res = res.ravel()
                for i in range(len(raveled_res)):
                    expression = sympy.simplify(raveled_res[i])
                    for a in sympy.preorder_traversal(expression):
                        if isinstance(a, sympy.Float):
                            rounded_float = round(a, decimals)
                            if abs(float(a) - 1) < 10 ** -(decimals):
                                expression = expression.subs(a, 1)
                            else:
                                expression = expression.subs(a, rounded_float)

                    raveled_res[i] = expression
            else:
                res = np.round(res, decimals)

        return res

    def get_depth_dic(self):
        from qrisp.misc import get_depth_dic

        return get_depth_dic(self)

    def cnot_count(self):
        """
        Method to determine the amount of CNOT gates used in this QuantumCircuit.

        Returns
        -------
        int
            The amount of CNOT gates.

        """

        from qrisp.misc import cnot_count

        return cnot_count(self)

    def transpile(self, transpilation_level=np.inf, **qiskit_kwargs):
        """
        Transpiles the QuantumCircuit in the sense that there are no longer any
        synthesized gate objects. Furthermore, we can call the `Qiskit transpiler
        <https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html>`_
        by supplying keyword arguments.

        The Qiskit transpiler is not called, if no keyword arguments are given.

        Parameters
        ----------
        **qiskit_kwargs :
            Keyword arguments for the Qiskit transpiler.

        Returns
        -------
        QuantumCircuit
            The transpiled QuantumCircuit.

        Examples
        --------

        We create a QuantumCircuit and append a synthesized gate. Afterwards we
        transpile to a given set of basis gates using the Qiskit transpiler:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(3)
        >>> qc.mcx([0,1], 2)
        >>> print(qc)

        .. code-block:: none

            qb_0: ──■──
                    │
            qb_1: ──■──
                  ┌─┴─┐
            qb_2: ┤ X ├
                  └───┘

        >>> print(qc.transpile(basis_gates = ["cx", "rz", "sx"]))

        .. code-block:: none

            global phase: 9π/8
                  ┌──────────┐                 ┌───┐┌─────────┐   ┌───┐   ┌──────────┐┌───┐»
            qb_1: ┤ Rz(-π/4) ├─────────────────┤ X ├┤ Rz(π/4) ├───┤ X ├───┤ Rz(-π/4) ├┤ X ├»
                  ├──────────┤                 └─┬─┘└─────────┘   └─┬─┘   └──────────┘└─┬─┘»
            qb_2: ┤ Rz(-π/4) ├───────────────────┼───────■──────────■──────────■────────┼──»
                  ├─────────┬┘┌────┐┌─────────┐  │     ┌─┴─┐   ┌─────────┐   ┌─┴─┐      │  »
            qb_3: ┤ Rz(π/2) ├─┤ √X ├┤ Rz(π/2) ├──■─────┤ X ├───┤ Rz(π/4) ├───┤ X ├──────■──»
                  └─────────┘ └────┘└─────────┘        └───┘   └─────────┘   └───┘         »
            «      ┌─────────┐┌───┐
            «qb_1: ┤ Rz(π/4) ├┤ X ├────────────
            «      └─────────┘└─┬─┘
            «qb_2: ─────────────■──────────────
            «      ┌─────────┐┌────┐┌─────────┐
            «qb_3: ┤ Rz(π/4) ├┤ √X ├┤ Rz(π/2) ├
            «      └─────────┘└────┘└─────────┘

        One can also transpile a specific composite gate in a QuantumCircuit, if desired. A Quantum
        Phase Estimation circuit also contains a `QFT_dg` gate. 

        >>> from qrisp import p, QuantumVariable, QPE, multi_measurement, h
        >>> import numpy as np
        >>>
        >>> def U(qv):
        >>>     x = 0.5
        >>>     y = 0.125
        >>>
        >>>     p(x*2*np.pi, qv[0])
        >>>     p(y*2*np.pi, qv[1])
        >>> 
        >>> qv = QuantumVariable(2)
        >>>
        >>> h(qv)
        >>> 
        >>> res = QPE(qv, U, precision = 3)
        >>>
        >>> print(qv.qs.compile())

        To transpile just `QFT_dg` in the compiled QuantumCircuit,

        >>> test_circuit = qv.qs.compile()
        >>>
        >>> def transpile_predicate(op):
        >>>    if op.name == "QFT_dg":
        >>>        return True
        >>>    else:
        >>>        return False
        >>>    
        >>> transpiled_qc = test_circuit.transpile(transpile_predicate = transpile_predicate)
        >>>
        >>> print(transpiled_qc)
        

        """
        from qrisp.circuit import transpile

        return transpile(self, transpilation_level, **qiskit_kwargs)

    # Counts the amount of operations self contains and returns
    # a dict {"operation_name" : operation_count, ...}
    def count_ops(self):
        """
        Counts the amount of operations of each kind. Note that operations are
        identified by their name.

        Returns
        -------
        count_dic : dict
            A dictionary containing the gate counts.

        Examples
        --------

        We create a QuantumCircuit containing a number of gates and evaluates the
        gate-counts:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(5)
        >>> qc.x(qc.qubits)
        >>> qc.cx(0, range(1,5))
        >>> qc.z(1)
        >>> qc.t(range(5))
        >>> qc.count_ops()
        {'x': 5, 'cx': 4, 'z': 1, 't': 5}

        """
        count_dic = {}

        for ins in self.data:
            if ins.op.name in ["qb_alloc", "qb_dealloc"]:
                continue
            try:
                count_dic[ins.op.name] += 1
            except KeyError:
                count_dic[ins.op.name] = 1

        return count_dic

    def control(self, amount):
        return self.to_gate().control(amount)

    def compose(self, other, qubits=[], clbits=[], inplace=True):
        if inplace:
            self.append(other.to_gate(), qubits, clbits)

        else:
            res = self.copy()
            res.append(other.to_gate(), qubits, clbits)
            return res

    def bind_parameters(self, subs_dic):
        """
        Returns a QuantumCircuit where the abstract parameters in ``subs_dic`` are bound
        to their specified values.

        Parameters
        ----------
        subs_dic : dict
            A dictionary containing the abstract parameters of this QuantumCircuit as
            keys and the desired parameters as values.

        Raises
        ------
        Exception
            ``subs_dic`` did not specify a value for all abstract parameters.

        Returns
        -------
        QuantumCircuit
            The QuantumCircuit with substituted parameters.

        Examples
        --------

        We create a QuantumCircuit with some abstract parameters and bind them
        subsequently:

        >>> from qrisp import QuantumCircuit
        >>> from sympy import symbols
        >>> qc = QuantumCircuit(3)

        Create some sympy symbols and use them as abstract parameters for phase gates:

        >>> abstract_parameters = symbols("a b c")
        >>> for i in range(3): qc.p(abstract_parameters[i], i)

        Create the substitution dictionary and bind the parameters:

        >>> subs_dic = {abstract_parameters[i] : i for i in range(3)}
        >>> bound_qc = qc.bind_parameters(subs_dic)
        >>> print(bound_qc)

        .. code-block:: none

                  ┌──────┐
            qb_0: ┤ P(0) ├
                  ├──────┤
            qb_1: ┤ P(1) ├
                  ├──────┤
            qb_2: ┤ P(2) ├
                  └──────┘

        """

        subs_circ = self.clearcopy()

        for ins in self.data:
            if len(ins.op.abstract_params):
                op = ins.op.bind_parameters(subs_dic)
            else:
                op = ins.op.copy()

            subs_circ.data.append(Instruction(op, ins.qubits, ins.clbits))

        subs_circ.abstract_params = {}
        return subs_circ

    def to_latex(self, **kwargs):
        """
        Deploys the Qiskit circuit drawer to generate LaTeX output.

        Parameters
        ----------
        **kwargs : dict
            Dictionary of keyword args for Qiskits `circuit_drawer
            <https://qiskit.org/documentation/stable/0.19/stubs/qiskit.visualization.circuit_drawer.html>`_
            function.

        Returns
        -------
        string
            A string containing the latex code.

        """
        from qrisp.interface import convert_to_qiskit

        qiskit_qc = convert_to_qiskit(self, transpile=False)

        from qiskit.visualization import circuit_drawer

        return circuit_drawer(qiskit_qc, output="latex_source", **kwargs)

    def to_qasm2(self, formatted=False, filename=None, encoding=None):
        """
        Returns the `OpenQASM <https://en.wikipedia.org/wiki/OpenQASM>`_ string of self.

        Parameters
        ----------
        formatted : bool, optional
            Return formatted Qasm string. The default is False.
        filename : string, optional
            Save Qasm to file with name ‘filename’. The default is None.
        encoding : TYPE, optional
            Optionally specify the encoding to use for the output file if filename is
            specified. By default, this is set to the system’s default encoding
            (i.e. whatever locale.getpreferredencoding() returns) and can be set to any
            valid codec or alias from stdlib’s codec module.

        Returns
        -------
        string
            The OPENQASM string.

        """
        qiskit_qc = self.to_qiskit()
        try:
            return qiskit_qc.qasm(formatted, filename, encoding)
        except:
            from qiskit.qasm2 import QASM2ExportError, dumps

            try:
                return dumps(qiskit_qc)
            except (QASM2ExportError, TypeError):
                from qiskit import transpile
                from qiskit.qasm3 import dumps

                transpiled_qiskit_qc = transpile(
                    qiskit_qc,
                    basis_gates=[
                        "x",
                        "y",
                        "z",
                        "h",
                        "s",
                        "t",
                        "s_dg",
                        "t_dg",
                        "cx",
                        "cz",
                        "rz",
                    ],
                )
                return dumps(qiskit_qc)

    def to_qasm3(self, formatted=False, filename=None, encoding=None):
        """
        Returns the `OpenQASM <https://en.wikipedia.org/wiki/OpenQASM>`_ string of self.

        Parameters
        ----------
        formatted : bool, optional
            Return formatted Qasm string. The default is False.
        filename : string, optional
            Save Qasm to file with name ‘filename’. The default is None.
        encoding : TYPE, optional
            Optionally specify the encoding to use for the output file if filename is
            specified. By default, this is set to the system’s default encoding
            (i.e. whatever locale.getpreferredencoding() returns) and can be set to any
            valid codec or alias from stdlib’s codec module.

        Returns
        -------
        string
            The OPENQASM string.

        """
        qiskit_qc = self.to_qiskit()
        from qiskit.qasm3 import dumps

        return dumps(qiskit_qc)

    def qasm(self, **kwargs):
        return self.to_qasm2(**kwargs)

    def depth(self, depth_indicator=lambda x: 1, transpile=True):
        """
        Returns the depth of the QuantumCircuit. Note that the depth on QuantumCircuit
        which are not transpiled, might have very little correlation with the runtime.

        Parameters
        ----------
        depth_indicator : function, optional
            A function which receives an :ref:`Operation` instance and returns the
            time/logical depth this operation takes. By default every Operation takes
            logical depth 1.
        transpile : bool, optional
            Boolean to indicate wether the QuantumCircuit should be transpiled before
            the depth is calculated. The default is True.

        Returns
        -------
        integer
            The depth of the QuantumCircuit.

        """

        from qrisp.misc import get_depth_dic

        if len(self.data) == 0:
            return 0

        depth_dic = get_depth_dic(
            self, transpile_qc=transpile, depth_indicator=depth_indicator
        )

        return max(depth_dic.values())

    def t_depth(self, epsilon=None):
        r"""
        Estimates the T-depth of self. T-depth is an important metric for fault tolerant
        quantum computing, because T gates are expected to be the bottleneck in fault tolerant
        architectures.
        
        According to `this paper <https://arxiv.org/abs/1403.2975>`_, the synthesis of an $RZ(\phi)$
        up to precision $\epsilon$ requires $3\text{log}_2(\frac{1}{\epsilon})$ 
        T-gates.
        
        Based on this formula, this method performs a conservative estimate of the T-depth
        of self.
            
        Parameters
        ----------
        epsilon : float, optional
            The precision up to which parametrized gates should be approximated. If not given,
            Qrisp will determine the parameter with the highest precision. For more information
            on this feature see the examples


        Returns
        -------
        float
            The estimated T-depth.

        Examples
        --------
        
        We create a QuantumCircuit and evaluate the T-depth:
            
        >>> import numpy as np
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.t(0)
        >>> qc.cx(0,1)
        >>> qc.rx(2*np.pi*3/2**4, 1)
        >>> qc.t_depth(epsilon = 2**-5)
        16
        
        In this example we execute a T-Gate on the 0-th qubit (T-depth : 1) followed
        by a CNOT gate (T-depth: 0) on qubits 0 and 1 and finally an RX gate.
        
        The RX-Gate can be executed by using the decomposition 
        
        .. math::
            
            RX(\phi) = \text{H } \text{RZ}(\phi) \text{ H}

        The T-depth of executing a parametrized RX gate is therefore the same
        as the parametrized RZ. To determine the T-depth of the RZ-gate, executed
        with precision $2^{-5}$ we use the above formula:
            
        .. math::
            
            \begin{align}
            \text{TDEPTH}(\text{RZ}(\phi), \epsilon = 2^-5) = 3 \text{log}_2(2^5)\\
            &= 15
            \end{align}
            
        We therefore arrive at 16.
        
        **Automatic precision determination**
        
        In the case of unknown precision, Qrisp assumes that every parameter in the 
        circuit is of the form.
        
        .. math::
            
            \phi = 2\pi\frac{m}{2^k}
        
        Where $m$ is an integer. Qrisp will then determine the parameter with the maximum $k$
        and set $\epsilon = 2^{k_{max} +3}$.
        
        >>> qc.t_depth()
        22
        
        In this circuit $k_{max} = 4$ therefore, $\epsilon = 2^{-7}$ implying the T-depth is 22.
        """

        if epsilon is None:
            transpiled_qc = self.transpile()

            max_circuit_prec = 15
            for instr in transpiled_qc.data:
                op = instr.op

                for par in op.params:

                    par = int(np.round((par % (2 * np.pi)) / (2 * np.pi) * 2**15))

                    for i in range(max_circuit_prec):
                        if par % (2**i):
                            max_circuit_prec = i
                            break

            max_circuit_prec = 16 - max_circuit_prec

            epsilon = 2 ** (-max_circuit_prec - 3)

        from qrisp.misc.utility import t_depth_indicator

        return self.depth(depth_indicator=lambda x: t_depth_indicator(x, epsilon))

    def cnot_depth(self):
        """
        This function returns the CNOT-depth of an self.

        In NISQ-era devices, CNOT gates are the restricting bottleneck for quantum
        circuit execution. This function can be used as a gate-speed specifier for
        the :meth:`compile <qrisp.QuantumSession.compile>` method.


        Returns
        -------
        int
            The CNOT depth of self.

        Examples
        --------

        We create a QuantumCircuit and evaluate it's CNOT depth.

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(4)
        >>> qc.cx(0,1)
        >>> qc.x(1)
        >>> qc.cx(1,2)
        >>> qc.y(2)
        >>> qc.cx(2,3)
        >>> qc.cx(1,0)
        >>> print(qc)

        .. code-block:: none

                                  ┌───┐
            qb_59: ──■────────────┤ X ├─────
                   ┌─┴─┐┌───┐     └─┬─┘
            qb_60: ┤ X ├┤ X ├──■────■───────
                   └───┘└───┘┌─┴─┐┌───┐
            qb_61: ──────────┤ X ├┤ Y ├──■──
                             └───┘└───┘┌─┴─┐
            qb_62: ────────────────────┤ X ├
                                       └───┘

        >>> qc.cnot_depth()
        3

        """
        from qrisp.misc.utility import cnot_depth_indicator

        return self.depth(depth_indicator=cnot_depth_indicator)

    def num_qubits(self):
        """
        Returns the amount of qubits.

        Returns
        -------
        int
            Amount of Qubits.

        """
        return len(self.qubits)

    # Interface for appending instructions
    # Can take either instruction or operations objects
    # Can apply multiple operations, if given the correct qubits
    # For instance if it is required to apply an x gate to qubits 4,5,6 execute
    # qc.append(XGate(), [qubit4, qubit5, qubit6])
    # If it is required to apply a cx gate to the qubit pairs (1,2), (3,4), (5,6)
    # execute qc.append(CXGate(), [[qubit1, qubit3, qubit5], [qubit2, qubit4, qubit6]])
    # If it is required to apply a cx gate to the qubit pairs (1,2), (1,3), (1,4)
    # execute qc.append(CXGate(), [qubit_1, [qubit2, qubit3, qubit4]])
    def append(self, operation_or_instruction, qubits=[], clbits=[]):
        r"""
        Method for appending Operation or Instruction objects to the QuantumCircuit.

        The parameter qubits can be an integer, a list of integers, a Qubit object or a
        list of Qubit objects. The same is valid for the clbit parameter.

        If given an Instruction object instead of an Operation, the given qubit and
        clbit parameters are ignored.

        Parameters
        ----------
        operation_or_instruction : Operation or Instruction
            The operation or instruction to be appended to the QuantumCircuit.
        qubits : integer, list[integer], Qubit, list[Qubit], optional
            The qubits on which to apply the operation. The default is [].
        clbits : integer, list[integer], Clbit, list[Clbit], optional
            The classical bits on which to apply the operation. The default is [].

        Examples
        --------

        We create a $H^{\otimes 4}$ gate and append it to every second qubit of another
        QuantumCircuit:


        >>> from qrisp import QuantumCircuit
        >>> multi_h_qc = QuantumCircuit(4, name = "multi h")
        >>> multi_h_qc.h(range(4))
        >>> multi_h = multi_h_qc.to_gate()
        >>> qc = QuantumCircuit(8)
        >>> qc.append(multi_h, [2*i for i in range(4)])
        >>> print(qc)

        .. code-block:: none

                   ┌──────────┐
             qb_4: ┤0         ├
                   │          │
             qb_5: ┤          ├
                   │          │
             qb_6: ┤1         ├
                   │          │
             qb_7: ┤  multi h ├
                   │          │
             qb_8: ┤2         ├
                   │          │
             qb_9: ┤          ├
                   │          │
            qb_10: ┤3         ├
                   └──────────┘
            qb_11: ────────────

        """

        # Check the type of the instruction/operation
        # from qrisp.circuit import Instruction, Operation

        if self.xla_mode > 0:
            if isinstance(operation_or_instruction, Instruction):
                self.data.append(operation_or_instruction)
            else:
                if self.xla_mode <= 1:
                    if not isinstance(qubits, list):
                        raise Exception(
                            f"Operation {operation_or_instruction.name} was appended with {qubits} in accelerated compilation mode (allowed is type List[Qubit])."
                        )
                    for qb in qubits:
                        if not isinstance(qb, Qubit):
                            raise Exception(
                                f"Operation {operation_or_instruction.name} was appended with {qubits} in accelerated compilation mode (allowed is type List[Qubit])."
                            )
                self.data.append(Instruction(operation_or_instruction, qubits, clbits))
            return

        if isinstance(operation_or_instruction, Instruction):
            instruction = operation_or_instruction
            self.append(instruction.op, instruction.qubits, instruction.clbits)

            return

        elif isinstance(operation_or_instruction, Operation):
            operation = operation_or_instruction

        else:
            raise Exception(
                "Tried to append object type "
                + str(type(operation_or_instruction))
                + " which is neither Instruction nor Operation"
            )

        # Convert arguments (possibly integers) to list
        # The logic here is that the list structure gets preserved ie.
        # [[0, 1] ,2] ==> [[qubit_0, qubit_1], qubit_2]
        # unless the input is a single qubit/integer.
        # In this case we have
        # qubit_0 ==> [qubit_0]

        qubits = convert_to_qb_list(qubits, circuit=self)
        clbits = convert_to_cb_list(clbits, circuit=self)

        # Now we check which of the arguments is a list
        # For user convenience we allow to execute multiple gates at the same time
        # This comes with some restrictions where the operation to execute could be
        # ambigous.
        # When appending n gates with a single call of this function,
        # each qubit argument must either be a list of n qubits or a single qubit

        # First we check which arguments are lists
        qb_argument_is_list = []
        for i in range(len(qubits)):
            if isinstance(qubits[i], list):
                qb_argument_is_list.append(i)

        # Same with classical bits
        cb_argument_is_list = []
        for i in range(len(clbits)):
            if isinstance(clbits[i], list):
                cb_argument_is_list.append(i)

        if qb_argument_is_list + cb_argument_is_list:
            # Determine the amount of gates to be applied
            if qb_argument_is_list:
                arg_list_len = len(qubits[qb_argument_is_list[0]])
            else:
                arg_list_len = len(clbits[cb_argument_is_list[0]])

            # Check that indeed every list argument that has been given has
            # arg_list_len entries
            for arg_list_index in qb_argument_is_list:
                if len(qubits[arg_list_index]) != arg_list_len:
                    raise Exception(
                        "Don't know how to combine appending arguments "
                        + str((qubits + clbits))
                    )

            for arg_list_index in cb_argument_is_list:
                if len(clbits[arg_list_index]) != arg_list_len:
                    raise Exception(
                        "Don't know how to combine appending arguments "
                        + str((qubits + clbits))
                    )

            # Create argument constellations
            for i in range(arg_list_len):
                qubit_constellation = []
                for j in range(len(qubits)):
                    if j in qb_argument_is_list:
                        qubit_constellation.append(qubits[j][i])
                    else:
                        qubit_constellation.append(qubits[j])

                clbit_constellation = []
                for j in range(len(clbits)):
                    if j in cb_argument_is_list:
                        clbit_constellation.append(clbits[j][i])
                    else:
                        clbit_constellation.append(clbits[j])

                # Append instruction (qubit_constellation and clbit_constellation) now
                # contains no lists but only qubit/clbit arguments
                QuantumCircuit.append(
                    self, operation, qubit_constellation, clbit_constellation
                )

            return

        if len(qubits) != operation.num_qubits:
            raise Exception(
                f"Provided incorrect amount ({len(qubits)}) of qubits for operation "
                + str(operation.name)
                + f" (requires {operation.num_qubits})"
            )

        if len(clbits) != operation.num_clbits:
            raise Exception(
                f"Provided incorrect amount ({len(clbits)}) of clbits for operation "
                + str(operation.name)
                + f" (requires {operation.num_clbits})"
            )

        if len(set(qubits)) != len(qubits):
            raise Exception(
                f"Duplicate qubit arguments in {qubits} for operation {operation.name}"
            )

        # Building up the list of identifiers seems to slow down this function
        # We therefore check first if the qubit objects match and if this is not the
        # case we check if the identifiers match
        if not set(qubits).issubset(set(self.qubits)):
            op_identifiers = [qb.identifier for qb in qubits]
            qc_identifiers = [qb.identifier for qb in self.qubits]

            if not set(op_identifiers).issubset(qc_identifiers):
                raise Exception(
                    "Instruction Qubits "
                    + str(set(qubits) - set(self.qubits))
                    + " not present in circuit"
                )
            else:
                qubits = [
                    self.qubits[qc_identifiers.index(op_id)] for op_id in op_identifiers
                ]

        if len(set([cb.identifier for cb in clbits])) != len(clbits):
            raise Exception("Duplicate clbit arguments")

        if not set([cb.identifier for cb in clbits]).issubset(
            set([cb.identifier for cb in self.clbits])
        ):
            raise Exception("Instruction Clbits not present in circuit")

        # Log which abstract parameters have been added to the circuit
        try:
            self.abstract_params.update(operation.abstract_params)
        except AttributeError:
            pass

        critical_qubits = []
        perm_critical_qubits = []

        for qb in qubits:
            if qb.lock:
                critical_qubits.append(qb)
            if qb.perm_lock:
                perm_critical_qubits.append(qb)

        critical_qubits = [qb for qb in qubits if qb.lock]
        if critical_qubits:
            if critical_qubits[0].lock_message:
                raise Exception(critical_qubits[0].lock_message)
            else:
                raise Exception(
                    f"Tried to perform operation {operation.name}"
                    "on locked qubit {critical_qubits[0]}"
                )

        # Check if there are non-permeable operations on pt_locked qubits
        critical_qubits = [qb for qb in qubits if qb.perm_lock]

        if critical_qubits:
            from qrisp.permeability import is_permeable

            critical_qubit_indices = [qubits.index(qb) for qb in critical_qubits]
            if not is_permeable(operation, critical_qubit_indices):
                if critical_qubits[0].perm_lock_message:
                    raise Exception(critical_qubits[0].perm_lock_message)
                else:
                    raise Exception(
                        f"Tried to perform non-permeable operation {operation.name} on"
                        f" perm_locked qubit {critical_qubits[0]}"
                    )

        self.data.append(Instruction(operation, qubits, clbits))

    def run(self, shots=None, backend=None):
        """
        Runs a QuantumCircuit on a given backend.

        Parameters
        ----------
        shots : int, optional
            The amount of shots to perform. The default is 10000.
        backend : BackendClient, optional
            The backend on which to evaluate the QuantumCircuit. The default is None.

        Returns
        -------
        dict
            The resulting counts for the given QuantumCircuit.

        Examples
        --------

        We create a GHZ QuantumCircuit and evaluate the results.

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(5)
        >>> qc.h(0)
        >>> qc.cx(0, range(1,5))
        >>> qc.measure(range(5))
        >>> qc.run()
        {'0': 5000, '1': 5000}

        """
        if backend is None:
            from qrisp.default_backend import def_backend

            backend = def_backend

        return backend.run(self, shots)

    def statevector_array(self):
        """
        Performs a simulation of the statevector of self and returns a numpy array of
        complex numbers.

        .. note::

            Qrisps qubit ordering convention is reversed when compared to Qiskit,
            because of simulation efficiency reasons.
            As a rule of thumb you can remember:

            The statevector array of the following circuit has the amplitude 1 at the
            index ``0010 = 2``

            .. code-block:: none

                qb.0: ─────

                qb.1: ─────
                      ┌───┐
                qb.2: ┤ X ├
                      └───┘
                qb.3: ─────

        Returns
        -------
        numpy.ndarray
            The statevector of this circuit.

        Examples
        --------

        We create a QuantumCircuit, perform some operations and retrieve the
        statevector array.

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(4)
        >>> qc.h(qc.qubits)
        >>> qc.z(-1)
        >>> qc.statevector_array()
        array([ 0.24999997+0.j, -0.24999997+0.j,  0.24999997+0.j, -0.24999997+0.j,
                0.24999997+0.j, -0.24999997+0.j,  0.24999997+0.j, -0.24999997+0.j,
                0.24999997+0.j, -0.24999997+0.j,  0.24999997+0.j, -0.24999997+0.j,
                0.24999997+0.j, -0.24999997+0.j,  0.24999997+0.j, -0.24999997+0.j],
              dtype=complex64)


        """
        from qrisp.simulator import statevector_sim

        return statevector_sim(self)

    def __hash__(self):
        from hashlib import sha256

        res = 0

        def hash_(x):
            temp = str(x).encode("utf-8")
            hex_value = sha256(temp).hexdigest()
            return int(hex_value, 16)

        transpiled_qc = self

        n = len(self.qubits)
        for i in range(len(transpiled_qc.data)):
            instr = transpiled_qc.data[i]

            qubit_indices = {}

            for j in range(n):
                try:
                    qubit_indices[instr.qubits.index(self.qubits[j])] = j
                except ValueError:
                    pass

            qubit_indices = [qubit_indices[j] for j in range(len(instr.qubits))]

            index_hash = hash(tuple(qubit_indices))

            params = []
            for j in range(len(instr.op.params)):
                p = hash((instr.op.params[j], i))
                params.append(p)

            param_hash = hash(tuple(params))

            if instr.op.definition:
                op_hash = hash(instr.op.definition)
            else:
                op_hash = hash(instr.op.name)

            res += hash((index_hash, param_hash, op_hash)) * (i + 1) ** 2

        res *= len(self.qubits) ** 2

        return hash(res)

    @classmethod
    def from_qasm_str(self, qasm_string):
        """
        Loads a QuantumCircuit from a QASM String.

        Parameters
        ----------
        qasm_string : string
            A string obeying the syntax of the OpenQASM specification.

        Returns
        -------
        QuantumCircuit
            The corresponding QuantumCircuit.

        """

        from qiskit import QuantumCircuit

        qiskit_qc = QuantumCircuit().from_qasm_str(qasm_string)

        from qrisp import QuantumCircuit

        return QuantumCircuit.from_qiskit(qiskit_qc)

    @classmethod
    def from_qasm_file(self, filename):
        """
        Loads a QuantumCircuit from a QASM file.

        Parameters
        ----------
        filename : string
            A string pointing to a file obeying the OpenQASM syntax.

        Returns
        -------
        QuantumCircuit
            The corresponding QuantumCircuit.

        """
        from qiskit import QuantumCircuit

        qiskit_qc = QuantumCircuit().from_qasm_file(filename)

        from qrisp import QuantumCircuit

        return QuantumCircuit.from_qiskit(qiskit_qc)

    @classmethod
    def from_qiskit(self, qiskit_qc):
        """
        Class method to create QuantumCircuits from Qiskit QuantumCircuits.

        Parameters
        ----------
        qiskit_qc : Qiskit QuantumCircuit
            The Qiskit QuantumCircuit to convert.

        Returns
        -------
        QuantumCircuit
            The converted QuantumCircuit.

        Examples
        --------

        We construct a fan-out QuantumCircuit in Qiskit:

        >>> from qiskit import QuantumCircuit as QiskitQuantumCircuit
        >>> qc_2 = QiskitQuantumCircuit(4)
        >>> qc_2.cx(0, range(1,4))
        >>> print(qc_2)

        .. code-block:: none

            q_0: ──■────■────■──
                 ┌─┴─┐  │    │
            q_1: ┤ X ├──┼────┼──
                 └───┘┌─┴─┐  │
            q_2: ─────┤ X ├──┼──
                      └───┘┌─┴─┐
            q_3: ──────────┤ X ├
                           └───┘

        Note that we don't need to create a QuantumCircuit object first as this is a
        class method.

        >>> qrisp_qc_2 = QuantumCircuit.from_qiskit(qc_2)
        >>> print(qrisp_qc_2)

        .. code-block:: none

             qb_8: ──■────■────■──
                   ┌─┴─┐  │    │
             qb_9: ┤ X ├──┼────┼──
                   └───┘┌─┴─┐  │
            qb_10: ─────┤ X ├──┼──
                        └───┘┌─┴─┐
            qb_11: ──────────┤ X ├
                             └───┘

        """
        from qrisp.interface import convert_from_qiskit

        return convert_from_qiskit(qiskit_qc)

    def to_qiskit(self):
        """
        Method to convert the given QuantumCircuit to a Qiskit QuantumCircuit.

        Returns
        -------
        Qiskit QuantumCircuit
            The converted circuit.

        """
        from qrisp.interface import convert_to_qiskit

        return convert_to_qiskit(self, transpile=False)

    def to_pennylane(self):
        """
        Method to convert the given QuantumCircuit to a `Pennylane <https://pennylane.ai/>`_ Circuit.

        Returns
        -------
        function
            A function representing a pennylane QuantumCircuit.

        """

        from qrisp.interface import qml_converter

        return qml_converter(self)
    
    def to_stim(self, return_measurement_map = False, return_detector_map = False, return_observable_map = False):
        """
        Method to convert the given QuantumCircuit to a `Stim <https://github.com/quantumlib/Stim/>`_ Circuit.

        .. note::
            
            Stim can only process/represent Clifford operations.

        Parameters
        ----------
        return_measurement_map : bool, optional
            If set to True, the function returns the measurement_map, as described below.
            The default is False.
        return_detector_map : bool, optional
            If set to True, the function returns the detector_map.
            The default is False.
        return_observable_map : bool, optional
            If set to True, the function returns the observable_map.
            The default is False.

        Returns
        -------
        stim_circuit : stim.Circuit
            The converted Stim circuit.
        measurement_map : dict
            (Optional) A dictionary mapping Qrisp Clbit objects to Stim measurement record indices. 
            For example, ``{Clbit(cb_1): 2, Clbit(cb_0): 1}`` means ``Clbit("cb_1")``
            corresponds to index 2 in Stim's measurement record.
        detector_map : dict
            (Optional) A dictionary mapping ParityHandle objects to Stim detector indices.
            ParityHandle objects are compared by their index, so handles from to_qc() can
            be used directly as keys.
        observable_map : dict
            (Optional) A dictionary mapping ParityHandle objects to Stim observable indices.
            ParityHandle objects are compared by their index, so handles from to_qc() can
            be used directly as keys.

        Examples
        --------
        Basic conversion:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2, 2)
        >>> qc.x(0)
        >>> qc.cz(0, 1)
        >>> qc.measure(0, 0)
        >>> qc.measure(1, 1)
        >>> print(qc)
              ┌───┐   ┌─┐   
        qb_0: ┤ X ├─■─┤M├───
              └───┘ │ └╥┘┌─┐
        qb_1: ──────■──╫─┤M├
                       ║ └╥┘
        cb_0: ═════════╩══╬═
                          ║ 
        cb_1: ════════════╩═
                 
        >>> stim_circuit = qc.to_stim()
        >>> print(stim_circuit)
        X 0
        CZ 0 1
        M 0 1
        
        Stim creates measurement indices in the order of how the measurements appear
        in the circuit. This is different in Qrisp: It is for instance possible 
        for the first measurement of the circuit to target the second ``Clbit``.
        The second measurement can in-principle then target either the first or
        the second ``Clbit``. In order to still identify which ``Clbit`` corresponds to
        which stim measurement index, we can use the ``return_measurement_map`` keyword
        argument.
        
        >>> qc = QuantumCircuit(2, 2)
        >>> qc.x(0)
        >>> qc.cz(0, 1) 
        >>> qc.measure(1, 1) # The first measurement of the circuit targets the second ClBit
        >>> qc.measure(0, 0) # The second measurement of the circuit targets the first ClBit
        >>> print(qc)
              ┌───┐      ┌─┐
        qb_0: ┤ X ├─■────┤M├
              └───┘ │ ┌─┐└╥┘
        qb_1: ──────■─┤M├─╫─
                      └╥┘ ║ 
        cb_0: ═════════╬══╩═
                       ║    
        cb_1: ═════════╩════
        >>> stim_circuit, measurement_map = qc.to_stim(return_measurement_map = True)
        >>> print(stim_circuit)
        X 0
        CZ 0 1
        M 1 0
            
        We see that Stim now measures the qubit with index 1 first (``M 1 0``),
        which is why in the measurement record the measurement result in ``Clbit("cb_1")``
        will appear at index 0 and ``Clbit("cb_0")`` at index 1.
        To retrieve the correct order, we inspect the ``measurement_map`` dictionary.

        >>> print(measurement_map)  # Maps Clbit objects to Stim measurement indices
        {Clbit(cb_1): 0, Clbit(cb_0): 1}
            
        We can now check the samples drawn from this circuit for a given ``Clbit``
        object by slicing the sampling result array.
        
        >>> sampler = stim_circuit.compile_sampler()
        >>> all_samples = sampler.sample(5)
        >>> samples = all_samples[:, measurement_map[qc.clbits[0]]]
        >>> print(samples)
        array([ True,  True,  True,  True,  True])
            
        """

        from qrisp.interface import qrisp_to_stim

        return qrisp_to_stim(self, return_measurement_map, return_detector_map, return_observable_map)

    def to_pytket(self):
        """
        Method to convert the given QuantumCircuit to a `PyTket <https://cqcl.github.io/tket/pytket/api/#>`_ Circuit.

        Returns
        -------
        function
            A function representing a PyTket QuantumCircuit.

        """
        from qrisp.interface import pytket_converter

        return pytket_converter(self)

    def to_cirq(self):
        """
        Method to convert the given QuantumCircuit to a Cirq Circuit.

        Returns
        -------
        function
            A function representing a Cirq QuantumCircuit.

        """
        from qrisp.interface import convert_to_cirq

        return convert_to_cirq(self)
    
    # Several methods to apply the standard operation defined in standard_operations.py
    def measure(self, qubits, clbits=None):
        """
        Instructs a measurement. If given no classical bits, the proper amount will be
        created.


        Parameters
        ----------
        qubits : Qubit
            The Qubit to be measured.
        clbits : ClBit, optional
            The Clbit to store the measurement result. The default is None.

        """

        if clbits is None:
            from qrisp import QuantumVariable

            if isinstance(qubits, (list, QuantumVariable)):
                clbits = []
                for i in range(len(qubits)):
                    clbits.append(self.add_clbit())
            else:
                clbits = self.add_clbit()

        self.append(ops.Measurement(), [qubits], [clbits])

    def cx(self, qubits_0, qubits_1):
        """
        Instruct a CX-gate.

        Parameters
        ----------
        qubits_0 : Qubit
            The Qubit to control on.
        qubits_1 : Qubit
            The target Qubit.

        """
        self.append(ops.CXGate(), [qubits_0, qubits_1])

    def cy(self, qubits_0, qubits_1):
        """
        Instruct a CY-gate.

        Parameters
        ----------
        qubits_0 : Qubit
            The Qubit to control on.
        qubits_1 : Qubit
            The target Qubit.

        """
        self.append(ops.CYGate(), [qubits_0, qubits_1])

    def cz(self, qubits_0, qubits_1):
        """
        Instruct a CZ-gate.

        Parameters
        ----------
        qubits_0 : Qubit
            The Qubit to control on.
        qubits_1 : Qubit
            The target Qubit.

        """
        self.append(ops.CZGate(), [qubits_0, qubits_1])

    def h(self, qubits):
        """
        Instruct a Hadamard-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """

        self.append(ops.HGate(), [qubits])

    def x(self, qubits):
        """
        Instruct a Pauli-X-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.XGate(), [qubits])

    def y(self, qubits):
        """
        Instruct a Pauli-Y-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.YGate(), [qubits])

    def z(self, qubits):
        """
        Instruct a Pauli-Z-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.ZGate(), [qubits])

    def rx(self, phi, qubits):
        """
        Instruct a parametrized RX-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits : Qubit
            The Qubit to apply the gate on.
        """
        if phi == 0:
            return
        self.append(ops.RXGate(phi), [qubits])

    def ry(self, phi, qubits):
        """
        Instruct a parametrized RY-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits : Qubit
            The Qubit to apply the gate on.
        """

        if phi == 0:
            return
        self.append(ops.RYGate(phi), [qubits])

    def rz(self, phi, qubits):
        """
        Instruct a parametrized RZ-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits : Qubit
            The Qubit to apply the gate on.
        """
        if phi == 0:
            return
        self.append(ops.RZGate(phi), [qubits])

    def cp(self, phi, qubits_0, qubits_1):
        """
        Instruct a controlled phase-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits_0 : Qubit
            The Qubit to apply the gate on.
        qubits_1 : Qubit
            The other Qubit to apply the gate on.
        """
        if phi == 0:
            return
        self.append(ops.CPGate(phi), [qubits_0, qubits_1])

    def p(self, phi, qubits):
        """
        Instruct a phase-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits : Qubit
            The Qubit to apply the gate on.
        """
        if phi == 0:
            return
        self.append(ops.PGate(phi), [qubits])

    def rxx(self, phi, qubits_0, qubits_1):
        """
        Instruct an RXX-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits_0 : Qubit
            The Qubit to apply the gate on.
        qubits_1 : Qubit
            The other Qubit to apply the gate on.
        """

        if phi == 0:
            return
        self.append(ops.RXXGate(phi), [qubits_0, qubits_1])

    def rzz(self, phi, qubits_0, qubits_1):
        """
        Instruct an RZZ-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits_0 : Qubit
            The Qubit to apply the gate on.
        qubits_1 : Qubit
            The other Qubit to apply the gate on.
        """

        if phi == 0:
            return
        self.append(ops.RZZGate(phi), [qubits_0, qubits_1])

    def xxyy(self, phi, beta, qubits_0, qubits_1):
        """
        Instruct an XXYY-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.
        beta : float or sympi.Symbol
            The other angle parameter
        qubits_0 : Qubit
            The Qubit to apply the gate on.
        qubits_1 : Qubit
            The other Qubit to apply the gate on.
        """

        if phi == 0:
            return
        self.append(ops.XXYYGate(phi, beta), [qubits_0, qubits_1])

    def swap(self, qubits_0, qubits_1):
        """
        Instruct a SWAP-gate.

        Parameters
        ----------
        qubits_0 : Qubit
            The qubit to swap.
        qubits_1 : Qubit
            The other qubit to swap.

        """
        self.append(ops.SwapGate(), [qubits_0, qubits_1])

    def mcx(self, control_qubits, target_qubits, method="gray", ctrl_state=-1):
        """
        Instruct a multi-controlled X-gate.

        Parameters
        ----------
        control_qubits : list
            The list of Qubits to control on.
        target_qubits : Qubit
            The target Qubit.
        method : str, optional
            The algorithm to synthesize the mcx gate. The default is "gray".
        ctrl_state : str or int, optional
            The state on which the X gate is activated. Can be supplied as a string
            (i.e. "010110...") or an integer. The default is all ones ("11111...").


        """
        self.append(
            ops.MCXGate(len(control_qubits), ctrl_state=ctrl_state, method=method),
            control_qubits + [target_qubits],
        )

    def ccx(self, ctrl_qubit_0, ctrl_qubit_1, target_qubit, method="gray"):
        """
        Instruct a Toffoli-gate.

        Parameters
        ----------
        ctrl_qubit_0 : list
            The first control Qubit.
        ctrl_qubit_1 : Qubit
            The second control Qubit.
        target_qubit : Qubit.
            The target Qubit.
        method : str, optional
            The algorithm to synthesize the mcx gate. The default is "gray".
        """

        self.mcx([ctrl_qubit_0, ctrl_qubit_1], target_qubit, method=method)

    def crx(self, phi, qubits_0, qubits_1):
        """
        Instruct a controlled rx-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.

        qubits_0 : Qubit
            The Qubit to apply the gate on.
        qubits_1 : Qubit
            The other Qubit to apply the gate on.
        """
        if phi == 0:
            return
        self.append(ops.MCRXGate(phi, 1), [qubits_0, qubits_1])

    def t(self, qubits):
        """
        Instruct a T-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.TGate(), [qubits])

    def t_dg(self, qubits):
        """
        Instruct a dagger T-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.TGate().inverse(), [qubits])

    def s(self, qubits):
        """
        Instruct an S-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.SGate(), [qubits])

    def s_dg(self, qubits):
        """
        Instruct a daggered S-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.SGate().inverse(), [qubits])

    def sx(self, qubits):
        """
        Instruct a SX-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.SXGate(), [qubits])

    def sx_dg(self, qubits):
        """
        Instruct a daggered SX-gate.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the gate on.
        """
        self.append(ops.SXGate().inverse(), [qubits])

    def barrier(self, qubits=None, clbits=None):
        """
        Instruct a Barrier onto the given Qubit. Barriers can be used as visual markers
        and compiler directives.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to apply the barrier on.
        clbits : Clbit
            The Clbits to apply the barrier on.
        """

        if qubits is None:
            qubits = self.qubits

        self.append(ops.Barrier(len(qubits)), qubits)

    def reset(self, qubits):
        r"""
        Instruct a reset. This resets this Qubit into the $\ket{0}$ state regardless
        of its previous state.

        Parameters
        ----------
        qubits : Qubit
            The Qubit to reset.
        """

        self.append(ops.Reset(), [qubits])

    def u3(self, theta, phi, lam, qubits):
        r"""
        Instruct a U3-gate from given Euler angles.

        A U3 gate has the unitary:

        .. math::

            U3(\theta, \phi, \lambda) = \begin{pmatrix} \cos{(\frac{\theta}{2})}
            & -\exp{(i\lambda)}\sin{(\frac{\theta}{2})} \\
            \exp{(i\phi)} \sin{(\frac{\theta}{2})}
            & \exp{(i(\phi+\lambda))}\cos{(\frac{\theta}{2})} \end{pmatrix}

        Parameters
        ----------
        theta : float or sympy.Symbol
            The theta parameter.
        phi : float or sympy.Symbol
            The phi parameter.
        lam : float or sympy.Symbol
            The lambda parameter.
        qubits : Qubit
            The Qubit to apply the u3 gate on.

        """
        self.append(ops.u3Gate(theta, phi, lam), [qubits])

    def r(self, phi, theta, qubits):
        self.append(ops.RGate(phi, theta), [qubits])

    def unitary(self, unitary_array, qubits):
        """
        Instruct a U3-gate from a given U3 matrix.

        Parameters
        ----------
        unitary_array : numpy.ndarray
            The U3 matrix to apply.
        qubits : Qubit
            The Qubit to apply the gate on.

        """

        mat = unitary_array
        coeff = 1 / np.sqrt(np.linalg.det(mat))
        gphase = -np.angle(coeff) % (2 * np.pi)
        tmp_10 = np.abs((coeff * mat[1][0]))
        tmp_00 = np.abs((coeff * mat[0][0]))
        theta = 2 * np.arctan2(tmp_10, tmp_00)
        phiplambda2 = np.angle(coeff * mat[1][1]) % (2 * np.pi)
        phimlambda2 = np.angle(coeff * mat[1][0]) % (2 * np.pi)
        phi = phiplambda2 + phimlambda2
        lam = phiplambda2 - phimlambda2

        # gphase -= (phi + lam)

        arg_max = np.argmax(np.abs(mat).flatten())
        from qrisp.simulator.unitary_management import u3matrix

        temp_u3 = u3matrix(theta, phi, lam, 0).flatten()

        gphase = (-np.angle(temp_u3[arg_max] / mat.flatten()[arg_max])) % (2 * np.pi)

        from qrisp.circuit import U3Gate
        from qrisp.simulator.unitary_management import u3matrix

        self.append(U3Gate(theta, phi, lam, global_phase=gphase), qubits)
        # self.u3(theta, phi, lam, [qubits], global_phase = gphase)

    def gphase(self, phi, qubits):
        """
        Instruct a global phase. Global phases do not directly influence the
        QuantumCircuits outcome however they can become physical if used as a base gate
        for a controlled operation.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.
        qubits : TYPE
            The Qubit to apply the gate on.
        """

        self.append(ops.GPhaseGate(phi), [qubits])

    def id(self, qubits):
        """
        Instruct an identity gate. Identity gates are simply placeholders and have no
        effect on the quantum state.

        Parameters
        ----------

        qubits : TYPE
            The Qubit to apply the gate on.
        """

        self.append(ops.IDGate(), [qubits])

    def to_pdag(self, remove_artificials=False):
        from qrisp.permeability import PermeabilityGraph

        return PermeabilityGraph(self, remove_artificials=remove_artificials)


# Converts various inputs (eg. integers, qubits or quantum variables) to lists of qubit
# used in the append method of QuantumCircuit and QuantumSession
def convert_to_qb_list(input, circuit=None, top_level=True):
    from qrisp import QuantumArray

    if issubclass(input.__class__, Qubit):
        if top_level:
            result = [input]
        else:
            result = input
    elif isinstance(input, QuantumArray):
        result = sum([qv.reg for qv in input.flatten()], [])

    elif hasattr(input, "__iter__"):
        result = []
        for i in range(len(input)):
            result.append(convert_to_qb_list(input[i], circuit, top_level=False))

    elif hasattr(input, "reg"):
        result = list(input.reg)

    elif isinstance(input, int):
        if isinstance(circuit, type(None)):
            raise Exception(
                "Tried to convert integer argument to qubit without given circuit"
            )

        if input >= len(circuit.qubits):
            raise Exception(
                f"Tried to adress qubit with index {input} in a circuit with {len(circuit.qubits)} qubits"
            )

        result = convert_to_qb_list(circuit.qubits[input], top_level=top_level)

    else:
        raise Exception("Couldn't convert type " + str(type(input)) + " to qubit list")

    return result


def convert_to_cb_list(input, circuit=None, top_level=True):
    from qrisp.circuit import Clbit

    if hasattr(input, "__iter__"):
        result = []
        for i in range(len(input)):
            result.append(convert_to_cb_list(input[i], circuit, top_level=False))

    elif isinstance(input, int):
        if isinstance(circuit, type(None)):
            raise Exception(
                "Tried to convert integer argument to qubit without given circuit"
            )

        result = convert_to_cb_list(circuit.clbits[input], top_level=top_level)

    elif issubclass(input.__class__, Clbit):
        if top_level:
            result = [input]
        else:
            result = input

    return result
