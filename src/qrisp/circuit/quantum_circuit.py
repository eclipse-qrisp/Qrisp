"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

This module contains the main class to describe quantum circuits in Qrisp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sympy
from numpy.linalg import norm
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import transpile as qiskit_transpile
from qiskit.qasm2 import QASM2ExportError
from qiskit.qasm2 import dumps as dumps_qasm2
from qiskit.qasm3 import dumps as dumps_qasm3
from qiskit.visualization import circuit_drawer

import qrisp.circuit.standard_operations as ops
from qrisp.circuit import Clbit, Instruction, Operation, Qubit, U3Gate
from qrisp.misc import (
    cnot_count,
    cnot_depth_indicator,
    get_depth_dic,
    t_depth_indicator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Set

    from qrisp.circuit.operation import ControlledOperation, PTControlledOperation
    from qrisp.interface import BackendLike, MeasurementResult
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import (
        ParityHandle,
    )
    from qrisp.typing import ClbitLike, FloatLike, QubitLike


TO_GATE_COUNTER = np.zeros(1)


def _check_qubit_locks(qubits: list, operation) -> None:
    """Raise RuntimeError if any qubit is locked or perm-locked for this operation."""
    critical_qubits = [qb for qb in qubits if qb.lock]
    if critical_qubits:
        msg = getattr(critical_qubits[0], "lock_message", None)
        if msg:
            raise RuntimeError(msg)
        raise RuntimeError(f"Tried to perform operation {operation.name} on locked qubit {critical_qubits[0]}")

    critical_qubits = [qb for qb in qubits if qb.perm_lock]
    if critical_qubits:
        from qrisp.permeability import is_permeable

        critical_qubit_indices = [qubits.index(qb) for qb in critical_qubits]
        if not is_permeable(operation, critical_qubit_indices):
            msg = getattr(critical_qubits[0], "perm_lock_message", None)
            if msg:
                raise RuntimeError(msg)
            raise RuntimeError(
                f"Tried to perform non-permeable operation {operation.name} on perm_locked qubit {critical_qubits[0]}"
            )


class QuantumCircuit:
    """This class describes quantum circuits. Many of the attribute and method names are
    oriented toward the `Qiskit QuantumCircuit
    <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html>`_ class
    in order to provide a high degree of compatibility.

    QuantumCircuits can be visualized by calling ``print`` on them.

    Qrisp QuantumCircuits can be quickly generated out of existing Qiskit
    QuantumCircuits with the :meth:`from_qiskit <qrisp.QuantumCircuit.from_qiskit>`
    method.


    Parameters
    ----------
    num_qubits : int, optional
        The amount of qubits this QuantumCircuit is initialized with. The default is 0.
    num_clbits : int, optional
        The amount of classical bits. The default is 0.


    Examples
    --------
    We create a QuantumCircuit containing a so-called fan-out gate:

    >>> from qrisp import QuantumCircuit
    >>> qc_0 = QuantumCircuit(4)
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
    >>> qc_1.append(qc_0.to_gate(name="fan-out"), qc_1.qubits)
    >>> print(qc_1)

    .. code-block:: none

              ┌───┐┌──────────┐
        qb_4: ┤ H ├┤0         ├
              └───┘│          │
        qb_5: ─────┤1         ├
                   │  fan-out │
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
                   │  fan-out │ ║ └╥┘┌─┐
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

    We construct the very same fan-out QuantumCircuit in Qiskit:

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
    need to create a QuantumCircuit object first as this is a classmethod.

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

    Abstract parameters are represented by `SymPy symbols
    <https://docs.sympy.org/latest/modules/core.html#module-sympy.core.symbol>`_
    in Qrisp.

    We create a QuantumCircuit with some abstract parameters and bind them subsequently.

    >>> from qrisp import QuantumCircuit
    >>> from sympy import symbols
    >>> qc = QuantumCircuit(3)

    Create some SymPy symbols and use them as abstract parameters for phase gates:

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

    qubit_index_counter: np.ndarray = np.zeros(1, dtype=int)
    clbit_index_counter: np.ndarray = np.zeros(1, dtype=int)
    xla_mode: int = 0

    def __init__(self, num_qubits: int = 0, num_clbits: int = 0) -> None:
        """Initializes the QuantumCircuit."""
        if not isinstance(num_qubits, int):
            raise TypeError(
                f"Tried to initialize QuantumCircuit with type {type(num_qubits).__name__} for num_qubits, expected int"
            )
        if not isinstance(num_clbits, int):
            raise TypeError(
                f"Tried to initialize QuantumCircuit with type {type(num_clbits).__name__} for num_clbits, expected int"
            )

        object.__setattr__(self, "data", [])
        object.__setattr__(self, "qubits", [])
        object.__setattr__(self, "clbits", [])

        self.abstract_params: Set = set()

        start_index = self.qubit_index_counter[0]
        self.qubits: list[Qubit] = [Qubit(f"qb_{start_index + i}") for i in range(num_qubits)]
        self.qubit_index_counter[0] += num_qubits

        start_index = self.clbit_index_counter[0]
        self.clbits: list[Clbit] = [Clbit(f"cb_{start_index + i}") for i in range(num_clbits)]
        self.clbit_index_counter[0] += num_clbits

    def add_qubit(self, qubit: Qubit | None = None) -> Qubit:
        """Adds a Qubit to the QuantumCircuit.

        Parameters
        ----------
        qubit : Qubit, optional
            The Qubit to be added. If None is provided, a new Qubit will be generated.

        Returns
        -------
        Qubit
            The added Qubit.

        Examples
        --------
        We create a QuantumCircuit and add a qubit to it:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit()
        >>> qc.add_qubit()
        >>> qc.qubits
        [Qubit(qb_0)]

        """
        self.qubit_index_counter[0] += 1

        if qubit is None:
            qubit = Qubit(f"qb_{self.qubit_index_counter[0]}")

        if not isinstance(qubit, Qubit):
            raise TypeError(f"Tried to add type {type(qubit)} as a qubit")

        if self.xla_mode < 2:
            if any(qb.identifier == qubit.identifier for qb in self.qubits):
                raise ValueError(f"Qubit name {qubit.identifier} already exists")

        self.qubits.append(qubit)

        return self.qubits[-1]

    def add_clbit(self, clbit: Clbit | None = None) -> Clbit:
        """Adds a classical bit to the QuantumCircuit.

        Parameters
        ----------
        clbit : Clbit, optional
            The classical bit to be added. If None is provided, a new Clbit will be generated.

        Returns
        -------
        Clbit
            The added Clbit.

        Examples
        --------
        We create a QuantumCircuit and add a classical bit to it:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit()
        >>> qc.add_clbit()
        >>> qc.clbits
        [Clbit(cb_0)]

        """
        self.clbit_index_counter[0] += 1

        if clbit is None:
            clbit = Clbit(f"cb_{self.clbit_index_counter[0]}")

        if not isinstance(clbit, Clbit):
            raise TypeError(f"Tried to add type {type(clbit)} as a classical bit")

        if self.xla_mode < 2:
            if any(cb.identifier == clbit.identifier for cb in self.clbits):
                raise ValueError(f"Clbit name {clbit.identifier} already exists")

        self.clbits.append(clbit)

        return self.clbits[-1]

    def to_op(self, name: str | None = None) -> Operation:
        """Method to return an Operation object generated out of this QuantumCircuit.

        Operation objects can be appended to other QuantumCircuits.

        An alias for Qiskit compatibility is the
        :meth:`to_gate<qrisp.QuantumCircuit.to_gate>` method.

        Parameters
        ----------
        name : str, optional
            The name of the gate. By default, the QuantumCircuit's name will be used.

        Returns
        -------
        Operation
            The Operation defined by this QuantumCircuit.

        Examples
        --------
        We create a QuantumCircuit and turn it into an Operation which we append to
        another QuantumCircuit:

        >>> from qrisp import QuantumCircuit
        >>> qc_0 = QuantumCircuit(4)
        >>> qc_0.x(qc_0.qubits)
        >>> operation = qc_0.to_op(name="converted_op")
        >>> qc_1 = QuantumCircuit(4)
        >>> qc_1.append(operation, qc_1.qubits)
        >>> print(qc_1)

        .. code-block:: none

                    ┌───────────────┐
            qb_107: ┤0              ├
                    │               │
            qb_108: ┤1              ├
                    │  converted_op │
            qb_109: ┤2              ├
                    │               │
            qb_110: ┤3              ├
                    └───────────────┘

        """
        if name is None:
            name = "circuit" + str(int(TO_GATE_COUNTER[0]))[:7].zfill(7)
            TO_GATE_COUNTER[0] += 1

        definition = self.copy()

        definition.data = [instr for instr in definition.data if instr.op.name not in ["qb_alloc", "qb_dealloc"]]

        return Operation(
            name=name,
            num_qubits=len(self.qubits),
            num_clbits=len(self.clbits),
            definition=definition,
            params=None,
        )

    # Wrapper to increase Qiskit compatibility
    def to_gate(self, name: str | None = None) -> Operation:
        """Similar to :meth:`to_op <qrisp.QuantumCircuit.to_op>` but raises an exception
        if self contains classical bits (like the
        `Qiskit equivalent
        <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.to_gate.html>`_).

        Parameters
        ----------
        name : str, optional
            A name for the resulting gate. The default is None.

        Raises
        ------
        ValueError
            Tried to turn a circuit including classical bits into unitary gate

        Returns
        -------
        Operation
            The QuantumCircuit turned into an :ref:`Operation` instance.

        Examples
        --------
        We create a QuantumCircuit and turn it into an Operation which we append to
        another QuantumCircuit:

        >>> from qrisp import QuantumCircuit
        >>> qc_0 = QuantumCircuit(4)
        >>> qc_0.x(qc_0.qubits)
        >>> gate = qc_0.to_gate(name="converted_gate")
        >>> qc_1 = QuantumCircuit(4)
        >>> qc_1.append(gate, qc_1.qubits)
        >>> print(qc_1)

        .. code-block:: none

                    ┌─────────────────┐
            qb_167: ┤0                ├
                    │                 │
            qb_168: ┤1                ├
                    │  converted_gate │
            qb_169: ┤2                ├
                    │                 │
            qb_170: ┤3                ├
                    └─────────────────┘

        """
        if len(self.clbits) != 0:
            raise ValueError("Tried to turn a circuit including classical bits into unitary gate")

        return self.to_op(name)

    def extend(self, other: QuantumCircuit, translation_dic: dict | None = None) -> None:
        """Extends this QuantumCircuit in-place by appending instructions from another QuantumCircuit.

        Parameters
        ----------
        other : QuantumCircuit
            The QuantumCircuit whose instructions will be appended to this circuit.

        translation_dic : dict, optional
            The dictionary containing the information about which Qubits and Clbits
            should be plugged into each other. This dictionary should contain qubits of
            `other` as keys and qubits of `self` as values.

            If None (default), uses identity mapping by matching identifiers.
            This only works if identifiers match between circuits.


        Examples
        --------
        We create two QuantumCircuits and extend the first with reversed qubit order by
        the other:

        >>> from qrisp import QuantumCircuit
        >>> extension_qc = QuantumCircuit(4)
        >>> extension_qc.cx(0, 1)
        >>> extension_qc.cy(0, 2)
        >>> extension_qc.cz(0, 3)
        >>> print(extension_qc)

        .. code-block:: none

            qb_0: ──■────■───■─
                  ┌─┴─┐  │   │
            qb_1: ┤ X ├──┼───┼─
                  └───┘┌─┴─┐ │
            qb_2: ─────┤ Y ├─┼─
                       └───┘ │
            qb_3: ───────────■─

        >>> qc_to_extend = QuantumCircuit(4)
        >>> translation_dic = {extension_qc.qubits[i] : qc_to_extend.qubits[-1-i] for i in range(4)}
        >>> qc_to_extend.extend(extension_qc, translation_dic)
        >>> print(qc_to_extend)

        .. code-block:: none


            qb_4: ────────────■──
                       ┌───┐  │
            qb_5: ─────┤ Y ├──┼──
                  ┌───┐└─┬─┘  │
            qb_6: ┤ X ├──┼────┼──
                  └─┬─┘  │    │
            qb_7: ──■────■────■──

        """
        if translation_dic is None:
            translation_dic = {qb.identifier: qb for qb in other.qubits}
            translation_dic.update({cb.identifier: cb for cb in other.clbits})
        else:
            translation_dic = {
                key.identifier if isinstance(key, (Qubit, Clbit)) else key: value
                for key, value in translation_dic.items()
            }

        for instruction_other in other.data:
            qubits = [translation_dic[qb.identifier] for qb in instruction_other.qubits]
            clbits = [translation_dic[cb.identifier] for cb in instruction_other.clbits]
            self.append(instruction_other.op, qubits, clbits)

    def copy(self) -> QuantumCircuit:
        """Returns a copy of the given QuantumCircuit.

        Returns
        -------
        QuantumCircuit
            The copied QuantumCircuit.

        """
        res = QuantumCircuit()

        object.__setattr__(res, "data", list(self.data))
        object.__setattr__(res, "qubits", list(self.qubits))
        object.__setattr__(res, "clbits", list(self.clbits))

        try:
            res.abstract_params = set(self.abstract_params)
        except AttributeError:
            # abstract_params may be absent on legacy unpickled instances
            pass

        return res

    def clearcopy(self) -> QuantumCircuit:
        """Returns a copy of the given QuantumCircuit but without any data
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

    # TODO write qiskit independent printer
    def __str__(self) -> str:

        # NOTE: This is here to avoid circular imports
        from qrisp.interface import convert_to_qiskit

        try:
            res_str = str(
                circuit_drawer(
                    convert_to_qiskit(self, transpile=False),
                    output="text",
                    cregbundle=False,
                )
            )
        except AttributeError as exc:
            raise RuntimeError("Tried to print QuantumSession with uncompiled QuantumEnvironments") from exc

        return res_str

    def compare_unitary(self, other: QuantumCircuit, precision: int = 4, ignore_gphase: bool = False) -> bool:
        """Compares the unitaries of two QuantumCircuits. This can be used to check if a
        QuantumCircuit transformation is valid.

        Parameters
        ----------
        other : QuantumCircuit
            The QuantumCircuit to compare to.

        precision : int, optional
            The precision of the comparison. This function will return True if the norm
            of the difference of the unitaries is below ``10**(-precision)``.
            The default is 4.

        ignore_gphase: bool, optional
            If set to True, this method returns True if the unitaries only differ in a
            global phase.

        Returns
        -------
        bool
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
            # Normalize by the phase of the largest amplitude element
            arg_max = np.argmax(np.abs(unitary_self.flatten()))
            phase_correction = unitary_other.flatten()[arg_max] / unitary_self.flatten()[arg_max]
            unitary_self = unitary_self * phase_correction

        return bool(norm(unitary_self - unitary_other) < 10**-precision)

    def inverse(self) -> QuantumCircuit:
        """Generates the inverse of this QuantumCircuit by applying the inverse gates
        in reversed order.

        Returns
        -------
        inverted_circuit : QuantumCircuit
            The inverted QuantumCircuit.

        Examples
        --------
        Daggering a QuantumCircuit reverses the order and daggers each operation:

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

    def get_unitary(self, decimals: int | None = None) -> np.ndarray:
        """Return the unitary matrix of this QuantumCircuit as a NumPy array.

        Works with both numeric and abstract (SymPy) parameters. When the
        circuit contains symbolic parameters, the returned array has
        ``dtype=object`` with SymPy expressions as entries.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. When not provided, full
            precision is returned. For symbolic arrays, floating-point
            coefficients inside each expression are rounded. Values within
            ``10**(-decimals)`` of 1 are snapped to exactly 1 to suppress
            floating-point noise.

        Returns
        -------
        numpy.ndarray
            The unitary matrix. ``dtype`` is ``complex64`` for numeric
            circuits and ``object`` for symbolic ones.


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

        We now synthesize the exact same QuantumCircuit, but this time ``phi`` is a SymPy
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
        # NOTE: This is here to avoid circular imports
        from qrisp.simulator import calc_circuit_unitary

        res = calc_circuit_unitary(self, res_type="numpy")
        if not isinstance(res, np.ndarray):
            raise TypeError(f"calc_circuit_unitary must return a numpy array, got {type(res).__name__}")

        if decimals is None:
            return res

        if res.dtype != np.dtype("O"):
            return np.round(res, decimals)

        raveled = res.ravel()
        snap_threshold = 10 ** (-decimals)

        for i, entry in enumerate(raveled):
            expression = sympy.simplify(entry)
            for leaf in sympy.preorder_traversal(expression):
                if isinstance(leaf, sympy.Float):
                    if abs(float(leaf) - 1) < snap_threshold:
                        expression = expression.subs(leaf, 1)
                    else:
                        expression = expression.subs(leaf, round(leaf, decimals))
            raveled[i] = expression

        return res

    def get_depth_dic(self) -> dict[Qubit, int]:
        """Returns the depth of each qubit in this QuantumCircuit.

        The circuit is transpiled before the depth is evaluated, so that composite
        gates are fully decomposed into primitive operations. The depth of a qubit
        is the length of the longest sequential chain of operations acting on it,
        where every operation contributes a depth of 1.

        Returns
        -------
        dict[Qubit, int]
            A dictionary mapping each :ref:`Qubit` to its depth.

        Examples
        --------
        We create a QuantumCircuit and inspect the per-qubit depth:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(3)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.x(1)
        >>> qc.get_depth_dic()
        {Qubit(qb_0): 2, Qubit(qb_1): 3, Qubit(qb_2): 0}

        ``qb_0`` has depth 2 (H followed by CX), ``qb_1`` has depth 3 (CX followed
        by X), and ``qb_2`` is idle so its depth is 0.

        See Also
        --------
        QuantumCircuit.depth : Returns the overall circuit depth
            (i.e. the maximum value in this dictionary).

        """
        return get_depth_dic(self)

    def cnot_count(self) -> int:
        """Returns the number of two-qubit Pauli-axis controlled gates (CX, CY, CZ) in
        this QuantumCircuit.

        The circuit is fully transpiled before counting, so that any composite gate
        containing CX/CY/CZ gates is decomposed first.

        Returns
        -------
        int
            The total number of CX, CY, and CZ gates after transpilation.

        Examples
        --------
        We build a small circuit and count its two-qubit Pauli controlled gates:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(3)
        >>> qc.cx(0, 1)
        >>> qc.h(1)
        >>> qc.cz(1, 2)
        >>> qc.cnot_count()
        2

        The H gate is a single-qubit gate and is not counted; the CX and CZ each
        contribute 1, giving a total of 2.

        See Also
        --------
        QuantumCircuit.count_ops : Returns a full breakdown of every gate type
            in the circuit.

        """
        return cnot_count(self)

    def transpile(self, transpilation_level: int | float = np.inf, **qiskit_kwargs) -> QuantumCircuit:
        """Transpiles the QuantumCircuit in the sense that there are no longer any
        synthesized gate objects. Furthermore, we can call the `Qiskit transpiler
        <https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html>`_
        by supplying keyword arguments.

        The Qiskit transpiler is not called, if no keyword arguments are given.

        Parameters
        ----------
        transpilation_level : int, optional
            The level of transpilation. If set to 0, no transpilation is performed.
            If set to 1, only the top-level gates are transpiled, and so on.
            The default is np.inf, which means that all gates are transpiled.


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
        Phase Estimation circuit also contains a ``QFT_dg`` gate.

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

        To transpile just ``QFT_dg`` in the compiled QuantumCircuit,

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
        # NOTE: This is here to avoid circular imports
        from qrisp.circuit import transpile

        return transpile(self, transpilation_level, **qiskit_kwargs)

    def count_ops(self) -> dict[str, int]:
        """Counts the amount of operations of each kind. Note that operations are
        identified by their name.

        Returns
        -------
        count_dic : dict[str, int]
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
            op_name = ins.op.name
            if op_name not in ["qb_alloc", "qb_dealloc"]:
                count_dic[op_name] = count_dic.get(op_name, 0) + 1

        return count_dic

    def control(self, amount: int) -> PTControlledOperation | ControlledOperation:
        """Returns a controlled version of this QuantumCircuit.

        Parameters
        ----------
        amount : int
            The amount of control qubits.

        Returns
        -------
        PTControlledOperation or ControlledOperation
            The controlled version of this QuantumCircuit.

        """
        return self.to_gate().control(amount)

    def compose(
        self,
        other: QuantumCircuit,
        qubits: Sequence[QubitLike] | None = None,
        clbits: Sequence[ClbitLike] | None = None,
        inplace: bool = True,
    ) -> QuantumCircuit | None:
        """Composes this QuantumCircuit with another QuantumCircuit by appending the other to self.

        Parameters
        ----------
        other : QuantumCircuit
            The QuantumCircuit to be appended to self.

        qubits : Sequence[QubitLike] | None, optional
            The qubits to be used for the composition.
            If None, the qubits of self and other will be matched by their identifiers.
            The default is None.

        clbits : Sequence[ClbitLike] | None, optional
            The classical bits to be used for the composition.
            If None, the clbits of self and other will be matched by their identifiers.
            The default is None.

        inplace : bool, optional
            If True, the composition is performed in-place and self is modified.
            If False, a new QuantumCircuit is returned and self is not modified.
            The default is True.

        Returns
        -------
        QuantumCircuit | None
            The composed QuantumCircuit. Only returned if inplace is False.

        """
        if inplace:
            self.append(other.to_gate(), qubits, clbits)
            return None

        qc = self.copy()
        qc.append(other.to_gate(), qubits, clbits)
        return qc

    def bind_parameters(self, subs_dic: dict) -> QuantumCircuit:
        """Returns a QuantumCircuit where the abstract parameters in ``subs_dic`` are bound
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

        subs_circ.abstract_params = set()
        return subs_circ

    def to_latex(self, **kwargs) -> str:
        """Deploys the Qiskit circuit drawer to generate LaTeX output.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to Qiskit's
            `circuit_drawer
            <https://docs.quantum.ibm.com/api/qiskit/qiskit.visualization.circuit_drawer>`_
            function.

        Returns
        -------
        str
            The LaTeX source code for the circuit diagram.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import convert_to_qiskit

        qiskit_qc = convert_to_qiskit(self, transpile=False)

        return cast(str, circuit_drawer(qiskit_qc, output="latex_source", **kwargs))

    def to_qasm2(
        self,
        formatted: bool = False,
        filename: str | None = None,
        encoding: str | None = None,
    ) -> str:
        """Returns the `OpenQASM 2.0 <https://en.wikipedia.org/wiki/OpenQASM>`_ string
        of this QuantumCircuit.

        If the circuit contains gates that cannot be represented in OpenQASM 2.0, it
        is first transpiled to a universal set of primitive gates before exporting.

        Parameters
        ----------
        formatted : bool, optional
            Return formatted Qasm string. The default is False.

        filename : str, optional
            If provided, the QASM string is also written to this file path.
            The default is None.

        encoding : str, optional
            The file encoding to use when writing to ``filename``. Defaults to the
            system’s preferred encoding. Only relevant when ``filename`` is given.

        Returns
        -------
        str
            The OpenQASM 2.0 string.

        Examples
        --------
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> print(qc.to_qasm2())
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg qb_77[1];
        qreg qb_78[1];
        h qb_77[0];
        cx qb_77[0],qb_78[0];

        """
        qiskit_qc = self.to_qiskit()
        try:
            return qiskit_qc.qasm(formatted, filename, encoding)
        except AttributeError:
            try:
                return dumps_qasm2(qiskit_qc)
            except (QASM2ExportError, TypeError):
                transpiled_qiskit_qc = qiskit_transpile(
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
                return dumps_qasm2(transpiled_qiskit_qc)

    def to_qasm3(
        self,
        formatted: bool = False,
        filename: str | None = None,
        encoding: str | None = None,
    ) -> str:
        """Returns the `OpenQASM 3.0 <https://en.wikipedia.org/wiki/OpenQASM>`_ string
        of this QuantumCircuit.

        Parameters
        ----------
        formatted : bool, optional
            Accepted for backward compatibility with the previous Qrisp API but
            has no effect. The default is False.

        filename : str, optional
            If provided, the QASM string is also written to this file path.
            The default is None.

        encoding : str, optional
            The file encoding to use when writing to ``filename``. Defaults to the
            system’s preferred encoding. Only relevant when ``filename`` is given.

        Returns
        -------
        str
            The OpenQASM 3.0 string.

        Examples
        --------
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> print(qc.to_qasm3())
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] qb_75;
        qubit[1] qb_76;
        h qb_75[0];
        cx qb_75[0], qb_76[0];

        """
        _ = formatted  # accepted for backward compatibility, has no effect
        qiskit_qc = self.to_qiskit()
        qasm_str = dumps_qasm3(qiskit_qc)

        if filename is not None:
            with open(filename, "w", encoding=encoding) as f:
                f.write(qasm_str)

        return qasm_str

    def qasm(self, **kwargs) -> str:
        """Alias for :meth:`to_qasm2`."""
        return self.to_qasm2(**kwargs)

    def depth(
        self,
        depth_indicator: Callable[[Operation], int] = lambda _: 1,
        transpile: bool = True,
    ) -> int:
        """Returns the depth of the QuantumCircuit.

        .. note::
            The depth of a circuit that has not been transpiled may have very
            little correlation with its actual runtime, since composite gates
            are counted as a single layer.

        Parameters
        ----------
        depth_indicator : Callable[[Operation], int], optional
            A function that receives an :ref:`Operation` instance and returns
            the time or logical depth that operation takes. By default every
            operation contributes a depth of 1.

        transpile : bool, optional
            If ``True``, the circuit is transpiled before the depth is
            calculated so that composite gates are fully decomposed into
            primitive operations. The default is True.

        Returns
        -------
        int
            The depth of the QuantumCircuit.

        Examples
        --------
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(3)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.cx(1, 2)
        >>> qc.depth()
        3

        """
        if len(self.data) == 0:
            return 0

        depth_dic = get_depth_dic(self, transpile_qc=transpile, depth_indicator=depth_indicator)

        return int(max(depth_dic.values()))

    def t_depth(self, epsilon: float | None = None) -> int:
        r"""Estimates the T-depth of this QuantumCircuit.

        T-depth is an important metric for fault-tolerant quantum computing,
        because T gates are expected to be the bottleneck in fault-tolerant
        architectures.

        According to `this paper <https://arxiv.org/abs/1403.2975>`_, the
        synthesis of an $RZ(\phi)$ up to precision $\epsilon$ requires
        $3\log_2(\frac{1}{\epsilon})$ T-gates.

        Based on this formula, this method performs a conservative estimate of
        the T-depth of this circuit.

        Parameters
        ----------
        epsilon : float, optional
            The precision up to which parametrized gates should be
            approximated. If not given, Qrisp will determine the precision
            from the parameter with the highest required precision. See the
            examples below for details.

        Returns
        -------
        int
            The estimated T-depth.

        Examples
        --------
        We create a QuantumCircuit and evaluate the T-depth:

        >>> import numpy as np
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.t(0)
        >>> qc.cx(0, 1)
        >>> qc.rx(2*np.pi*3/2**4, 1)
        >>> qc.t_depth(epsilon=2**-5)
        16

        In this example we execute a T-gate on qubit 0 (T-depth: 1), followed
        by a CNOT (T-depth: 0), and finally an RX gate on qubit 1.

        The RX gate can be decomposed as

        .. math::

            RX(\phi) = H \cdot RZ(\phi) \cdot H

        so its T-depth equals that of the parametrized RZ. To determine the
        T-depth of $RZ(\phi)$ with precision $\epsilon = 2^{-5}$ we use the
        formula above:

        .. math::

            \text{T-depth}(RZ(\phi),\; \epsilon = 2^{-5})
            = 3 \log_2(2^5) = 15

        Adding the 1 T-depth contribution from the T-gate gives a total of
        16.

        **Automatic precision determination**

        When ``epsilon`` is not provided, Qrisp assumes every parameter has
        the form

        .. math::

            \phi = 2\pi \frac{m}{2^k}

        where $m$ is an integer. It determines the maximum $k$ across all
        parameters and sets $\epsilon = 2^{-(k_{\max}+3)}$, where the extra
        $+3$ is a conservative buffer that slightly overestimates the required
        precision.

        >>> qc.t_depth()
        22

        In this circuit $k_{\max} = 4$, so $\epsilon = 2^{-7}$, giving a
        T-depth of 22.

        """
        if epsilon is None:
            transpiled_qc = self.transpile()

            max_circuit_prec = 15
            for instr in transpiled_qc.data:
                op = instr.op

                for par in op.params:
                    # Normalize parameter to range [0, 2π) and convert to fixed-point representation
                    normalized_par = (par % (2 * np.pi)) / (2 * np.pi)
                    fixed_point_par = int(np.round(normalized_par * 2**15))

                    # Find the position of the least significant bit
                    for idx in range(max_circuit_prec):
                        if fixed_point_par % (2**idx):
                            max_circuit_prec = idx
                            break

            # Convert precision index to actual precision value
            max_circuit_prec = 16 - max_circuit_prec

            # Set epsilon based on the maximum precision across all parameters
            epsilon = 2 ** (-max_circuit_prec - 3)

        return self.depth(depth_indicator=lambda x: t_depth_indicator(x, epsilon))

    def cnot_depth(self) -> int:
        """Returns the CNOT depth of this QuantumCircuit.

        In NISQ-era devices, CNOT gates are the restricting bottleneck for
        quantum circuit execution. This method can be used as a gate-speed
        specifier for the :meth:`compile <qrisp.QuantumSession.compile>`
        method.

        Returns
        -------
        int
            The CNOT depth of this QuantumCircuit.

        Examples
        --------
        We create a QuantumCircuit and evaluate its CNOT depth:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(4)
        >>> qc.cx(0, 1)
        >>> qc.x(1)
        >>> qc.cx(1, 2)
        >>> qc.y(2)
        >>> qc.cx(2, 3)
        >>> qc.cx(1, 0)
        >>> print(qc)

        .. code-block:: none

                                 ┌───┐
            qb_0: ──■────────────┤ X ├─────
                  ┌─┴─┐┌───┐     └─┬─┘
            qb_1: ┤ X ├┤ X ├──■────■───────
                  └───┘└───┘┌─┴─┐┌───┐
            qb_2: ──────────┤ X ├┤ Y ├──■──
                            └───┘└───┘┌─┴─┐
            qb_3: ────────────────────┤ X ├
                                      └───┘

        >>> qc.cnot_depth()
        3

        """
        return self.depth(depth_indicator=cnot_depth_indicator)

    def num_qubits(self) -> int:
        """Returns the number of qubits in this QuantumCircuit.

        Returns
        -------
        int
            The number of qubits.

        Examples
        --------
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(5)
        >>> qc.num_qubits()
        5

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

    def _append_xla_fast_path(
        self,
        operation_or_instruction: Operation | Instruction,
        qubits: list,
        clbits: list,
    ) -> None:
        """Handle the XLA accelerated compilation fast path."""
        if isinstance(operation_or_instruction, Instruction):
            self.data.append(operation_or_instruction)
            return
        if self.xla_mode <= 1:
            if not isinstance(qubits, list):
                raise TypeError(
                    f"Operation {operation_or_instruction.name} was appended with "
                    f"{qubits} in accelerated compilation mode "
                    "(allowed is type List[Qubit])."
                )
            for qb in qubits:
                if not isinstance(qb, Qubit):
                    raise TypeError(
                        f"Operation {operation_or_instruction.name} was appended with "
                        f"{qubits} in accelerated compilation mode "
                        "(allowed is type List[Qubit])."
                    )
        self.data.append(Instruction(operation_or_instruction, qubits, clbits))

    def _apply_broadcast(
        self,
        operation: Operation,
        qubits: list,
        clbits: list,
        qb_argument_is_list: list,
        cb_argument_is_list: list,
    ) -> None:
        """Apply the operation once per broadcast index."""
        if qb_argument_is_list:
            arg_list_len = len(qubits[qb_argument_is_list[0]])
        else:
            arg_list_len = len(clbits[cb_argument_is_list[0]])

        for arg_list_index in qb_argument_is_list:
            if len(qubits[arg_list_index]) != arg_list_len:
                raise ValueError(f"Don't know how to combine appending arguments {qubits + clbits}")

        for arg_list_index in cb_argument_is_list:
            if len(clbits[arg_list_index]) != arg_list_len:
                raise ValueError(f"Don't know how to combine appending arguments {qubits + clbits}")

        for i in range(arg_list_len):
            qubit_constellation = [qubits[j][i] if j in qb_argument_is_list else qubits[j] for j in range(len(qubits))]
            clbit_constellation = [clbits[j][i] if j in cb_argument_is_list else clbits[j] for j in range(len(clbits))]
            QuantumCircuit.append(self, operation, qubit_constellation, clbit_constellation)

    def _resolve_qubits(self, qubits: list, operation: Operation) -> list:
        """Validate qubit count and membership; return the resolved qubit list."""
        if len(qubits) != operation.num_qubits:
            raise ValueError(
                f"Provided incorrect amount ({len(qubits)}) of qubits for operation "
                f"{operation.name} (requires {operation.num_qubits})"
            )

        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Duplicate qubit arguments in {qubits} for operation {operation.name}")

        # Comparing object identity first is fast; falling back to identifier comparison
        # handles qubits equal by name but not by identity (e.g. after unpickling).
        if not set(qubits).issubset(set(self.qubits)):
            op_identifiers = [qb.identifier for qb in qubits]
            qc_identifiers = [qb.identifier for qb in self.qubits]

            if not set(op_identifiers).issubset(qc_identifiers):
                raise ValueError(f"Instruction Qubits {set(qubits) - set(self.qubits)} not present in circuit")

            qubits = [self.qubits[qc_identifiers.index(op_id)] for op_id in op_identifiers]

        return qubits

    def append(
        self,
        operation_or_instruction: Operation | Instruction,
        qubits: Sequence[QubitLike] | None = None,
        clbits: Sequence[ClbitLike] | None = None,
    ):
        r"""Append an :class:`.Operation` or :class:`.Instruction` to this QuantumCircuit.

        Each qubit or classical-bit argument may be specified as a :class:`.Qubit` /
        :class:`.Clbit` object, an integer index into ``self.qubits`` / ``self.clbits``,
        or a (possibly nested) list thereof.  When a list argument contains *n* elements,
        the operation is broadcast and applied *n* times — once per element — with the
        remaining scalar arguments reused for every application.

        If an :class:`.Instruction` is given instead of an :class:`.Operation`, the
        *qubits* and *clbits* arguments are ignored; the instruction's own qubit and
        classical-bit lists are used directly.

        Parameters
        ----------
        operation_or_instruction : Operation or Instruction
            The operation or instruction to append.
        qubits : QubitLike, optional
            The qubit(s) on which to apply the operation. The default is ``[]``.
        clbits : ClbitLike, optional
            The classical bit(s) on which to apply the operation. The default is ``[]``.

        Returns
        -------
        None

        Examples
        --------
        We create a $H^{\otimes 4}$ gate and append it to every second qubit of another
        QuantumCircuit:

        >>> from qrisp import QuantumCircuit
        >>> multi_h_qc = QuantumCircuit(4)
        >>> multi_h_qc.h(range(4))
        >>> multi_h = multi_h_qc.to_gate(name="multi h")
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
        qubits = [] if qubits is None else qubits
        clbits = [] if clbits is None else clbits

        if self.xla_mode > 0:
            self._append_xla_fast_path(operation_or_instruction, qubits, clbits)
            return

        if isinstance(operation_or_instruction, Instruction):
            instruction = operation_or_instruction
            self.append(instruction.op, instruction.qubits, instruction.clbits)
            return

        if not isinstance(operation_or_instruction, Operation):
            raise TypeError(
                f"Tried to append object of type {type(operation_or_instruction)} "
                "which is neither Instruction nor Operation"
            )

        operation = operation_or_instruction
        # Convert arguments (possibly integers) to lists of Qubit / Clbit objects.
        # The list structure is preserved:
        #   [[0, 1], 2]  →  [[qubit_0, qubit_1], qubit_2]
        # A scalar input is wrapped in a list:
        #   qubit_0  →  [qubit_0]
        qubits = convert_to_qb_list(qubits, circuit=self)
        clbits = convert_to_cb_list(clbits, circuit=self)

        # Identify which positional arguments are lists (broadcast dimensions).
        # For user convenience, a single append() call can apply the same operation
        # n times by passing a list of n qubits for one or more arguments.
        # Every list argument must have the same length n; scalar arguments are
        # reused for all n applications.
        qb_argument_is_list = [i for i, qb in enumerate(qubits) if isinstance(qb, list)]
        cb_argument_is_list = [i for i, cb in enumerate(clbits) if isinstance(cb, list)]

        if qb_argument_is_list or cb_argument_is_list:
            self._apply_broadcast(operation, qubits, clbits, qb_argument_is_list, cb_argument_is_list)
            return

        qubits = self._resolve_qubits(qubits, operation)

        if len(clbits) != operation.num_clbits:
            raise ValueError(
                f"Provided incorrect amount ({len(clbits)}) of clbits for operation "
                f"{operation.name} (requires {operation.num_clbits})"
            )

        if len({cb.identifier for cb in clbits}) != len(clbits):
            raise ValueError("Duplicate clbit arguments")

        if not {cb.identifier for cb in clbits}.issubset({cb.identifier for cb in self.clbits}):
            raise ValueError("Instruction Clbits not present in circuit")

        try:
            self.abstract_params.update(operation.abstract_params)
        except AttributeError:
            pass

        _check_qubit_locks(qubits, operation)

        self.data.append(Instruction(operation, qubits, clbits))

    def run(
        self,
        shots: int | None = None,
        backend: BackendLike | None = None,
    ) -> MeasurementResult:
        """Executes a QuantumCircuit on a backend and returns the measurement results.

        Parameters
        ----------
        shots : int or None, optional
            Number of shots to sample. When set to ``None`` (default), the behaviour
            depends on the backend. For simulators, the exact probability distribution
            is returned. For real quantum devices, the number of shots is determined
            by the backend's default settings.

        backend : BackendLike, optional
            The backend on which to evaluate the QuantumCircuit. When not provided,
            Qrisp's built-in statevector simulator is used.

        Returns
        -------
        MeasurementResult
            A lazy mapping from measurement outcome strings to integer counts
            (when *shots* is given) or to exact float probabilities (when
            *shots* is ``None`` and the backend is a simulator).

        Examples
        --------
        In this example, we prepare a 3-qubit GHZ state and retrieve the exact
        probability distribution by omitting *shots*:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(3)
        >>> qc.h(0)
        >>> qc.cx(0, [1, 2])
        >>> qc.measure([0, 1, 2])
        >>> qc.run()
        {'000': 0.5, '111': 0.5}

        We can also pass an explicit shot count to obtain sampled integer counts instead.
        In this example we prepare a 2-qubit state where we expect to get
        the outcome ``11`` in all shots:

        >>> qc_det = QuantumCircuit(2)
        >>> qc_det.x([0, 1])
        >>> qc_det.measure([0, 1])
        >>> qc_det.run(shots=100)
        {'11': 100}

        """
        if backend is None:
            # NOTE: This is here to avoid circular imports
            from qrisp.default_backend import def_backend

            backend = def_backend

        return backend.run(self, shots)

    def statevector_array(self) -> np.ndarray:
        r"""Simulates the circuit and returns its statevector as a NumPy array of
        complex amplitudes.

        .. note::

            The returned array uses **big-endian index ordering**. The array index
            ``i`` maps to qubit values as

            .. math::

                i = \sum_{k=0}^{n-1} q_k \, 2^{\,n-1-k},

            so :math:`q_0` is the most significant qubit. For two qubits this yields:

            - ``i = 0``  → :math:`|q_0=0, q_1=0\rangle`
            - ``i = 1``  → :math:`|q_0=0, q_1=1\rangle`
            - ``i = 2``  → :math:`|q_0=1, q_1=0\rangle`
            - ``i = 3``  → :math:`|q_0=1, q_1=1\rangle`

            This differs from Qrisp’s internal little-endian convention (only the
            index-to-basis mapping changes).

        Returns
        -------
        numpy.ndarray
            A 1-D ``complex64`` array of statevector amplitudes in big-endian
            order. The array has length :math:`2^n` where *n* is the number of
            qubits.

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

        In this example, we create a :ref:`QuantumFloat` and prepare the normalized state
        $\sum_{i=0}^3 \tilde b_i\ket{i}$ for $\tilde b=(0,1,2,3)/\sqrt{14}$.

        >>> import numpy as np
        >>> from qrisp import QuantumFloat
        >>> b = np.array([0, 1, 2, 3], dtype=float)
        >>> b /= np.linalg.norm(b)
        >>> qf = QuantumFloat(2)
        >>> qf.init_state(b)
        >>> sv_array = qf.qs.statevector_array()
        >>> print(f"b[1]: {b[1]:.6f} -> {sv_array[2]:.6f}")
        b[1]: 0.267261 -> 0.267261-0.000000j
        >>> print(f"b[2]: {b[2]:.6f} -> {sv_array[1]:.6f}")
        b[2]: 0.534522 -> 0.534522-0.000000j

        Here ``sv_array[2]`` corresponds to :math:`\ket{q_0=1, q_1=0}` and
        ``sv_array[1]`` to :math:`\ket{q_0=0, q_1=1}`.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.simulator import statevector_sim

        return statevector_sim(self)

    def __hash__(self) -> int:
        """Compute a structural hash of this QuantumCircuit.

        Two circuits are intended to hash identically when they apply the
        same sequence of operations to the same qubit *positions*, regardless
        of qubit names or identifiers.  The hash captures four aspects:

        Qubit count
            Circuits with a different number of qubits are scaled by
            different factors (``n²``), making same-length collisions far
            less likely.

        Instruction order
            Each instruction's contribution is multiplied by ``(i + 1)²``
            (1-based squared position), so reordering instructions changes
            the total.

        Qubit positions
            Each instruction records the circuit-global index of every qubit
            it acts on (i.e. the 0-based position in ``self.qubits``), not
            the qubit's name. Only the *positional* slot matters, not the
            identity of the :class:`Qubit` object.

        Gate identity and parameters
            Composite gates (those with a sub-circuit ``definition``) are
            identified by recursively hashing their definition. Primitive
            gates are identified by their name string. Each gate parameter
            is hashed together with the instruction's position so that the
            same angle at two different circuit positions produces a
            different contribution.

        Returns
        -------
        int
            The hash value.

        """
        n = len(self.qubits)
        total = 0
        qubit_index_map = {qb: idx for idx, qb in enumerate(self.qubits)}

        for i, instr in enumerate(self.data):
            qubit_indices = tuple(qubit_index_map[qb] for qb in instr.qubits)
            index_hash = hash(qubit_indices)

            # Couple each parameter value to the instruction's position so
            # that the same angle at different positions is distinguished.
            param_hash = hash(tuple(hash((p, i)) for p in instr.op.params))

            # Composite gates are identified by the hash of their
            # sub-circuit, while primitive gates are identified by their name.
            op_hash = hash(instr.op.definition) if instr.op.definition else hash(instr.op.name)

            # Weight by (i+1)² so that swapping two instructions changes
            # the total, making the hash order-sensitive.
            total += hash((index_hash, param_hash, op_hash)) * (i + 1) ** 2

        # Scale by n² so that circuits with different qubit counts are
        # unlikely to collide even when their instruction sequences match.
        return hash(total * n**2)

    @classmethod
    def from_qasm_str(cls, qasm_string: str) -> QuantumCircuit:
        """Loads a QuantumCircuit from a QASM String.

        Parameters
        ----------
        qasm_string : str
            A string obeying the syntax of the OpenQASM specification.

        Returns
        -------
        QuantumCircuit
            The corresponding QuantumCircuit.

        """
        qiskit_qc = QiskitQuantumCircuit().from_qasm_str(qasm_string)
        return cls.from_qiskit(qiskit_qc)

    @classmethod
    def from_qasm_file(cls, filename: str) -> QuantumCircuit:
        """Loads a QuantumCircuit from a QASM file.

        Parameters
        ----------
        filename : str
            A string pointing to a file obeying the OpenQASM syntax.

        Returns
        -------
        QuantumCircuit
            The corresponding QuantumCircuit.

        """
        qiskit_qc = QiskitQuantumCircuit().from_qasm_file(filename)
        return cls.from_qiskit(qiskit_qc)

    @classmethod
    def from_qiskit(cls, qiskit_qc):
        """Class method to create QuantumCircuits from Qiskit QuantumCircuits.

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

        >>> from qrisp import QuantumCircuit
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
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import convert_from_qiskit

        return convert_from_qiskit(qiskit_qc)

    def to_qiskit(self) -> QiskitQuantumCircuit:
        """Method to convert the given QuantumCircuit to a Qiskit QuantumCircuit.

        Returns
        -------
        Qiskit QuantumCircuit
            The converted circuit.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import convert_to_qiskit

        return convert_to_qiskit(self, transpile=False)

    def to_pennylane(self):
        """Method to convert the given QuantumCircuit to a
        `Pennylane <https://pennylane.ai/>`_ Circuit.

        Returns
        -------
        function
            A function representing a pennylane QuantumCircuit.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import qml_converter

        return qml_converter(self)

    def to_stim(
        self,
        return_measurement_map: bool = False,
        return_detector_map: bool = False,
        return_observable_map: bool = False,
    ):
        """Method to convert the given QuantumCircuit to a
        `Stim <https://github.com/quantumlib/Stim/>`_ Circuit.

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
            (Optional) A dictionary mapping :class:`~qrisp.jasp.ParityHandle`
            objects to Stim detector indices.
            ParityHandle objects are compared by their content, so handles returned by
            :meth:`parity` can be used directly as keys.
        observable_map : dict
            (Optional) A dictionary mapping :class:`~qrisp.jasp.ParityHandle`
            objects to Stim observable indices.
            ParityHandle objects are compared by their content, so handles returned by
            :meth:`parity` can be used directly as keys.

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
        >>> samples
        array([ True,  True,  True,  True,  True])

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import qrisp_to_stim

        return qrisp_to_stim(self, return_measurement_map, return_detector_map, return_observable_map)

    def to_pytket(self):
        """Method to convert the given QuantumCircuit to a
        `PyTket <https://cqcl.github.io/tket/pytket/api/#>`_ Circuit.

        Returns
        -------
        pytket.Circuit
            The converted PyTket circuit.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import pytket_converter

        return pytket_converter(self)

    def to_cirq(self):
        """Method to convert the given QuantumCircuit to a Cirq Circuit.

        Returns
        -------
        cirq.Circuit
            A Cirq Circuit equivalent to the Qrisp QuantumCircuit.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import convert_to_cirq

        return convert_to_cirq(self)

    @classmethod
    def from_cirq(cls, cirq_circuit):
        """Class method to create QuantumCircuits from Cirq Circuits.

        Parameters
        ----------
        cirq_circuit : cirq.Circuit
            The Cirq Circuit to convert.

        Returns
        -------
        QuantumCircuit
            The converted QuantumCircuit.

        Examples
        --------
        We construct a Bell-state circuit in Cirq:

        >>> import cirq
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> cirq_circ = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
        >>> from qrisp import QuantumCircuit
        >>> qrisp_qc = QuantumCircuit.from_cirq(cirq_circ)
        >>> print(qrisp_qc)

        .. code-block:: none

             qb_0: ──H──■──
                       ┌─┴─┐
             qb_1: ────┤ X ├
                       └───┘

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.interface import convert_from_cirq

        return convert_from_cirq(cirq_circuit)

    def to_pdag(self, remove_artificials: bool = False):
        """Method to convert the given QuantumCircuit to a PermeabilityGraph.

        Parameters
        ----------
        remove_artificials : bool, optional
            Whether to remove artificial nodes from the PermeabilityGraph. The default is False.

        Returns
        -------
        PermeabilityGraph
            The resulting PermeabilityGraph.

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.permeability import PermeabilityGraph

        return PermeabilityGraph(self, remove_artificials=remove_artificials)

    def measure(
        self,
        qubits: QubitLike,
        clbits: ClbitLike | None = None,
    ) -> None:
        """Append a measurement instruction to the circuit.

        For each qubit in *qubits* a :class:`~qrisp.circuit.Measurement`
        operation is added that stores the binary outcome in the corresponding
        entry of *clbits*.  When *clbits* is omitted the required classical
        bits are allocated automatically.

        Parameters
        ----------
        qubits : QubitLike
            The qubit(s) to measure.  A single :ref:`Qubit` object or
            integer index measures one qubit; any sequence (``list``,
            ``tuple``, ``range``, :ref:`QuantumVariable`, …) measures each
            element independently.

        clbits : ClbitLike or None, optional
            The classical bit(s) that receive the measurement results.  When
            ``None`` (default), fresh classical bits are created automatically
            (one per qubit being measured).

        Examples
        --------
        In this example, we measure a single qubit.
        One classical bit is allocated automatically:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(1)
        >>> qc.x(0)
        >>> qc.measure(0)
        >>> len(qc.clbits)
        1

        Now we measure several qubits at once.
        One classical bit is created per qubit:

        >>> qc = QuantumCircuit(3)
        >>> qc.measure([0, 1, 2])
        >>> len(qc.clbits)
        3

        Finally, we provide explicit classical bits
        to control where results are stored:

        >>> qc = QuantumCircuit(2)
        >>> cb0, cb1 = qc.add_clbit(), qc.add_clbit()
        >>> qc.measure(0, cb0)
        >>> qc.measure(1, cb1)
        >>> qc.clbits == [cb0, cb1]
        True

        """
        if clbits is None:
            if isinstance(qubits, (Qubit, int)):
                # For single-qubit measurement,
                # we allocate exactly one classical bit.
                clbits = self.add_clbit()
            else:
                # For multi-qubit measurement,
                # we allocate one classical bit per qubit.
                clbits = [self.add_clbit() for _ in qubits]

        self.append(ops.Measurement(), [qubits], [clbits])

    # TODO: Extend to accept integer indices for clbits as well
    def parity(
        self,
        clbits: Clbit | Sequence[Clbit],
        expectation: int = 0,
        observable: bool = False,
    ) -> ParityHandle:
        """Append a parity (XOR) check over classical bits to the circuit.

        Computes ``p = b_0 ⊕ b_1 ⊕ … ⊕ b_{n-1} ⊕ expectation``, so
        ``p = 0`` whenever the measured parity matches the expected value.
        This is useful for quantum error correction and when interfacing with Stim.
        When the circuit is converted via :meth:`to_stim`, a parity
        instruction becomes either a ``DETECTOR`` (if ``observable=False``)
        or an ``OBSERVABLE_INCLUDE`` (if ``observable=True``) instruction.

        Parameters
        ----------
        clbits : Clbit or Sequence[Clbit]
            The classical bit(s) to compute parity over.  A single
            :class:`Clbit` measures the parity of one bit; any sequence
            (``list``, ``tuple``, ``range``, …) computes the XOR of all
            elements.

        expectation : int, optional
            The expected parity value (``0`` or ``1``), XORed into the result
            so that ``p = 0`` when the measured parity equals the expectation.
            Default is ``0``.

        observable : bool, optional
            If ``True``, this parity is treated as a Stim observable rather
            than a detector.  Default is ``False``.

        Returns
        -------
        :class:`~qrisp.jasp.ParityHandle`
            A handle representing the parity result.  Use it as a key to look
            up detector/observable indices in the maps returned by
            :meth:`to_stim`.

        Examples
        --------
        Create a simple detector checking that two qubits have even parity:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2, 2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.measure([0, 1], [0, 1])
        >>> handle = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0)
        >>> print(handle)
        ParityHandle(Clbit(cb_2), Clbit(cb_3))

        Convert to Stim and check the detector:

        >>> stim_circuit, meas_map, det_map = qc.to_stim(
        ...     return_measurement_map=True,
        ...     return_detector_map=True
        ... )
        >>> det_map[handle]  # Get the Stim detector index
        0

        See Also
        --------
        :func:`qrisp.parity` : The gate function version for use in QuantumSessions
        :meth:`to_stim` : Convert to Stim circuit with detector/observable maps
        :class:`qrisp.jasp.ParityHandle` : Documentation of the ParityHandle class

        """
        # NOTE: This is here to avoid circular imports
        from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import (
            ParityHandle,
        )

        # NOTE: This is here to avoid circular imports
        from qrisp.jasp.primitives.parity_primitive import ParityOperation

        clbits = [clbits] if isinstance(clbits, Clbit) else clbits

        parity_op = ParityOperation(len(clbits), expectation=expectation, observable=observable)
        self.append(parity_op, clbits=clbits)

        return ParityHandle(self.data[-1])

    def cx(self, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct a CX-gate.

        Parameters
        ----------
        qubits_0 : QubitLike
            The Qubit to control on.
        qubits_1 : QubitLike
            The target Qubit.

        """
        self.append(ops.CXGate(), [qubits_0, qubits_1])

    def cy(self, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct a CY-gate.

        Parameters
        ----------
        qubits_0 : QubitLike
            The Qubit to control on.
        qubits_1 : QubitLike
            The target Qubit.

        """
        self.append(ops.CYGate(), [qubits_0, qubits_1])

    def cz(self, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct a CZ-gate.

        Parameters
        ----------
        qubits_0 : QubitLike
            The Qubit to control on.
        qubits_1 : QubitLike
            The target Qubit.

        """
        self.append(ops.CZGate(), [qubits_0, qubits_1])

    def h(self, qubits: QubitLike):
        """Instruct a Hadamard-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.HGate(), [qubits])

    def x(self, qubits: QubitLike):
        """Instruct a Pauli-X-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.XGate(), [qubits])

    def y(self, qubits: QubitLike):
        """Instruct a Pauli-Y-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.YGate(), [qubits])

    def z(self, qubits: QubitLike):
        """Instruct a Pauli-Z-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.ZGate(), [qubits])

    def rx(self, phi: FloatLike, qubits: QubitLike):
        """Instruct a parametrized RX-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.RXGate(phi), [qubits])

    def ry(self, phi: FloatLike, qubits: QubitLike):
        """Instruct a parametrized RY-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.RYGate(phi), [qubits])

    def rz(self, phi: FloatLike, qubits: QubitLike):
        """Instruct a parametrized RZ-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.RZGate(phi), [qubits])

    def cp(self, phi: FloatLike, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct a controlled phase-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits_0 : QubitLike
            The Qubit to apply the gate on.

        qubits_1 : QubitLike
            The other Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.CPGate(phi), [qubits_0, qubits_1])

    def p(self, phi: FloatLike, qubits: QubitLike):
        """Instruct a phase-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.PGate(phi), [qubits])

    def rxx(self, phi: FloatLike, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct an RXX-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits_0 : QubitLike
            The Qubit to apply the gate on.

        qubits_1 : QubitLike
            The other Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.RXXGate(phi), [qubits_0, qubits_1])

    def rzz(self, phi: FloatLike, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct an RZZ-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits_0 : QubitLike
            The Qubit to apply the gate on.

        qubits_1 : QubitLike
            The other Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.RZZGate(phi), [qubits_0, qubits_1])

    def xxyy(self, phi: FloatLike, beta: FloatLike, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct an XXYY-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.
        beta : FloatLike
            The other angle parameter
        qubits_0 : QubitLike
            The Qubit to apply the gate on.
        qubits_1 : QubitLike
            The other Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.XXYYGate(phi, beta), [qubits_0, qubits_1])

    def swap(self, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct a SWAP-gate.

        Parameters
        ----------
        qubits_0 : QubitLike
            The qubit to swap.
        qubits_1 : QubitLike
            The other qubit to swap.

        """
        self.append(ops.SwapGate(), [qubits_0, qubits_1])

    def mcx(
        self,
        control_qubits: list[QubitLike],
        target_qubits: QubitLike,
        method="gray",
        ctrl_state=-1,
    ):
        """Instruct a multi-controlled X-gate.

        Parameters
        ----------
        control_qubits : list[QubitLike]
            The list of Qubits to control on.
        target_qubits : QubitLike
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
    
    def mcz(self, qubits, method="gray", ctrl_state=-1):
        """
        Instruct a multi-controlled Z-gate.

        Parameters
        ----------
        qubits : list
            The list of Qubits to apply the gate to.
        method : str, optional
            The algorithm to synthesize the mcz gate. The default is "gray".
        ctrl_state : str or int, optional
            The state on which the X gate is activated. Can be supplied as a string
            (i.e. "010110...") or an integer. The default is all ones ("11111...").


        """
        self.append(
            ops.MCZGate(len(qubits)-1, ctrl_state=ctrl_state, method=method),
            qubits,
        )
    
    def mcp(self, phi, qubits, method="gray", ctrl_state=-1):
        """
        Instruct a multi-controlled P-gate.

        Parameters
        ----------
        phi : float or sympy.Symbol
            The angle parameter.
        qubits : list
            The list of Qubits to apply the gate to.
        method : str, optional
            The algorithm to synthesize the mcp gate. The default is "gray".
        ctrl_state : str or int, optional
            The state on which the X gate is activated. Can be supplied as a string
            (i.e. "010110...") or an integer. The default is all ones ("11111...").


        """
        if not isinstance(phi, (float, sympy.Symbol)):
            raise ValueError("Input parameter phi must be of type float or sympy.Symbol.")

        self.append(
            ops.MCPGate(phi, len(qubits)-1, ctrl_state=ctrl_state, method=method),
            qubits,
        )

    def ccx(
        self,
        ctrl_qubit_0: QubitLike,
        ctrl_qubit_1: QubitLike,
        target_qubit: QubitLike,
        method="gray",
    ):
        """Instruct a Toffoli-gate.

        Parameters
        ----------
        ctrl_qubit_0 : QubitLike
            The first control Qubit.
        ctrl_qubit_1 : QubitLike
            The second control Qubit.
        target_qubit : QubitLike
            The target Qubit.
        method : str, optional
            The algorithm to synthesize the mcx gate. The default is "gray".

        """
        self.mcx([ctrl_qubit_0, ctrl_qubit_1], target_qubit, method=method)

    def crx(self, phi: FloatLike, qubits_0: QubitLike, qubits_1: QubitLike):
        """Instruct a controlled RX-gate.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.

        qubits_0 : QubitLike
            The Qubit to apply the gate on.

        qubits_1 : QubitLike
            The other Qubit to apply the gate on.

        """
        if phi == 0:
            return
        self.append(ops.MCRXGate(phi, 1), [qubits_0, qubits_1])
    
    def crz(self, phi, qubits_0, qubits_1):
        """
        Instruct a controlled rz-gate.

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
        if not isinstance(phi, (float, sympy.Symbol)):
            raise ValueError("Input parameter phi must be of type float or sympy.Symbol.")
        self.append(ops.MCRZGate(phi, 1), [qubits_0, qubits_1])

    def t(self, qubits: QubitLike):
        """Instruct a T-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.TGate(), [qubits])

    def t_dg(self, qubits: QubitLike):
        """Instruct a dagger T-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.TGate().inverse(), [qubits])

    def s(self, qubits: QubitLike):
        """Instruct an S-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.SGate(), [qubits])

    def s_dg(self, qubits: QubitLike):
        """Instruct a daggered S-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.SGate().inverse(), [qubits])

    def sx(self, qubits: QubitLike):
        """Instruct a SX-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.SXGate(), [qubits])

    def sx_dg(self, qubits: QubitLike):
        """Instruct a daggered SX-gate.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.SXGate().inverse(), [qubits])

    def barrier(self, qubits: QubitLike | None = None):
        """Instruct a Barrier onto the given Qubit. Barriers can be used as visual markers
        and compiler directives.

        Parameters
        ----------
        qubits : QubitLike | None
            The qubits to apply the barrier on. If ``None``, the barrier spans all
            qubits in the circuit. The default is ``None``.

        """
        if qubits is None:
            qubits = self.qubits

        self.append(ops.Barrier(len(qubits)), qubits)

    def reset(self, qubits: QubitLike):
        r"""Instruct a reset. This resets this Qubit into the $\ket{0}$ state regardless
        of its previous state.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to reset.

        """
        self.append(ops.Reset(), [qubits])

    def u3(self, theta: FloatLike, phi: FloatLike, lam: FloatLike, qubits: QubitLike):
        r"""Instruct a U3-gate from given Euler angles.

        A U3 gate has the unitary:

        .. math::

            U3(\theta, \phi, \lambda) = \begin{pmatrix} \cos{(\frac{\theta}{2})}
            & -\exp{(i\lambda)}\sin{(\frac{\theta}{2})} \\
            \exp{(i\phi)} \sin{(\frac{\theta}{2})}
            & \exp{(i(\phi+\lambda))}\cos{(\frac{\theta}{2})} \end{pmatrix}

        Parameters
        ----------
        theta : FloatLike
            The theta parameter.
        phi : FloatLike
            The phi parameter.
        lam : FloatLike
            The lambda parameter.
        qubits : QubitLike
            The Qubit to apply the u3 gate on.

        """
        self.append(ops.u3Gate(theta, phi, lam), [qubits])

    def r(self, theta: FloatLike, phi: FloatLike, qubits: QubitLike):
        r"""Instruct an R-gate.

        The unitary is

        .. math::

            R(\theta, \phi) =
            \begin{pmatrix}
                \cos\!\tfrac{\theta}{2} & e^{-i\phi}\sin\!\tfrac{\theta}{2} \\
                -e^{i\phi}\sin\!\tfrac{\theta}{2} & \cos\!\tfrac{\theta}{2}
            \end{pmatrix}

        Parameters
        ----------
        theta : FloatLike
            Rotation angle in radians. Controls the amount of rotation.
        phi : FloatLike
            Axis angle in radians. Selects the rotation axis in the XY plane.
        qubits : QubitLike
            The qubit to apply the gate on.

        """
        self.append(ops.RGate(theta, phi), [qubits])

    def unitary(self, unitary_array, qubits: QubitLike):
        """Instruct a gate from a given unitary matrix.

        Parameters
        ----------
        unitary_array : numpy.ndarray
            The 2×2 unitary matrix to apply.
        qubits : QubitLike
            The qubit to apply the gate on.

        """
        from qrisp.simulator.unitary_management import u3matrix

        mat = unitary_array
        coeff = 1 / np.sqrt(np.linalg.det(mat))
        theta = 2 * np.arctan2(np.abs(coeff * mat[1][0]), np.abs(coeff * mat[0][0]))
        phiplambda2 = np.angle(coeff * mat[1][1]) % (2 * np.pi)
        phimlambda2 = np.angle(coeff * mat[1][0]) % (2 * np.pi)
        phi = phiplambda2 + phimlambda2
        lam = phiplambda2 - phimlambda2

        arg_max = np.argmax(np.abs(mat).flatten())
        temp_u3 = u3matrix(theta, phi, lam, 0).flatten()
        gphase = (-np.angle(temp_u3[arg_max] / mat.flatten()[arg_max])) % (2 * np.pi)

        self.append(U3Gate(theta, phi, lam, global_phase=gphase), qubits)

    def gphase(self, phi: FloatLike, qubits: QubitLike):
        """Instruct a global phase. Global phases do not directly influence the
        QuantumCircuits outcome however they can become physical if used as a base gate
        for a controlled operation.

        Parameters
        ----------
        phi : FloatLike
            The angle parameter.
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.GPhaseGate(phi), [qubits])

    def id(self, qubits: QubitLike):
        """Instruct an identity gate. Identity gates are simply placeholders and have no
        effect on the quantum state.

        Parameters
        ----------
        qubits : QubitLike
            The Qubit to apply the gate on.

        """
        self.append(ops.IDGate(), [qubits])


def _convert_qb_item(
    value: Any,
    circuit: QuantumCircuit | None = None,
) -> Qubit | list[Any]:
    """Recursive helper for convert_to_qb_list; may return a bare Qubit or a list."""
    # NOTE: This is here to avoid circular imports
    from qrisp import QuantumArray

    if isinstance(value, Qubit):
        return value

    if isinstance(value, QuantumArray):
        return [qb for qv in value.flatten() for qb in qv.reg]

    if hasattr(value, "__iter__"):
        return [_convert_qb_item(item, circuit) for item in value]

    if hasattr(value, "reg"):
        return list(value.reg)

    if isinstance(value, int):
        if circuit is None:
            raise ValueError("Tried to convert integer index to Qubit without a circuit")
        if value >= len(circuit.qubits):
            raise ValueError(
                f"Tried to address qubit with index {value} in a circuit with {len(circuit.qubits)} qubits"
            )
        return _convert_qb_item(circuit.qubits[value], circuit)

    raise TypeError(f"Cannot convert type {type(value)} to a qubit list")


def convert_to_qb_list(
    value: Any,
    circuit: QuantumCircuit | None = None,
) -> list[Any]:
    """Convert a qubit specification to a (possibly nested) list of :class:`.Qubit` objects.

    This is the internal helper used by :meth:`.QuantumCircuit.append` to normalise the
    *qubits* argument before an instruction is recorded.  The function accepts every form
    that ``append`` advertises:

    * A :class:`.Qubit` instance, returned as ``[qubit]``.
    * An :class:`int` and then re-processed.
    * A :class:`~qrisp.QuantumArray`, flattened to a single list of its constituent
      qubits.
    * Any object with a ``.reg`` attribute (e.g. a :class:`~qrisp.QuantumVariable`) —
      converted via ``list(value.reg)``.
    * Any other iterable. Each element is recursively processed and the results are
      collected into a list. This is what enables the multi-gate broadcasting feature
      of ``append``.

    Parameters
    ----------
    value : Any
        The qubit specification to convert.
    circuit : QuantumCircuit, optional
        The circuit whose ``qubits`` list is used to resolve integer indices.
        Required when *value* (or any nested element) is an integer.
        The default is ``None``.

    Returns
    -------
    list[Any]
        A flat or nested list of :class:`.Qubit` objects.  Nesting only occurs
        when *value* is itself a nested iterable, enabling the broadcasting logic
        in :meth:`.QuantumCircuit.append`.

    Raises
    ------
    ValueError
        If an integer index is provided but no *circuit* is given, or if the
        index is out of range.
    TypeError
        If *value* cannot be converted to a qubit list.

    """
    item = _convert_qb_item(value, circuit)
    return item if isinstance(item, list) else [item]


def _convert_cb_item(
    value: Any,
    circuit: QuantumCircuit | None = None,
) -> Clbit | list[Any]:
    """Recursive helper for convert_to_cb_list. May return a bare Clbit or a list."""
    if isinstance(value, Clbit):
        return value

    if hasattr(value, "__iter__"):
        return [_convert_cb_item(item, circuit) for item in value]

    if isinstance(value, int):
        if circuit is None:
            raise ValueError("Tried to convert integer index to Clbit without a circuit")
        return _convert_cb_item(circuit.clbits[value], circuit)

    raise TypeError(f"Cannot convert type {type(value)} to a classical-bit list")


def convert_to_cb_list(
    value: Any,
    circuit: QuantumCircuit | None = None,
) -> list[Any]:
    """Convert a classical-bit specification to a (possibly nested) list of
    :class:`.Clbit` objects.

    This is the internal helper used by :meth:`.QuantumCircuit.append` to normalise the
    *clbits* argument before an instruction is recorded.  It mirrors the behaviour of
    :func:`convert_to_qb_list` for classical bits:

    * A :class:`.Clbit` instance, returned as ``[clbit]``.
    * An :class:`int` and then re-processed.
    * Any other iterable. Each element is recursively processed and the results are
      collected into a list.

    Parameters
    ----------
    value : Any
        The classical-bit specification to convert.
    circuit : QuantumCircuit, optional
        The circuit whose ``clbits`` list is used to resolve integer indices.
        Required when *value* (or any nested element) is an integer.
        The default is ``None``.

    Returns
    -------
    list[Any]
        A flat or nested list of :class:`.Clbit` objects.

    Raises
    ------
    ValueError
        If an integer index is provided but no *circuit* is given.
    TypeError
        If *value* cannot be converted to a classical-bit list.

    """
    item = _convert_cb_item(value, circuit)
    return item if isinstance(item, list) else [item]
