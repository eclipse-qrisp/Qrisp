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
import weakref

import matplotlib.pyplot as plt
import numpy as np

from qrisp.core.compilation import qompiler

from qrisp.jax import TracingQuantumSession

class QuantumVariable:
    """
    The QuantumVariable is the quantum equivalent of a regular variable in classical
    programming languages. All :ref:`quantum types <QuantumTypes>` inherit from this
    class. The QuantumVariable allows many automizations and quality of life
    improvements such as hidden qubit management, de/encoding to human readable labels
    or typing.

    Each QuantumVariable is registered in a :ref:`QuantumSession`. It can be accessed
    using the ``.qs`` attribute:

    >>> from qrisp import QuantumVariable
    >>> example_qv = QuantumVariable(3)
    >>> quantum_session = example_qv.qs

    The qubits of the QuantumVariable are stored as a list in the ``.reg`` attribute

    >>> qubits = example_qv.reg

    To quickly access the qubits of a given variable, we use the [ ] operator:

    >>> qubit_2 = example_qv[2]

    We can find out about the amount of qubits in the QuantumVariable with the ``.size``
    attribute

    >>> example_qv.size
    3

    **Naming**

    QuantumVariables can be given names to identify them independently of their naming
    as Python objects.

    >>> example_qv_2 = QuantumVariable(3, name = "alice")
    >>> example_qv_2.name
    'alice'

    If not explicitely specified during construction, a name is determined
    automatically. Qrisp will try to infer the name of the Python variable and if that
    fails, a generic name is given.

    >>> example_qv.name
    'example_qv'

    In order to keep the generated quantum circuits comprehensive, the qubits are named
    after their containing QuantumVariable with an extra number, which indicates their
    index.

    >>> from qrisp import cx
    >>> cx(example_qv, example_qv_2)
    >>> print(example_qv.qs)
    
    ::
    
        QuantumCircuit:
        --------------
        example_qv.0: ──■────────────
                        │
        example_qv.1: ──┼────■───────
                        │    │
        example_qv.2: ──┼────┼────■──
                      ┌─┴─┐  │    │
             alice.0: ┤ X ├──┼────┼──
                      └───┘┌─┴─┐  │
             alice.1: ─────┤ X ├──┼──
                           └───┘┌─┴─┐
             alice.2: ──────────┤ X ├
                                └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable example_qv
        QuantumVariable alice

    QuantumSessions can only contain uniquely named QuantumVariables. If two
    QuantumSessions are :ref:`merged <SessionMerging>` containing identically named
    QuantumVariables, the more recently created QuantumVariable will be renamed:

    ::

        from qrisp import QuantumFloat

        s = QuantumFloat(5)

        for i in range(4):
            temp = QuantumFloat(4)
            temp[:] = 2**i
            s += temp

    >>> print(s.qs)
    
    ::
    
        QuantumCircuit:
        --------------
                       ┌───────────┐┌───────────┐┌───────────┐┌───────────┐
             s.0: ─────┤0          ├┤0          ├┤0          ├┤0          ├
                       │           ││           ││           ││           │
             s.1: ─────┤1          ├┤1          ├┤1          ├┤1          ├
                       │           ││           ││           ││           │
             s.2: ─────┤2          ├┤2          ├┤2          ├┤2          ├
                       │           ││           ││           ││           │
             s.3: ─────┤3          ├┤3          ├┤3          ├┤3          ├
                       │           ││           ││           ││           │
             s.4: ─────┤4 __iadd__ ├┤4          ├┤4          ├┤4          ├
                  ┌───┐│           ││           ││           ││           │
          temp.0: ┤ X ├┤5          ├┤           ├┤           ├┤           ├
                  └───┘│           ││           ││           ││           │
          temp.1: ─────┤6          ├┤  __iadd__ ├┤           ├┤           ├
                       │           ││           ││           ││           │
          temp.2: ─────┤7          ├┤           ├┤           ├┤           ├
                       │           ││           ││           ││           │
          temp.3: ─────┤8          ├┤           ├┤  __iadd__ ├┤           ├
                       └───────────┘│           ││           ││           │
        temp_1.0: ──────────────────┤5          ├┤           ├┤           ├
                  ┌───┐             │           ││           ││           │
        temp_1.1: ┤ X ├─────────────┤6          ├┤           ├┤  __iadd__ ├
                  └───┘             │           ││           ││           │
        temp_1.2: ──────────────────┤7          ├┤           ├┤           ├
                                    │           ││           ││           │
        temp_1.3: ──────────────────┤8          ├┤           ├┤           ├
                                    └───────────┘│           ││           │
        temp_2.0: ───────────────────────────────┤5          ├┤           ├
                                                 │           ││           │
        temp_2.1: ───────────────────────────────┤6          ├┤           ├
                  ┌───┐                          │           ││           │
        temp_2.2: ┤ X ├──────────────────────────┤7          ├┤           ├
                  └───┘                          │           ││           │
        temp_2.3: ───────────────────────────────┤8          ├┤           ├
                                                 └───────────┘│           │
        temp_3.0: ────────────────────────────────────────────┤5          ├
                                                              │           │
        temp_3.1: ────────────────────────────────────────────┤6          ├
                                                              │           │
        temp_3.2: ────────────────────────────────────────────┤7          ├
                  ┌───┐                                       │           │
        temp_3.3: ┤ X ├───────────────────────────────────────┤8          ├
                  └───┘                                       └───────────┘
        Live QuantumVariables:
        ---------------------
        QuantumFloat s
        QuantumFloat temp
        QuantumFloat temp_1
        QuantumFloat temp_2
        QuantumFloat temp_3


    Renaming does not happen for names given through the ``name`` keyword, unless the
    name ends with a ``*``.

    >>> example_qv_3 = QuantumVariable(3, name = "alice")
    >>> cx(example_qv, example_qv_3)
    Exception: Tried to merge QuantumSession containing identically named
    QuantumVariables
    >>> example_qv_4 = QuantumVariable(3, name = "alice*")
    >>> cx(example_qv, example_qv_4)
    >>> example_qv_4.name
    'alice_1'

    Examples
    --------

    Writing a function that brings an arbitrary QuantumVariable into a GHZ state

    ::

        from qrisp import QuantumVariable, h, cx
        def GHZ(qv):
             h(qv[0])
             for i in range(1, qv.size):
                 cx(qv[0], qv[i])

    Evaluation:

    >>> qv = QuantumVariable(5)
    >>> GHZ(qv)
    >>> print(qv)
    {'00000': 0.5, '11111': 0.5}


    """

    live_qvs = []
    creation_counter = np.zeros(1)
    name_tracker = {}

    def __init__(self, size, qs=None, name=None):
        r"""
        Constructs a QuantumVariable - possibly with a given name or in a given
        QuantumSession.

        Parameters
        ----------
        size : int
            The amount of qubits this QuantumVariable contains.
        qs : QuantumSession, optional
            A QuantumSession object, where the QuantumVariable is supposed to be
            registered. The default is None.
        name : string, optional
            A name which uniquely identifies the QuantumVariable. If ended with a
            \*, name is allowed to be updated if a naming collision arises. By default,
            Qrisp will try to infer the name of the Python variable - otherwise a
            generic name is given.

        """

        # Store quantum session
        from qrisp.core import QuantumSession, merge_sessions
        from qrisp.jax import check_for_tracing_mode, get_tracing_qs

        if check_for_tracing_mode():
            self.qs = get_tracing_qs()
        else:
            if qs is not None:
                self.qs = qs
            else:
                self.qs = QuantumSession()

        # self.size = size

        self.user_given_name = False
        # If name is given, register variable in session manager
        if name is not None:
            self.user_given_name = True

            if name[-1] == "*":
                name = name[:-1]
                self.user_given_name = False

                try:
                    self.name = name
                    self.qs.register_qv(self, size)

                except RuntimeError:
                    i = int(self.creation_counter)
                    while True:
                        try:
                            self.name = name + "_" + str(i)
                            self.qs.register_qv(self)
                        except RuntimeError:
                            i += 1
                            continue
                        break

            else:
                self.name = name
                self.qs.register_qv(self, size)

        # Otherwise try to infer from code inspection
        else:
            from qrisp.misc import find_calling_line

            if type(self) is QuantumVariable:
                line = find_calling_line(1)
            else:
                line = find_calling_line(2)
            split_line = line.split("=")
            name_found = False

            if len(split_line) >= 2:
                python_var_name = split_line[0]

                if split_line[1].replace(" ", "")[:7] == "Quantum":
                    python_var_name = python_var_name.split(" ")[0]
                    python_var_name = python_var_name.split(" ")[-1]

                    # name = self.get_unique_name(python_var_name)
                    name = python_var_name

                    self.name = name
                    self.qs.register_qv(self, size)
                    name_found = True

            # If this didn't work, generate a generic, unique name
            if not name_found:
                while True:
                    try:
                        self.name = self.get_unique_name()
                        self.qs.register_qv(self, size)
                        break
                    except RuntimeError:
                        pass

        import weakref

        # This attribute tracks the created QuantumVariables for the
        # auto_uncompute decorator
        # We use weak references as some qrisp modules rely on reference counting
        QuantumVariable.live_qvs.append(weakref.ref(self))
        self.creation_time = int(self.creation_counter[0])
        self.creation_counter += 1
    
    def __or__(self, other):
        from qrisp import mcx, x, cx
        
        if len(self) > len(other):
            or_res = self.duplicate()
        else:
            or_res = other.duplicate()
        
        for i in range(min(len(self), len(other))):
            mcx([self[i], other[i]], or_res[i], ctrl_state = 0)
            x(or_res[i])
        
        for i in range(min(len(self), len(other)), len(self)):
            cx(self[i], or_res[i])
            
        for i in range(min(len(self), len(other)), len(other)):
            cx(other[i], or_res[i])
        
        return or_res
        
    def __and__(self, other):
        from qrisp import mcx
        
        if len(self) > len(other):
            and_res = self.duplicate()
        else:
            and_res = other.duplicate()
        
        for i in range(min(len(self), len(other))):
            mcx([self[i], other[i]], and_res[i])
        
        return and_res
    
    def __xor__(self, other):
        from qrisp import cx
        
        if len(self) > len(other):
            and_res = self.duplicate()
        else:
            and_res = other.duplicate()
        
        for i in range(min(len(self), len(other))):
            cx(self[i], and_res[i])
            cx(other[i], and_res[i])
        
        return and_res
        
    def delete(self, verify=False, recompute=False):
        r"""
        This method is for deleting a QuantumVariable and thus freeing up and resetting
        the used qubits.

        Note that this method has a different function than the destructor. Calling this
        method will tell the QuantumSession to mark the used qubits as free and apply a
        reset gate.
        If set to True, the keyword verify will cause a simulation to check, wether the
        deleted qubits are in the $\ket{0}$ state prior to resetting. This is helpfull
        during debugging, as it indicates wether the uncomputation of this
        QuantumVariable was successfull.

        After deletion, the QuantumVariable object is basically unchanged but an error
        will be raised if further operations on the deleted qubits are attempted.

        Parameters
        ----------
        verify : bool, optional
            If this bool is set to True, Qrisp will verify that the deleted qubits are
            indeed in the $\ket{0}$ state. The default is ``False``.
        recompute : bool, optional
            If set to ``True``, this QuantumVariable can be recomputed if it is required
            for the uncomputation of another QuantumVariable. For more information on
            the (dis)advantages check :ref:`recomputation <recomputation>`. The default
            is ``False``.

        Raises
        ------
        Exception
            Tried to delete qubits not in \|0> state.


        Examples
        --------

        We create a QuantumVariable, execute some gates and try to delete with
        verify = True

        >>> from qrisp import QuantumVariable, x, h
        >>> qv = QuantumVariable(2)
        >>> x(qv[0])
        >>> h(qv[1])
        >>> qv.delete(verify = True)
        Exception: Tried to delete qubits not in |0> state.

        We now (manually) uncompute the gates

        >>> x(qv[0])
        >>> h(qv[1])
        >>> qv.delete(verify = True)
        >>> qv.is_deleted()
        True
        >>> x(qv[0])
        Exception: Tried to perform operation x on unallocated qubit qv_1.0.

        """
        
        if not isinstance(self.qs, TracingQuantumSession) and self.is_deleted():
            return

        self.qs.delete_qv(self, verify)

        i = 0
        while i < len(QuantumVariable.live_qvs):
            if QuantumVariable.live_qvs[i]() is None:
                QuantumVariable.live_qvs.pop(i)
                continue

            if QuantumVariable.live_qvs[i]().name == self.name:
                QuantumVariable.live_qvs.pop(i)
                break

            i += 1

        if recompute:
            for qb in self.reg:
                qb.recompute = True

    def is_deleted(self):
        for qb in self.reg:
            if not qb.allocated:
                return True
        else:
            return False

    def duplicate(self, name=None, qs=None, init=False):
        r"""
        Duplicates the QuantumVariable in the sense that a new QuantumVariable is
        created with same type and parameters but initialized in the $\ket{0}$ state.

        Parameters
        ----------
        name : string, optional
            A unique name to identify that QuantumVariable. If not given, a name will be
            generated.
        qs : QuantumSession, optional
            A QuantumSession, where the result should be registered. If not given, a new
            QuantumSession will be generated.
        init : bool, optional
            If set to True, the :meth:`init_from <qrisp.QuantumVariable.init_from>`
            method of the result will be called on self. The default is False.

        Returns
        -------
        duplicate : Type of self
            The duplicated QuantumVariable.

        Examples
        --------

        We create a QuantumFloat and duplicate:

        >>> from qrisp import QuantumFloat
        >>> qf_0 = QuantumFloat(4, signed = False)
        >>> qf_1 = qf_0.duplicate()
        >>> type(qf_1)
        qrisp.qtypes.quantum_float.QuantumFloat
        >>> qf_1.size
        4

        """

        duplicate = copy.copy(self)

        from qrisp.core import QuantumSession

        new_qs = QuantumSession()

        # Register duplicate variable in session manager

        if name is not None:
            
            if name[-1] == "*":
                self.user_given_name = False
                name = name[:-1]
            else:
                duplicate.user_given_name = True
            
            duplicate.name = name
            new_qs.register_qv(duplicate)
            
            

        else:
            duplicate.user_given_name = False

            try:
                duplicate.name = self.name + "_dupl"
                new_qs.register_qv(duplicate)
            except NameError:
                i = 0
                while True:
                    try:
                        duplicate.name = self.name + "_dupl" + str(i)
                        new_qs.register_qv(duplicate)
                        break
                    except NameError:
                        pass
                    i += 1

        from qrisp import merge

        duplicate.qs = new_qs

        # This attribute tracks the created QuantumVariables for the
        # auto_uncompute decorator
        # We use weak references as some qrisp modules rely on reference counting
        QuantumVariable.live_qvs.append(weakref.ref(duplicate))
        duplicate.creation_time = int(self.creation_counter[0])
        duplicate.creation_counter += 1

        if qs is not None:
            merge(qs, new_qs)

        if init:
            duplicate.init_from(self)

        return duplicate

    def decoder(self, i):
        """
        The decoder method specifies how a QuantumVariable turns the outcomes of
        measurements into human-readable values. It recieves an integer ``i`` and
        returns a human-readable value.

        This method is supposed to be overloaded when defining new
        :ref:`quantum types <QuantumTypes>`.


        Parameters
        ----------
        i : int
            Integer representing the outcome of a measurement of the qubits of this
            QuantumVariable.

        Returns
        -------

            A human-readable value. Has to be hashable.

        Examples
        --------

        We create a QuantumFloat and inspect its decoder:

        >>> from qrisp import QuantumFloat
        >>> qf = QuantumFloat(3, -1, signed = False)
        >>> print(qf.decoder(1))
        0.5

        This implies that if the 3 qubits of this QuantumFloat are measured in state
        001, this outcome corresponds to the value 0.5.
        """

        from qrisp.misc import bin_rep

        return bin_rep(i, self.size)[::-1]

    def encoder(self, value):
        """
        The encoder reverses the decoder, it turns human-readable values into integers.

        If not overloaded, the encoder will perform a linear search on decoder inputs to
        match the given value.

        Parameters
        ----------
        label :
            A human-readable value.

        Raises
        ------
        Exception
            Unknown input value.

        Returns
        -------
        i : int
            The integer encoding the given value.

        Examples
        --------

        We create a QuantumChar and inspect it's encoder:

        >>> from qrisp import QuantumChar
        >>> q_ch = QuantumChar()
        >>> print(q_ch.encoder("f"))
        5

        This implies that if the 5 qubits of this QuantumChar are measured to
        ``5 = 00101``, the out come will be displayed as f.

        """
        for i in range(2**self.size):
            if self.decoder(i) == value:
                return i

        raise Exception("Value " + str(value) + " not supported by encoder.")

    def encode(self, value, permit_dirtyness = False):
        """
        The encode method allows to quickly bring a QuantumVariable in a desired
        computational basis state.

        A shorthand for this method is given by the ``[:]`` operator.

        Note that the qubits to initialize have to be fresh (i.e. no operations
        performed on them).

        Parameters
        ----------
        value :
            A value supported by the encoder.
        permit_dirtyness : bool, optional
            Surpresses the error message when calling encode on dirty qubits.

        Returns
        -------
        None.

        Examples
        --------

        We create two quantum floats and encode the value 2.5. For one of them, we
        perform an x gate onto the corresponding qubits, resulting in an error.

        >>> from qrisp import QuantumFloat, x
        >>> qf_0 = QuantumFloat(3, -1, signed = False)
        >>> qf_1 = QuantumFloat(3, -1, signed = False)
        >>> x(qf_0)
        >>> qf_0.encode(2.5)
        Exception: Tried to initialize qubits which are not fresh anymore.
        >>> qf_1[:] = 2.5
        >>> print(qf_1)
        {2.5: 1.0}

        """

        from qrisp.misc import check_if_fresh, int_encoder
        
        if not permit_dirtyness:
            if not check_if_fresh(self.reg, self.qs):
                raise Exception("Tried to initialize qubits which are not fresh anymore.")

        int_encoder(self, self.encoder(value))

    def init_state(self, state_dic):
        r"""
        The ``init_state`` method allows the initialization of arbitrary quantum states.
        It recieves a dictionary of the type

        **{value : complex number}**

        and initializes the **normalized** state. Amplitudes not specified are assumed
        to be zero.

        Note that the state initialization algorithm requires it's qubits to be in
        state $\ket{0}$.

        A shorthand for this method is the ``[:]`` operator, when handed the
        corresponding dictionary

        Parameters
        ----------
        state_dic : dict
            Dictionary describing the wave function to be initialized.

        Raises
        ------
        Exception
            Tried to initialize qubits which are not fresh anymore.

        Examples
        --------

        We create a QuantumFloat and encode the state

        .. math::

            \ket{\psi} = \sqrt{\frac{1}{3}} \ket{0.5} + i\sqrt{\frac{2}{3}} \ket{2}

        >>> from qrisp import QuantumFloat
        >>> qf = QuantumFloat(3, -1)

        We can now use either

        >>> qf.init_state({0.5: (1/3)**0.5, 2.0 : 1j*(2/3)**0.5})

        or:

        >>> qf[:] = {0.5: (1/3)**0.5, 2.0 : 1j*(2/3)**0.5}

        To acquire the expected result

        >>> print(qf)
        {2.0: 0.6667, 0.5: 0.3333}

        """

        from qrisp.misc import check_if_fresh

        if not check_if_fresh(self.reg, self.qs):
            raise Exception("Tried to initialize qubits which are not fresh anymore.")

        from qrisp import init_state

        target_array = np.zeros(2**self.size, dtype=np.complex128)

        for key in state_dic.keys():
            target_array[self.encoder(key)] = state_dic[key]

        target_array = target_array / np.vdot(target_array, target_array) ** 0.5

        init_state(self, target_array)

    def append(self, operation):
        self.qs.append(operation, self)

    def extend(self, amount, position=-1):
        """
        This method is used to add more qubits to the QuantumVariable. Using the
        position keyword it is possible to specify the position where the qubits should
        be added. By default, the qubits are added at the end.

        Parameters
        ----------
        amount : int
            The amount of qubits to add.
        position : int, optional
            The position of where to add the qubits. By default, qubits are added at the
            end.
            st of qubits which are to be added to the QuantumVariable.
            The default is None.

        Raises
        ------
        Exception
            Missmatch between proposed qubits and amount integer.

        Returns
        -------
        None.

        Examples
        --------

        We create a QuantumVariable and extend it with some extra qubits.

        >>> from qrisp import QuantumVariable
        >>> qv = QuantumVariable(3)
        >>> print(qv.reg)
        [Qubit(qv.0), Qubit(qv.1), Qubit(qv.2)]
        >>> qv.extend(3)
        >>> print(qv.reg)
        [Qubit(qv.0), Qubit(qv.1), Qubit(qv.2), Qubit(qv.6), Qubit(qv.6), Qubit(qv.6)]

        """

        if position == -1:
            position = self.size

        insertion_qubits = self.qs.request_qubits(amount)

        for i in range(amount):
            insertion_qubits[i].identifier =  self.name + "_ext_" + str(self.qs.qubit_index_counter[0]) + "." + str(self.size)
            self.reg.insert(position + i, insertion_qubits[i])
            self.size += 1

    def reduce(self, qubits, verify=False):
        r"""
        Reduces the qubit count of the QuantumVariable by removing a specified set of
        qubits.

        Parameters
        ----------
        qubits : list
            The qubits to remove from the QuantumVariable.

        verify : bool
            Boolean value which indicates wether Qrisp should verify that the reduced
            qubits are in the $\ket{0}$ state.

        Raises
        ------
        Exception
            Qubits not present in QuantumVariable.

        Exception
            Verification that the given qubits are in $\ket{0}$ state failed.

        Examples
        --------

        We create a QuantumVariable with 5 qubits and remove the first 2

        >>> from qrisp import QuantumVariable
        >>> qv = QuantumVariable(5)
        >>> print(qv.reg)
        [Qubit(qv.0), Qubit(qv.1), Qubit(qv.2), Qubit(qv.3), Qubit(qv.4)]
        >>> qv.reduce(qv[:2])
        >>> print(qv.reg)
        [Qubit(qv.2), Qubit(qv.3), Qubit(qv.4)]

        """

        try:
            len(qubits)
        except TypeError:
            qubits = [qubits]

        if not set(qubits).issubset(self.reg):
            raise Exception("Tried to reduce QuantumVariable by invalid qubits")

        # Find Qubits to be cleared
        for i in range(len(qubits)):
            for j in range(self.size):
                if self.reg[j] == qubits[i]:
                    self.reg[j].identifier = "reduced_" + str(self.qs.qubit_index_counter[0])
                    self.qs.qubit_index_counter += 1
                    self.reg.pop(j)
                    break

        self.qs.clear_qubits(qubits, verify)
        # Adjust variable size
        self.size -= len(qubits)

    def get_measurement(
        self,
        plot=False,
        backend=None,
        shots=100000,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        circuit_preprocessor=None,
        filename=None,
        precompiled_qc = None
    ):
        r"""
        Method for quick access to the measurement results of the state of the variable.
        This method returns a dictionary of the type {value : p} where p indicates the
        probability with which that value is measured.


        Parameters
        ----------
        plot : Bool, optional
            Plots the measurement results as a historgram. The default is False.
        backend : BackendClient, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        shots : integer, optional
            The amount of shots to evaluate the circuit. The default is 10000.
        compile : bool, optional
            Boolean indicating if the .compile method of the underlying QuantumSession
            should be called before. The default is True.
        compilation_kwargs  : dict, optional
            Keyword arguments for the compile method. For more details check
            :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`. The default
            is ``{}``.
        subs_dic : dict, optional
            A dictionary of Sympy symbols and floats to specify parameters in the case
            of a circuit with unspecified, :ref:`abstract parameters<QuantumCircuit>`.
            The default is {}.
        circuit_preprocessor : Python function, optional
            A function which recieves a QuantumCircuit and returns one, which is applied
            after compilation and parameter substitution. The default is None.
        filename : string, optional
            The location of where to save a generated plot. The default is None.

        Raises
        ------
        Exception
            If the containing QuantumSession is in a quantum environment, it is not
            possible to execute measurements.

        Returns
        -------
        dict
            A dictionary of values and their corresponding measurement probabilities.

        Examples
        --------

        We create an integer :ref:`QuantumFloat`, encode the value 1 and bring the qubit
        with significance 2 in superposition. We utilize the Qiskit transpiler by
        transpiling into the gate set $\{\text{CX}, \text{U}\}$

        >>> from qrisp import QuantumFloat, h
        >>> qf = QuantumFloat(3,-1)
        >>> qf[:] = 1
        >>> h(qf[2])
        >>> mes_results = qf.get_measurement(transpilation_kwargs = {"basis_gates" : ["cx", "u"]})  # noqa:501
        >>> print(mes_results)
        {1.0: 0.5, 3.0: 0.5}
        """

        if backend is None:
            if self.qs.backend is None:
                from qrisp.default_backend import def_backend

                backend = def_backend
            else:
                backend = self.qs.backend

        if len(self.qs.env_stack) != 0:
            raise Exception("Tried to get measurement within open environment")

        if self.is_deleted():
            raise Exception("Tried to get measurement from deleted QuantumVariable")

        if self.size == 0:
            return {"": 1.0}

        if precompiled_qc is None:        
            if compile:
                qc = qompiler(
                    self.qs, intended_measurements=self.reg, **compilation_kwargs
                )
            else:
                qc = self.qs.copy()
        else:
            qc = precompiled_qc.copy()

        # Bind parameters
        if subs_dic:
            qc = qc.bind_parameters(subs_dic)
            from qrisp.core.compilation import combine_single_qubit_gates
            qc = combine_single_qubit_gates(qc)

        # Copy circuit in over to prevent modification
        # from qrisp.quantum_network import QuantumNetworkClient

        # if isinstance(backend, QuantumNetworkClient):
        #     self.qs.data = []
        #     shots = 1

        # Execute user specified circuit_preprocessor
        if circuit_preprocessor is not None:
            qc = circuit_preprocessor(qc)

        qc = qc.transpile()

        from qrisp.misc import get_measurement_from_qc

        counts = get_measurement_from_qc(qc, self.reg, backend, shots)

        # Insert outcome labels (if available and hashable)
        try:
            new_counts_dic = {}

            sorted_keys = list(counts.keys())
            sorted_keys.sort()

            for key in sorted_keys:
                new_counts_dic[self.decoder(key)] = counts[key]

            counts = new_counts_dic

            # Sort keys
            sorted_key_list = list(counts.keys())
            sorted_key_list.sort(key=lambda x: -counts[x])
            counts = {key: counts[key] for key in sorted_key_list}

        except TypeError:
            counts_tuple_list = []

            for key in counts.keys():
                counts_tuple_list.append((key, counts[key]))

            counts = counts_tuple_list

            counts.sorted(key=lambda x: x[1])

        if plot:
            outcome_labels = []
            for i in range(2**self.size):
                temp = self.decoder(i)
                
                try:
                    hash(temp)
                except TypeError:
                    raise Exception(
                        "Outcome value " + str(self.decoder(i)) + " is not hashable"
                    )

                outcome_labels.append(temp)

            plot_histogram(outcome_labels, counts, filename)
            plt.show()

        # Return dictionary of measurement results
        return counts

    def most_likely(self, **kwargs):
        """
        Performs a measurement and returns the most likely outcome.

        Parameters
        ----------
        **kwargs : Keyword arguments for the get_measurement call.

        Examples
        --------

        >>> from qrisp import QuantumFloat, ry
        >>> import numpy as np
        >>> qf = QuantumFloat(3)
        >>> ry(np.pi*9/8, qf[0])
        >>> print(qf)
        {1: 0.9619, 0: 0.0381}
        >>> qf.most_likely()
        1

        """

        return list(self.get_measurement())[0]

    def __getitem__(self, key):
        if isinstance(self.reg, list):
            return self.reg[key]
        else:
            from qrisp.jax import get_qubit
            from qrisp import Qubit
            qb = Qubit(self.name + "_abs")
            qb.qs = self.qs
            qb.abstract = get_qubit(self.reg, key)
            qb.allocated = True
            return qb

    def __str__(self):
        return str(self.get_measurement())
    
    def __repr__(self):
        return "<" + str(type(self)).split(".")[-1][:-2] + " '" + self.name + "'>"
        return str(type(self)).split(".")[-1][:-2] + "(name = " + self.name + ")"
        return str(self)
    
    def __del__(self):
        i = 0
        while i < len(self.live_qvs):
            if self.live_qvs[i]() is None or id(self) == id(self.live_qvs[i]()):
                self.live_qvs.pop(i)
                continue
            i += 1
        

    def __len__(self):
        return self.size
    
    @property
    def size(self):
        if isinstance(self.reg, list):
            return len(self.reg)
        else:
            from qrisp.jax import get_size
            return get_size(self.reg)
        

    # Overload equality operator to use python syntax for if environments?
    # Not sure if the possible user confusion is worth it
    def __eq__(self, other):
        from qrisp.environments import q_eq

        return q_eq(self, other)
    
    def __ne__(self, other):
        from qrisp.environments import q_eq

        return q_eq(self, other, invert = True)

    def __hash__(self):
        return self.creation_time

    def __setitem__(self, key, value):
        if key != slice(None, None, None):
            raise Exception(
                "Tried to encode value into QuantumVariable using non-trivial slicing."
            )

        if isinstance(type(value), type(None)):
            return

        if isinstance(value, dict):
            self.init_state(value)
            return

        if isinstance(value, QuantumVariable):
            self.init_from(value)
            return

        self.encode(value)

    def app_phase_function(self, phi):
        r"""
        Applies a previously specified phase function to each computational basis state
        of the QuantumVariable using Gray-Synthesis.

        For a given phase function $\phi(x)$ and a QuantumVariable in state
        $\ket{\psi} = \sum_{x \in \text{Labels}} a_x \ket{x}$  this method acts as:

        .. math::

            U_{\phi} \sum_{x \in \text{Labels}} a_x \ket{x} =
            \sum_{x \in \text{Labels}} \text{exp}(i\phi(x)) a_x \ket{x}

        Parameters
        ----------
        phi : Python function
            A Python function which turns the labels of the QuantumVariable into floats.

        Examples
        --------

        We create a QuantumFloat and encode the k-th basis state of the Fourier basis.
        Finally, we will apply an inverse Fourier transformation to measure k in the
        computational basis.

        >>> import numpy as np
        >>> from qrisp import QuantumFloat, h, QFT
        >>> n = 5
        >>> qf = QuantumFloat(n, signed = False)
        >>> h(qf)

        After this, qf is in the state

        .. math::

            \ket{\text{qf}} = \frac{1}{\sqrt{2^n}} \sum_{x = 0}^{2^n} \ket{x}

        We specify phi

        >>> k = 4
        >>> def phi(x):
        >>>     return 2*np.pi*x*k/2**n

        And apply phi as a phase function

        >>> qf.app_phase_function(phi)

        qf is now in the state

        .. math::

            \ket{\text{qf}} = \frac{1}{\sqrt{2^n}} \sum_{x = 0}^{2^n}
            \text{exp}\left( \frac{2\pi ikx}{2^n}\right) \ket{x}


        Finally we apply the inverse Fourier transformation and measure:

        >>> QFT(qf, inv = True)
        >>> print(qf)
        {4: 1.0}


        """

        from qrisp.misc import app_phase_function

        app_phase_function([self], phi)

    def uncompute(self, do_it=True, recompute=False):
        """
        Method for automatic uncomputation. Uses a generalized form of
        `this algorithm <https://dl.acm.org/doi/10.1145/3453483.3454040>`_.

        For more information check the
        :ref:`uncomputation documentation<uncomputation>`.

        Parameters
        ----------
        do_it : bool, optional
            If set to False, this variable will be appended to the uncomputation stack
            of it's QuantumSession and uncomputed once an uncompute call with
            ``do_it = True`` is performed. The default is True.
        recompute : bool, optional
            If set to True, this QuantumVariable will be uncomputed but temporarily
            recomputed, if it is required for the uncomputation of another
            QuantumVariable. For more information check
            :ref:`recomputation <recomputation>`. The default is False.

        Examples
        --------

        We create two QuantumVariables, apply some gates and perform automatic
        uncomputation:

        >>> from qrisp import QuantumVariable, x, cx, h, p, mcx
        >>> a = QuantumVariable(3)
        >>> b = QuantumVariable(2)
        >>> mcx(a, b[0])
        >>> h(a[:2])
        >>> x(b[0])
        >>> cx(b[0], b[1])
        >>> p(0.5, b[1])
        >>> print(a.qs)
        
        ::
        
            QuantumCircuit:
            --------------
                      ┌───┐
            a.0: ──■──┤ H ├───────────────
                   │  ├───┤
            a.1: ──■──┤ H ├───────────────
                   │  └───┘
            a.2: ──■──────────────────────
                 ┌─┴─┐┌───┐
            b.0: ┤ X ├┤ X ├──■────────────
                 └───┘└───┘┌─┴─┐┌────────┐
            b.1: ──────────┤ X ├┤ P(0.5) ├
                           └───┘└────────┘
            Live QuantumVariables:
            ---------------------
            QuantumVariable a
            QuantumVariable b

        >>> b.uncompute()
        >>> print(b.qs)
        
        ::
        
            QuantumCircuit:
            --------------
                 ┌────────┐                              ┌────────┐┌───┐
            a.0: ┤0       ├──────────────────────────────┤0       ├┤ H ├
                 │        │                              │        │├───┤
            a.1: ┤1       ├──────────────────────────────┤1       ├┤ H ├
                 │  pt3cx │                              │  pt3cx │└───┘
            a.2: ┤2       ├──────────────────────────────┤2       ├─────
                 │        │┌───┐                    ┌───┐│        │
            b.0: ┤3       ├┤ X ├──■──────────────■──┤ X ├┤3       ├─────
                 └────────┘└───┘┌─┴─┐┌────────┐┌─┴─┐└───┘└────────┘
            b.1: ───────────────┤ X ├┤ P(0.5) ├┤ X ├────────────────────
                                └───┘└────────┘└───┘
            Live QuantumVariables:
            ---------------------
            QuantumVariable a


        """

        if self.is_deleted():
            raise Exception("Tried to uncompute deleted QuantumVariable")

        if do_it:
            from qrisp.uncomputation import uncompute

            uncompute(self.qs, self.qs.uncomp_stack + [self], recompute)
            self.qs.uncomp_stack = []
        else:
            self.qs.uncomp_stack.append(self)

    def get_unique_name(self, name=None):
        if name is None:
            from qrisp import QuantumBool, QuantumChar, QuantumFloat

            if isinstance(self, QuantumBool):
                name = "qbl"
            elif isinstance(self, QuantumFloat):
                name = "qf"
            elif isinstance(self, QuantumChar):
                name = "qch"
            else:
                name = "qv"

        while True:
            try:
                naming_number = self.name_tracker[name]
                self.name_tracker[name] += 1
                name = name + "_" + str(naming_number)
            except KeyError:
                self.name_tracker[name] = 1
                name = name + "_0"

            i = 0
            while i < len(QuantumVariable.live_qvs):
                qv = QuantumVariable.live_qvs[i]()
                if qv is None:
                    QuantumVariable.live_qvs.pop(i)
                    continue
                if qv.name == name:
                    break
                i += 1
            else:
                break

        return name

    def init_from(self, other):
        r"""
        Method to initiate a QuantumVariable based on the state of another. This method
        does NOT copy the state. Much rather it performs the operation


        .. math::

            U_{\text{init_from}} \left( \sum_{x \in \text{labels}} a_x \ket{x}
            \right)  \ket{0} = \sum_{x \in \text{labels}} a_x \ket{x} \ket{x}


        This is different from a state copying operation:

        .. math::

            U_{\text{copy}} \left( \sum_{x \in \text{labels}} a_x \ket{x} \right)
            \ket{0} = \left( \sum_{x \in \text{labels}} a_x \ket{x} \right)
            \left( \sum_{x \in \text{labels}} a_x \ket{x} \right)


        A shorthand for initiating this way is the ``[:]`` operator.


        Parameters
        ----------
        other : QuantumVariable
            The QuantumVariable from which to initiate.

        Raises
        ------
        Exception
            Tried to initialize qubits which are not fresh anymore.

        Examples
        --------

        We create a QuantumFloat, and bring it into superposition.

        >>> from qrisp import QuantumFloat, h, multi_measurement
        >>> qf_a = QuantumFloat(8)
        >>> qf_a[:] = 6
        >>> h(qf_a[0])
        >>> print(qf_a)
        {6: 0.5, 7: 0.5}

        We now duplicate and initiate the duplicate

        >>> qf_b = qf_a.duplicate()
        >>> print(qf_b)
        {0: 1.0}
        >>> qf_b.init_from(qf_a)
        >>> print(multi_measurement([qf_a, qf_b]))
        {(6, 6): 0.5, (7, 7): 0.5}

        The slicing operator achieves the same:

        >>> qf_c = qf_a.duplicate()
        >>> qf_c[:] = qf_b
        >>> print(multi_measurement([qf_a, qf_b, qf_c]))
        {(6, 6, 6): 0.5, (7, 7, 7): 0.5}



        """

        if not type(self) == type(other):
            raise Exception(
                "Tried to initialize " + str(type(self)) + " from " + str(type(other))
            )

        from qrisp.misc import check_if_fresh

        if not check_if_fresh(self.reg, self.qs):
            raise Exception("Tried to initialize qubits which are not fresh anymore")

        self.qs.cx(other.reg, self.reg)

    @classmethod
    def custom(self, label_list, decoder=None, qs=None, name=None):
        """
        Creates a QuantumVariable with customized outcome labels.

        Note that this is a class method, implying there is no need to create another
        QuantumVariable first to call this method.

        Parameters
        ----------
        label_list : list
            A list of outcome labels.
        decoder : function, optional
            The decoder function. If given none, the labels will be encoded according to
            their placement in the ``label_list``.
        qs : QuantumSession, optional
            The :ref:`QuantumSession` in which to register the customized
            QuantumVariable. If given none, the QuantumVariable will be registered in a
            new QuantumSession.
        name : string, optional
            The name of the QuantumVariable. If given none, a suited name will be
            generated.

        Returns
        -------
        CustomQuantumVariable
            A QuantumVariable with the desired outcome labels.

        Examples
        --------

        We create a QuantumVariable with some examples values as outcome labels and
        bring it into uniform superposition.

        >>> from qrisp import QuantumVariable, h
        >>> qv = QuantumVariable.custom(["lorem", "ipsum", "dolor", "sit", 42, (1,2,3)])
        >>> h(qv)
        >>> print(qv)
        {'lorem': 0.125, 'ipsum': 0.125, 'dolor': 0.125, 'sit': 0.125, 42: 0.125,
        (1, 2, 3): 0.125, 'undefined_label_6': 0.125, 'undefined_label_7': 0.125}

        """
        from qrisp.misc import custom_qv

        return custom_qv(label_list, decoder=decoder, qs=qs, name=name)


def plot_histogram(outcome_labels, counts, filename=None):
    res_list = []

    for k in range(len(outcome_labels)):
        try:
            res_list.append(counts[outcome_labels[k]])
        except KeyError:
            res_list.append(0)

    plt.bar(outcome_labels, res_list, width = 0.8/len(outcome_labels))
    plt.grid()
    plt.ylabel("Measurement probability")
    plt.xlabel("QuantumVariable value")

    if filename:
        plt.savefig(filename, dpi=400, bbox_inches="tight")
    else:
        plt.show()


from jax import tree_util
from qrisp.jax.tracing_quantum_session import get_tracing_qs
from builtins import id


def flatten_qv(qv):
    # return the tracers and auxiliary data (structure of the object)
    children = (qv.reg,)
    aux_data = (id(qv), qv.name)  # No auxiliary data in this simple example
    return children, aux_data

def unflatten_qv(aux_data, children):
    # reconstruct the object from children and auxiliary data
    res = QuantumVariable.__new__(QuantumVariable)
    
    res.reg = children[0]
    res.name = aux_data[1]
    res.qs = get_tracing_qs(check_validity = False)
    
    return res

# Register as a PyTree with JAX
tree_util.register_pytree_node(QuantumVariable, flatten_qv, unflatten_qv)