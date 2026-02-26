"""
********************************************************************************
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
"""

# Abstract class to describe environments
# The idea is to start dumping the circuit data into a container
# (the .env_data attribute) once the environment is entered.
# Meanwhile, the original data of the session is stored in the .original_data attribute.
# The dumping is ended when the environment is left or a child environment is entered.
# The child environment will append itself into the container, once it is left.
# As soon as the most outer environment is left all the circuit data will be compiled.
# The compilation process is specified by the inheritor of this class.
# For instance conditional environments will turn every quantum operation
# that took place in that environment into it's controlled version.

# Another level of complexity is introduced by handling the quantum sessions.
# The problem here is that we want (almost) all quantum sessions,
# that operate inside the environment, to be merged into the environment session.
# The sessions we dont want merged are the ones that get CREATED inside the environment.
# This situtation represents the case that we have a foreign function from a module,
# which creates some gate objects, using a quantum session. If we merge this quantum
# session into our environment, the foreign function no longer works as intended.
# The strategy for solving this kind of problem is now:

# 1. At environment entry, all currently live quantum sessions are logged and the
# environments very own QuantumSession (.env_qs) is created. This QuantumEnvironment
# appends itself to the environment stack of these QuantumSessions.

# 2. Once the append method of a QuantumSession is executed, the environment stack of
# this QuantumSession is checked for Environment where the env_qs QuantumSession
# is not merged into. If this is the case, the QuantumSession's data will be transfered
# into the original_data attribute of the oldest env_qs environment
# that has not been merged into self.

# 3. At environment exit, every quantum session, that operated inside this environment
# is merged together.

# 4. Apart from the (de)allocation gates, all the collected data is stored inside
# the .env_data attribute

from contextvars import ContextVar

from qrisp.circuit import QubitAlloc, QubitDealloc, fast_append
from qrisp.core.quantum_session import QuantumSession
from qrisp.jasp import AbstractQuantumState, QuantumPrimitive, TracingQuantumSession


class QuantumEnvironment(QuantumPrimitive):
    """

    QuantumEnvironments are blocks of code that undergo a user-specified compilation
    process. They can be entered using the ``with`` statement:

    ::

       from qrisp import QuantumEnvironment, QuantumVariable, x

       qv = QuantumVariable(5)

       with QuantumEnvironment():
          x(qv)

    In this case, no special compilation is applied since the base class simply
    returns its content:

    >>> print(qv.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
              ┌───┐
        qv.0: ┤ X ├
              ├───┤
        qv.1: ┤ X ├
              ├───┤
        qv.2: ┤ X ├
              ├───┤
        qv.3: ┤ X ├
              ├───┤
        qv.4: ┤ X ├
              └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv

    More advanced environments offer a variety of features that can significantly
    simplify code development and maintenance.

    The most important built-in QuantumEnvironments are:

    * :ref:`ConditionEnvironment`
    * :ref:`ControlEnvironment`
    * :ref:`InversionEnvironment`
    * :ref:`GateWrapEnvironment`

    Due to sophisticated condition evaluation of nested :ref:`conditionenvironment` and
    :ref:`controlenvironment`, using QuantumEnvironments can even bring an increase in
    performance, compared to the `control method
    <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.control.html>`_
    which is commonly implemented by QuantumCircuit-based approaches.

    **Uncomputation within QuantumEnvironments**

    Uncomputation via the :meth:`uncompute <qrisp.QuantumVariable.uncompute>` method is
    possible only if the :ref:`QuantumVariable` has been created within the same or a
    sub-environment:

    ::

        from qrisp import QuantumVariable, QuantumEnvironment, cx

        a = QuantumVariable(1)

        with QuantumEnvironment():

            b = QuantumVariable(1)

            cx(a, b)

            with QuantumEnvironment():

                c = QuantumVariable(1)

                cx(b, c)

            c.uncompute()   # works because c was created in a sub-environment
            b.uncompute()   # works because b was created in the same environment
            # a.uncompute() # doesn't work because a was created outside this environment.


    >>> print(a.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
        a.0: ──■──────────────■──
             ┌─┴─┐          ┌─┴─┐
        b.0: ┤ X ├──■────■──┤ X ├
             └───┘┌─┴─┐┌─┴─┐└───┘
        c.0: ─────┤ X ├┤ X ├─────
                  └───┘└───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable a

    **Visualization within QuantumEnvironments**

    Calling ``print`` on a :ref:`QuantumSession` inside a QuantumEnvironment displays
    only instructions performed within that environment:

    ::

        from qrisp import x, y, z
        a = QuantumVariable(3)

        x(a[0])

        with QuantumEnvironment():

            y(a[1])

            with QuantumEnvironment():

                z(a[2])

                print(a.qs)

            print(a.qs)

        print(a.qs)

    Executing this snippet yields

    .. code-block:: none

        QuantumCircuit:
        --------------
        a.0: ─────

        a.1: ─────
             ┌───┐
        a.2: ┤ Z ├
             └───┘
        QuantumEnvironment Stack:
        ------------------------
        Level 0: QuantumEnvironment
        Level 1: QuantumEnvironment

        Live QuantumVariables:
        ---------------------
        QuantumVariable a
        QuantumCircuit:
        --------------
        a.0: ─────
             ┌───┐
        a.1: ┤ Y ├
             ├───┤
        a.2: ┤ Z ├
             └───┘
        QuantumEnvironment Stack:
        ------------------------
        Level 0: QuantumEnvironment

        Live QuantumVariables:
        ---------------------
        QuantumVariable a
        QuantumCircuit:
        --------------
             ┌───┐
        a.0: ┤ X ├
             ├───┤
        a.1: ┤ Y ├
             ├───┤
        a.2: ┤ Z ├
             └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable a


    .. warning::

        Calling ``print`` within a QuantumEnvironment causes all sub-environments to be
        compiled. While this doesn't change the semantics of the resulting circuit,
        especially nested :ref:`Condition <conditionenvironment>`- and
        :ref:`ControlEnvironments <controlenvironment>` lose a lot of efficiency if
        compiled prematurely. Therefore, ``print``-calls within QuantumEnvironments are
        usefull for debugging purposes but should be removed, if efficiency is a
        concern.


    **Creating custom QuantumEnvironments**

    More interesting QuantumEnvironments can be created by inheriting and modifying
    the compile method. In the following code snippet, we will demonstrate how to
    set up a QuantumEnvironment that skips every second instruction. We do this
    by inheriting from the QuantumEnvironment class. This will provide us with
    the necessary attributes for writing the compile method:

    * ``.env_data``: List of instructions appended in this environment. Child
      environments append themselves here upon exiting.
    * ``.env_qs``: QuantumSession where all QuantumVariables operating inside
      this environment are registered.

    The ``compile`` method is then called once all environments of ``.env_qs`` have been
    exited. Note that this doesn't necessarily imply that all QuantumEnvironments have
    been left. For more information about the interplay between QuantumSessions and
    QuantumEnvironments check the :ref:`session merging <SessionMerging>` documentation.

    ::

       class ExampleEnvironment(QuantumEnvironment):

          def compile(self):

             for idx, instruction in enumerate(self.env_data):

                if idx % 2:
                   continue

                # If the instruction is an environment, we compile this environment
                if isinstance(instruction, QuantumEnvironment):
                   instruction.compile()
                # Otherwise we append
                else:
                    self.env_qs.append(instruction)

    Check the result: ::

       from qrisp import x, y, z, t, s, h
       qv = QuantumVariable(6)

       with ExampleEnvironment():
           x(qv[0])
           y(qv[1])
           with ExampleEnvironment():
               z(qv[2])
               t(qv[3])
           with ExampleEnvironment():
               s(qv[4])
           h(qv[5])

    >>> print(qv.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
              ┌───┐
        qv.0: ┤ X ├
              └───┘
        qv.1: ─────
              ┌───┐
        qv.2: ┤ Z ├
              └───┘
        qv.3: ─────

        qv.4: ─────
              ┌───┐
        qv.5: ┤ H ├
              └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv

    """

    _deepest_environment: ContextVar["QuantumEnvironment | None"] = ContextVar(
        "deepest_q_env",
        default=None,
    )

    def __init__(self, env_args=None) -> None:
        """Initialize a new quantum environment."""

        QuantumPrimitive.__init__(self, name="q_env")
        self.multiple_results = True

        @self.def_abstract_eval
        def _(*_, **__):
            """Abstract evaluation of the primitive.

            This function does not need to be JAX traceable. It will be invoked with
            abstractions of the actual arguments.
            """

            return (AbstractQuantumState(),)

        env_args = [] if env_args is None else env_args
        self.env_args = env_args

    # The methods to start the dumping process for this environment
    # The dumping basically consists of copying the original data into a temporary
    # container (here the list .original_data) and then clearing the data of
    # the quantum session. Once the dumping ends the data which has been appended
    # in the meantime is appended to the environments' data list .env_data and
    # the original circuit data is reinstated
    def _start_dumping(self) -> None:
        """Start dumping circuit data into the environment."""

        qs = self.env_qs

        # Temporarily store the qs circuit data
        self.original_data += qs.data

        # Clear the qs circuit data to collect what is coming
        qs.clear_data()

        qs.data.extend(self.env_data)

        self.env_data = []

    def _stop_dumping(self) -> None:
        """Stop dumping circuit data into the environment."""

        qs = self.env_qs

        # Collect circuit data into the environment data list
        self.env_data += qs.data

        qs.clear_data()

        # Reinstate original circuit data
        qs.data.extend(self.original_data)

        self.original_data = []

    def __enter__(self) -> "QuantumEnvironment | None":
        """Enter the quantum environment."""

        from qrisp.jasp import check_for_tracing_mode

        if check_for_tracing_mode():
            tr_qs = TracingQuantumSession.get_instance()
            self.temp_qubit_cache = tr_qs.qubit_cache
            tr_qs.qubit_cache = {}
            tr_qs.abs_qst = self.bind(
                *(self.env_args + [tr_qs.abs_qst]),
                stage="enter",
                type=str(type(self)).rsplit(".", maxsplit=1)[-1][:-2],
            )[0]
            return None

        # The QuantumSessions operating inside this environment will be merged
        # into this QuantumSession
        self.env_qs = QuantumSession()

        # This list stores the original data of the quantum session tracked
        self.original_data = []

        # This list stores the data that is appended inside the environment
        self.env_data = []

        # This list stores the qubits that have been deallocated in this environment
        # This information is required because they need to be temporarily reallocated
        # to prevent compilation errors at compile time.
        self.deallocated_qubits = []

        # Set the new relationships and
        # store a token so we can restore the previous value reliably in __exit__
        self.parent = self._deepest_environment.get()
        self._deepest_env_token = self._deepest_environment.set(self)

        # Acquire a list of all active quantum sessions
        self.active_qs_list = QuantumSession.get_active_quantum_sessions()
        for qs_ref in self.active_qs_list:
            qs = qs_ref()
            if qs is not None:
                qs.env_stack.append(self)

        self._start_dumping()

        # Manual allocation management means that the compile method can process allocation
        # and deallocation gates.
        # If set to False, these gates will be filtered out of the env_data attribute before
        # compile is called.
        # In this case, the (de)allocation gates that happened inside this environment
        # will be collected and execute before (after) the compile method is called.
        self.manual_allocation_management = isinstance(self, QuantumEnvironment)

        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Exit the quantum environment."""

        from qrisp.jasp import check_for_tracing_mode

        if check_for_tracing_mode():

            if exception_value:
                raise exception_value

            abs_qs = TracingQuantumSession.get_instance()
            abs_qs.qubit_cache = self.temp_qubit_cache
            abs_qs.abs_qst = self.bind(
                abs_qs.abs_qst,
                stage="exit",
                type=str(type(self)).rsplit(".", maxsplit=1)[-1][:-2],
            )[0]
            return None

        try:
            self._deepest_environment.reset(self._deepest_env_token)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to reset the QuantumEnvironment. "
                "This may indicate improper nesting or re-entrance of QuantumEnvironments."
            ) from e

        self._stop_dumping()

        for qs_ref in self.active_qs_list:
            qs = qs_ref()
            if qs is not None:
                qs.env_stack.pop(-1)

        if exception_value:
            raise exception_value

        # Create lists to store allocation/deallocation gates
        dealloc_qubit_list = []
        alloc_qubit_list = []

        manual_allocation = getattr(self, "manual_allocation_management", False)

        # We now iterate through the collected data. We do this in order
        # to make sure that the (de)allocation gates are not processed
        # by the environment compiler as this might disturb their functionality
        idx = 0
        while idx < len(self.env_data):

            instr = self.env_data[idx]

            if isinstance(instr, QuantumEnvironment):
                idx += 1
                continue

            op_name = instr.op.name

            if op_name == "qb_alloc":
                if not manual_allocation:
                    alloc_qubit_list.append(self.env_data.pop(idx).qubits[0])
                    continue
                alloc_qubit_list.append(instr.qubits[0])

            elif op_name == "qb_dealloc":
                if not manual_allocation:
                    dealloc_qubit_list.append(self.env_data.pop(idx).qubits[0])
                    continue
                dealloc_qubit_list.append(instr.qubits[0])

            idx += 1

        if not manual_allocation:
            for qb in set(alloc_qubit_list):
                self.env_qs.append(QubitAlloc(), [qb])

        # If this was the outermost environment, we compile
        if len(self.env_qs.env_stack) == 0:
            self.deallocated_qubits.extend(dealloc_qubit_list)
            for qb in self.deallocated_qubits:
                qb.allocated = True

            with fast_append(3):
                self.compile()

        # Otherwise, we append self to the data of the parent environment
        else:
            if self.env_data:
                self.env_qs.data.append(self)
            self.parent.deallocated_qubits.extend(dealloc_qubit_list)

        # Append deallocation gates after compilation
        if not manual_allocation:
            for qb in set(dealloc_qubit_list):
                self.env_qs.append(QubitDealloc(), [qb])

    def compile(self) -> None:
        """
        Default compilation method to compile the quantum environment.

        It simply compiles all sub-environments and the collected data to the session.
        """

        for instruction in self.env_data:

            if isinstance(instruction, QuantumEnvironment):
                instruction.compile()

            else:
                self.env_qs.append(instruction)

    def jcompile(self, eqn, context_dic) -> None:
        """
        Default jasp compilation method to compile the quantum environment.

        It simply compiles all sub-environments and the collected data to the session.
        """

        from qrisp.jasp import eval_jaxpr, extract_invalues, insert_outvalues

        args = extract_invalues(eqn, context_dic)
        flattened_envs = eqn.params["jaspr"].flatten_environments()

        res = eval_jaxpr(flattened_envs)(*args)
        res = (res,) if not isinstance(res, tuple) else res

        insert_outvalues(eqn, context_dic, res)
