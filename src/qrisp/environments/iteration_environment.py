# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:37:05 2023

@author: sea
"""

from qrisp.environments import QuantumEnvironment, GateWrapEnvironment
from qrisp.core.quantum_variable import QuantumVariable
from qrisp.core.compilation import qompiler
from qrisp.misc.utility import retarget_instructions
from qrisp.circuit import QubitAlloc, transpile


class IterationEnvironment(QuantumEnvironment):
    """
    This QuantumEnvironment can be used for reducing bottlenecks in compilation time. 
    Many algorithms such as Grover or QPE require repeated execution of the same
    quantum circuit. When scaling up complex algorithms that perform a lot of 
    non-trivial logic many iterations can significantly slow down the compilation
    speed. The ``IterationEnvironment`` remedies this flaw by recording the circuit
    of a single iteration and then duplicating the instructions the required 
    amount of times.

    Another bottleneck that can appear is the :meth:`compile <qrisp.QuantumSession.compile>` 
    method as the qubit allocation algorithm can also scale bad for really large
    algorithms. For this problem the ``IterationEnvironment`` exposes the ``precompile``
    keyword. Setting this keyword to ``True`` will perform the Qubit allocation algorithm
    on the QuantumEnvironments content and then (if necessary) allocate another
    :ref:`QuantumVariable` to accomodate the workspace qubits of the compilation 
    result. This way there is only a single (de)allocation per 
    ``IterationEnvironment``.

    .. note::

        Code that is executed within a ``IterationEnvironment`` may not 
        :meth:`delete <qrisp.QuantumVariable.delete>`
        previously created ``QuantumVariable`` and every created ``QuantumVariable``
        inside this :ref:`QuantumEnvironment` has to be deleted before exit.

    Parameters
    ----------

    qs : QuantumSession
        The ``QuantumSession`` in which the iterated code should be performed.
        QuantumVariables that have been created outside this ``QuantumEnvironment``
        need to be :ref:`registered <SessionMerging>` in this ``QuantumSession``.
    iteration_amount : integer
        The amount of iterations to perform.
    precompile : bool
        If set to ``True``, the qubit allocation algorithm will be run, once this
        ``QuantumEnvironment`` is compiled. This can significantly reduce the
        workload for a later compile call as there is only a single allocation
        per ``IterationEnvironment`` to handle. The default is False.

    Examples
    --------

    We perform a simple addition circuit multiple times:

    ::

        from qrisp import QuantumFloat, IterationEnvironment

        a = QuantumFloat(5)

        with IterationEnvironment(a.qs, iteration_amount = 10):

            a += 1

    Evaluate the result

    >>> print(a)
    {10: 1.0}

    **Precompilation**

    Squaring a :ref:`QuantumFloat` uses the :meth:`qrisp.sbp_mult` function,
    which has a high demand of ancilla qubits.
    Therefore many iterations can quickly overload the allocation algorithm.

    ::

        def benchmark_function(use_iter_env = False):
            qf = QuantumFloat(5)

            if use_iter_env:

                with IterationEnvironment(qf.qs, 20, precompile = True):

                    temp = qf*qf
                    temp.uncompute()
            else:

                for i in range(20):

                    temp = qf*qf
                    temp.uncompute()

            compiled_qc = qf.qs.compile()


    Benchmark results:

    ::

        import time

        start_time = time.time()
        benchmark_function(False)
        print("Time taken without IterationEnvironment: ", time.time() - start_time)
        #Takes 55s

        start_time = time.time()
        benchmark_function(True)
        print("Time taken with IterationEnvironment: ", time.time() - start_time)
        #Takes 6s

    """

    def __init__(self, qs, iteration_amount, precompile=False):

        if iteration_amount < 1:
            raise Exception(
                "Tried to create IterationEnvironment with < 1 iterations")

        self.iteration_amount = iteration_amount
        self.precompile = precompile

        QuantumEnvironment.__init__(self)

        self.env_qs = qs

        # Manual allocation management = False means that the compile function
        # pulls out all allocation gates to the front and all deallocation gates
        # to the back. This way the user doesn't have to worry about the (sometimes)
        # complicated logic of arranging them.
        # In this case we enable manual allocation management because the compilation
        # function is simple enough that we can ignore the allocation logic.
        self.manual_allocation_management = True

    def __enter__(self):

        self.inital_qvs = set(self.env_qs.qv_list)

        QuantumEnvironment.__enter__(self)

    def __exit__(self, exception_type, exception_value, traceback):

        if set(self.env_qs.qv_list) != self.inital_qvs:

            if exception_value is None:
                raise Exception(
                    "Tried to invoke IterationEnvironment with code creating/deleting QuantumVariables")

        QuantumEnvironment.__exit__(
            self, exception_type, exception_value, traceback)

    def compile(self):

        # Stow away the environment data to facicility environment compilation
        temp_qs_data = list(self.env_qs.data)
        self.env_qs.data = []

        # If the code executed in the Iteration environment contains many
        # (de)allocations the allocation algorithm in the compile method can
        # be a bottleneck.
        # The precompilation feature calls the compile method on the quantum session
        # and appends the compiled data instead. This way there is only a single
        # (de)allocation for an arbitrary amount of iterations.
        # This comes at the cost that the allocation algorithm might find better
        # ways if it has insight into the internal allocation structure.
        if self.precompile:

            # Compile the quantum environment to retrieve the compiled data
            QuantumEnvironment.compile(self)
            
            compiled_data = list(self.env_qs.data)
            
            self.env_qs.data = []
            
            # The idea is now to create a new quantum session, convert the collected
            # data to this quantum session and compile this quantum session.
            # This gives us a quantum circuit whose data we again convert to the
            # original QuantumSession.
            
            anc_qv = QuantumVariable(len(self.env_qs.qubits))
            
            translation_dic = {self.env_qs.qubits[i] : anc_qv[i] for i in range(len(anc_qv))}

            anc_qv.qs.data = []
            # We append the previously executed allocation calls such that
            # the compile method knows which variables are allocated
            for qb in self.env_qs.qubits:
                if qb.allocated:
                    anc_qv.qs.append(QubitAlloc(), [translation_dic[qb]])
            
            anc_qv.qs.barrier()
            # Convert the compiled data to the new quantum session
            retarget_instructions(compiled_data, self.env_qs.qubits, anc_qv.reg)
            
            # Append the data to the new QuantumSession
            anc_qv.qs.data.extend(compiled_data)
            
            compiled_qc = qompiler(anc_qv.qs, cancel_qfts = False, use_dirty_anc_for_mcx_recomp = False)
            
            # Remove previously added allocation calls from the compiled quantum circuit
            compiled_data = []
            for instr in compiled_qc.data:
                if "alloc" not in instr.op.name or instr.op.name == "barrier":
                    compiled_data.append(instr)

            # Retarget the resulting instructions from the compilation
            retarget_instructions(compiled_data, anc_qv.reg, self.env_qs.qubits)

            # Reinstate the original data
            self.env_qs.data = temp_qs_data
            
            # We now need to locate ancilla qubits of the compilation result and
            # create a new ancilla variable to hold these qubits
            
            # Determine the workspace qubits from the compiled qc
            workspace_qubits = list(
                set(compiled_qc.qubits) - set(anc_qv.reg))
            
            if len(workspace_qubits):
                # Allocate a QuantumVariable that will hold the workspace
                workspace_var = QuantumVariable(
                    len(workspace_qubits), qs=self.env_qs, name="workspace_var*")
            else:
                workspace_var = []

            # We now prepare the qubit lists for the retarget_instructions function

            source_qubits = workspace_qubits
            target_qubits = list(workspace_var)

            # Perform instruction retargeting
            retarget_instructions(compiled_data, source_qubits, target_qubits)

            # Perform iterated instruction execution
            for i in range(self.iteration_amount):
                compiled_data = [instr.copy() for instr in compiled_data]
                self.env_qs.data.extend(compiled_data)

            # Delete workspace variable
            if isinstance(workspace_var, QuantumVariable):
                workspace_var.delete()

        # The non-precompiled case is much simpler
        else:

            QuantumEnvironment.compile(self)

            compiled_data = list(self.env_qs.data)

            self.env_qs.data = temp_qs_data

            for i in range(self.iteration_amount):
                compiled_data = [instr.copy() for instr in compiled_data]
                self.env_qs.data.extend(compiled_data)
