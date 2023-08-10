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

from qrisp.core.library import mcx, p, rz, x, z
from qrisp.core.quantum_variable import QuantumVariable
from qrisp.core.session_merging_tools import merge
from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.environments.quantum_inversion import invert
from qrisp.misc import (
    find_calling_line,
    perm_lock,
    perm_unlock,
    redirect_qfunction,
    unlock,
)


def quantum_condition(function):
    def q_env_generator(*args, **kwargs):
        return ConditionEnvironment(function, args, kwargs=kwargs)

    return q_env_generator


# Class to describe if-environments
# For more information on how Environments work, check the QuantumEnvironment class.
# This class is instantiated with a function returning a qubit,
# which indicates wther the condition is met.
# On entering this environment, the function will be evaluated and the content
# (i.e. the quantum operations) will be collected.
# When compiled, this environment will check for parent conditional environments and
# include their truth value into the compilation process. Subsequently, every operation
# that happened inside this environment is turned into it's controlled version.
# Finally, the function evaluating the truth value will be uncomputed and
# the environment is reset for it's next use.
class ConditionEnvironment(QuantumEnvironment):
    r"""

    This class enables the usage of *if*-conditionals as we are used to from
    classical programming: ::

        from qrisp import QuantumChar, QuantumFloat, h, multi_measurement

        q_ch = QuantumChar()
        qf = QuantumFloat(3, signed = True)

        h(q_ch[0])

        with q_ch == "a":
            qf += 2


    >>> print(multi_measurement([q_ch,qf]))
    {('a', 2): 0.5, ('b', 0): 0.5}

    In this code snippet, we first bring the :ref:`QuantumChar` ``q_ch`` into the
    superposition

    .. math::

        \ket{\text{q_ch}} = \frac{1}{\sqrt{2}} \left( \ket{\text{"a"}}
        + \ket{\text{"b"}} \right)

    After that, we enter the ConditionEnvironment, which controls the operations
    on the condition that ``q_ch`` is in state $\ket{a}$. Finally, we simultaneously
    measure both :ref:`QuantumVariables <QuantumVariable>`. We see that the
    incrementation of ``qf`` only occured on the branch, where ``q_ch`` is equal to the
    character ``"a"``. The resulting quantum state is:

    .. math::

        \ket{\psi} = \frac{1}{\sqrt{2}} \left( \ket{\text{"a"}}\ket{2}
        + \ket{\text{"b"}}\ket{0} \right)

    It is furthermore possible to invert the condition truth value or apply phases.
    For this we acquire the :ref:`QuantumBool` containing the truth value using the *as*
    statement ::

        from qrisp import x, p
        import numpy as np

        with q_ch == "a" as cond_bool:
            qf += 2
            cond_bool.flip()
            qf -= 2
            p(np.pi/4, cond_bool)

    >>> qf.qs.statevector()
    sqrt(2)*(|a>*|4> + exp(I*pi/4)*|b>*|-2>)/2

    **Constructing custom conditional environments**

    Apart from infix notation like the equality operator, Qrisp also provides an
    interface for creating custom conditonal environments.

    Parameters
    ----------
    cond_eval_function : function
        A function which evaluates the truth value. Must return a :ref:`QuantumBool`.
        Intermediate results do not have to be uncomputed or deleted, as this is
        automatically performed, when the condition truth value is uncomputed.
    args : list
        The arguments on which to evaluate.
    kwargs : dict, optional
        A dictionary of keyword arguments for ``cond_eval_function``. The default is {}.


    Examples
    --------

    We will now demonstrate how a ConditionEnvironment, that evaluates the equality of
    two :ref:`QuantumVariables <QuantumVariable>` can be constructed: ::

        from qrisp import QuantumBool, QuantumVariable, x, cx, mcx

        def quantum_eq(qv_0, qv_1):



            if qv_0.size != qv_1.size:
                raise Exception("Tried to evaluate equality condition for
                QuantumVariables of differing size")

            temp_qv = QuantumVariable(qv_0.size)

            cx(qv_0, temp_qv)
            cx(qv_1, temp_qv)
            x(temp_qv)

            res = QuantumBool()

            mcx(temp_qv, res)

            return res

    In this function, we create a temporary variable where we apply CX gates controlled
    on the inputs onto. The qubits where ``qv_0`` and ``qv_1`` agree, will then be in
    state $\ket{0}$. After this, we apply regular X gates onto ``temp_qv`` such that
    the qubits where the inputs agree, are in state $\ket{1}$. Finally,
    we apply a multi-controlled X gate onto the result to flip the result, if
    all of qubits of the qubits in ``temp_qv`` are in the $\ket{0}$ state.

    We inspect the resulting QuantumCircuit

    >>> from qrisp import QuantumChar
    >>> q_ch_0 = QuantumChar()
    >>> q_ch_1 = QuantumChar()
    >>> res_bool = quantum_eq(q_ch_0, q_ch_1)
    >>> print(q_ch_0.qs)
    QuantumCircuit:
    ---------------
     q_ch_0.0: ──■─────────────────────────────────────────────────────────
                 │
     q_ch_0.1: ──┼────■────────────────────────────────────────────────────
                 │    │
     q_ch_0.2: ──┼────┼────■───────────────────────────────────────────────
                 │    │    │
     q_ch_0.3: ──┼────┼────┼────■──────────────────────────────────────────
                 │    │    │    │
     q_ch_0.4: ──┼────┼────┼────┼────■─────────────────────────────────────
                 │    │    │    │    │
     q_ch_1.0: ──┼────┼────┼────┼────┼────■────────────────────────────────
                 │    │    │    │    │    │
     q_ch_1.1: ──┼────┼────┼────┼────┼────┼────■───────────────────────────
                 │    │    │    │    │    │    │
     q_ch_1.2: ──┼────┼────┼────┼────┼────┼────┼────■──────────────────────
                 │    │    │    │    │    │    │    │
     q_ch_1.3: ──┼────┼────┼────┼────┼────┼────┼────┼────■─────────────────
                 │    │    │    │    │    │    │    │    │
     q_ch_1.4: ──┼────┼────┼────┼────┼────┼────┼────┼────┼────■────────────
               ┌─┴─┐  │    │    │    │  ┌─┴─┐  │    │    │    │  ┌───┐
    temp_qv.0: ┤ X ├──┼────┼────┼────┼──┤ X ├──┼────┼────┼────┼──┤ X ├──■──
               └───┘┌─┴─┐  │    │    │  └───┘┌─┴─┐  │    │    │  ├───┤  │
    temp_qv.1: ─────┤ X ├──┼────┼────┼───────┤ X ├──┼────┼────┼──┤ X ├──■──
                    └───┘┌─┴─┐  │    │       └───┘┌─┴─┐  │    │  ├───┤  │
    temp_qv.2: ──────────┤ X ├──┼────┼────────────┤ X ├──┼────┼──┤ X ├──■──
                         └───┘┌─┴─┐  │            └───┘┌─┴─┐  │  ├───┤  │
    temp_qv.3: ───────────────┤ X ├──┼─────────────────┤ X ├──┼──┤ X ├──■──
                              └───┘┌─┴─┐               └───┘┌─┴─┐├───┤  │
    temp_qv.4: ────────────────────┤ X ├────────────────────┤ X ├┤ X ├──■──
                                   └───┘                    └───┘└───┘┌─┴─┐
        res.0: ───────────────────────────────────────────────────────┤ X ├
                                                                      └───┘
    Live QuantumVariables:
    ----------------------
    QuantumChar q_ch_0
    QuantumChar q_ch_1
    QuantumVariable temp_qv
    QuantumBool res

    We can now construct the conditional environment from this function ::

        from qrisp import ConditionEnvironment, multi_measurement, h

        #Create some sample arguments on which to evaluate the condition

        q_bool_0 = QuantumBool()
        q_bool_1 = QuantumBool()
        q_bool_2 = QuantumBool()

        h(q_bool_0)

        with ConditionEnvironment(cond_eval_function = quantum_eq,
                                  args = [q_bool_0, q_bool_1]):
            q_bool_2.flip()

    >>> print(multi_measurement([q_bool_0, q_bool_1, q_bool_2]))
    {(False, False, True): 0.5, (True, False, False): 0.5}

    This agrees with our expectation, that ``q_bool_2`` is only ``True`` if the other
    two agree.

    **The quantum_condition decorator**

    Creating quantum conditions like this seems a bit unwieldy.
    For a more convenient solution, we provide the ``quantum_condition`` decorator.
    This decorator can be applied to a function returning a :ref:`QuantumBool`,
    which is then returning the corresponding ConditionEnvironment instead.
    To demonstrate, we construct a "less than" condition for QuantumFloats ::

        from qrisp import quantum_condition, cx

        @quantum_condition
        def less_than(qf_0, qf_1):

            temp_qf = qf_0 - qf_1

            res = QuantumBool()

            cx(temp_qf.sign(), res)

            return res

        qf_0 = QuantumFloat(3)
        qf_1 = qf_0.duplicate()

        qf_0[:] = 2
        h(qf_1[:2])

        res_q_bool = QuantumBool()

        with less_than(qf_0, qf_1):
            res_q_bool.flip()

    >>> print(multi_measurement([qf_0, qf_1, res_q_bool]))
    {(2, 0, False): 0.25, (2, 1, False): 0.25, (2, 2, False): 0.25, (2, 3, True): 0.25}


    **Quantum-Loops**

    An interesting application of conditional environments is the ``qRange`` iterator.
    Using this construct, we can mimic a loop as we are used from classical computing,
    where the end of the loop is determined by a quantum state: ::

        from qrisp import QuantumFloat, qRange, h

        n = QuantumFloat(3)
        qf = QuantumFloat(5)

        n[:] = 4

        h(n[0])

        n_results = n.get_measurement()

        for i in qRange(n):
            qf += i


    >>> print(qf)
    {10: 0.5, 15: 0.5}

    This script calculates the sum of all integers up to a certain threshold.
    The threshold (n) is a :ref:`QuantumFloat` in superposition, implying the result of
    the sum is also in a superposition. The expected results can be quickly determined
    by using Gauß's formula:

    .. math::

        \sum_{i = 0}^n i = \frac{n(n+1)}{2}


    >>> print("Excpected outcomes:", [n*(n+1)/2 for n in n_results.keys()])
    Excpected outcomes: [10.0, 15.0]

    """

    # Constructor of the class

    def __init__(self, cond_eval_function, args, kwargs={}):
        # The function which evaluates the condition - should return a QuantumBool
        self.cond_eval_function = cond_eval_function

        # Save the arguments on which the function should be evaluated
        self.args = args

        # Save the keyword arguments
        self.kwargs = kwargs

        # Note the QuantumSession of the arguments of the arguments

        self.arg_qs = merge(args)

        self.manual_allocation_management = True

    # Method to enter the environment
    def __enter__(self):
        from qrisp.qtypes.quantum_bool import QuantumBool

        # For more information on why this attribute is neccessary check the comment
        # on the line containing subcondition_truth_values = []
        self.sub_condition_envs = []

        self.qbool = QuantumBool(name="cond_env*", qs=self.arg_qs)
        self.condition_truth_value = self.qbool[0]

        super().__enter__()

        merge(self.env_qs, self.arg_qs)
        return self.qbool

    def __exit__(self, exception_type, exception_value, traceback):
        from qrisp.environments import ControlEnvironment, InversionEnvironment

        # We determine the parent condition environment
        self.parent_cond_env = None

        QuantumEnvironment.__exit__(self, exception_type, exception_value, traceback)

        for env in self.env_qs.env_stack[::-1]:
            if isinstance(env, (ConditionEnvironment, ControlEnvironment)):
                self.parent_cond_env = env
                break
            if not isinstance(env, (QuantumEnvironment, InversionEnvironment)):
                break

    # Compile method
    def compile(self):
        from qrisp.environments import ControlEnvironment
        from qrisp.qtypes.quantum_bool import QuantumBool

        # Create the quantum variable where the condition truth value should be saved
        # Incase we have a parent environment we create two qubits because
        # we use the second qubit to compute the toffoli of this one and the parent
        # environments truth value in order to not have the environment operations
        # controlled on two qubits
        if len(self.env_data) == 0:
            self.qbool.delete()
        else:
            # The first step we have to perform is calculating the truth value of
            # this environments quantum condition. For this we differentiate between
            # the case that this condition is embedded in another condition or not

            if self.parent_cond_env is not None:
                # In the parent case we also need to make sure that the code is executed
                # if the parent environment is executed. For this a possible approach
                # would be to control the content on both, the parent and
                # the chield truth value. However, for each nesting level the gate count
                # to generate the controlled-controlled-controlled... version
                # of the gates inside the environment increases exponentially.
                # Because of this we compute the toffoli of the parent and
                # child truth value and control the environment gates on this qubit.

                cond_eval_bool = self.cond_eval_function(*self.args, **self.kwargs)

                if not isinstance(cond_eval_bool, QuantumBool):
                    raise Exception(
                        "Tried to compile QuantumCondition environment with"
                        "a condition evaluation function not returning a QuantumBool"
                    )

                # Create and execute phase tolerant toffoli gate
                toffoli_qb_list = [
                    self.parent_cond_env.condition_truth_value,
                    cond_eval_bool,
                ]

                from qrisp import mcx

                mcx(toffoli_qb_list, self.condition_truth_value, method="gray_pt")

            else:
                # Without any parent environment we can simply synhesize the
                # truth value of the quantum condition into it's qubit

                redirect_qfunction(self.cond_eval_function)(
                    *self.args, target=self.qbool, **self.kwargs
                )

                if isinstance(self.env_qs.data[-1], QuantumEnvironment):
                    env = self.env_qs.data.pop(-1)
                    env.compile()

                cond_eval_bool = self.qbool

            from qrisp import recursive_qv_search

            perm_lock(recursive_qv_search(self.args))

            unlock(self.condition_truth_value)

            inversion_tracker = 1

            # This list will contain the qubits holding the truth values of
            # conditional/control environments within this environment. The instruction
            # from the subcondition environments do not need to be controlled,
            # since their compile method compiles their condition truth value based
            # on the truth value of the parent environment.
            subcondition_truth_values = [
                env.condition_truth_value for env in self.sub_condition_envs
            ]

            # Now we need to recover the instructions from the data list and
            # perform their controlled version on the condition_truth_value qubit
            while self.env_data:
                instruction = self.env_data.pop(0)

                # If the instruction == conditional environment, compile the environment
                if isinstance(instruction, (ControlEnvironment, ConditionEnvironment)):
                    instruction.compile()

                    subcondition_truth_values = [
                        env.condition_truth_value for env in self.sub_condition_envs
                    ]
                    continue

                # If the instruction == general environment, compile the instruction and
                # add the compilation result to the list of instructions that need to be
                # conditionally executed
                elif issubclass(instruction.__class__, QuantumEnvironment):
                    temp_data_list = list(self.env_qs.data)
                    self.env_qs.clear_data()
                    instruction.compile()
                    self.env_data = list(self.env_qs.data) + self.env_data
                    self.env_qs.clear_data()
                    self.env_qs.data.extend(temp_data_list)

                    subcondition_truth_values = [
                        env.condition_truth_value for env in self.sub_condition_envs
                    ]
                    continue

                if instruction.op.name in ["qb_alloc", "qb_dealloc"]:
                    self.env_qs.append(instruction)
                    continue

                if set(instruction.qubits).intersection(subcondition_truth_values):
                    self.env_qs.append(instruction)
                    continue

                # Support for inversion of a condition without opening a new environment
                if set(instruction.qubits).issubset([self.condition_truth_value]):
                    if instruction.op.name == "x":
                        inversion_tracker *= -1
                        perm_unlock(self.condition_truth_value)
                        x(self.condition_truth_value)
                        perm_lock(self.condition_truth_value)
                    elif instruction.op.name == "p":
                        p(instruction.op.params[0], self.condition_truth_value)
                    elif instruction.op.name == "rz":
                        rz(instruction.op.params[0], self.condition_truth_value)
                    elif instruction.op.name == "z":
                        z(self.condition_truth_value)
                    else:
                        raise Exception(
                            "Tried to perform invalid operations"
                            "on condition truth value (allowed are x, p, z, rz)"
                        )
                    continue

                # Create controlled instruction
                instruction.op = instruction.op.control(1)

                # Add condition truth value qubit to the instruction qubit list
                instruction.qubits = [self.condition_truth_value] + list(
                    instruction.qubits
                )
                # Append instruction
                self.env_qs.append(instruction)

            unlock(self.condition_truth_value)
            perm_unlock(self.condition_truth_value)
            if inversion_tracker == -1:
                x(self.condition_truth_value)

            perm_unlock(recursive_qv_search(self.args))

            # Uncompute truth values

            # If we had a parent environment, we first uncompute the
            # "actual truth value", i.e. the mcx of the parent and
            # this environments truth value
            if self.parent_cond_env is not None:
                self.parent_cond_env.sub_condition_envs.extend(
                    self.sub_condition_envs + [self]
                )
                mcx(toffoli_qb_list, self.qbool, method="gray_pt_inv")
                self.qbool.delete()

            # We now uncompute the environments' truth value

            # For this we can use the uncompute method, which will however not
            # recompute any intermediate values, therefore blocking alot of qubits
            # during execution. Especially in nested environments this can quickly
            # become a problem, because the blocked ancillae can not be reused
            # for further condition evaluations.

            # For the condition environment examples we saw an increase from 36 to 44
            # qubits without recomputation while only lowering the depth by about 25%

            # There might be cases where recomputation based uncomputation
            # is not worth it but for now, we leave it

            recompute = True
            # if not recompute:
            #     try:
            #         cond_eval_bool.uncompute()
            #     except:
            #         recompute = True

            if recompute:
                with invert():
                    redirected_qfunction = redirect_qfunction(self.cond_eval_function)
                    redirected_qfunction(
                        *self.args, target=cond_eval_bool, **self.kwargs
                    )

                if isinstance(self.env_qs.data[-1], QuantumEnvironment):
                    env = self.env_qs.data.pop(-1)
                    env.compile()
                
                cond_eval_bool.delete()


# This decorator allows to have conditional evaluations to return condition environments
# when called after a "with" statement but QuantumBools otherwise.

# from qrisp import QuantumFloat
# a = QuantumFloat(2)
# b = QuantumFloat(2)

# temp = (a == b) # this is a QuantumBool

# with temp: # QuantumBools have an enter method,
# but they use the less efficient control environment
#     pass

# with a == b: # This is a ConditionEnvironment entered
#     pass


# We do this mainly for performance reasons. If a QuantumBool is returned,
# it can also be entered, but it uses the less efficient (almost a factor 2)
# control Environment.

# The detection is based on the inspect module.
# This implies that the detection only works if the with statements is exactly
# two levels above the cond_eval_function call.
# That means, a class like QuantumFloat can call __eq__ and then the result
# of this decorator in order for this to work


def adaptive_condition(cond_eval_function):
    def new_cond_eval_function(*args, **kwargs):
        from qrisp import auto_uncompute

        calling_line = find_calling_line(2)

        uncomputed_function = auto_uncompute(cond_eval_function)

        if calling_line.split(" ")[0] == "with" and "&" not in calling_line and "|" not in calling_line and "~" not in calling_line:
            return quantum_condition(uncomputed_function)(*args, **kwargs)
        else:
            return uncomputed_function(*args, **kwargs)

    return new_cond_eval_function


@adaptive_condition
def q_eq(input_0, input_1):
    from qrisp import cx
    from qrisp.qtypes.quantum_bool import QuantumBool

    res = QuantumBool(name="eq_cond*")

    if isinstance(input_1, QuantumVariable):
        if input_0.size != input_1.size:
            raise Exception(
                "Tried to evaluate equality conditional"
                "for QuantumVariables of differing size"
            )

        temp_qv = QuantumVariable(input_0.size)
        cx(input_0, temp_qv)
        cx(input_1, temp_qv)
        x(temp_qv)
        mcx(temp_qv, res, method="gray_pt")

        return res

    else:
        label_int = input_0.encoder(input_1)

        mcx(input_0, res, ctrl_state=label_int)

        return res
