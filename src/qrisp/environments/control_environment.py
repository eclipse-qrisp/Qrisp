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

from jax.core import ShapedArray
from jax._src.array import ArrayImpl
from jax.extend.core import ClosedJaxpr

from qrisp.circuit import Qubit, QuantumCircuit, XGate
from qrisp.core.session_merging_tools import merge, merge_sessions, multi_session_merge
from qrisp.environments import QuantumEnvironment, ClControlEnvironment
from qrisp.misc import perm_lock, perm_unlock, bin_rep
from qrisp.jasp import check_for_tracing_mode, get_last_equation
from qrisp.core import mcx, p, rz, x

import numpy as np


class ControlEnvironment(QuantumEnvironment):
    """
    This class behaves similarly to ConditionEnvironment but instead of a function
    calculating a truth value, we supply a list of qubits.
    The environment's content is then controlled on these qubits.

    An alias for this QuantumEnvironment is "control".


    Parameters
    ----------
    ctrl_qubits : list[Qubit]
        A list of qubits on which to control the environment's content.
    ctrl_state : int/str, optional
        The computational basis state which is supposed to activate the environment.
        Can be supplied as a bitstring or integer. The default is "1111..".


    Examples
    --------

    We create a QuantumVariable and control on some of it's qubits
    using the control alias ::

        from qrisp import QuantumVariable, QuantumString, multi_measurement, control, h

        qv = QuantumVariable(3)
        q_str = QuantumString()

        qv[:] = "011"
        h(qv[0])

        with control(qv[:2], "11"):
            q_str += "hello world"


    >>> print(multi_measurement([qv, q_str]))
    {('011', 'aaaaaaaaaaa'): 0.5, ('111', 'hello world'): 0.5}


    """

    def __init__(self, ctrl_qubits, ctrl_state=-1, ctrl_method=None, invert=False):

        if isinstance(ctrl_state, int):
            if ctrl_state < 0:
                ctrl_state += 2 ** len(ctrl_qubits)

            self.ctrl_state = bin_rep(ctrl_state, len(ctrl_qubits))[::-1]
        else:
            self.ctrl_state = str(ctrl_state)

        if check_for_tracing_mode():

            QuantumEnvironment.__init__(self, list(ctrl_qubits))
            if not isinstance(ctrl_qubits, list):
                ctrl_qubits = [ctrl_qubits]

        else:

            if isinstance(ctrl_qubits, list):
                self.arg_qs = multi_session_merge([qb.qs() for qb in ctrl_qubits])
            else:
                self.arg_qs = ctrl_qubits.qs()
            self.arg_qs = merge(ctrl_qubits)

            self.ctrl_method = ctrl_method
            if isinstance(ctrl_qubits, Qubit):
                ctrl_qubits = [ctrl_qubits]

            self.ctrl_qubits = ctrl_qubits
            self.invert = invert

            self.manual_allocation_management = True

            # For more information on why this attribute is neccessary check the comment
            # on the line containing subcondition_truth_values = []
            self.sub_condition_envs = []

            QuantumEnvironment.__init__(self)

    # Method to enter the environment
    def __enter__(self):

        if check_for_tracing_mode():
            QuantumEnvironment.__enter__(self)
            return

        from qrisp.qtypes.quantum_bool import QuantumBool

        if len(self.ctrl_qubits) == 1:
            self.condition_truth_value = self.ctrl_qubits[0]
        else:
            self.qbool = QuantumBool(name="ctrl_env*", qs=self.arg_qs)
            self.condition_truth_value = self.qbool[0]

        QuantumEnvironment.__enter__(self)

        merge_sessions(self.env_qs, self.arg_qs)

        return self.condition_truth_value

    def __exit__(self, exception_type, exception_value, traceback):
        from qrisp.environments import (
            ConditionEnvironment,
            ControlEnvironment,
            InversionEnvironment,
        )

        if check_for_tracing_mode():
            QuantumEnvironment.__exit__(
                self, exception_type, exception_value, traceback
            )
            return
        self.parent_cond_env = None

        QuantumEnvironment.__exit__(self, exception_type, exception_value, traceback)

        from qrisp import ConjugationEnvironment

        # Determine the parent environment
        for env in self.env_qs.env_stack[::-1]:
            if isinstance(env, (ControlEnvironment, ConditionEnvironment)):
                self.parent_cond_env = env
                break
            if not isinstance(env, (InversionEnvironment, ConjugationEnvironment)):

                if not type(env) == QuantumEnvironment:
                    break

    def compile(self):
        from qrisp import QuantumBool
        from qrisp.environments import ConditionEnvironment, CustomControlOperation

        # Create the quantum variable where the condition truth value should be saved
        # Incase we have a parent environment we create two qubits because
        # we use the second qubit to compute the toffoli of this one and the parent
        # environments truth value in order to not have the environment operations
        # controlled on two qubits

        if len(self.env_data):
            # The first step we have to perform is calculating the truth value of the
            # environments quantum condition. For this we differentiate between
            # the case that this condition is embedded in another condition or not

            ctrl_qubits = list(self.ctrl_qubits)
            cond_compile_ctrl_state = self.ctrl_state

            if self.parent_cond_env is not None:

                # In the parent case we also need to make sure that the code is executed
                # if the parent environment is executed. A possible approach would be
                # to control the content on both, the parent and the chield truth value.
                # However, for each nesting level the gate count to generate
                # the controlled-controlled-controlled... version of the gates inside
                # the environment increases exponentially. Because of this we compute
                # the toffoli of the parent and child truth value
                # and controll the environment gates on this qubit.

                # Synthesize the condition of the environment
                # into the condition truth value qubit
                if len(ctrl_qubits) == 1:
                    from qrisp.misc import retarget_instructions

                    self.qbool = QuantumBool(name="ctrl_env*", qs=self.env_qs)
                    retarget_instructions(
                        self.env_data, [self.condition_truth_value], [self.qbool[0]]
                    )

                    if len(self.env_qs.data):
                        if isinstance(self.env_qs.data[-1], QuantumEnvironment):
                            env = self.env_qs.data.pop(-1)
                            env.compile()

                    self.condition_truth_value = self.qbool[0]

                ctrl_qubits.append(self.parent_cond_env.condition_truth_value)

                parent_ctrl_state = "1"

                if isinstance(self.parent_cond_env, ControlEnvironment):
                    if len(self.parent_cond_env.ctrl_qubits) == 1:
                        if self.parent_cond_env.ctrl_state == "0" and not hasattr(
                            self.parent_cond_env, "qbool"
                        ):
                            parent_ctrl_state = "0"

                    if self.parent_cond_env.invert:
                        if parent_ctrl_state == "0":
                            parent_ctrl_state = "1"
                        elif parent_ctrl_state == "1":
                            parent_ctrl_state = "0"

                cond_compile_ctrl_state = cond_compile_ctrl_state + parent_ctrl_state

            if len(ctrl_qubits) > 1:

                if len(ctrl_qubits) > 5:
                    method = "auto"
                else:
                    method = "gray_pt"

                mcx(
                    ctrl_qubits,
                    self.condition_truth_value,
                    ctrl_state=cond_compile_ctrl_state,
                    method=method,
                )

            perm_lock(ctrl_qubits)
            # unlock(self.condition_truth_value)

            # This list will contain the qubits holding the truth values of
            # conditional/control environments within this environment.
            # The instruction from the subcondition environments do not need to be
            # controlled, since their compile method compiles their condition
            # truth value based on the truth value of the parent environment.
            subcondition_truth_values = [
                env.condition_truth_value for env in self.sub_condition_envs
            ]
            inversion_tracker = 1

            # Now we need to recover the instructions from the data list
            # and perform their controlled version on the condition_truth_value qubit
            while self.env_data:
                instruction = self.env_data.pop(0)

                # If the instruction == conditional environment, compile the environment
                if isinstance(instruction, (ControlEnvironment, ConditionEnvironment)):
                    instruction.compile()

                    subcondition_truth_values = [
                        env.condition_truth_value for env in self.sub_condition_envs
                    ]
                    continue

                # If the instruction is a general environment, compile the instruction
                # and add the compilation result to the list of instructions
                # that need to be conditionally executed.
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

                if (
                    instruction.op.name in ["qb_alloc", "qb_dealloc"]
                    and instruction.qubits[0] != self.condition_truth_value
                ) or instruction.op.name == "barrier":
                    self.env_qs.append(instruction)
                    continue

                if set(instruction.qubits).intersection(subcondition_truth_values):
                    # REWORK required: Condition environments evaluate their condition
                    # and (if within another control/condition environment) combine
                    # this QuantumBool (via pt toffoli) with the superior condition.
                    # This "combining" is done, so the condition evaluation doesn't
                    # have to be controlled. However, this is not properly realized
                    # in this method. The condition evaluation could probably work
                    # better using the custom control feature.
                    self.env_qs.append(instruction)
                    continue

                # Support for inversion of the condition without opening a new
                # environment
                # if set(instruction.qubits).issubset(self.user_exposed_qbool):
                if set(instruction.qubits).issubset(
                    [self.condition_truth_value]
                ) and not isinstance(instruction.op, CustomControlOperation):
                    if instruction.op.name == "x":
                        inversion_tracker *= -1
                        x(self.condition_truth_value)
                    elif instruction.op.name == "p":
                        p(instruction.op.params[0], self.condition_truth_value)
                    elif instruction.op.name == "rz":
                        rz(instruction.op.params[0], self.condition_truth_value)
                    elif instruction.op.name in ["qb_alloc", "qb_dealloc"]:
                        pass
                    else:
                        raise Exception(
                            f"Tried to perform invalid operation {instruction.op.name} "
                            "on condition truth value (allowed are x, p, rz)"
                        )
                    continue

                if len(ctrl_qubits) == 1:
                    ctrl_state = self.ctrl_state
                else:
                    ctrl_state = "1"

                if self.invert:

                    new_ctrl_state = ""
                    for c in ctrl_state:
                        if c == "1":
                            new_ctrl_state += "0"
                        else:
                            new_ctrl_state += "1"

                    ctrl_state = new_ctrl_state

                if self.condition_truth_value in instruction.qubits:
                    self.env_qs.append(
                        convert_to_custom_control(
                            instruction,
                            self.condition_truth_value,
                            invert_control=ctrl_state == "0",
                        )
                    )
                    continue

                else:
                    # Create controlled instruction
                    instruction.op = instruction.op.control(
                        num_ctrl_qubits=1,
                        method=self.ctrl_method,
                        ctrl_state=ctrl_state,
                    )

                    # Add condition truth value qubit to the instruction qubit list
                    instruction.qubits = [self.condition_truth_value] + list(
                        instruction.qubits
                    )
                # Append instruction
                self.env_qs.append(instruction)

            perm_unlock(ctrl_qubits)

            if inversion_tracker == -1:
                x(self.condition_truth_value)

            if len(ctrl_qubits) > 1:

                if len(ctrl_qubits) > 5:
                    method = "auto"
                else:
                    method = "gray_pt_inv"

                mcx(
                    ctrl_qubits,
                    self.qbool,
                    method=method,
                    ctrl_state=cond_compile_ctrl_state,
                )
                self.qbool.delete()

        if self.parent_cond_env is not None:
            self.parent_cond_env.sub_condition_envs.extend(
                self.sub_condition_envs + [self]
            )

    def jcompile(self, eqn, context_dic):

        from qrisp.jasp import extract_invalues, insert_outvalues

        args = extract_invalues(eqn, context_dic)
        body_jaspr = eqn.params["jaspr"]

        num_ctrl = len(args) - len(body_jaspr.invars)
        flattened_jaspr = body_jaspr.flatten_environments()
        controlled_jaspr = flattened_jaspr.control(num_ctrl, ctrl_state=self.ctrl_state)

        import jax

        res = jax.jit(controlled_jaspr.eval)(*args)

        # Retrieve the equation
        jit_eqn = get_last_equation()

        jit_eqn.params["jaxpr"] = controlled_jaspr
        jit_eqn.params["name"] = "ctrl_env"

        if not isinstance(res, tuple):
            res = (res,)

        insert_outvalues(eqn, context_dic, res)


# This function turns instructions where the definition contains CustomControlOperations
# into CustomControlOperations. For this it checks that all Instructions that are acting
# on the control_qubit are also CustomControlOperations.
# For the conversion process, this function turns all Operations which are NO custom
# controls into their regular controls.
def convert_to_custom_control(instruction, control_qubit, invert_control=False):

    from qrisp.environments import CustomControlOperation

    if invert_control:
        qc = QuantumCircuit(len(instruction.qubits))
        cusc_x = CustomControlOperation(XGate(), targeting_control=True)
        qc.append(cusc_x, [qc.qubits[instruction.qubits.index(control_qubit)]])
        qc.append(instruction.op, qc.qubits)
        qc.append(cusc_x, [qc.qubits[instruction.qubits.index(control_qubit)]])

        res = instruction.copy()
        res.op = qc.to_gate()

        return convert_to_custom_control(res, control_qubit)

    # If the Operation is already a CustomControlOperation, do nothing
    if isinstance(instruction.op, CustomControlOperation):
        return instruction

    # If the Operation is primitive, an error happened during compilation,
    # since Operations which are not CustomControlledOperations should not target
    # the control qubit
    if instruction.op.definition is None:
        print(instruction)
        raise Exception

    # We now generate the new Instruction
    new_definition = instruction.op.definition.clearcopy()
    new_control_qubit = new_definition.qubits[instruction.qubits.index(control_qubit)]

    # Iterate through the data
    for def_instr in instruction.op.definition.data:

        if new_control_qubit in def_instr.qubits:
            # If the instruction is targeting the control qubit, we call the function
            # recursively to make sure that we are indeed appending a custom_control
            # operation
            new_definition.append(
                convert_to_custom_control(def_instr, new_control_qubit)
            )
        else:
            # Else, we generate the operations regular control
            new_op = def_instr.op.control(1)
            new_definition.append(
                new_op, [new_control_qubit] + def_instr.qubits, def_instr.clbits
            )

    # Create the result and modify the definition
    res = instruction.copy()
    res.op.definition = new_definition

    res.op = CustomControlOperation(
        res.op, targeting_control=control_qubit in instruction.qubits
    )

    return res


def control(*args, **kwargs):
    args = list(args)
    from qrisp import Qubit, QuantumBool, QuantumVariable
    from qrisp.jasp import AbstractQubit, check_for_tracing_mode

    if isinstance(args[0], QuantumVariable):
        if isinstance(args[0], QuantumBool):
            args[0] = [args[0][0]]
        else:
            args[0] = list(args[0])
    if not isinstance(args[0], list):
        args[0] = [args[0]]

    if check_for_tracing_mode():
        if all(isinstance(obj, bool) for obj in [x for x in args[0]]):
            return ClControlEnvironment(*args, **kwargs)
        elif all(isinstance(obj, AbstractQubit) for obj in [x.aval for x in args[0]]):
            return ControlEnvironment(*args, **kwargs)
        elif all(isinstance(obj, ShapedArray) for obj in [x.aval for x in args[0]]):
            return ClControlEnvironment(*args, **kwargs)
        else:
            raise Exception(f"Don't know how to control from input type {args[0]}")
    else:
        if all(isinstance(obj, (Qubit, QuantumBool)) for obj in args[0]):
            return ControlEnvironment(*args, **kwargs)
        elif all(isinstance(obj, (bool, np.bool)) for obj in [x for x in args[0]]):
            return ClControlEnvironment(*args, **kwargs)
        elif all(isinstance(obj, ArrayImpl) for obj in [x for x in args[0]]):
            args[0] = [bool(bit) for bit in args[0]]
            return ClControlEnvironment(*args, **kwargs)
        else:
            raise Exception(f"Don't know how to control from input type {args[0]}")
