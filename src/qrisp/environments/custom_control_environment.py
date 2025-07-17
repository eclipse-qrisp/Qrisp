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

import inspect

import jax
import jax.numpy as jnp

from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.environments.gate_wrap_environment import GateWrapEnvironment
from qrisp.circuit import Operation, QuantumCircuit, Instruction
from qrisp.environments.iteration_environment import IterationEnvironment
from qrisp.core import merge

from qrisp.jasp import check_for_tracing_mode, qache, AbstractQubit, make_jaspr, get_last_equation


def custom_control(*func, **cusc_kwargs):
    """
    The ``custom_control`` decorator allows to specify the controlled version of
    the decorated function. If this function is called within a :ref:`ControlEnvironment`
    or a :ref:`ConditionEnvironment` the controlled version is executed instead.

    Specific controlled versions of quantum functions are very common in many
    scientific publications. This is because the general control procedure can
    signifcantly increase resource demands.

    In order to use the ``custom_control`` decorator, you need to add the ``ctrl``
    keyword to your function signature. If called within a controlled context,
    this keyword will receive the corresponding control qubit.

    For more details consult the examples section.


    Parameters
    ----------
    func : function
        A function of QuantumVariables, which has the ``ctrl`` keyword.

    Returns
    -------
    adaptive_control_function : function
        A function which will execute it's controlled version, if called
        within a :ref:`ControlEnvironment` or a :ref:`ConditionEnvironment`.

    Examples
    --------

    We create a swap function with custom control.

    ::

        from qrisp import mcx, cx, custom_control

        @custom_control
        def swap(a, b, ctrl = None):

            if ctrl is None:

                cx(a, b)
                cx(b, a)
                cx(a, b)

            else:

                cx(a, b)
                mcx([ctrl, b], a)
                cx(a, b)


    Test the non-controlled version:

    ::

        from qrisp import QuantumBool

        a = QuantumBool()
        b = QuantumBool()

        swap(a, b)

        print(a.qs)


    ::

        QuantumCircuit:
        --------------
                  ┌───┐
        a.0: ──■──┤ X ├──■──
             ┌─┴─┐└─┬─┘┌─┴─┐
        b.0: ┤ X ├──■──┤ X ├
             └───┘     └───┘
        Live QuantumVariables:
        ---------------------
        QuantumBool a
        QuantumBool b


    Test the controlled version:

    ::

        from qrisp import control

        a = QuantumBool()
        b = QuantumBool()
        ctrl_qbl = QuantumBool()

        with control(ctrl_qbl):

            swap(a,b)

        print(a.qs.transpile(1))

    ::

                         ┌───┐
               a.0: ──■──┤ X ├──■──
                    ┌─┴─┐└─┬─┘┌─┴─┐
               b.0: ┤ X ├──■──┤ X ├
                    └───┘  │  └───┘
        ctrl_qbl.0: ───────■───────


    """

    if len(func) == 0:
        return lambda x: custom_control(x, **cusc_kwargs)
    else:
        func = func[0]

    # The idea to realize the custom control feature in traced mode is to
    # first trace the non-controlled version into a pjit primitive using
    # the qache feature and the trace the controlled version.
    # The controlled version is then stored in the params attribute

    # Qache the function (in non-traced mode, this has no effect)
    
    # Make sure the inv keyword argument is treated as a static argument
    new_static_argnames = list(cusc_kwargs.get("static_argnames", []))
    new_static_argnames.append("inv")
    
    qache_kwargs = dict(cusc_kwargs)
    qache_kwargs["static_argnames"] = new_static_argnames
    
    func = qache(func, **qache_kwargs)

    def adaptive_control_function(*args, **kwargs):

        if not check_for_tracing_mode():

            from qrisp.core import recursive_qs_search
            from qrisp import (
                merge,
                ControlEnvironment,
                ConditionEnvironment,
                QuantumEnvironment,
                InversionEnvironment,
                ConjugationEnvironment,
            )

            qs_list = recursive_qs_search(args)

            if "ctrl" in kwargs:
                if kwargs["ctrl"] is not None:
                    qs_list.append(kwargs["ctrl"].qs())

            merge(qs_list)

            if len(qs_list) == 0:
                return func(*args, **kwargs)

            qs = qs_list[0]

            # Search for a Control/Condition Environment and get the control qubit
            control_qb = None
            for env in qs.env_stack[::-1]:
                if type(env) == QuantumEnvironment:
                    continue
                if isinstance(env, (ControlEnvironment, ConditionEnvironment)):
                    control_qb = env.condition_truth_value
                    break
                if not isinstance(
                    env,
                    (InversionEnvironment, ConjugationEnvironment, GateWrapEnvironment),
                ):
                    if isinstance(env, IterationEnvironment):
                        if env.precompile:
                            break
                        else:
                            continue
                    break

            # If no control qubit was found, simply execute the function
            if control_qb is None:
                return func(*args, **kwargs)

            # Check whether the function supports the ctrl_method kwarg and adjust
            # the kwargs accordingly
            if "ctrl_method" in list(inspect.getfullargspec(func))[0] and isinstance(
                env, ControlEnvironment
            ):
                kwargs.update({"ctrl_method": env.ctrl_method})

            # In the case that a qubit was found, we use the CustomControlEnvironent (definded below)
            # This environments gatewraps the function and compiles it to a specific Operation subtype
            # called CustomControlledOperation.
            # The Condition/Control Environment compiler recognizes this Operation type
            # and processes it accordingly

            with CustomControlEnvironment(control_qb, func.__name__):
                if "ctrl" in kwargs:
                    kwargs["ctrl"] = control_qb
                    res = func(*args, **kwargs)
                else:
                    res = func(*args, ctrl=control_qb, **kwargs)

        else:

            args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], bool):
                    args[i] = jnp.array(args[i], dtype=jnp.bool)
                elif isinstance(args[i], int):
                    args[i] = jnp.array(args[i], dtype=jnp.int64)
                elif isinstance(args[i], float):
                    args[i] = jnp.array(args[i], dtype=jnp.float64)
                elif isinstance(args[i], complex):
                    args[i] = jnp.array(args[i], dtype=jnp.complex64)

            # Call the (qached) function
            res = func(*args, **kwargs)

            # Retrieve the pjit equation
            jit_eqn = get_last_equation()

            if not jit_eqn.params["jaxpr"].jaxpr.ctrl_jaspr:
                # Trace the controlled version

                # Make sure the inv keyword argument is treated as a static argument
                # (this is important for the interaction with custom_inverse)
                if "inv" in kwargs:
                    custom_inversion = True
                    custom_inv_value = kwargs["inv"]
                    del kwargs["inv"]
                else:
                    custom_inversion = False

                def ammended_func(*ammended_args, **kwargs):
                    new_kwargs = dict(kwargs)
                    new_kwargs["ctrl"] = ammended_args[0]
                    args = ammended_args[1:]
                    if custom_inversion:
                        return func(*args, inv = custom_inv_value, **new_kwargs)
                    else:
                        return func(*args, **new_kwargs)

                ctrl_aval = AbstractQubit()
                ammended_args = [ctrl_aval] + list(args)
                
                controlled_jaspr = make_jaspr(ammended_func, **cusc_kwargs)(
                    *ammended_args, **kwargs
                )
                
                # Store controlled version
                jit_eqn.params["jaxpr"].jaxpr.ctrl_jaspr = controlled_jaspr

        return res

    return adaptive_control_function


class CustomControlEnvironment(QuantumEnvironment):

    def __init__(self, control_qb, name):

        self.control_qb = control_qb
        self.manual_allocation_management = True
        QuantumEnvironment.__init__(self, env_args=[])

    def __enter__(self):

        QuantumEnvironment.__enter__(self)
        merge([self.env_qs, self.control_qb.qs()])

    def compile(self):

        original_data = list(self.env_qs.data)
        self.env_qs.data = []
        QuantumEnvironment.compile(self)

        temp = list(self.env_qs.data)
        self.env_qs.data = []

        for instr in temp:

            if instr.op.name in ["qb_alloc", "qb_dealloc"]:
                self.env_qs.append(instr)
            elif self.control_qb in instr.qubits:
                cusc_op = CustomControlOperation(instr.op, targeting_control=True)
                self.env_qs.append(cusc_op, instr.qubits, instr.clbits)
            else:
                cusc_op = CustomControlOperation(instr.op)
                self.env_qs.append(
                    cusc_op, [self.control_qb] + instr.qubits, instr.clbits
                )

        self.env_qs.data = original_data + list(self.env_qs.data)


class CustomControlOperation(Operation):

    def __init__(self, init_op, targeting_control=False):

        self.targeting_control = targeting_control

        if not targeting_control:
            definition = QuantumCircuit(init_op.num_qubits + 1, init_op.num_clbits)
            definition.data.append(
                Instruction(init_op, definition.qubits[1:], definition.clbits)
            )

            Operation.__init__(
                self,
                name="cusc_" + init_op.name,
                num_qubits=init_op.num_qubits + 1,
                num_clbits=init_op.num_clbits,
                definition=definition,
            )

            self.init_op = init_op

            self.permeability = {
                i + 1: init_op.permeability[i] for i in range(init_op.num_qubits)
            }
            self.permeability[0] = True

            self.is_qfree = init_op.is_qfree
        else:

            definition = QuantumCircuit(init_op.num_qubits, init_op.num_clbits)
            definition.append(init_op, definition.qubits, definition.clbits)

            Operation.__init__(
                self,
                name="cusc_" + init_op.name,
                num_qubits=init_op.num_qubits,
                num_clbits=init_op.num_clbits,
                definition=definition,
            )

            self.init_op = init_op

            self.permeability = dict(init_op.permeability)
            self.is_qfree = bool(init_op.is_qfree)

    def inverse(self):
        temp = self.init_op.inverse()
        return CustomControlOperation(temp, targeting_control=self.targeting_control)

    def copy(self):
        return CustomControlOperation(
            self.init_op.copy(), targeting_control=self.targeting_control
        )
