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

import jax
from jax.extend.core import ClosedJaxpr, JaxprEqn

from qrisp.environments import QuantumEnvironment, control
from qrisp.environments.custom_control_environment import custom_control
from qrisp.circuit import Operation
from qrisp.core.session_merging_tools import recursive_qs_search, merge
from qrisp.misc import get_depth_dic
from qrisp.jasp import check_for_tracing_mode, get_last_equation


class ConjugationEnvironment(QuantumEnvironment):
    r"""
    This :ref:`QuantumEnvironment<QuantumEnvironment>` can be used for perfoming conjugated operations.
    An arbitrary unitary :math:`U \in SU(2^n)` can be conjugated by another unitary
    :math:`V \in SU(2^n)`:

    .. math::

        \text{conj}(U,V) = V^\dagger U V


    This structure appears in many quantum algorithms such as `Grover <https://arxiv.org/abs/quant-ph/9605043>`_,
    `Quantum backtracking <https://arxiv.org/abs/1509.02374>`_ or
    `Fourier arithmetic <https://arxiv.org/abs/quant-ph/0008033>`_.

    Using the ``ConjugationEnvironment`` not only helps to structure the code,
    but can also grant performance advantages.

    This is because the controlled circuit of such a conjugation is can be
    realized by just controlling :math:`U` instead of all three operations.

    .. math::

        C\text{conj}(U,V) = V^\dagger CU V


    The ``ConjugationEnvironment`` can be called using the alias ``conjugate``.
    Conjugate takes the conjugation function (in our example :math:`V`) and returns
    a function that takes the arguments for the conjugation function and returns
    the corresponding ``ConjugationEnvironment``.
    For more information consult the examples section.

    .. note::

        Note that every QuantumVariable that is created by the conjugation
        function :math:`V` must be deleted/uncomputed before function conclusion.

    Parameters
    ----------

    conjugation_function : function
        The function performing the operation :math:`V`.
    args : iterable
        The arguments for the conjugation function.
    kwargs : dict
        The keyword arguments for the conjugation function.

    Examples
    --------

    We perform Fourier addition on a :ref:`QuantumFloat`

    ::

        from qrisp import conjugate, QuantumFloat, p, QFT

        def fourier_adder(qf, n):

            with conjugate(QFT)(qf):

                for i in range(qf.size):
                    p(n*np.pi*2**(i-qf.size+1), qf[i])

    >>> qf = QuantumFloat(5)
    >>> fourier_adder(qf, 3)
    >>> print(qf)
    {3: 1.0}
    >>> fourier_adder(qf, 2)
    {5: 1.0}

    Investigate the effects of a controlled addition:

    ::

        from qrisp import control

        ctrl = QuantumFloat(1)
        qf = QuantumFloat(5)

        with control(ctrl):
            fourier_adder(qf, 3)


    To see that indeed only the conjugand has been controlled we take a look
    at the circuit:

    >>> print(qf.qs.transpile(1))

    .. code-block:: none

        ctrl.0: ─────────■──────────■─────────■─────────■─────────■─────────────────
                ┌──────┐ │P(3π/16)  │         │         │         │      ┌─────────┐
          qf.0: ┤0     ├─■──────────┼─────────┼─────────┼─────────┼──────┤0        ├
                │      │            │P(3π/8)  │         │         │      │         │
          qf.1: ┤1     ├────────────■─────────┼─────────┼─────────┼──────┤1        ├
                │      │                      │P(3π/4)  │         │      │         │
          qf.2: ┤2 QFT ├──────────────────────■─────────┼─────────┼──────┤2 QFT_dg ├
                │      │                                │P(3π/2)  │      │         │
          qf.3: ┤3     ├────────────────────────────────■─────────┼──────┤3        ├
                │      │                                          │P(3π) │         │
          qf.4: ┤4     ├──────────────────────────────────────────■──────┤4        ├
                └──────┘                                                 └─────────┘


    """

    def __init__(self, conjugation_function, args, kwargs, allocation_management=True):

        self.conjugation_function = conjugation_function

        self.args = args

        self.kwargs = kwargs

        self.manual_allocation_management = allocation_management

        QuantumEnvironment.__init__(self)

    def __enter__(self):

        QuantumEnvironment.__enter__(self)

        if check_for_tracing_mode():
            with PJITEnvironment():
                res = self.conjugation_function(*list(self.args), **self.kwargs)
                # res = qache(self.conjugation_function)(*list(self.args), **self.kwargs)
            return res

        merge(recursive_qs_search(self.args) + [self.env_qs])

        qv_set_before = set(self.env_qs.qv_list)

        try:
            res = self.conjugation_function(*self.args, **self.kwargs)
        except Exception as e:
            QuantumEnvironment.__exit__(self, type(e), e, "")
            raise e

        temp_data = list(self.env_qs.data)
        self.env_qs.data = []
        i = 0

        while temp_data:
            instr = temp_data.pop(i)
            if isinstance(instr, QuantumEnvironment):
                instr.compile()
            else:
                self.env_qs.append(instr)

        creation_dic = {}
        for i in range(len(self.env_qs.data)):
            instr = self.env_qs.data[i]
            if instr.op.name == "qb_alloc":
                creation_dic[instr.qubits[0]] = 1
            elif instr.op.name == "qb_dealloc":
                if instr.qubits[0] not in creation_dic:
                    raise Exception(
                        f"Tried to destroy qubit {instr.qubits[0]} within a conjugator."
                    )
                else:
                    creation_dic[instr.qubits[0]] -= 1

        for k, v in creation_dic.items():
            if v != 0:
                raise Exception(f"Tried to create qubit {k} within a conjugator.")

        self.conjugation_circ = self.env_qs.copy()

        self.env_qs.data = []

        return res

    def __exit__(self, exception_type, exception_value, traceback):

        if exception_value:
            QuantumEnvironment.__exit__(
                self, exception_type, exception_value, traceback
            )

        if not check_for_tracing_mode():
            conjugation_center_data = list(self.env_qs.data)
            self.env_qs.data = []
            self.perform_conjugation(conjugation_center_data)

        else:
            from qrisp.environments import invert

            with invert():
                with PJITEnvironment():
                    self.conjugation_function(*list(self.args), **self.kwargs)
                    # qache(self.conjugation_function)(*list(self.args), **self.kwargs)

        QuantumEnvironment.__exit__(self, exception_type, exception_value, traceback)

    @custom_control
    def perform_conjugation(self, conjugation_center_data, ctrl=None, ctrl_method=None):

        for instr in self.conjugation_circ.data:
            self.env_qs.append(instr)

        if ctrl is not None:
            with control(ctrl, ctrl_method=ctrl_method):
                self.env_qs.data.extend(conjugation_center_data)
        else:
            self.env_qs.data.extend(conjugation_center_data)

        for instr in self.conjugation_circ.inverse().data:
            self.env_qs.append(instr)

    def jcompile(self, eqn, context_dic):

        # This function transforms a collected ConjugationEnvironment
        # into the proper unitary, i.e. U^\dagger V U
        # Next to this transformation, we also have to make sure
        # to properly set the ctrl_jaspr attribute to reflect the
        # controlled version: U^\dagger cV U

        from qrisp.jasp import extract_invalues, insert_outvalues

        args = extract_invalues(eqn, context_dic)
        body_jaspr = eqn.params["jaspr"]

        flattened_jaspr = body_jaspr.flatten_environments()

        # We now generate the controlled version.
        # The implementation of the environment ensures that the conjugator,
        # (i.e. U) is traced into a pjit primitive. This ensures that the
        # first and last equation of the environments body corresponds to
        # pjit calls of the conjugator.
        # To generate the efficient conjugated version, we first generate the
        # naively controlled version and subsequently replace the equations
        # describing the controlled conjugators with their uncontrolled version.

        controlled_flattened_jaspr = flattened_jaspr.control(1)

        controlled_eqn_list = list(controlled_flattened_jaspr.eqns)

        # Replace the controlled conjugators
        controlled_eqn_list[0] = copy_jaxpr_eqn(controlled_flattened_jaspr.eqns[0])

        # Remove the invar (control qubits are always added as the first arguments)
        controlled_eqn_list[0].invars.pop(0)

        new_params = controlled_eqn_list[0].params

        # Replace the pjit body
        new_params["jaxpr"] = flattened_jaspr.eqns[0].params["jaxpr"]

        # We also need to update these parameters - otherwise Jax raises errors
        # during MLIR lowering
        new_params["in_shardings"] = new_params["in_shardings"][1:]
        new_params["in_layouts"] = new_params["in_layouts"][1:]
        new_params["donated_invars"] = new_params["donated_invars"][1:]

        # Do the same for the last equation
        controlled_eqn_list[-1] = copy_jaxpr_eqn(controlled_flattened_jaspr.eqns[-1])
        controlled_eqn_list[-1].invars.pop(0)
        new_params = controlled_eqn_list[-1].params
        new_params["jaxpr"] = flattened_jaspr.eqns[-1].params["jaxpr"]
        new_params["in_shardings"] = new_params["in_shardings"][1:]
        new_params["in_layouts"] = new_params["in_layouts"][1:]
        new_params["donated_invars"] = new_params["donated_invars"][1:]

        # Set the ctrl_jaspr attribute to use enable custom control behavior
        flattened_jaspr.ctrl_jaspr = controlled_flattened_jaspr.update_eqns(
            controlled_eqn_list
        )

        # Trace the jaxpr and subsequently update the equation (so it contains
        # the controlled version)
        res = jax.jit(flattened_jaspr.eval)(*args)

        # Retrieve the equation
        jit_eqn = get_last_equation()
        jit_eqn.params["jaxpr"] = flattened_jaspr
        jit_eqn.params["name"] = "conjugation_env"

        if not isinstance(res, tuple):
            res = (res,)

        insert_outvalues(eqn, context_dic, res)

    def compile_(self, ctrl=None):

        temp = list(self.env_qs.data)
        self.env_qs.data = []

        for instr in self.conjugation_circ.data:
            if isinstance(instr, QuantumEnvironment):
                instr.compile()
            else:
                self.env_qs.append(instr)

        self.conjugation_circ = self.env_qs.copy()
        self.env_qs.data = []

        QuantumEnvironment.compile(self)

        content_circ = self.env_qs.copy()
        self.conjugation_circ.qubits = list(content_circ.qubits)

        conjugation_depth_dic = get_depth_dic(self.conjugation_circ)
        content_depth_dic = get_depth_dic(content_circ)

        added_depth_dic = {
            qb: conjugation_depth_dic[qb] + content_depth_dic[qb]
            for qb in content_circ.qubits
        }

        instruction_qubits = []

        i = 0
        while i < len(content_circ.qubits):

            qb = content_circ.qubits[i]

            if added_depth_dic[qb]:
                instruction_qubits.append(qb)
                i += 1
            else:
                content_circ.qubits.pop(i)
                self.conjugation_circ.qubits.pop(i)

        self.env_qs.data = temp

        conj_op = ConjugatedOperation(self.conjugation_circ, content_circ)

        alloc_instr = [
            instr
            for instr in self.conjugation_circ.data + content_circ.data
            if instr.op.name == "qb_alloc"
        ]

        for instr in alloc_instr:
            self.env_qs.append(instr)

        self.env_qs.append(conj_op, content_circ.qubits)

        dealloc_instr = [
            instr
            for instr in self.conjugation_circ.data + content_circ.data
            if instr.op.name == "qb_dealloc"
        ]

        for instr in dealloc_instr:
            self.env_qs.append(instr)


class ConjugatedOperation(Operation):

    def __init__(self, conjugation_circ, content_circ):

        self.conjugation_gate = conjugation_circ.to_gate(name="conjugator")
        self.content_gate = content_circ.to_gate(name="conjugand")

        definition = conjugation_circ.clearcopy()

        definition.append(self.conjugation_gate, definition.qubits)
        definition.append(self.content_gate, definition.qubits)
        definition.append(self.conjugation_gate.inverse(), definition.qubits)

        Operation.__init__(
            self,
            name="conjugation_env",
            definition=definition,
            num_qubits=definition.num_qubits(),
        )

    def control(self, num_ctrl_qubits=1, ctrl_state=-1, method=None):

        controlled_conjugand = self.content_gate.control(
            num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state, method=None
        )

        res = type(controlled_conjugand)(
            self, num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state, method=method
        )

        res.definition.data = []

        res.definition.append(self.conjugation_gate, self.definition.qubits)
        res.definition.append(
            controlled_conjugand,
            res.definition.qubits[:num_ctrl_qubits] + self.definition.qubits,
        )
        res.definition.append(self.conjugation_gate.inverse(), self.definition.qubits)

        return res

    def inverse(self):
        return ConjugatedOperation(
            self.conjugation_gate.definition, self.content_gate.inverse().definition
        )


def conjugate(conjugation_function, allocation_management=True):

    def conjugation_env_creator(*args, **kwargs):

        return ConjugationEnvironment(
            conjugation_function,
            args,
            kwargs,
            allocation_management=allocation_management,
        )

    return conjugation_env_creator


class PJITEnvironment(QuantumEnvironment):

    def jcompile(self, eqn, context_dic):

        from qrisp.jasp import extract_invalues, insert_outvalues, Jaspr

        args = extract_invalues(eqn, context_dic)
        body_jaspr = eqn.params["jaspr"]

        flattened_jaspr = body_jaspr.flatten_environments()

        res = jax.jit(flattened_jaspr.eval)(*args)

        jit_eqn = get_last_equation()

        jit_eqn.params["jaxpr"] = Jaspr.from_cache(jit_eqn.params["jaxpr"])

        if not isinstance(res, tuple):
            res = (res,)

        insert_outvalues(eqn, context_dic, res)


def copy_jaxpr_eqn(jaxpr_eqn):
    return JaxprEqn(
        invars=list(jaxpr_eqn.invars),
        outvars=list(jaxpr_eqn.outvars),
        params=dict(jaxpr_eqn.params),
        primitive=jaxpr_eqn.primitive,
        effects=jaxpr_eqn.effects,
        source_info=jaxpr_eqn.source_info,
        ctx=jaxpr_eqn.ctx,
    )
