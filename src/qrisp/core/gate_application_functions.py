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

import sympy

import qrisp.circuit.standard_operations as std_ops
from qrisp.jasp import check_for_tracing_mode

def append_operation(operation, qubits=[], clbits=[], param_tracers = []):
    from qrisp import find_qs
    
    qs = find_qs(qubits)
    
    qs.append(operation, qubits, clbits, param_tracers = param_tracers)


def cx(control, target):
    """
    Applies a CX gate.

    Parameters
    ----------
    control : Qubit or list[Qubit] or QuantumVariable
        The Qubit to control on.
    target : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the X gate on.
    """
    append_operation(std_ops.CXGate(), [control, target])

    # std_ops.CXGate().append([qubits_0, qubits_1])

    return control, target


def cy(control, target):
    """
    Applies a CY gate.

    Parameters
    ----------
    control : Qubit or list[Qubit] or QuantumVariable
        The Qubit to control on.
    target : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Y gate on.
    """

    append_operation(std_ops.CYGate(), [control, target])
    return control, target


def cz(control, target):
    """
    Applies a CZ gate.

    Parameters
    ----------
    control : Qubit or list[Qubit] or QuantumVariable
        The Qubit to control on.
    target : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Z gate on.
    """

    append_operation(std_ops.CZGate(), [control, target])
    return control, target


def h(qubits):
    """
    Applies an H gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the H gate on.
    """
    append_operation(std_ops.HGate(), [qubits])

    return qubits


def x(qubits):
    """
    Applies an X gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the X gate on.
    """

    append_operation(std_ops.XGate(), [qubits])

    return qubits


def y(qubits):
    """
    Applies a Y gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Y gate on.
    """

    append_operation(std_ops.YGate(), [qubits])

    return qubits


def z(qubits):
    """
    Applies a Z gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Z gate on.
    """
    append_operation(std_ops.ZGate(), [qubits])

    return qubits


def mcx(controls, target, method="auto", ctrl_state=-1, num_ancilla=1):
    r"""
    Applies a multi-controlled X gate.

    The following methods are available:

    
    .. list-table::
        :widths: 20 80
        :header-rows: 1

        *   - Method
            - Description
        *   - ``gray`` 
            - Performs a gray code traversal which requires no ancillae but is rather inefficient for large numbers of control qubits.
        *   - ``gray_pt``/``gray_pt_inv`` 
            - More efficient but introduce extra phases that need to be uncomputed by performing the inverse of this gate on the same inputs. For more information on phase tolerance, check `this paper <https://iopscience.iop.org/article/10.1088/2058-9565/acaf9d/meta>`__.
        *   - ``balauca`` 
            - Method based on this `paper <https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf>`__ with logarithmic depth but requires many ancilla qubits.
        *   - ``maslov``
            - Documented `here <https://arxiv.org/abs/1508.03273>`_, requires less ancilla qubits but is only available for 4 or less control qubits.
        *   - ``yong`` 
            - Can be found int this `article <https://link.springer.com/article/10.1007/s10773-017-3389-4>`__.This method requires only a single ancilla and has moderate scaling in depth and gate count.
        *   - ``amy``
            - A Toffoli-circuit (ie. only two control qubits are possible), which (temporarily) requires one ancilla qubit. However, instead of the no-ancilla T-depth 4, this circuit achieves a T-depth of 2. Find the implementation details in `this paper <https://arxiv.org/pdf/1206.0758.pdf>`__.
        *   - ``jones``
            - Similar to ``amy`` but uses two ancilla qubits, and has a T-depth of 1. Read about it `here <https://arxiv.org/abs/1212.5069>`__.
        *   - ``gidney``
            - A very unique way for synthesizing a logical AND. The Gidney Logical AND performs a circuit with T-depth 1 to compute the truth value and performs another circuit involving a measurement and a classically controlled CZ gate for uncomputation. The uncomputation circuit has T-depth 0, such that the combined T-depth is 1. Requires no ancillae. More details `here <https://arxiv.org/abs/1709.06648>`__. Works only for two control qubits.
        *   - ``hybrid``
            - A flexible method which combines the other available methods, such that the amount of used ancillae is customizable. After several ``balauca``-layers, the recursion is canceled by either a ``yong``, ``maslov`` or ``gray`` mcx, depending on what fits the most.
        *   - ``auto`` 
            - Recompiles the mcx gate at compile time using the hybrid algorithm together with the information about how many clean/dirty ancillae qubits are available. For more information check :meth:`qrisp.QuantumSession.compile`.
  
    .. note::
        Due to Qrisp's automatic qubit management, clean ancilla qubits are not as much
        of a sparse resource as one might think. Even though the ``balauca`` method
        requires a considerable amount of ancillae, many other functions also do,
        implying there is alot of recycling potential. The net effect is that in more
        complex programs, the amount of qubits of the circuit returned by the
        :meth:`compile method <qrisp.QuantumSession.compile>` increases only slightly.


    Parameters
    ----------
    controls : list[Qubits] or QuantumVariable
        The Qubits to control on.
    target : Qubit
        The Qubit to perform the X gate on.
    method : str, optional
        The synthesis method. Available are ``auto``, ``gray``, ``gray_pt``,
        ``gray_pt_inv``, ``maslov``, ``balauca`` and ``yong``. The default is ``auto``.
    ctrl_state : int or str, optional
        The state on which to activate the X gate. The default is "1111..".
    num_ancilla : int, optional
        Specifies the amount of ancilla qubits to use. This parameter is used only if
        the method is set to ``hybrid``. The default is 1.


    Examples
    --------

    We apply a 3-contolled X gate

    >>> from qrisp import QuantumVariable, mcx
    >>> control = QuantumVariable(3)
    >>> target = QuantumVariable(1)
    >>> mcx(control, target, method = "gray")
    >>> print(control.qs)
    
    ::
    
        QuantumCircuit:
        --------------
        control.0: ──■──
                     │
        control.1: ──■──
                     │
        control.2: ──■──
                   ┌─┴─┐
         target.0: ┤ X ├
                   └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable control
        QuantumVariable target

    We compare different performance indicators. ::

        from qrisp import QuantumVariable, mcx

        def benchmark_mcx(n, methods):
            for method in methods:

                controls = QuantumVariable(n)
                target = QuantumVariable(1)

                mcx(controls, target, method = method)

                compiled_qc = controls.qs.compile()

                print(f"==================\nMethod: {method}\n------------------")
                print(f"Depth: \t\t\t{compiled_qc.depth()}")
                print(f"CNOT count: \t{compiled_qc.cnot_count()}")
                print(f"Qubit count: \t{compiled_qc.num_qubits()}")


    >>> benchmark_mcx(4, methods = ["gray", "gray_pt", "maslov", "balauca", "yong"])
    ==================
    Method: gray
    ------------------
    Depth:          50
    CNOT count:     30
    Qubit count:    5
    ==================
    Method: gray_pt
    ------------------
    Depth:          34
    CNOT count:     16
    Qubit count:    5
    ==================
    Method: maslov
    ------------------
    Depth:          43
    CNOT count:     18
    Qubit count:    6
    ==================
    Method: balauca
    ------------------
    Depth:          22
    CNOT count:     18
    Qubit count:    7
    ==================
    Method: yong
    ------------------
    Depth:          77
    CNOT count:     30
    Qubit count:    6
    >>> benchmark_mcx(10, methods = ["gray", "gray_pt", "balauca", "yong"])
    ==================
    Method: gray
    ------------------
    Depth:          3106
    CNOT count:     2046
    Qubit count:    11
    ==================
    Method: gray_pt
    ------------------
    Depth:          2050
    CNOT count:     1024
    Qubit count:    11
    ==================
    Method: balauca
    ------------------
    Depth:          53
    CNOT count:     54
    Qubit count:    18
    ==================
    Method: yong
    ------------------
    Depth:          621
    CNOT count:     264
    Qubit count:    12
    
    **Mid circuit measurement based methods**
    
    The ``gidney`` and ``jones`` method are unique in the way that they require
    mid circuit measurements. The measurements are inserted retroactively by the 
    :meth:`.compile <qrisp.QuantumSession.compile>` method, because immediate compilation
    would prevent evaluation of the statevector (since a measurement is involved).
    
    Instead a tentative (measurement free) representative is inserted and replaced
    at compile time.
    
    To get a better understanding consider the following script:
        
    >>> from qrisp import QuantumVariable, mcx
    >>> control = QuantumVariable(2)
    >>> target = QuantumVariable(1)
    >>> mcx(control, target, method = "jones")
    >>> print(control.qs)
    QuantumCircuit:
    ---------------
                     ┌───────────────────────────┐
          control.0: ┤0                          ├
                     │                           │
          control.1: ┤1                          ├
                     │                           │
           target.0: ┤2 uncompiled_jones_toffoli ├
                     │                           │
    jones_ancilla.0: ┤3                          ├
                     │                           │
    jones_ancilla.1: ┤4                          ├
                     └───────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumVariable target    
    
    We see that there is no classical bit and therefore also no measurement.
    The statevector can still be accessed:
    
    >>> print(control.qs.statevector())
    |00>*|0>
    
    To introduce the measurement we simply call the :meth:`.compile <qrisp.QuantumSession.compile>` method
    with the keyword ``compile_mcm = True``:
        
    >>> qc = control.qs.compile(compile_mcm = True)
    >>> print(qc)
                 ┌─────────────────────────┐
      control.0: ┤0                        ├
                 │                         │
      control.1: ┤1                        ├
                 │                         │
       target.0: ┤2                        ├
                 │  compiled_jones_toffoli │
    workspace_0: ┤3                        ├
                 │                         │
    workspace_1: ┤4                        ├
                 │                         │
           cb_0: ╡0                        ╞
                 └─────────────────────────┘
                 
    Because there is a measurement now, the statevector can no longer be accessed.

    >>> qc.statevector_array()    
    Exception: Unitary of operation measure not defined.
    
    However the T-depth went down by 50%:
        
    >>> print(qc.t_depth())
    1
    >>> print(control.qs.compile(compile_mcm = False).t_depth())
    2
    
    A similar construction holds for the `Gidney's temporary logical AND <https://arxiv.org/abs/1709.06648>`_. 
    However there are additional details: This technique always comes in pairs. A computation
    and an uncomputation. The computation circuit has a T-depth of 1 and the uncomputation
    circuit has a T-depth of 0. The uncomputation circuit however contains a measurement.
    
    As you can imagine, this measurement is also inserted at compile time.
    
    Even though both circuits are not the inverses of each other, Qrisp will use
    the respective partner if called to invert:
        
    >>> control = QuantumVariable(2)
    >>> target = QuantumVariable(1)
    >>> mcx(control, target, method = "gidney")
    >>> print(control.qs)
    QuantumCircuit:
    ---------------
               ┌────────────────────────┐
    control.0: ┤0                       ├
               │                        │
    control.1: ┤1 uncompiled_gidney_mcx ├
               │                        │
     target.0: ┤2                       ├
               └────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumVariable target
    
    This even works in conjunction with the :ref:`uncomputation module <Uncomputation>`:
        
    >>> target.uncompute()
    >>> print(target.qs)
    QuantumCircuit:
    ---------------
               ┌────────────────────────┐┌────────────────────────────┐
    control.0: ┤0                       ├┤0                           ├
               │                        ││                            │
    control.1: ┤1 uncompiled_gidney_mcx ├┤1 uncompiled_gidney_mcx_inv ├
               │                        ││                            │
     target.0: ┤2                       ├┤2                           ├
               └────────────────────────┘└────────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control

    To introduce the measurement, we call the compile method.
    
    >>> print(target.qs.compile(compile_mcm = True))
                 ┌──────────────────────┐┌──────────────────────────┐
      control.0: ┤0                     ├┤0                         ├
                 │                      ││                          │
      control.1: ┤1 compiled_gidney_mcx ├┤1                         ├
                 │                      ││  compiled_gidney_mcx_inv │
    workspace_0: ┤2                     ├┤2                         ├
                 └──────────────────────┘│                          │
           cb_0: ════════════════════════╡0                         ╞
                                         └──────────────────────────┘
    
    Apart from uncomputation, the inverted Gidney mcx can also be accessed via,
    the :ref:`InversionEnvironment`:
        
    ::
        
        from qrisp import invert
        
        control = QuantumVariable(2)
        target = QuantumVariable(1)
        
        with invert():
            mcx(control, target, method = "gidney")
            
    
    >>> print(control.qs)
    QuantumCircuit:
    ---------------
               ┌────────────────────────────┐
    control.0: ┤0                           ├
               │                            │
    control.1: ┤1 uncompiled_gidney_mcx_inv ├
               │                            │
     target.0: ┤2                           ├
               └────────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumVariable target
    
    
    """

    from qrisp.misc import bin_rep
    from qrisp.alg_primitives.mcx_algs import GidneyLogicalAND, amy_toffoli, jones_toffoli
    from qrisp.core import QuantumVariable
    from qrisp.qtypes import QuantumBool

    new_controls = []

    for qbl in controls:
        if isinstance(qbl, QuantumBool):
            new_controls.append(qbl[0])
        else:
            new_controls.append(qbl)
    
    if isinstance(target, (list, QuantumVariable)):
        
        if len(target) > 1:
            raise Exception("Target of mcx contained more than one qubit")
        target = target[0]
        
        
    qubits_0 = new_controls
    qubits_1 = [target]

    n = len(qubits_0)

    if n == 0:
        return controls, target

    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2**n
        ctrl_state = bin_rep(ctrl_state, n)[::-1]

    if len(ctrl_state) != n:
        raise Exception(
            f"Given control state {ctrl_state} does not match control qubit amount {n}"
        )

    from qrisp.alg_primitives.mcx_algs import (
        balauca_dirty,
        balauca_mcx,
        hybrid_mcx,
        maslov_mcx,
        yong_mcx,
    )

    if method in ["gray", "gray_pt", "gray_pt_inv"] or len(qubits_0) == 1:
        if len(qubits_0) == 1:
            method = "gray"
        append_operation(
            std_ops.MCXGate(len(qubits_0), ctrl_state, method=method),
            qubits_0 + qubits_1,
        )
    elif method == "gidney":
        if len(qubits_0) != 2:
            raise Exception(f"Tried to call Gidney logical AND with {len(qubits_0)} controls instead of two")
        
        append_operation(
            GidneyLogicalAND(ctrl_state = ctrl_state),
            qubits_0 + qubits_1,
        )
    
    elif method == "gidney_inv":
        if len(qubits_0) != 2:
            raise Exception(f"Tried to call Gidney logical AND with {len(qubits_0)} controls instead of two")
        
        append_operation(
            GidneyLogicalAND(ctrl_state = ctrl_state, inv = True),
            qubits_0 + qubits_1,
        )

    elif method == "maslov":
        from qrisp import QuantumBool

        if n >= 3:
            ancilla = [QuantumBool(name="maslov_anc_")]
        else:
            ancilla = []
        append_operation(maslov_mcx(n, ctrl_state), qubits_0 + ancilla + qubits_1)

        [qv.delete() for qv in ancilla]

    elif method == "balauca":
        balauca_mcx(qubits_0, qubits_1, ctrl_state=ctrl_state)

    elif method == "balauca_dirty":
        balauca_dirty(qubits_0, qubits_1, k=num_ancilla, ctrl_state=ctrl_state)

    elif method == "yong":
        yong_mcx(qubits_0, qubits_1, ctrl_state=ctrl_state)

    elif method == "hybrid":
        hybrid_mcx(qubits_0, qubits_1, ctrl_state=ctrl_state, num_ancilla=num_ancilla)
    
    elif method == "amy":
        if len(qubits_0) != 2:
            raise Exception(f"Tried to call Amy MCX with {len(qubits_0)} controls instead of two")
        amy_toffoli(qubits_0, qubits_1, ctrl_state = ctrl_state)
    
    elif method == "jones":
        if len(qubits_0) != 2:
            raise Exception(f"Tried to call Jones MCX with {len(qubits_0)} controls instead of two")
        jones_toffoli(qubits_0, qubits_1, ctrl_state = ctrl_state)
        
    elif method == "auto":
        # if n <= 3:
        #     return mcx(qubits_0, qubits_1, method = "gray", ctrl_state = ctrl_state)
        # if 3 < n < 5:
        #     return mcx(qubits_0, qubits_1, method = "maslov", ctrl_state = ctrl_state)
        # else:
        #     return mcx(qubits_0, qubits_1, method = "balauca", ctrl_state = ctrl_state) # noqa:501

        gate = std_ops.MCXGate(len(qubits_0), ctrl_state, method="auto")
        append_operation(gate, qubits_0 + qubits_1)

    return controls, target


def mcz(qubits, method="auto", ctrl_state=-1, num_ancilla=1):
    """
    Applies a multi-controlled Z gate.

    For more information on the available methods, check
    :meth:`the mcx documentation page <qrisp.mcx>`.

    Parameters
    ----------
    qubits : QuantumVariable or list[Qubits]
        The Qubits to control on.
    method : str, optional
        The synthesis method. Available are ``auto``, ``gray``, ``gray_pt``,
        ``gray_pt_inv``, ``maslov``, ``balauca``, ``yong`` and ``hybrid``. The default
        is ``auto``.
    ctrl_state : int or str, optional
        The state on which to activate the Z gate. The default is "1111...".
    num_ancilla : int, optional
        Specifies the amount of ancilla qubits to use. This parameter is used only if
        the method is set to ``hybrid``. The default is 1.

    """

    from qrisp.misc import gate_wrap

    @gate_wrap(permeability="full", is_qfree=True, name="anc supported mcz")
    def mcz_inner(qubits, method="auto", ctrl_state=-1):
        if len(ctrl_state) != n:
            raise Exception(
                f"Given control state {ctrl_state} does not match"
                f"control qubit amount {n}"
            )

        from qrisp import h, x

        if ctrl_state[-1] == "0":
            x(qubits[-1])

        h(qubits[-1])
        mcx(qubits[:-1], qubits[-1], method=method, ctrl_state=ctrl_state[:-1])
        h(qubits[-1])

        if ctrl_state[-1] == "0":
            x(qubits[-1])

        return qubits

    n = len(qubits)

    from qrisp import bin_rep

    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2**n
        ctrl_state = bin_rep(ctrl_state, n)

    if method in ["gray", "auto"]:
        if ctrl_state[-1] == "0":
            x(qubits[-1])

        if len(qubits) > 1:
            append_operation(
                std_ops.ZGate().control(
                    len(qubits) - 1, method=method, ctrl_state=ctrl_state[:-1]
                ),
                qubits,
            )
        else:
            z(qubits[0])

        if ctrl_state[-1] == "0":
            x(qubits[-1])

        return qubits

    return mcz_inner(qubits, method, ctrl_state)


def mcp(phi, qubits, method="auto", ctrl_state=-1):
    """
    Applies a multi-controlled phase gate.

    The available methods are:

    * ``gray`` , which performs a traversal of the gray code.

    * ``balauca`` , which is a modified version of the algorithm presented `here
      <https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf>`_.

    * ``auto`` , which picks ``gray`` for any qubit count less than 4 and ``balauca``
      otherwise.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits : list[Qubit] or QuantumVariable
        The qubits to apply the multi-controlled phase gate on.
    method : str, optional
        The method to deploy. The default is "auto".
    ctrl_state : str or int, optional
        The control state on which to apply the phase. The default is "111...".

    """

    from qrisp.alg_primitives.mcx_algs import hybrid_mcx
    from qrisp import QuantumBool
    from qrisp.misc import bin_rep, gate_wrap
    import numpy as np
    @gate_wrap(permeability="full", is_qfree=True, name="anc supported mcp")
    def balauca_mcp(phi, qubits, ctrl_state):
        from qrisp.circuit.quantum_circuit import convert_to_qb_list
        qubits = convert_to_qb_list(qubits)

        temp = QuantumBool()
        hybrid_mcx(qubits,
                   temp[0], 
                   ctrl_state=ctrl_state, 
                   phase=phi, 
                   num_ancilla=np.inf, 
                   use_mcm = True)
        
        temp.delete()

    n = len(qubits)

    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2**n
        ctrl_state = bin_rep(ctrl_state, n)[::-1]

    n = len(qubits)

    if method == "gray" or method == "gray_pt":
        if ctrl_state[-1] == "0":
            x(qubits[-1])
            
        if check_for_tracing_mode():
            mcp_gate = std_ops.PGate(sympy.Symbol("alpha")).control(n - 1, ctrl_state=ctrl_state[:-1], method = method)
            append_operation(mcp_gate, qubits, param_tracers = [phi])
        else:
            mcp_gate = std_ops.PGate(phi).control(n - 1, ctrl_state=ctrl_state[:-1], method = method)
            append_operation(mcp_gate, qubits)


        if ctrl_state[-1] == "0":
            x(qubits[-1])
        return qubits

    elif method == "balauca":
        balauca_mcp(phi, qubits, ctrl_state=ctrl_state)
        return qubits

    elif method == "auto":
        if n < 4:
            return mcp(phi, qubits, method="gray", ctrl_state=ctrl_state)
        else:
            return mcp(phi, qubits, method="balauca", ctrl_state=ctrl_state)

    else:
        raise Exception(f"Don't know method {method}")


def p(phi, qubits):
    """
    Applies a phase gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit on which to apply the phase gate.

    """
    
    if check_for_tracing_mode():
        append_operation(std_ops.PGate(sympy.Symbol("alpha")), [qubits], param_tracers = [phi])
    else:
        append_operation(std_ops.PGate(phi), [qubits])

    # std_ops.PGate(phi).append([qubits])

    return qubits


def cp(phi, qubits_0, qubits_1):
    """
    Applies a controlled phase gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit.

    """
    
    if check_for_tracing_mode():
        cp_gate = std_ops.CPGate(sympy.Symbol("alpha"))
        append_operation(cp_gate, [qubits_0, qubits_1], param_tracers = [phi])    
    else:
        cp_gate = std_ops.CPGate(phi)
        append_operation(cp_gate, [qubits_0, qubits_1])    
    
    return qubits_0, qubits_1


def rx(phi, qubits):
    """
    Applies an RX gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the RX gate on.

    """

    if check_for_tracing_mode():
        append_operation(std_ops.RXGate(sympy.Symbol("alpha")), [qubits], param_tracers = [phi])
    else:
        append_operation(std_ops.RXGate(phi), [qubits])

    return qubits


def ry(phi, qubits):
    """
    Applies an RY gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the RY gate on.

    """

    if check_for_tracing_mode():
        append_operation(std_ops.RYGate(sympy.Symbol("alpha")), [qubits], param_tracers = [phi])
    else:
        append_operation(std_ops.RYGate(phi), [qubits])

    return qubits


def rz(phi, qubits):
    """
    Applies an RY gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the RY gate on.

    """

    if check_for_tracing_mode():
        append_operation(std_ops.RZGate(sympy.Symbol("alpha")), [qubits], param_tracers = [phi])
    else:
        append_operation(std_ops.RZGate(phi), [qubits])

    return qubits


def crz(phi, qubits_0, qubits_1):
    """
    Applies controled RZ gate

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit.

    """

    if check_for_tracing_mode():
        crz_gate = std_ops.RZGate(sympy.Symbol("alpha")).control(1)
        append_operation(crz_gate, [qubits_0, qubits_1], param_tracers = [phi])
    else:
        crz_gate = std_ops.RZGate(phi).control(1)
        append_operation(crz_gate, [qubits_0, qubits_1])
        
    return qubits_0, qubits_1


def s(qubits):
    """
    Applies an S gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the S gate on.
    """

    append_operation(std_ops.SGate(), [qubits])
    return qubits


def t(qubits):
    """
    Applies a T gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the T gate on.
    """
    append_operation(std_ops.TGate(), [qubits])
    return qubits


def s_dg(qubits):
    """
    Applies a daggered S gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the daggered S gate on.
    """

    append_operation(std_ops.SGate().inverse(), [qubits])
    return qubits


def t_dg(qubits):
    """
    Applies a daggered T gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the daggered T gate on.
    """
    append_operation(std_ops.TGate().inverse(), [qubits])
    return qubits


def sx(qubits):
    """
    Applies an SX gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the SX gate on.
    """

    append_operation(std_ops.SXGate().inverse(), [qubits])

    return qubits


def sx_dg(qubits):
    """
    Applies a daggered SX gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the daggered SX gate on.
    """
    append_operation(std_ops.SXDGGate().inverse(), [qubits])
    return qubits


def gphase(phi, qubits):
    """
    Applies a global phase. This gate turns into a phase gate when controlled.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The global phase to apply.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the global phase gate on.
    """

    if check_for_tracing_mode():
        append_operation(std_ops.GPhaseGate(sympy.Symbol("alpha")), [qubits], param_tracers = [phi])
    else:
        append_operation(std_ops.GPhaseGate(phi), [qubits])
    return qubits



def xxyy(phi, beta, qubits_0, qubits_1):
    """
    Applies an XXYY interaction gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The global phase to apply.
    beta : float or sympy.Symbol
        The other angle parameter.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit to perform the XXYY gate on.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit to perform the XXYY gate on.
    """

    if check_for_tracing_mode():
        xxyy_gate = std_ops.XXYYGate(sympy.Symbol("alpha"), sympy.Symbol("beta"))
        append_operation(xxyy_gate, [qubits_0, qubits_1], param_tracers = [phi, beta])
        
    else:
        xxyy_gate = std_ops.XXYYGate(phi, beta)
        append_operation(xxyy_gate, [qubits_0, qubits_1])
    
    return qubits_0, qubits_1


def rzz(phi, qubits_0, qubits_1):
    """
    Applies an RZZ gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    beta : float or sympy.Symbol
        The other angle parameter.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first argument to perform the RZZ gate one.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second argument to perform the RZZ gate one.
    """

    if check_for_tracing_mode():
        rzz_gate = std_ops.RZZGate(sympy.Symbol("alpha"))
        append_operation(rzz_gate, [qubits_0, qubits_1], param_tracers = [phi])
    else:
        rzz_gate = std_ops.RZZGate(phi)
        append_operation(rzz_gate, [qubits_0, qubits_1])
    return qubits_0, qubits_1

def rxx(phi, qubits_0, qubits_1):
    """
    Applies an RXX gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    beta : float or sympy.Symbol
        The other angle parameter.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first argument to perform the RXX gate one.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second argument to perform the RXX gate one.
    """

    if check_for_tracing_mode():
        rxx_gate = std_ops.RXXGate(sympy.Symbol("alpha"))
        append_operation(rxx_gate, [qubits_0, qubits_1], param_tracers = [phi])
    else:
        rxx_gate = std_ops.RZZGate(phi)
        append_operation(rxx_gate, [qubits_0, qubits_1])
        
    append_operation(rxx_gate, [qubits_0, qubits_1])
    return qubits_0, qubits_1

def u3(theta, phi, lam, qubits):
    """
    Applies an U3 gate.

    Parameters
    ----------
    theta : float or sympy.Symbol
        The first angle parameter.
    phi : float or sympy.Symbol
        The second angle parameter.
    lambda : float or sympy.Symbol
        The third angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the U3 gate on.
    """

    if check_for_tracing_mode():
        append_operation(std_ops.U3Gate(sympy.Symbol("alpha"), sympy.Symbol("beta"), sympy.symbol("gamma")), [qubits], param_tracers = [phi, theta, lam])
    else:
        append_operation(std_ops.U3Gate(theta, phi, lam), [qubits])

    return qubits


def measure(qubits):
    """
    Performs a measurement of the specified Qubit.

    Parameters
    ----------
    qubit : Qubit or list[Qubit] or QuantumVariable
        The Qubit to measure.
    clbit : Clbit, optional
        The Clbit to store the result in. By default, a new Clbit will be created.

    """
    from qrisp import find_qs
    from qrisp.jasp import TracingQuantumSession
    qs = find_qs(qubits)
    
    if not isinstance(qs, TracingQuantumSession):
        raise Exception("measure function is available only in Jasp mode")
    else:
        from qrisp.jasp import Measurement_p, AbstractQubit, AbstractQubitArray
        from qrisp import QuantumVariable
        
        
        if isinstance(qubits, QuantumVariable):
            abs_qc, res = Measurement_p.bind(qs.abs_qc, qubits.reg.tracer)
            res = qubits.jdecoder(res)
        elif isinstance(qubits.aval, AbstractQubitArray):
            abs_qc, res = Measurement_p.bind(qs.abs_qc, qubits.tracer)
        elif isinstance(qubits.aval, AbstractQubit):
            abs_qc, res = Measurement_p.bind(qs.abs_qc, qubits)
        else:
            raise Exception(f"Tried to measure type {type(qubits.aval)}")
        
        qs.abs_qc = abs_qc
        
        return res


def barrier(qubits):
    """
    A visual marker for structuring the QuantumCircuit.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to apply the barrier to.

    Examples
    --------

    >>> from qrisp import QuantumVariable, x, y, barrier
    >>> qv = QuantumVariable(5)
    >>> x(qv)
    >>> barrier(qv)
    >>> y(qv)
    >>> print(qv.qs)
    
    ::
    
        QuantumCircuit:
        --------------
              ┌───┐ ░ ┌───┐
        qv.0: ┤ X ├─░─┤ Y ├
              ├───┤ ░ ├───┤
        qv.1: ┤ X ├─░─┤ Y ├
              ├───┤ ░ ├───┤
        qv.2: ┤ X ├─░─┤ Y ├
              ├───┤ ░ ├───┤
        qv.3: ┤ X ├─░─┤ Y ├
              ├───┤ ░ ├───┤
        qv.4: ┤ X ├─░─┤ Y ├
              └───┘ ░ └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv

    """
    from qrisp import Qubit
    if isinstance(qubits, Qubit):
        qubits = [qubits]
    
    append_operation(std_ops.Barrier(len(qubits)), qubits)

    return qubits


def swap(qubits_0, qubits_1):
    """
    Applies a SWAP gate.

    Parameters
    ----------
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit.

    """
    append_operation(std_ops.SwapGate(), [qubits_0, qubits_1])

    return qubits_0, qubits_1