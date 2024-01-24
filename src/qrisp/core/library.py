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


import numpy as np

import qrisp.circuit.standard_operations as std_ops
from qrisp.core import recursive_qs_search

def append_operation(operation, qubits=[], clbits=[]):
    from qrisp import find_qs
    
    qs = find_qs(qubits)
    
    qs.append(operation, qubits, clbits)


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
            - More efficient but introduce extra phases that need to be uncomputed by performing the inverse of this gate on the same inputs. For more information on phase tolerance, check `this paper <https://iopscience.iop.org/article/10.1088/2058-9565/acaf9d/meta>`_.
        *   - ``balauca`` 
            - Method based on this `paper <https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf>`_ with logarithmic depth but requires many ancilla qubits.
        *   - ``maslov``
            - Documented `here <https://arxiv.org/abs/1508.03273>`_, requires less ancilla qubits but is only available for 4 or less control qubits.
        *   - ``yong`` 
            - Can be found int this `article <https://link.springer.com/article/10.1007/s10773-017-3389-4>`_.This method requires only a single ancilla and has moderate scaling in depth and gate count.
        *   - ``amy``
            - A Toffoli-circuit (ie. only two control qubits are possible), which (temporarily) requires one ancilla qubit. However, instead of the no-ancilla T-depth 4, this circuit achieves a T-depth of 2. Find the implementation details in `this paper <https://arxiv.org/pdf/1206.0758.pdf>`_.
        *   - ``jones``
            - Similar to ``amy`` but uses two ancilla qubits, and has a T-depth of 1. Read about it `here <https://arxiv.org/abs/1212.5069>`_.
        *   - ``gidney``
            - A very unique way for synthesizing a logical AND. The Gidney Logical AND performs a circuit with T-depth 1 to compute the truth value and performs another circuit involving a measurement and a classically controlled CZ gate for uncomputation. The uncomputation circuit has T-depth 0, such that the combined T-depth is 1. Requires no ancillae. More details `here <https://arxiv.org/abs/1709.06648>`_. Works only for two control qubits.
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

    from qrisp.circuit.quantum_circuit import convert_to_qb_list
    from qrisp.misc import bin_rep
    from qrisp.mcx_algs import GidneyLogicalAND, amy_toffoli, jones_toffoli

    qubits_0 = convert_to_qb_list(controls)
    qubits_1 = convert_to_qb_list(target)

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

    from qrisp.mcx_algs import (
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

    from qrisp.mcx_algs import balauca_mcx
    from qrisp.misc import bin_rep, gate_wrap

    @gate_wrap(permeability="full", is_qfree=True, name="anc supported mcp")
    def balauca_mcp(phi, qubits, ctrl_state):
        from qrisp.circuit.quantum_circuit import convert_to_qb_list

        qubits = convert_to_qb_list(qubits)
        if ctrl_state[-1] == "0":
            x(qubits[-1])

        balauca_mcx(qubits[:-1], [qubits[-1]], ctrl_state=ctrl_state, phase=phi)

        if ctrl_state[-1] == "0":
            x(qubits[-1])

    n = len(qubits)

    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2**n
        ctrl_state = bin_rep(ctrl_state, n)[::-1]

    n = len(qubits)

    if method == "gray" or method == "gray_pt":
        if ctrl_state[-1] == "0":
            x(qubits[-1])

        append_operation(
            std_ops.PGate(phi).control(n - 1, ctrl_state=ctrl_state[:-1], method = method), qubits
        )

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

    append_operation(std_ops.CPGate(phi), [qubits_0, qubits_1])

    # std_ops.CPGate(phi).append([qubits_0, qubits_1])

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

    append_operation(std_ops.RZGate(phi).control(1), [qubits_0, qubits_1])
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

    append_operation(std_ops.GPhaseGate(phi), qubits)
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

    append_operation(std_ops.XXYYGate(phi, beta), [qubits_0, qubits_1])
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

    append_operation(std_ops.RZZGate(phi), [qubits_0, qubits_1])
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

    append_operation(std_ops.RXXGate(phi), [qubits_0, qubits_1])
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

    append_operation(std_ops.RZGate(phi), [qubits])

    return qubits


def measure(qubits, clbits=None):
    """
    Performs a measurement of the specified Qubit.

    Parameters
    ----------
    qubit : Qubit or list[Qubit] or QuantumVariable
        The Qubit to measure.
    clbit : Clbit, optional
        The Clbit to store the result in. By default, a new Clbit will be created.

    """
    if clbits is None:
        clbits = []
        if hasattr(qubits, "__len__"):
            for qb in qubits:
                try:
                    clbits.append(qubits[0].qs.add_clbit())
                except AttributeError:
                    clbits.append(qubits[0].qs().add_clbit())

        else:
            clbits = qubits.qs.add_clbit()
    append_operation(std_ops.Measurement(), [qubits], [clbits])

    return qubits


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


def id(qubits):
    """
    Applies an ID gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The qubits to perform the ID gate on.

    """
    append_operation(std_ops.IDGate(), [qubits])

    return qubits


def QFT_inner(qv, exec_swap=True, qiskit_endian=True, inplace_mult=1, use_gms=False, inpl_adder = None):
    from qrisp.misc import is_inv

    qv = list(qv)
    n = len(qv)

    if qiskit_endian:
        qv = qv[::-1]

    if not use_gms:
        from qrisp.environments.quantum_environments import QuantumEnvironment

        env = QuantumEnvironment

    else:
        from qrisp.environments.GMS_environment import GMSEnvironment

        env = GMSEnvironment

    if not is_inv(inplace_mult, n):
        raise Exception(
            "Tried to perform non-invertible inplace multiplication"
            "during Fourier-Transform"
        )


    if inpl_adder is None:
        accumulated_phases = np.zeros(n)
        for i in range(n):
            if accumulated_phases[i] and not use_gms:
                p(accumulated_phases[i], qv[i])
                accumulated_phases[i] = 0
            
            h(qv[i])
    
            if i == n - 1:
                break
    
            with env():
                for k in range(n - i - 1):
                    # cp(inplace_mult * 2 * np.pi / 2 ** (k + 2), qv[k + i + 1], qv[i])
                    
                    if use_gms:
                        cp(inplace_mult * 2 * np.pi / 2 ** (k + 2), qv[i], qv[k + i + 1])
                    else:
                        phase = inplace_mult * 2 * np.pi / 2 ** (k + 2)
                        
                        # cx(qv[k + i + 1], qv[i])
                        # p(-phase/2, qv[i])
                        # cx(qv[k + i + 1], qv[i])
                        
                        
                        cx(qv[i], qv[k + i + 1])
                        p(-phase/2, qv[k + i + 1])
                        cx(qv[i], qv[k + i + 1])
                        
                        
                        accumulated_phases[i] += phase/2
                        accumulated_phases[k + i + 1] += phase/2
        
        
                    
        for i in range(n):
            if accumulated_phases[i] and not use_gms:
                p(accumulated_phases[i], qv[i])
                accumulated_phases[i] = 0
                
    else:
        
        from qrisp import QuantumFloat, conjugate
        reservoir = QuantumFloat(n+1)
        
        def prepare_reservoir(reservoir):
            n = len(reservoir)
            h(reservoir)
            for i in range(n):
                p(np.pi*2**(i-n+1), reservoir[i])
        
        
        with conjugate(prepare_reservoir)(reservoir):

            for i in range(n):
                
                h(qv[i])
        
                if i == n - 1:
                    break
        
                phase_qubits = []
                for k in range(n - i - 1):
                    cx(qv[i], qv[k + i + 1])
                    phase_qubits.append(qv[k + i + 1])
                
                inpl_adder(phase_qubits[::-1], reservoir[-len(phase_qubits)-2:])
                    
                for k in range(n - i - 1):
                    cx(qv[i], qv[k + i + 1])
                
                x(reservoir)
                inpl_adder(phase_qubits[::-1], reservoir[-len(phase_qubits)-2:])
                x(reservoir)
            
            s(qv)
            inpl_adder(qv, reservoir[-n-1:])
        
        reservoir.delete()

    if exec_swap:
        for i in range(n // 2):
            swap(qv[i], qv[n - i - 1])

    return qv


def QFT(
    qv, inv=False, exec_swap=True, qiskit_endian=True, inplace_mult=1, use_gms=False, inpl_adder=None
):
    """
    Performs the quantum fourier transform on the input.

    Parameters
    ----------
    qv : QuantumVariable
        QuantumVariable to transform (in-place).
    inv : bool, optional
        If set to True, the inverse transform will be applied. The default is False.
    exec_swap : bool, optional
        If set to False, the swaps at the end of the transformation will be skipped.
        The default is True.
    qiskit_endian : bool, optional
        If set to False the order of bits will be reversed. The default is True.
    inplace_mult : int, optional
        Allows multiplying the QuantumVariable with an extra factor during the
        transformation. For more information check `the publication
        <https://ieeexplore.ieee.org/document/9815035>`_. The default is 1.
    use_gms : bool, optional
        If set to True, the QFT will be compiled using only GMS gates as entangling
        gates. The default is False.
    inpl_adder : callable, optional
        Uses an adder and a reservoir state to perform the QFT. Read more about 
        it :ref:`here <adder_based_qft>`. The default is None


    """
    from qrisp import gate_wrap, invert

    name = "QFT"
    if not exec_swap:
        name += " no swap"
    if inplace_mult != 1:
        name += " inpl mult " + str(inplace_mult)
    if inpl_adder is not None:
        name += "_adder_supported"

    if inv:
        with invert():
            gate_wrap(permeability=[], is_qfree=False, name=name)(QFT_inner)(
                qv,
                exec_swap=exec_swap,
                qiskit_endian=qiskit_endian,
                inplace_mult=inplace_mult,
                use_gms=use_gms,
                inpl_adder=inpl_adder
            )
    else:
        gate_wrap(permeability=[], is_qfree=False, name=name)(QFT_inner)(
            qv,
            exec_swap=exec_swap,
            qiskit_endian=qiskit_endian,
            inplace_mult=inplace_mult,
            use_gms=use_gms,
            inpl_adder=inpl_adder
        )

    return qv


def QPE(
    args, U, precision=None, target=None, iter_spec=False, ctrl_method=None, kwargs={}
):
    r"""
    Evaluates the `quantum phase estimation algorithm
    <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`_.

    The unitary to estimate is expected to be given as Python function, which is called
    on ``args``.

    Parameters
    ----------
    args : list
        A list of arguments (could be QuantumVariables) which represent the state,
        the quantum phase estimation is performed on.
    U : function
        A Python function, which will receive the list ``args`` as arguments in the
        course of this algorithm.
    precision : int, optional
        The precision of the estimation. The default is None.
    target : QuantumFloat, optional
        A target QuantumFloat to perform the estimation into. The default is None.
        If given neither a precision nor a target, an Exception will be raised.
    iter_spec : bool, optional
        If set to ``True``, ``U`` will be called with the additional keyword
        ``iter = i`` where ``i`` is the amount of iterations to perform (instead of
        simply calling ``U`` for ``i`` times). The default is False.
    ctrl_method : string, optional
        Allows to specify which method should be used to generate the
        controlled U circuit. For more information check
        :meth:`.control <qrisp.Operation.control>`. The default is None.
    kwargs : dict, optional
        A dictionary of keyword arguments to pass to ``U``. The default is {}.

    Raises
    ------
    Exception
        Tried to perform quantum phase estimation without precision specification.

    Returns
    -------
    res : QuantumFloat
        The QuantumFloat containing the estimated phase as a fraction of $2 \pi$.

    Examples
    --------

    We define a function that applies two phase gates onto its input and estimate the
    applied phase. ::

        from qrisp import p, QuantumVariable, QPE, multi_measurement

        def U(qv):
            x = 0.5
            y = 0.125

            p(x*2*np.pi, qv[0])
            p(y*2*np.pi, qv[1])

        qv = QuantumVariable(2)

        h(qv)

        res = QPE(qv, U, precision = 3)

    >>> multi_measurement([qv, res])
    {('00', 0.0): 0.25,
     ('10', 0.5): 0.25,
     ('01', 0.125): 0.25,
     ('11', 0.625): 0.25}
    >>> res.qs.depth()
    66

    During the phase estimation, ``U`` is called $2^{\text{precision}}$ times. We can
    reduce that number by abusing that we can bundle repeated calls into a single call
    with a modified phase. ::

        def U(qv, iter = None):
            x = 0.5
            y = 0.125

            p(x*2*np.pi*iter, qv[0])
            p(y*2*np.pi*iter, qv[1])

        qv = QuantumVariable(2)

        h(qv)

        res = QPE(qv, U, precision = 3, iter_spec = True)

    >>> multi_measurement([qv, res])
    {('00', 0.0): 0.25,
     ('10', 0.5): 0.25,
     ('01', 0.125): 0.25,
     ('11', 0.625): 0.25}
    >>> res.qs.depth()
    34

    """

    from qrisp import QuantumFloat, control

    if target is None:
        if precision is None:
            raise Exception(
                "Tried to perform quantum phase estimation without"
                "precision specification"
            )
        qpe_res = QuantumFloat(precision, -precision, signed=False)
    else:
        qpe_res = target

    h(qpe_res)

    for i in range(qpe_res.size):
        if iter_spec:
            with control(qpe_res[i], ctrl_method=ctrl_method):
                U(args, iter=2**i, **kwargs)
        else:
            with control(qpe_res[i], ctrl_method=ctrl_method):
                for j in range(2**i):
                    U(args, **kwargs)

    QFT(qpe_res, inv=True)

    return qpe_res


def quantum_counting(qv, oracle, precision):
    """
    This algorithm estimates the amount of solutions for a given Grover oracle.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable on which to evaluate.
    oracle : function
        The oracle function.
    precision : int
        The precision to perform the quantum phase estimation with.

    Returns
    -------
    M : float
        An estimate of the amount of solutions.

    Examples
    --------

    We create an oracle, which performs a simple phase flip on the last qubit. ::

        from qrisp import quantum_counting, z, QuantumVariable

        def oracle(qv):
            z(qv[-1])


    We expect half of the state-space of the input to be a solution.

    For 3 qubits, the state space is $2^3 = 8$ dimensional.

    >>> quantum_counting(QuantumVariable(3), oracle, 3)
    3.999999999999999

    For 4 qubits, the state space is $2^4 = 16$ dimensional.

    >>> quantum_counting(QuantumVariable(4), oracle, 3)
    7.999999999999998


    """

    from qrisp import gate_wrap
    from qrisp.grover import diffuser

    @gate_wrap
    def grover_operator(qv):
        oracle(qv)
        diffuser(qv)

    h(qv)
    res = QPE(qv, grover_operator, precision=precision)

    mes_res = res.get_measurement()

    theta = min(list(mes_res.keys())[:1]) * 2 * np.pi

    N = 2**qv.size
    M = N * np.sin(theta / 2) ** 2

    return M


def HHL(qv, hamiltonian_evolution, ev_inversion, precision):
    """
    Evaluates the HHL algorithm.

    Parameters
    ----------
    qv : TYPE
        DESCRIPTION.
    hamiltonian_evolution : TYPE
        DESCRIPTION.
    ev_inversion : TYPE
        DESCRIPTION.
    precision : TYPE
        DESCRIPTION.

    Returns
    -------
    ancilla : TYPE
        DESCRIPTION.

    """
    from qrisp import QuantumBool, invert

    # Perform quantum phase estimation
    res = QPE(qv, hamiltonian_evolution, precision)

    ancilla = QuantumBool()

    ev_inversion(res, ancilla)

    # Perform quantum phase estimation inverse
    with invert():
        QPE(qv, hamiltonian_evolution, target=res)

    res.delete()

    return ancilla


def fredkin_qc(num_ctrl_qubits=1, ctrl_state=-1, method="gray"):
    from qrisp import QuantumCircuit, XGate
    mcx_gate = XGate().control().control(ctrl_state=ctrl_state, method=method)

    qc = QuantumCircuit(num_ctrl_qubits + 2)
    qc.cx(qc.qubits[-1], qc.qubits[-2])
    qc.append(mcx_gate, qc.qubits)
    qc.cx(qc.qubits[-1], qc.qubits[-2])

    return qc


def demux(input, ctrl_qv, output=None, ctrl_method=None, permit_mismatching_size=False, parallelize_qc = False):
    """
    This functions allows moving an input value into an iterable output, where the
    position is specified by a ``QuantumFloat``. Demux is short for demultiplexer and
    is a standard component in `classical electrical circuitry
    <https://en.wikipedia.org/wiki/Multiplexer>`_.

    Demux can either move qubit states into a QuantumVariable or ``QuantumVariables``
    into ``QuantumArrays``.

    This function can also be used to "in-place demux" the 0-th entry of an iterable to
    the position specified by ``ctrl_qv``. For more information on this, check the
    second example.

    Parameters
    ----------
    input : Qubit or QuantumVariable
        The input value that is supposed to be moved.
    ctrl_qv : QuantumFloat
        The QuantumFloat specifying to which output the input should be moved.
    output : QuantumVariable or QuantumArray, optional
        The output object, where the input should end up. By default, a new object
        (QuantumVariable or QuantumArray) is created. Note that when this parameter is
        given, it is guaranteed, that the 0-th entry will be moved to the desired
        position, the other entries can also be permuted away from their original
        position.
    ctrl_method : string, optional
        The ``ctrl_method`` string passed to the
        :ref:`control environment <ControlEnvironment>` to generate controlled swaps.
    permit_mismatching_size : bool, optional
        If set to False, an exception will be raised, if the state-space dimension of
        `ctrl_qv`` is differing from the amount of outputs. The default is False.
    parallelize_qc : bool, optional
        If set to True, this option reduces (de)allocates additional qubits to
        reduce the depth. The default is False.

    Raises
    ------
    Exception
        Tried to demux with mismatchingly sized control input.

    Returns
    -------
    output : QuantumVariable or QuantumArray
        The output object with the input signal placed at the index specified by
        ``ctrl_qv``.

    Examples
    --------

    We create a ``QuantumBool`` and demux it into a ``QuantumArray`` ::

        from qrisp import *

        qb = QuantumBool()
        qb.flip()

        index = QuantumFloat(2)

        h(index[1])

        res_array = demux(qb, index)

    >>> print(multi_measurement([index, res_array]))
    {(0, OutcomeArray([1., 0., 0., 0.])): 0.5, (2, OutcomeArray([0., 0., 1., 0.])): 0.5}

    Demux can also be used to move the 0-th entry of a ``QuantumArray`` in-place. ::

        qa = QuantumArray(shape = 4, qtype = qb)

        qa[0].flip()

        demux(qa[0], index, qa)

    >>> print(multi_measurement([index, qa]))
    {(0, OutcomeArray([1., 0., 0., 0.])): 0.5, (2, OutcomeArray([0., 0., 1., 0.])): 0.5}

    For low-level manipulations, demux can move information within ``QuantumVariables``.
    ::

        qf = QuantumVariable(4)

        qf[:] = "1000"

        demux(qf[0], index, qf)

    >>> print(multi_measurement([index, qf]))
    {(0, '1000'): 0.5, (2, '0010'): 0.5}
    """

    from qrisp import QuantumArray, QuantumVariable, Qubit, control, swap

    if output is None:
        if isinstance(input, QuantumVariable):
            output = QuantumArray(input, 2 ** len(ctrl_qv))
        elif isinstance(input, Qubit):
            output = QuantumVariable(2 ** len(ctrl_qv))
        else:
            raise Exception("Don't know how to handle input type " + str(type(input)))
    else:
        if isinstance(output, QuantumArray):
            for qv in output.flatten()[1:]:
                if qv.name == input.name:
                    raise Exception(
                        "Tried to in-place demux QuantumArray entry,"
                        "which is not a 0-th position"
                    )
        elif isinstance(output, QuantumVariable):
            for qb in output.reg[1:]:
                if qb.identifier == input.identifier:
                    raise Exception(
                        "Tried to in-place demux QuantumVariable entry,"
                        "which is not a 0-th position"
                    )

    n = int(np.ceil(np.log2(len(output))))
    N = 2**n

    if len(output) != 2 ** len(ctrl_qv) and not permit_mismatching_size:
        raise Exception("Tried to demux with mismatching sized control input")

    if hash(input) != hash(output[0]):
        swap(input, output[0])

    if not len(ctrl_qv):
        return output

    if len(output) > 2 ** (len(ctrl_qv) - 1):
        with control(ctrl_qv[-1], ctrl_method=ctrl_method):
            swap(output[0], output[N // 2])
    else:
        demux(
            output[0],
            ctrl_qv[:-1],
            output,
            ctrl_method=ctrl_method,
            permit_mismatching_size=permit_mismatching_size,
        )
        return output

    if n > 1:
        
        if parallelize_qc:
            demux_ancilla = QuantumVariable(len(ctrl_qv)-1)
            cx(ctrl_qv[:-1], demux_ancilla)
            ctrl_qubits = list(demux_ancilla)
        else:
            ctrl_qubits = ctrl_qv[:-1]
            
        
        demux(
            output[0],
            ctrl_qubits,
            #ctrl_qv[:-1],
            output[: N // 2],
            ctrl_method=ctrl_method,
            permit_mismatching_size=permit_mismatching_size,
            parallelize_qc=parallelize_qc
        )
        
        
        demux(
            output[N // 2],
            ctrl_qv[:-1],
            output[N // 2 :],
            ctrl_method=ctrl_method,
            permit_mismatching_size=permit_mismatching_size,
            parallelize_qc=parallelize_qc
        )
        if parallelize_qc:
            cx(ctrl_qv[:-1], demux_ancilla)
            demux_ancilla.delete()
        


    return output


def q_indexing(q_array, index):
    from qrisp import invert

    with invert():
        demux(q_array[0], index, q_array, ctrl_method="gray_pt")

    res = q_array[0].duplicate(init=True)

    demux(q_array[0], index, q_array, ctrl_method="gray_pt")

    return res


def q_swap_into(q_array, index, qv):
    from qrisp import invert, swap

    with invert():
        demux(q_array[0], index, q_array, ctrl_method="gray_pt")

    swap(q_array[0], qv)

    demux(q_array[0], index, q_array, ctrl_method="gray_pt")



def cyclic_shift(iterable, shift_amount = 1):
    r"""
    Performs a cyclic shift of the values of an iterable with logarithmic depth. 
    The shifting amount can be specified.

    
    Parameters
    ----------
    iterable : list[Qubit] or list[QuantumVariable] or QuantumArray
        The iterable to be shifted.
    shift_amount : integer or QuantumFloat, optional
        The iterable will be shifted by that amount. The default is 1.

    Examples
    --------
    
    We create a QuantumArray, initiate a sequence of increments and perform a cyclic shift.
    
    >>> from qrisp import QuantumFloat, QuantumArray, cyclic_shift
    >>> import numpy as np
    >>> qa = QuantumArray(QuantumFloat(3), 8)
    >>> qa[:] = np.arange(8)
    >>> cyclic_shift(qa, shift_amount = 2)
    >>> print(qa)
    {OutcomeArray([6, 7, 0, 1, 2, 3, 4, 5]): 1.0}
    
    We do something similar to demonstrate the shift by quantum values.
    For this we initiate a :ref:`QuantumFloat` in the superposition of 0, 1 and -3.
    
    >>> shift_amount = QuantumFloat(3, signed = True)
    >>> shift_amount[:] = {0 : 3**-0.5, 1: 3**-0.5, -3 : 3**-0.5}
    >>> qa = QuantumArray(QuantumFloat(3), 8)
    >>> qa[:] = np.arange(8)
    >>> cyclic_shift(qa, shift_amount)
    >>> print(qa)
    {OutcomeArray([0, 1, 2, 3, 4, 5, 6, 7]): 0.3333, OutcomeArray([7, 0, 1, 2, 3, 4, 5, 6]): 0.3333, OutcomeArray([3, 4, 5, 6, 7, 0, 1, 2]): 0.3333}
    """
    
    from qrisp import QuantumFloat, control, QuantumBool, cx
    
    if isinstance(shift_amount, QuantumFloat):
        
        if shift_amount.mshape[0] < 0:
            raise Exception("Tried to quantum shift by non-integer QuantumFloat")
        
        if shift_amount.signed:
            with control(shift_amount.sign()):
                cyclic_shift(iterable, -2**(shift_amount.mshape[1]))
            
        for i in range(*shift_amount.mshape):
            with control(shift_amount.significant(i)):
                cyclic_shift(iterable, 2**i)
    
        return
            
    N = len(iterable)
    n = int(np.floor(np.log2(N)))
    
    if N == 0 or not shift_amount%N:
        return
    if shift_amount < 0:
        return cyclic_shift(iterable[::-1], -shift_amount)

    if shift_amount != 1:
        
        perm = np.arange(N)
        perm = (perm - shift_amount)%(N)
        
        permute_iterable(iterable, perm)
        return
    
    singular_shift(iterable[:2**n])
    singular_shift([iterable[0]] + list(iterable[2**n:]), use_saeedi = True)
    

def singular_shift(iterable, use_saeedi = False):
    
    N = len(iterable)
    
    if N in [0,1]:
        return
    
    if use_saeedi:
        #Strategy from https://arxiv.org/abs/1304.7516
        #Seems to perform worse when shifting by a quantum float
        #But better when the shift length is not a power of 2
        for i in range(N//2):
            if (-i)%N == i+1 or i+1 >= N:
                continue
            swap(iterable[-i], iterable[i+1])
            
        for i in range(N//2):
            if (-i)%N == i+2 or i+2 >= N:
                continue
            swap(iterable[-i], iterable[i+2])
        
    else:        
        correction_indices = []
        for i in range(len(iterable)//2):
            swap_tuple = (2*i, 2*i+1)
            swap(iterable[swap_tuple[0]], iterable[swap_tuple[1]])
            correction_indices.append(swap_tuple[0])
            
        singular_shift([iterable[i] for i in correction_indices])


def to_cycles(perm):
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

def permute_iterable(iterable, perm):
    """
    Applies an arbitrary permutation to an iterable with logarithmic depth.

    Parameters
    ----------
    iterable : QuantumArray or List[QuantumVariable] or QuantumVariable or List[Qubit]
        The iterable to perform the permutation on.
    perm : List[integer]
        A list specifying the permutation.

    Examples
    --------
    
    We create a QuantumArray containing increments and apply a specified permutation.
    
    >>> from qrisp import QuantumFloat, QuantumArray, permute_iterable
    >>> import numpy as np
    >>> qa = QuantumArray(QuantumFloat(3), 8)
    >>> qa[:] = np.arange(8)
    >>> permute_iterable(qa, perm = [1,0,3,7,5,2,6,4])
    >>> print(qa)
    {OutcomeArray([1, 0, 3, 7, 5, 2, 6, 4]): 1.0}
    >>> print(qa.qs)
    
    ::
    
        QuantumCircuit:
        --------------
          qa.0: ────────────X──────────────────────
                            │                      
          qa.1: ──────X─────┼──────────────────────
                      │     │                      
          qa.2: ──────┼──X──┼──────────────────────
                ┌───┐ │  │  │                      
        qa_1.0: ┤ X ├─┼──┼──X──────────────────────
                └───┘ │  │                         
        qa_1.1: ──────X──┼─────────────────────────
                         │                         
        qa_1.2: ─────────X─────────────────────────
                                          
        qa_2.0: ───────────────────────────X───────
                ┌───┐                      │       
        qa_2.1: ┤ X ├──────────────────────┼──X────
                └───┘                      │  │    
        qa_2.2: ───────────────────────────┼──┼──X─
                ┌───┐                      │  │  │ 
        qa_3.0: ┤ X ├──────────X───────────┼──┼──┼─
                ├───┤          │           │  │  │ 
        qa_3.1: ┤ X ├──────────┼──X────────┼──┼──┼─
                └───┘          │  │        │  │  │ 
        qa_3.2: ───────────────┼──┼──X─────┼──┼──┼─
                               │  │  │     │  │  │ 
        qa_4.0: ──────X────────┼──┼──┼─────┼──┼──┼─
                      │        │  │  │     │  │  │ 
        qa_4.1: ──────┼──X─────┼──┼──┼─────┼──┼──┼─
                ┌───┐ │  │     │  │  │     │  │  │ 
        qa_4.2: ┤ X ├─┼──┼──X──┼──┼──┼─────┼──┼──┼─
                ├───┤ │  │  │  │  │  │     │  │  │ 
        qa_5.0: ┤ X ├─X──┼──┼──┼──┼──┼──X──X──┼──┼─
                └───┘    │  │  │  │  │  │     │  │ 
        qa_5.1: ─────────X──┼──┼──┼──┼──┼──X──X──┼─
                ┌───┐       │  │  │  │  │  │     │ 
        qa_5.2: ┤ X ├───────X──┼──┼──┼──┼──┼──X──X─
                └───┘          │  │  │  │  │  │    
        qa_6.0: ───────────────┼──┼──┼──┼──┼──┼────
                ┌───┐          │  │  │  │  │  │    
        qa_6.1: ┤ X ├──────────┼──┼──┼──┼──┼──┼────
                ├───┤          │  │  │  │  │  │    
        qa_6.2: ┤ X ├──────────┼──┼──┼──┼──┼──┼────
                ├───┤          │  │  │  │  │  │    
        qa_7.0: ┤ X ├──────────X──┼──┼──X──┼──┼────
                ├───┤             │  │     │  │    
        qa_7.1: ┤ X ├─────────────X──┼─────X──┼────
                ├───┤                │        │    
        qa_7.2: ┤ X ├────────────────X────────X────
                └───┘                              
        Live QuantumVariables:
        ---------------------
        QuantumFloat qa
        QuantumFloat qa_1
        QuantumFloat qa_2
        QuantumFloat qa_3
        QuantumFloat qa_4
        QuantumFloat qa_5
        QuantumFloat qa_6
        QuantumFloat qa_7
    
    """
    
    from sympy.combinatorics import Permutation
    
    inv_perm = list(Permutation(perm)**-1)
    
    cycles = to_cycles(inv_perm)
    
    for c in cycles:
        cyclic_shift([iterable[i] for i in c], 1)


def amplitude_amplification(args, state_function, oracle_function, kwargs_oracle={}, iter=1):
    r"""
    This method performs `quantum amplitude amplification <https://arxiv.org/abs/quant-ph/0005055>`_.

    The problem of quantum amplitude amplification is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}`.
    * Write :math:`\ket{\Psi}=\ket{\Psi_1}+\ket{\Psi_0}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Enhance the probability :math:`a=\langle\Psi_1|\Psi_1\rangle` that a measurement of $\ket{\Psi}$ yields a good state.

    Let $\theta_a\in [0,\pi/2]$ such that $\sin^2(\theta_a)=a$. Then the amplitude amplification operator $\mathcal Q$ acts as

    .. math::

        \mathcal Q^j\ket{\Psi}=\frac{1}{\sqrt{a}}\sin((2j+1)\theta_a)\ket{\Psi_1}+\frac{1}{\sqrt{1-a}}\cos((2j+1)\theta_a)\ket{\Psi_0}.

    Therefore, after $m$ iterations the probability of measuring a good state is $\sin^2((2m+1)\theta_a)$. 
    
    Parameters
    ----------

    args : QuantumVariable or list[QuantumVariable]
        The (list of) QuantumVariables which represent the state,
        the amplitude amplification is performed on.
    state_function : function
        A Python function preparing the state $\ket{\Psi}$.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    oracle_function : function
        A Python function tagging the good state $\ket{\Psi_1}$.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is {}.
    iter : int, optional
        The amount of amplitude amplification iterations to perform. The default is 1.

    Examples
    --------

    We define a function that prepares the state :math:`\ket{\Psi}=\cos(\frac{\pi}{16})\ket{0}+\sin(\frac{\pi}{16})\ket{1}`
    and an oracle that tags the good state :math:`\ket{1}`. In this case, we have :math:`a=\sin^2(\frac{\pi}{16})\approx 0.19509`.
     
    ::

        from qrisp import z, ry, QuantumBool, amplitude_amplification
        import numpy as np

        def state_function(qb):
            ry(np.pi/8,qb)

        def oracle_function(qb):   
            z(qb)
        
        qb = QuantumBool()

        state_function(qb)

    >>> qb.qs.statevector(decimals=5)
    0.98079∣False⟩+0.19509∣True⟩

    We can enhance the probability of measuring the good state with amplitude amplification:

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.83147*|False> + 0.55557*|True> 

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.55557*|False> + 0.83147*|True> 

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.19509*|False> + 0.98079*|True>

    """

    from qrisp import merge, IterationEnvironment
    from qrisp.grover import diffuser

    merge(args)
    qs = recursive_qs_search(args)[0]
    with IterationEnvironment(qs, iter):
        oracle_function(*args, **kwargs_oracle)
        diffuser(args, state_function=state_function)


def QAE(args, state_function, oracle_function, kwargs_oracle={}, precision=None, target=None):
    r"""
    This method implements the canonical quantum amplitude estimation (QAE) algorithm by `Brassard et al. <https://arxiv.org/abs/quant-ph/0005055>`_.

    The problem of quantum amplitude estimation is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}`.
    * Write :math:`\ket{\Psi}=\ket{\Psi_1}+\ket{\Psi_0}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Find an estimate for :math:`a=\langle\Psi_1|\Psi_1\rangle`, the probability that a measurement of $\ket{\Psi}$ yields a good state.

    Parameters
    ----------
    args : QuantumVariable or list[QuantumVariable]
        The (list of) QuantumVariables which represent the state,
        the quantum amplitude estimation is performed on.
    state_function : function
        A Python function preparing the state :math:`\ket{\Psi}`.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    oracle_function : function
        A Python function tagging the good state :math:`\ket{\Psi_1}`.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is {}.
    precision : int, optional
        The precision of the estimation. The default is None.
    target : QuantumFloat, optional
        A target QuantumFloat to perform the estimation into. The default is None.
        If given neither a precision nor a target, an Exception will be raised.

    Returns
    -------
    
    res : QuantumFloat
        A QuantumFloat encoding the angle :math:`\theta` as a fraction of :math:`\pi`,
        such that :math:`\tilde{a}=\sin^2(\theta)` is an estimate for :math:`a`. 

        More precisely, we have :math:`\theta=\pi\frac{y}{M}` for :math:`y\in\{0,\dotsc,M-1\}` and :math:`M=2^{\text{precision}}`.
        After measurement, the estimate :math:`\tilde{a}=\sin^2(\theta)` satisfies

        .. math::

            |a-\tilde{a}|\leq\frac{2\pi}{M}+\frac{\pi^2}{M^2}

        with probability of at least :math:`8/\pi^2`.

    Examples
    --------

    We define a function that prepares the state :math:`\ket{\Psi}=\cos(\frac{\pi}{8})\ket{0}+\sin(\frac{\pi}{8})\ket{1}`
    and an oracle that tags the good state :math:`\ket{1}`. In this case, we have :math:`a=\sin^2(\frac{\pi}{8})`.
     
    ::

        from qrisp import z, ry, QuantumBool, QAE
        import numpy as np

        def state_function(qb):
            ry(np.pi/4,qb)

        def oracle_function(qb):   
            z(qb)

        qb = QuantumBool()

        res = QAE([qb], state_function, oracle_function, precision=3)

    >>> res.get_measurement()
    {0.125: 0.5, 0.875: 0.5}

    That is, after measurement we find $\theta=\frac{\pi}{8}$ or $\theta=\frac{7\pi}{8}$ with probability $\frac12$, respectively.
    Therefore, we obtain the estimate $\tilde{a}=\sin^2(\frac{\pi}{8})$ or $\tilde{a}=\sin^2(\frac{7\pi}{8})$.
    In this case, both results coincide with the exact value $a$.

    
    **Numerical integration**

    
    Here, we demonstarate how to use QAE for numerical integration. 

    Consider a continuous function $f\colon[0,1]\rightarrow[0,1]$. We wish to evaluate

    .. math::

        A=\int_0^1f(x)\mathrm dx.

    For this, we set up the corresponding ``state_function`` acting on the ``input_list``:

    ::

        from qrisp import QuantumFloat, QuantumBool, control, z, h, ry, QAE
        import numpy as np

        n = 6 
        inp = QuantumFloat(n,-n)
        tar = QuantumBool()
        input_list = [inp, tar]

    Here, $N=2^n$ is the number of sampling points the function $f$ is evaluated on. The ``state_function`` acts as

    .. math::

        \ket{0}\ket{0}\rightarrow\frac{1}{\sqrt{N}}\sum\limits_{x=0}^{N-1}\ket{x}\left(\sqrt{1-f(x/N)}\ket{0}+\sqrt{f(x/N)}\ket{1}\right).
    
    Then the probability of measuring $1$ in the target state ``tar`` is

    .. math::

            p(1)=\frac{1}{N}\sum\limits_{x=0}^{N-1}f(x/N),

    which acts as an approximation for the value of the integral $A$.

    The ``oracle_function``, therefore, tags the $\ket{1}$ state of the target state:

    ::

        def oracle_function(inp, tar):
            z(tar)

    For example, if $f(x)=\sin^2(x)$ the ``state_function`` can be implemented as follows:

    ::

        def state_function(inp, tar):
            h(inp)
    
            N = 2**inp.size
            for k in range(inp.size):
                with control(inp[k]):
                    ry(2**(k+1)/N,tar)

    Finally, we apply QAE and obtain an estimate $a$ for the value of the integral $A=0.27268$.

    ::

        prec = 6
        res = QAE(input_list, state_function, oracle_function, precision=prec)
        meas_res = res.get_measurement()

        theta = np.pi*max(meas_res, key=meas_res.get)
        a = np.sin(theta)**2

    >>> a
    0.26430

    """

    state_function(*args)
    res = QPE(args, amplitude_amplification, precision, target, iter_spec=True,
                kwargs={'state_function':state_function, 'oracle_function':oracle_function, 'kwargs_oracle':kwargs_oracle})
  
    return res

