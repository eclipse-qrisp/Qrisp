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


import sys

from qrisp.core.quantum_variable import QuantumVariable

class QuantumBool(QuantumVariable):
    """
    QuantumBools are the quantum type, which represents boolean truth values.
    They are the return type of comparison operators like the equality ``==``.

    Apart from their behavior as a QuantumVariable, they can also be treated like
    :ref:`ControlEnvironments <ControlEnvironment>`.

    .. note:
        QuantumBools that are evaluated directly after a ``with`` statement are
        uncomputed automatically upon leaving the ControlEnvironment.

    Examples
    --------

    We create a QuantumBool and set it to uniform superposition

    >>> from qrisp import QuantumBool, h
    >>> q_bool_0 = QuantumBool()
    >>> h(q_bool_0)
    >>> print(q_bool_0)
    {False: 0.5, True: 0.5}

    We create a second QuantumBool and evaluate some logical functions

    >>> q_bool_1 = QuantumBool()
    >>> print(q_bool_1 | q_bool_0)
    {False: 0.5, True: 0.5}
    >>> print(q_bool_1 & q_bool_0)
    {False: 1.0}

    QuantumBools are the results of comparisons:

    >>> from qrisp import QuantumFloat, QuantumChar
    >>> q_ch = QuantumChar()
    >>> q_ch[:] = {"g" : 1, "l" : -1}
    >>> q_bool_2 = (q_ch == "g")
    >>> q_bool_2.qs.statevector()
    sqrt(2)*(|g>*|True> - |l>*|False>)/2

    For :ref:`QuantumFloats <QuantumFloat>`, numeric comparison is also possible:


    >>> qf = QuantumFloat(4)
    >>> h(qf[3])
    >>> print(qf)
    {0: 0.5, 8: 0.5}
    >>> q_bool_3 = (qf >=  4)
    >>> print(q_bool_3)
    {False: 0.5, True: 0.5}

    To use a QuantumBool as a :ref:`ControlEnvironment`, we simply put it in a ``with``
    statement:

    ::

        with q_bool_3:
            qf += 2

    >>> print(qf)
    {0: 0.5, 10: 0.5}

    QuantumBools that are created directly after a ``with`` statement are uncomputed
    automatically:

    ::

        with qf == 10:
            q_bool_3.flip()

    >>> print(qf.qs)
    
    ::
    
        QuantumCircuit:
        --------------
                           ┌────────────┐     ┌───────────┐
                qf.0: ─────┤0           ├─────┤0          ├──o─────────o──
                           │            │     │           │  │         │
                qf.1: ─────┤1           ├─────┤1          ├──■─────────■──
                           │            │     │  __iadd__ │  │         │
                qf.2: ─────┤2           ├─────┤2          ├──o─────────o──
                      ┌───┐│  less_than │     │           │  │         │
                qf.3: ┤ H ├┤3           ├─────┤3          ├──■─────────■──
                      └───┘│            │┌───┐└─────┬─────┘  │  ┌───┐  │
            lt_qbl.0: ─────┤4           ├┤ X ├──────■────────┼──┤ X ├──┼──
                           │            │└───┘               │  └─┬─┘  │
        lt_ancilla.0: ─────┤5           ├────────────────────┼────┼────┼──
                           └────────────┘                  ┌─┴─┐  │  ┌─┴─┐
          cond_env.0: ─────────────────────────────────────┤ X ├──■──┤ X ├
                                                           └───┘     └───┘
        Live QuantumVariables:
        ---------------------
        QuantumFloat qf
        QuantumBool lt_qbl

    Note that there is only a single QuantumBool listed in the "Live QuantumVariables"
    section, because the QuantumBool of the comparison
    ``qf == 10`` (called ``cond_env``) has been uncomputed.

    """

    def __init__(self, qs=None, name=None):
        QuantumVariable.__init__(self, 1, qs=qs, name=name)

        self.qfloat_comparison = False

    def decoder(self, integer):
        return bool(integer)

    def __and__(self, other):
        from qrisp import mcx

        and_qbl = QuantumBool()

        mcx(self.reg + other.reg, and_qbl.reg)

        return and_qbl

    def __or__(self, other):
        from qrisp import mcx, x

        or_qbl = QuantumBool()

        x(self)
        x(other)

        mcx(self.reg + other.reg, or_qbl.reg)

        x(self)
        x(other)

        x(or_qbl)

        return or_qbl

    def __xor__(self, other):
        from qrisp import cx

        xor_qbl = QuantumBool()

        cx(self, xor_qbl)
        cx(other, xor_qbl)

        return xor_qbl

    def flip(self):
        """
        Flips the QuantumBool's value.

        """

        from qrisp import x

        x(self)
        return self
    
    def __invert__(self):
        inverted_qbl = QuantumBool()
        from qrisp import cx
        cx(self, inverted_qbl)
        inverted_qbl.flip()
        return inverted_qbl
        

    def __enter__(self):
        from qrisp.environments import control
        self.env = control(self)
        self.env.__enter__()

    def __exit__(self, a, b, c):
        ref_count = sys.getrefcount(self)

        # If the refcount is 5, this means that the QuantumBool has been
        # created in a "with" statement and can therefore no longer be reached
        # after exiting. We therefore uncompute

        self.env.__exit__(a, b, c)
        if ref_count == 5:
            self.uncompute()
            
    def __bool__(self):
        raise Exception("Tried to convert QuantumBool to classical bool (probable due using the and + or keywords - try using & + | instead)")
