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

from qrisp.misc.utility import lifted
from qrisp.environments import adaptive_condition, conjugate, control


def less_than_gate(a, b):
    from qrisp import QuantumBool, QuantumFloat, cx, inpl_add

    a = a.duplicate()

    if isinstance(b, QuantumFloat):
        b = b.duplicate(qs=a.qs)

    # a[:] = -5
    added_sign = False
    if not a.signed:
        a.extend(1, position=-1)
        a.signed = True
        added_sign = True

    added_qubits_upper = 0

    while True:
        a.extend(1, position=a.size - 1)

        if not added_sign:
            cx(a[-1], a[-2])

        added_qubits_upper += 1

        if isinstance(b, QuantumFloat):
            if not a.mshape[1] < b.mshape[1] + 1:
                break
        else:
            break

    added_qubits_lower = 0

    if isinstance(b, QuantumFloat):
        while a.mshape[0] > b.mshape[0]:
            a.extend(1, position=0)

            a.exponent -= 1

            added_qubits_lower += 1

    a -= b
    # x(a)
    # inpl_add(a, b, ignore_rounding_error=True, ignore_overflow_error=False)
    # x(a)

    if added_sign:
        res = a[-1]

        a.reduce(a[-1])
        a.qs.data.pop(-1)
    else:
        res_qbl = QuantumBool()
        res = res_qbl[0]
        cx(a[-1], res)

    a += b
    # inpl_add(a, b, ignore_rounding_error=True, ignore_overflow_error=False)

    for i in range(added_qubits_upper):
        if not added_sign:
            cx(a[-1], a[-2])

        if added_sign:
            a.reduce(a[-1], verify=False)
        else:
            a.reduce(a[-2], verify=False)

    for i in range(added_qubits_lower):
        a.reduce(a[0], verify=False)
        a.exponent += 1

    if added_sign:
        a.signed = False

    if isinstance(b, QuantumFloat):
        reordered_qubits = a.reg + b.reg + [res]
    else:
        reordered_qubits = a.reg + [res]
    qc = a.qs.compile(cancel_qfts=False)
    # qc = a.qs

    for qb in qc.qubits:
        if qb not in reordered_qubits:
            reordered_qubits.append(qb)

    qc.qubits = reordered_qubits

    res_gate = qc.to_gate("less_than")
    res_gate.is_qfree = True
    res_gate.permeability = {
        i: i < qc.qubits.index(res) for i in range(res_gate.num_qubits)
    }

    return res_gate


@lifted
def less_than(a, b):
    from qrisp import QuantumBool, QuantumFloat, QuantumVariable, x

    if isinstance(a, QuantumFloat) and isinstance(b, QuantumFloat):
        lt_gate = less_than_gate(a, b)

        lt_qbl = QuantumBool(qs=a.qs, name="lt_qbl*")

        anc_amount = lt_gate.num_qubits - a.size - b.size - 1

        if anc_amount:
            lt_ancilla = QuantumVariable(anc_amount, qs=a.qs, name="lt_ancilla*")
            ancillae = lt_ancilla.reg
        else:
            ancillae = []

        a.qs.append(lt_gate, a.reg + b.reg + lt_qbl.reg + ancillae)

        if anc_amount:
            lt_ancilla.delete(verify=False)

        return lt_qbl

    elif isinstance(a, QuantumFloat):
        labels = [a.decoder(i) for i in range(2**a.size)]
        labels.sort()

        if labels[-1] < b:
            return always_true()
        if labels[0] >= b:
            return always_false()

        for i in range(len(labels) - 1):
            if labels[i] < b <= labels[i + 1]:
                b = labels[i + 1]
                break
        else:
            b = labels[-1]

        lt_gate = less_than_gate(a, b)

        lt_qbl = QuantumBool(qs=a.qs, name="lt_qbl*")

        anc_amount = lt_gate.num_qubits - a.size - 1

        if anc_amount:
            lt_ancilla = QuantumVariable(anc_amount, qs=a.qs, name="lt_ancilla*")
            ancillae = lt_ancilla.reg
        else:
            ancillae = []

        a.qs.append(lt_gate, a.reg + lt_qbl.reg + ancillae)

        if anc_amount:
            lt_ancilla.delete(verify=False)

        return lt_qbl

    elif isinstance(b, QuantumFloat):
        added_sign = False
        if not b.signed:
            b.extend(1, b.size)
            b.signed = True
            added_sign = True

        x(b)
        res = less_than(b, -a - 2**b.exponent)

        x(b)

        if added_sign:
            b.reduce(b.reg[-1])
            b.signed = False

        return res


@lifted
def equal(qf_0, qf_1):
    from qrisp import QuantumBool, QuantumFloat, cx, mcx
    from qrisp.jasp import jrange
    import jax.numpy as jnp

    eq_qbl = QuantumBool(qs=qf_0.qs, name="eq_qbl*")

    if isinstance(qf_1, QuantumFloat):
        if qf_1.signed and not qf_0.signed:
            qf_0, qf_1 = qf_1, qf_0

        mcx_qubits = []
        mcx_qubits += qf_0.reg

        if qf_1.signed and qf_0.signed:
            cx(qf_1.sign(), qf_0.sign())

        m0 = qf_0.msize
        m1 = qf_1.msize
        e0 = qf_0.exponent
        e1 = qf_1.exponent

        # We calculate the overlap of bits with the same significance of qf_0 and qf_1.
        l = jnp.max(jnp.array([e0, e1]))
        r = jnp.min(jnp.array([e0 + m0, e1 + m1]))
        d = r - l

        def conjugator(qf_0, qf_1):
            for i in jrange(d):
                cx(qf_1[i + l - e1], qf_0[i + l - e0])

        mcx_qubits += qf_1.reg[:l - e1]
        mcx_qubits += qf_1.reg[e0 + m0 : e1 + m1]

        with conjugate(conjugator)(qf_0, qf_1):
            mcx(mcx_qubits, eq_qbl, ctrl_state=0)

        if qf_1.signed and qf_0.signed:
            cx(qf_1.sign(), qf_0.sign())

        return eq_qbl
    
    #if qf_0.truncate(qf_1) != qf_1:
    #    return always_false()

    # We compare a QuantumFloat to a classical float. If the condition is False, the result is always False.
    with control(qf_0.truncate(qf_1) == qf_1):
        mcx(qf_0, eq_qbl, ctrl_state=qf_0.encoder(qf_1))
    
    return eq_qbl


def always_true():
    from qrisp import QuantumBool

    true_qbl = QuantumBool()
    true_qbl.flip()
    return true_qbl


def always_false():
    from qrisp import QuantumBool

    false_qbl = QuantumBool()
    return false_qbl


@adaptive_condition
def lt(a, b):
    return less_than(a, b)


@adaptive_condition
def gt(a, b):
    return less_than(b, a)


@adaptive_condition
def geq(a, b):
    return lt(a, b).flip()


@adaptive_condition
def leq(a, b):
    return gt(a, b).flip()


@adaptive_condition
def eq(a, b):
    return equal(a, b)


@adaptive_condition
def neq(a, b):
    return equal(a, b).flip()
