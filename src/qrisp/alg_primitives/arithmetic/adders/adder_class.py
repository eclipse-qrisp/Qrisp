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

import numpy as np

import jax.numpy as jnp

from qrisp.qtypes import QuantumFloat, QuantumVariable
from qrisp.jasp import jlen


def _static_inpl_adder_test(inpl_adder):
    """
    This function runs tests on a desired inplace addition function
    in static mode.
    An inplace addition function is a function mapping (a, b) to (a, a+b),
    where a is a :ref:`QuantumVariable`, list[:ref:`Qubit`] or an integer
    and b is either a :ref:`QuantumVariable` or a list[:ref:`Qubit`].

    Parameters
    ----------
    inpl_adder : callable
        A quantum inplace addition function that can either act on single QuantumVariables or on lists of Qubits
        by adding the first one to the second.

    Returns
    -------
    Bool:
        True if all tests are passed, else False/ Exceptions.

    Examples
    --------

    We test the built-in Cuccaro adder:

    ::

        from qrisp import cuccaro_adder, inpl_adder_test

        inpl_adder_test(cuccaro_adder)
        print("The cuccaro adder passed the tests without errors.")

    And now a new user-defined qcla adder:
    ::

        from qrisp import inpl_adder_test, qcla

        qcla_2_0 = lambda x, y : qcla(x, y, radix_base = 2, radix_exponent = 0)
        inpl_adder_test(qcla_2_0)
        print("The qcla_2_0 adder passed the tests without errors.")

    """
    from qrisp import QuantumBool, QuantumFloat, control, h, multi_measurement

    for i in range(1, 7):

        for j in range(1, i + 1):
            a = QuantumFloat(j)
            b = QuantumFloat(i)
            c = QuantumFloat(i)

            h(a)
            h(b)

            c[:] = b

            inpl_adder(a, c)

            statevector_arr = a.qs.compile().statevector_array()
            angles = np.angle(
                statevector_arr[
                    np.abs(statevector_arr) > 1 / 2 ** ((a.size + b.size) / 2 + 1)
                ]
            )

            assert (
                np.sum(np.abs(angles)) < 0.1
            ), f"Quantum-quantum adder produced a faulty phase shift on input sizes, {i},{j}."

            mes_res = multi_measurement([a, b, c])

            for a, b, c in mes_res.keys():
                assert (a + b) % (
                    2**i
                ) == c, f"Quantum-quantum addition result was incorrect for input values {a} += {c} on input sizes, {i},{j}."

        if i < 6:
            for j in range(1, 2**i):
                a = QuantumFloat(i)
                b = QuantumFloat(i)

                h(a)

                b[:] = a

                inpl_adder(j, a)

                statevector_arr = a.qs.compile().statevector_array()
                angles = np.angle(
                    statevector_arr[
                        np.abs(statevector_arr) > 1 / 2 ** ((a.size) / 2 + 1)
                    ]
                )
                assert (
                    np.sum(np.abs(angles)) < 0.1
                ), f"Classical-quantum adder produced a faulty phase shift on input size {i}."

                mes_res = multi_measurement([a, b])

                for a, b in mes_res.keys():
                    assert (b + j) % (
                        2**i
                    ) == a, f"Classical-quantum addition result was incorrect for input values {a} += {c} on input size {i}."

    for i in range(1, 7):

        for j in range(1, i + 1):
            a = QuantumFloat(j)
            b = QuantumFloat(i)
            c = QuantumFloat(i)
            qbl = QuantumBool()

            h(qbl)
            h(a)
            h(b)

            c[:] = b

            with control(qbl):
                inpl_adder(a, c)

            statevector_arr = a.qs.compile().statevector_array()
            angles = np.angle(
                statevector_arr[
                    np.abs(statevector_arr) > 1 / 2 ** ((a.size + b.size) / 2 + 1)
                ]
            )
            assert (
                np.sum(np.abs(angles)) < 0.1
            ), f"Controlled quantum-quantum adder produced a faulty phase shift on input sizes, {i},{j}."

            mes_res = multi_measurement([a, b, c, qbl])

            for a, b, c, qbl in mes_res.keys():

                if qbl:
                    assert (a + b) % (
                        2**i
                    ) == c, f"Controlled quantum-quantum addition result was incorrect for input values {a} += {c} on input sizes, {i},{j}."
                else:
                    assert (
                        c == b
                    ), f"Controlled quantum-quantum addition behaviour was incorrect; an operation was performed without the control qubit in |1> state.Faulty input sizes: {i},{j}"

        if i < 6:
            for j in range(1, 2**i):
                a = QuantumFloat(i)
                b = QuantumFloat(i)
                qbl = QuantumBool()

                h(qbl)
                h(a)

                b[:] = a

                with control(qbl):
                    inpl_adder(j, a)

                statevector_arr = a.qs.compile().statevector_array()
                angles = np.angle(
                    statevector_arr[
                        np.abs(statevector_arr) > 1 / 2 ** ((a.size) / 2 + 1)
                    ]
                )
                assert (
                    np.sum(np.abs(angles)) < 0.1
                ), f"Controlled classical-quantum adder produced a faulty phase shift on input size {i}."

                mes_res = multi_measurement([a, b, qbl])

                for a, b, qbl in mes_res.keys():
                    if qbl:
                        assert (b + j) % (
                            2**i
                        ) == a, f"Controlled classical-quantum addition result was incorrect for input values {b} += {j} on input size, {i}."
                    else:
                        assert (
                            b == a
                        ), f"Controlled classical-quantum addition behaviour was incorrect; an operation was performed without the control qubit in |1> state. Faulty input sizes: {i}"


def _dynamic_inpl_adder_test(inpl_adder):
    """
    This function runs tests on a desired inplace addition function
    in dynamic (Jasp) mode.
    An inplace addition function is a function mapping (a, b) to (a, a+b),
    where a is a :ref:`QuantumVariable`, list[:ref:`Qubit`] or an integer
    and b is either a :ref:`QuantumVariable` or a list[:ref:`Qubit`].

    Parameters
    ----------
    inpl_adder : callable
        A quantum inplace addition function that can either act on single QuantumVariables or on lists of Qubits
        by adding the first one to the second.

    Returns
    -------
    Bool:
        True if all tests are passed, else False/ Exceptions.
    """
    from qrisp import QuantumBool, QuantumFloat, control, measure
    from qrisp.jasp import make_jaspr, boolean_simulation

    # Test 1: Quantum-Quantum addition
    def qq_func(i, j, a_val, b_val):
        a = QuantumFloat(j)
        b = QuantumFloat(i)
        a.encode(a_val % 2 ** j)
        b.encode(b_val % 2 ** i)
        inpl_adder(a, b)
        return measure(b)

    jaspr_qq = make_jaspr(qq_func)(1, 1, 0, 0)
    for i in range(1, 7):
        for j in range(1, i + 1):
            max_a = min(2 ** j, 4)
            max_b = min(2 ** i, 4)
            for a_val in range(max_a):
                for b_val in range(max_b):
                    res = jaspr_qq(i, j, a_val, b_val)
                    assert (
                        res == (a_val + b_val) % 2 ** i
                    ), f"Dynamic quantum-quantum addition result was incorrect for input values {a_val} + {b_val} on input sizes, {i},{j}."

    # Test 2: Classical-Quantum addition
    def cq_func(i, a_val, j_val):
        a = QuantumFloat(i)
        a.encode(a_val % 2 ** i)
        inpl_adder(j_val, a)
        return measure(a)

    jaspr_cq = make_jaspr(cq_func)(1, 0, 0)
    for i in range(1, 6):
        for a_val in range(min(2 ** i, 4)):
            for j_val in range(min(2 ** i, 4)):
                res = jaspr_cq(i, a_val, j_val)
                assert (
                    res == (a_val + j_val) % 2 ** i
                ), f"Dynamic classical-quantum addition result was incorrect for input values {a_val} += {j_val} on input size {i}."

    # Test 3: Controlled Quantum-Quantum addition
    @boolean_simulation
    def ctrl_qq_func(i, j, a_val, b_val, ctrl_val):
        qbl = QuantumBool()
        qbl.encode(ctrl_val)
        a = QuantumFloat(j)
        b = QuantumFloat(i)
        a.encode(a_val % 2 ** j)
        b.encode(b_val % 2 ** i)
        with control(qbl):
            inpl_adder(a, b)
        return measure(b), measure(qbl)

    for i in range(1, 7):
        for j in range(1, i + 1):
            max_a = min(2 ** j, 3)
            max_b = min(2 ** i, 3)
            for a_val in range(max_a):
                for b_val in range(max_b):
                    res_true = ctrl_qq_func(i, j, a_val, b_val, True)
                    assert (
                        res_true[0] == (a_val + b_val) % 2 ** i
                    ), f"Dynamic controlled quantum-quantum addition result was incorrect for input values {a_val} += {b_val} on input sizes, {i},{j}."

                    res_false = ctrl_qq_func(i, j, a_val, b_val, False)
                    assert (
                        res_false[0] == b_val
                    ), f"Dynamic controlled quantum-quantum addition behaviour was incorrect; an operation was performed without the control qubit in |1> state. Faulty input sizes: {i},{j}"

    # Test 4: Controlled Classical-Quantum addition
    @boolean_simulation
    def ctrl_cq_func(i, a_val, j_val, ctrl_val):
        qbl = QuantumBool()
        qbl.encode(ctrl_val)
        a = QuantumFloat(i)
        a.encode(a_val % 2 ** i)
        with control(qbl):
            inpl_adder(j_val, a)
        return measure(a), measure(qbl)

    for i in range(1, 6):
        for a_val in range(min(2 ** i, 3)):
            for j_val in range(min(2 ** i, 3)):
                res_true = ctrl_cq_func(i, a_val, j_val, True)
                assert (
                    res_true[0] == (a_val + j_val) % 2 ** i
                ), f"Dynamic controlled classical-quantum addition result was incorrect for input values {a_val} += {j_val} on input size, {i}."

                res_false = ctrl_cq_func(i, a_val, j_val, False)
                assert (
                    res_false[0] == a_val
                ), f"Dynamic controlled classical-quantum addition behaviour was incorrect; an operation was performed without the control qubit in |1> state. Faulty input sizes: {i}"


def inpl_adder_test(inpl_adder, mode="static"):
    """
    Runs the inplace adder test in the specified mode(s).

    Parameters
    ----------
    inpl_adder : callable
        A quantum inplace addition function.
    mode : str, optional
        ``"static"``, ``"dynamic"``, or ``"both"`` (default).
        Determines which test(s) to run.

    Returns
    -------
    bool
        ``True`` if the adder passes the requested test(s).

    Raises
    ------
    AssertionError
        If the adder fails the requested test(s).
    ValueError
        If ``mode`` is not one of ``"static"``, ``"dynamic"``, ``"both"``.
    """
    if mode not in ("static", "dynamic", "both"):
        raise ValueError(f"Unknown mode '{mode}'. Expected 'static', 'dynamic', or 'both'.")

    static_pass = True
    dynamic_pass = True

    if mode in ("static", "both"):
        try:
            _static_inpl_adder_test(inpl_adder)
        except AssertionError:
            static_pass = False

    if mode in ("dynamic", "both"):
        try:
            _dynamic_inpl_adder_test(inpl_adder)
        except Exception:
            dynamic_pass = False

    if mode == "both":
        if static_pass and dynamic_pass:
            return True
        if static_pass:
            raise AssertionError("Adder failed the dynamic test.")
        if dynamic_pass:
            raise AssertionError("Adder failed the static test.")
        raise AssertionError("Adder failed both the static and the dynamic test.")

    if mode == "static":
        if static_pass:
            return True
        raise AssertionError("Adder failed the static test.")

    if dynamic_pass:
        return True
    raise AssertionError("Adder failed the dynamic test.")


def amend_inpl_adder(raw_inpl_adder, amend_cl_int=True):

    def amended_adder(qf2, qf1, *args, **kwargs):

        if isinstance(qf2, (list, QuantumVariable)):
            dim_2 = jlen(qf2)
            dim_1 = jlen(qf1)

            qf2 = qf2[:jnp.minimum(dim_2, dim_1)]

            ancilla_var = QuantumVariable(
                jnp.maximum(0, dim_1 - dim_2),
                name="add_amend_anc*",
                qs=qf1.qs,
            )
            qf2 = qf2[:] + ancilla_var[:]

            raw_inpl_adder(qf2, qf1, *args, **kwargs)

            try:
                ancilla_var.delete(verify=False)
            except NameError:
                pass

        elif isinstance(qf2, int) and amend_cl_int:
            from qrisp.misc import int_encoder
            from qrisp.environments import conjugate

            q_a = qf1.duplicate()
            with conjugate(int_encoder)(q_a, qf2 % (2 ** qf1.size)):
                raw_inpl_adder(q_a, qf1, *args, **kwargs)
            q_a.delete()

        else:
            raw_inpl_adder(qf2, qf1, *args, **kwargs)

    return amended_adder
