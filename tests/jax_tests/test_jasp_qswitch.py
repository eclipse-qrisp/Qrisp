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


def test_jasp_qswitch_case_hamiltonian_simulation():
    from qrisp import QuantumFloat, h, qswitch, terminal_sampling
    import numpy as np
    from qrisp.operators import X, Y, Z

    H1 = Z(0)*Z(1)
    H2 = Y(0)+Y(1)

    # Some sample case functions
    def f0(x): H1.trotterization()(x)
    def f1(x): H2.trotterization()(x, t=np.pi/4)
    case_function_list = [f0, f1]

    @terminal_sampling
    def main():
        # Create operand and case variable
        operand = QuantumFloat(2)
        case = QuantumFloat(1)
        h(case)

        # Execute switch_case function
        qswitch(operand, case, case_function_list)

        return case, operand

    meas_res = main()

    assert np.round(meas_res[0, 0], 2) == 0.5
    for i in [0, 1, 2, 3]:
        assert np.round(meas_res[1, i], 3) == 0.125


def test_jasp_qswitch_inversion():

    from qrisp import QuantumFloat, qswitch, jaspify, measure
    import jax.numpy as jnp
    from jax import jit

    @jit
    def extract_boolean_digit(integer, digit):
        return (integer >> digit) & 1

    def fake_inversion(qf, precision, res_qf=None):
        if res_qf is None:
            res_qf = QuantumFloat(2*precision+1, -precision)

        def case_function(i, operand):
            curr = 2**(2*precision)//(jnp.maximum(i, 1))
            operand[:] = curr/2**(2*precision)
        qswitch(res_qf, qf, case_function)
        return res_qf

    @jaspify
    def main():

        qf = QuantumFloat(5)
        qf[:] = 5
        inv = fake_inversion(qf, 5)
        return measure(inv)

    assert main() == 0.1875


def test_jasp_qswitch_function():
    from qrisp import QuantumFloat, qswitch, boolean_simulation, measure

    # tree
    @boolean_simulation
    def main(num, case_size, case_val):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)

        def case_function(i, operand):
            operand += i

        qswitch(operand_qf, case_qf, case_function, "tree", num)

        return measure(operand_qf)

    for case_size in range(1, 6):
        for num in range(1, 2**case_size+1):
            for case_val in range(0, 2**case_size):
                r = main(num, case_size, case_val)
                if num <= case_val:
                    assert r == 0
                else:
                    assert r == case_val

    # sequential
    @boolean_simulation
    def main(num, case_size, case_val):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)

        def case_function(i, operand):
            operand += i

        qswitch(operand_qf, case_qf, case_function, "sequential", num)

        return measure(operand_qf)

    for case_size in range(1, 6):
        for num in range(1, 2**case_size+1):
            for case_val in range(0, 2**case_size):
                r = main(num, case_size, case_val)
                if num <= case_val:
                    assert r == 0
                else:
                    assert r == case_val


def test_jasp_qswitch_function_control():
    from qrisp import QuantumFloat, qswitch, control, boolean_simulation, measure

    # tree
    @boolean_simulation
    def main(num, case_size, case_val, c):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        def case_function(i, operand):
            operand += i

        with control(control_qf[0]):
            qswitch(operand_qf, case_qf, case_function, "tree", num)

        return measure(operand_qf)

    for case_size in range(1, 6):
        for num in range(1, 2**case_size+1):
            for case_val in range(0, 2**case_size):
                for c in [0, 1]:
                    r = main(num, case_size, case_val, c)
                    if c == 0 or num <= case_val:
                        assert r == 0
                    else:
                        assert r == case_val

    # sequential
    @boolean_simulation
    def main(num, case_size, case_val, c):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        def case_function(i, operand):
            operand += i

        with control(control_qf[0]):
            qswitch(operand_qf, case_qf, case_function, "sequential", num)

        return measure(operand_qf)

    for case_size in range(1, 6):
        for num in range(1, 2**case_size+1):
            for case_val in range(0, 2**case_size):
                for c in [0, 1]:
                    r = main(num, case_size, case_val, c)
                    if c == 0 or num <= case_val:
                        assert r == 0
                    else:
                        assert r == case_val


def test_jasp_qswitch_tree_list():
    from qrisp import QuantumFloat, qswitch, boolean_simulation, measure

    def case_function(i, operand):
        operand += i

    case_function_list = [(lambda arg, i=i: case_function(i, arg)) for i in range(3**2)]

    # tree
    @boolean_simulation
    def main(num, case_size, case_val):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)

        qswitch(operand_qf, case_qf, case_function_list, "tree", num)

        return measure(operand_qf)

    for case_size in range(1, 3):
        for num in range(1, 2**case_size+1):
            for case_val in range(2**case_size//2, 2**case_size):
                r = main(num, case_size, case_val)
                if num <= case_val:
                    assert r == 0
                else:
                    assert r == case_val

    case_function_list = [(lambda arg, i=i: case_function(i, arg)) for i in range(2**2)]

    # sequential
    @boolean_simulation
    def main(case_size, case_val):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)

        qswitch(operand_qf, case_qf, case_function_list, "sequential")

        return measure(operand_qf)

    for case_size in [2]:
        for case_val in range(2**case_size//2, 2**case_size):
            r = main(case_size, case_val)
            assert r == case_val


def test_jasp_qswitch_tree_list_control():
    from qrisp import QuantumFloat, qswitch, control, boolean_simulation, measure

    def case_function(i, operand):
        operand += i

    case_function_list = [(lambda arg, i=i: case_function(i, arg)) for i in range(3**2)]

    # tree
    @boolean_simulation
    def main(num, case_size, case_val, c):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        with control(control_qf[0]):
            qswitch(operand_qf, case_qf, case_function_list, "tree", num)

        return measure(operand_qf)

    for case_size in range(1, 3):
        for num in range(1, 2**case_size+1):
            for case_val in range(2**case_size//2, 2**case_size):
                for c in [0, 1]:
                    r = main(num, case_size, case_val, c)
                    if c == 0 or num <= case_val:
                        assert r == 0
                    else:
                        assert r == case_val

    case_function_list = [(lambda arg, i=i: case_function(i, arg)) for i in range(2**2)]

    # sequential
    @boolean_simulation
    def main(case_size, case_val, c):
        case_qf = QuantumFloat(case_size)
        case_qf[:] = case_val
        operand_qf = QuantumFloat(case_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        with control(control_qf[0]):
            qswitch(operand_qf, case_qf, case_function_list, "sequential")

        return measure(operand_qf)

    for case_size in [2]:
        for case_val in range(2**case_size//2, 2**case_size):
            for c in [0, 1]:
                r = main(case_size, case_val, c)
                if c == 0:
                    assert r == 0
                else:
                    assert r == case_val
