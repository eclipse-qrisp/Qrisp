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

# classical indexed switch tests

def test_jasp_q_switch_classical_index():
    from qrisp import QuantumFloat, jaspify, measure, q_switch

    @jaspify
    def main(index_val):

        def f0(x): pass
        def f1(x): x += 1
        def f2(x): x += 2
        def f3(x): x += 3
        branches = [f0, f1, f2, f3]
        operand = QuantumFloat(2)

        q_switch(index_val, branches, operand)
        return measure(operand)

    for index_val in range(4):
        res = main(index_val)
        assert res == index_val


# quantum indexed switch tests

def test_jasp_q_switch_quantum_index():
    from qrisp import QuantumFloat, jaspify, measure, q_switch

    @jaspify
    def main(index_val):

        def f0(x): pass
        def f1(x): x += 1
        def f2(x): x += 2
        def f3(x): x += 3
        index = QuantumFloat(2)
        index[:] = index_val
        branches = [f0, f1, f2, f3]
        operand = QuantumFloat(2)

        q_switch(index, branches, operand)
        return measure(operand)

    for index_val in range(4):
        res = main(index_val)
        assert res == index_val


def test_jasp_q_switch_hamiltonian_simulation():
    from qrisp import QuantumFloat, h, q_switch, terminal_sampling
    import numpy as np
    from qrisp.operators import X, Y, Z

    H1 = Z(0) * Z(1)
    H2 = Y(0) + Y(1)

    # Some sample functions
    def f0(x):
        H1.trotterization()(x)

    def f1(x):
        H2.trotterization()(x, t=np.pi / 4)

    branches = [f0, f1]

    @terminal_sampling
    def main():
        # Create operand and index variable
        operand = QuantumFloat(2)
        index = QuantumFloat(1)
        h(index)

        # Execute switch_case function
        q_switch(index, branches, operand)

        return index, operand

    meas_res = main()

    assert np.round(meas_res[0, 0], 2) == 0.5
    for i in [0, 1, 2, 3]:
        assert np.round(meas_res[1, i], 3) == 0.125


def test_jasp_q_switch_inversion():

    from qrisp import QuantumFloat, q_switch, jaspify, measure
    import jax.numpy as jnp
    from jax import jit

    @jit
    def extract_boolean_digit(integer, digit):
        return (integer >> digit) & 1

    def fake_inversion(qf, precision, res_qf=None):
        if res_qf is None:
            res_qf = QuantumFloat(2 * precision + 1, -precision)

        def branches(i, operand):
            curr = 2 ** (2 * precision) // (jnp.maximum(i, 1))
            operand[:] = curr / 2 ** (2 * precision)

        q_switch(qf, branches, res_qf)
        return res_qf

    @jaspify
    def main():

        qf = QuantumFloat(5)
        qf[:] = 5
        inv = fake_inversion(qf, 5)
        return measure(inv)

    assert main() == 0.1875


def test_jasp_q_switch_function():
    from qrisp import QuantumFloat, q_switch, boolean_simulation, measure

    # tree
    @boolean_simulation
    def main(num, index_size, index_val):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)

        def branches(i, operand):
            operand += i

        q_switch(index_qf, branches, operand_qf, branch_amount=num, method="tree")

        return measure(operand_qf)

    for index_size in range(1, 6):
        for num in range(1, 2**index_size + 1):
            for index_val in range(0, 2**index_size):
                r = main(num, index_size, index_val)
                if num <= index_val:
                    assert r == 0
                else:
                    assert r == index_val

    # sequential
    @boolean_simulation
    def main(num, index_size, index_val):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)

        def branches(i, operand):
            operand += i

        q_switch(index_qf, branches, operand_qf, branch_amount=num, method="sequential")

        return measure(operand_qf)

    for index_size in range(1, 6):
        for num in range(1, 2**index_size + 1):
            for index_val in range(0, 2**index_size):
                r = main(num, index_size, index_val)
                if num <= index_val:
                    assert r == 0
                else:
                    assert r == index_val


def test_jasp_q_switch_function_control():
    from qrisp import QuantumFloat, q_switch, control, boolean_simulation, measure

    # tree
    @boolean_simulation
    def main(num, index_size, index_val, c):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        def branches(i, operand):
            operand += i

        with control(control_qf[0]):
            q_switch(index_qf, branches, operand_qf, branch_amount=num, method="tree")

        return measure(operand_qf)

    for index_size in range(1, 6):
        for num in range(1, 2**index_size + 1):
            for index_val in range(0, 2**index_size):
                for c in [0, 1]:
                    r = main(num, index_size, index_val, c)
                    if c == 0 or num <= index_val:
                        assert r == 0
                    else:
                        assert r == index_val

    # sequential
    @boolean_simulation
    def main(num, index_size, index_val, c):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        def branches(i, operand):
            operand += i

        with control(control_qf[0]):
            q_switch(index_qf, branches, operand_qf, branch_amount=num, method="sequential")

        return measure(operand_qf)

    for index_size in range(1, 6):
        for num in range(1, 2**index_size + 1):
            for index_val in range(0, 2**index_size):
                for c in [0, 1]:
                    r = main(num, index_size, index_val, c)
                    if c == 0 or num <= index_val:
                        assert r == 0
                    else:
                        assert r == index_val


def test_jasp_q_switch_tree_list():
    from qrisp import QuantumFloat, q_switch, boolean_simulation, measure

    def branches(i, operand):
        operand += i

    branches_list = [(lambda arg, i=i: branches(i, arg)) for i in range(3**2)]

    # tree
    @boolean_simulation
    def main(num, index_size, index_val):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)

        q_switch(index_qf, branches_list, operand_qf, branch_amount=num, method="tree")

        return measure(operand_qf)

    for index_size in range(1, 3):
        for num in range(1, 2**index_size + 1):
            for index_val in range(2**index_size // 2, 2**index_size):
                r = main(num, index_size, index_val)
                if num <= index_val:
                    assert r == 0
                else:
                    assert r == index_val

    branches_list = [(lambda arg, i=i: branches(i, arg)) for i in range(2**2)]

    # sequential
    @boolean_simulation
    def main(index_size, index_val):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)

        q_switch(index_qf, branches_list, operand_qf, method="sequential")

        return measure(operand_qf)

    for index_size in [2]:
        for index_val in range(2**index_size // 2, 2**index_size):
            r = main(index_size, index_val)
            assert r == index_val


def test_jasp_q_switch_tree_list_control():
    from qrisp import QuantumFloat, q_switch, control, boolean_simulation, measure

    def branches(i, operand):
        operand += i

    branches_list = [(lambda arg, i=i: branches(i, arg)) for i in range(3**2)]

    # tree
    @boolean_simulation
    def main(num, index_size, index_val, c):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        with control(control_qf[0]):
            q_switch(index_qf, branches_list, operand_qf, branch_amount=num, method="tree")

        return measure(operand_qf)

    for index_size in range(1, 3):
        for num in range(1, 2**index_size + 1):
            for index_val in range(2**index_size // 2, 2**index_size):
                for c in [0, 1]:
                    r = main(num, index_size, index_val, c)
                    if c == 0 or num <= index_val:
                        assert r == 0
                    else:
                        assert r == index_val

    branches_list = [(lambda arg, i=i: branches(i, arg)) for i in range(2**2)]

    # sequential
    @boolean_simulation
    def main(index_size, index_val, c):
        index_qf = QuantumFloat(index_size)
        index_qf[:] = index_val
        operand_qf = QuantumFloat(index_size)
        control_qf = QuantumFloat(1)
        control_qf[:] = c

        with control(control_qf[0]):
            q_switch(index_qf, branches_list, operand_qf, method="sequential")

        return measure(operand_qf)

    for index_size in [2]:
        for index_val in range(2**index_size // 2, 2**index_size):
            for c in [0, 1]:
                r = main(index_size, index_val, c)
                if c == 0:
                    assert r == 0
                else:
                    assert r == index_val
