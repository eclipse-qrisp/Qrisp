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


def test_mcx():
    from qrisp import h, mcx, QuantumVariable, multi_measurement, bin_rep
    import numpy as np

    def test_mcx_inner(n, ctrl_state, method, test_phase=True, num_ancilla=0):
        ctrl = QuantumVariable(n)

        target = QuantumVariable(1)

        h(ctrl)

        if method in ["hybrid", "balauca_dirty"]:
            mcx(
                ctrl,
                target,
                method=method,
                ctrl_state=ctrl_state,
                num_ancilla=num_ancilla,
            )
        else:
            mcx(ctrl, target, method=method, ctrl_state=ctrl_state)
        mes_res = multi_measurement([ctrl, target])
        statevector = ctrl.qs.statevector("array")

        # Test correct flipping behavior
        for i in range(2**n):
            ctrl_var = bin_rep(i, n)
            if ctrl_var != ctrl_state:
                assert (ctrl_var, "0") in mes_res
            else:
                assert (ctrl_var, "1") in mes_res

        if not test_phase:
            return

        angles = np.angle(statevector[np.abs(statevector) > 1 / 2 ** (n / 2 + 1)])

        # Test correct phase behavior
        assert np.sum(np.abs(angles)) < 0.1

    for n in range(1, 8):
        print("n: ", n)
        for ctrl_state in [
            n * "1",
            n * "0",
            bin_rep(int(np.random.randint(0, 2**n - 1)), n),
        ]:
            print("Ctrl state: ", ctrl_state)
            for method in ["gray", "gray_pt", "maslov", "balauca", "yong"]:
                print(method)

                if method == "maslov" and n > 4:
                    continue

                test_mcx_inner(n, ctrl_state, method, test_phase=(method != "gray_pt"))

            for method in ["hybrid", "balauca_dirty"]:
                for num_ancilla in range(8):
                    print("Ancilla count: ", num_ancilla)
                    test_mcx_inner(n, ctrl_state, method, num_ancilla=num_ancilla)
