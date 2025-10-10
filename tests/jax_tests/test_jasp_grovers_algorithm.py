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

def test_jasp_grovers_algorithm():
    from qrisp import QuantumFloat, QuantumArray
    from qrisp.jasp import terminal_sampling
    from qrisp.grover import tag_state, grovers_alg
    import numpy as np

    def test_oracle(qf_list, phase = np.pi):
        tag_dic = {qf_list[0] : 0, qf_list[1] : 0.5}
        tag_state(tag_dic, phase=phase)

    @terminal_sampling
    def main():
        qf_list = [QuantumFloat(2,-2), QuantumFloat(2,-2)]
        grovers_alg(qf_list, test_oracle)
        return qf_list[0], qf_list[1]

    res_dict = main()
    assert res_dict[(0,0.5)]>0.95

    # Exact Grover's algorithm
    @terminal_sampling
    def main():
        qf_list = [QuantumFloat(2,-2), QuantumFloat(2,-2)]
        grovers_alg(qf_list, test_oracle, winner_state_amount=1, exact=True)
        return qf_list[0], qf_list[1]

    res_dict = main()
    assert np.abs(res_dict[(0,0.5)]-1) < 1e-4

    # Test for input of type QuantumArray

    def oracle(qa):
        tag_state({qa[0]:0, qa[1]:0, qa[2]:0})

    @terminal_sampling
    def main():

        qa = QuantumArray(QuantumFloat(2), shape=(3,))

        grovers_alg(qa, oracle)

        return qa[0], qa[1], qa[2]

    mes_res = main()

    assert mes_res[(0.0, 0.0, 0.0)] > 0.99