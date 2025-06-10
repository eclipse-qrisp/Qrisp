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



from qrisp.alg_primitives.dicke_state_prep import dicke_divide_and_conquer, iterative_dicke_state_sampling



from random import randint
from qrisp import QuantumFloat, QuantumVariable
import scipy
from qrisp.jasp import terminal_sampling

n = 7
k_rand = randint(1,3)
expect_num_res = scipy.special.binom(n,k_rand)
expect_ampl = 1/expect_num_res


def test_dicke_state_preps_dnq():
    n = 7
    k = 3
    expect_num_res = scipy.special.binom(n,k)
    expect_ampl = 1/expect_num_res

    # dicke_divide_and_conquer - NON-JASP
    n = 7
    k = 3
    qv = QuantumVariable(n)
    dicke_divide_and_conquer(qv, k)
    res_dnq = qv.get_measurement()

    # check number of results and amplitude
    assert len(res_dnq.keys()) == expect_num_res
    for k,v in res_dnq.items():
        assert expect_ampl-expect_ampl*0.1 < v < expect_ampl+expect_ampl*0.1



def test_dicke_state_preps_dnq_jasp():

    n = 7
    k_rand = 3
    expect_num_res = scipy.special.binom(n,k_rand)
    expect_ampl = 1/expect_num_res

    # dicke_divide_and_conquer - JASP
    @terminal_sampling
    def main():
            
        n = 7
        k = 3
        q_test = QuantumVariable(n)
        q_test = dicke_divide_and_conquer(q_test,k)

        return q_test

    res_dnq_jasp = main()
    # check number of results and amplitude
    assert len(res_dnq_jasp.keys()) == expect_num_res
    for k,v in res_dnq_jasp.items():
        assert expect_ampl-expect_ampl*0.1 < v < expect_ampl+expect_ampl*0.1



def test_dicke_state_preps_iterative():

    n = 7
    k = 3
    expect_num_res = scipy.special.binom(n,k)
    expect_ampl = 1/expect_num_res
    
    # iterative_dicke_state_sampling - JASP
    @terminal_sampling
    def main():
            
        n = 7
        k = 3
        qv_iter = QuantumFloat(n)
        qv_iter = iterative_dicke_state_sampling(qv_iter,k)

        return qv_iter

    res_iter = main()
    # check number of results and amplitude
    assert len(res_iter.keys()) == expect_num_res
    for k,v in res_iter.items():
        assert expect_ampl-expect_ampl*0.1 < v < expect_ampl+expect_ampl*0.1


