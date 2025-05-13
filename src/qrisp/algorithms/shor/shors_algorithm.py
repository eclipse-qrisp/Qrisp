"""
\********************************************************************************
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
********************************************************************************/
"""

import numpy as np
from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational

from qrisp.interface import QiskitBackend
from qrisp.alg_primitives.arithmetic.modular_arithmetic import find_optimal_m, modinv
from qrisp.alg_primitives import QFT
from qrisp import QuantumModulus, QuantumFloat, h, control

depths = []
cnot_count = []
qubits = []

def find_optimal_a(N):
    n = int(np.ceil(np.log2(N)))
    proposals = []
    
    # Search through the first O(1) possibilities to find a good a
    for a in range(2, min(100, N-1)):
        # We only append non-trivial proposals
        if np.gcd(a, N) == 1:
            proposals.append(a)
    
    cost_dic = {}
    for a in proposals:
        m_values = []
        for k in range(2*n+1):
            inpl_multiplier = (a**(2**k))%N
            
            if inpl_multiplier == 1:
                continue
            
            # find_optimal_m is a function that determines the lowest possible
            # Montgomery shift for a given number. The higher the montgomery shift,
            # the more qubits and the more effort is needed.
            m_values.append(find_optimal_m(inpl_multiplier, N))
            m_values.append(find_optimal_m(modinv((-inpl_multiplier)%N, N), N))
        
        cost_dic[a] = sum(m_values) + max(m_values)*1E-5
        
    proposals.sort(key = lambda a : cost_dic[a])
    
    
    optimal_a = proposals[0]
    
    m_values = []
    
    for k in range(2*n+1):
        inpl_multiplier = ((optimal_a)**(2**k))%N
        
        if inpl_multiplier == 1:
            continue
        
        m_values.append(find_optimal_m(inpl_multiplier, N))
    
    return proposals

def find_order(a, N, inpl_adder = None, mes_kwargs = {}):
    qg = QuantumModulus(N, inpl_adder)
    qg[:] = 1
    qpe_res = QuantumFloat(2*qg.size + 1, exponent = -(2*qg.size + 1))
    h(qpe_res)
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= a
            a = (a*a)%N
    QFT(qpe_res, inv = True, inpl_adder = inpl_adder)
    
    mes_res = qpe_res.get_measurement(**mes_kwargs)
    
    return extract_order(mes_res, a, N)


def extract_order(mes_res, a, N):
    
    collected_r_values = []
    
    approximations = list(mes_res.keys())
    
    try:
        approximations.remove(0)
    except ValueError:
        pass
    
    while True:
        
        r_values = get_r_values(approximations.pop(0))
        
        for r in r_values:  
            if (a**r)%N == 1:
                return r
        
        collected_r_values.append(r_values)
        from itertools import product
        
        for comb in product(*collected_r_values):
            r = np.lcm.reduce(comb)
            if (a**r)%N == 1:
                return r
    
def get_r_values(approx):
    rationals = continued_fraction_convergents(continued_fraction_iterator(Rational(approx)))
    return [rat.q for rat in rationals if 1 < rat.q]
    

def shors_alg(N, inpl_adder = None, mes_kwargs = {}):
    """
    Performs `Shor's factorization algorithm <https://arxiv.org/abs/quant-ph/9508027>`_ on a given integer N.
    The adder used for factorization can be customized. To learn more about this feature, please read :ref:`QuantumModulus`

    Parameters
    ----------
    N : integer
        The integer to be factored.
    inpl_adder : callable, optional
        A function that performs in-place addition. The default is None.
    mes_kwargs : dict, optional
        A dictionary of keyword arguments for :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`. This especially allows you to specify an execution backend. The default is {}.

    Returns
    -------
    res : integer
        A factor of N.

    Examples
    --------
    
    We factor 65:
        
    >>> from qrisp.shor import shors_alg
    >>> shors_alg(65)
    5

    """
    if not N%2:
        return 2
    
    a_proposals = find_optimal_a(N)
    
    for a in a_proposals:
        
        K = np.gcd(a, N)
        
        if K != 1:
            res = K
            break
        
        r = find_order(a, N, inpl_adder, mes_kwargs)
        
        if r%2:
            continue
        
        g = int(np.gcd(a**(r//2)+1, N))
        
        if g not in[N, 1]:
            res = g
            break
    return res