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

from qrisp.algorithms.dqi import dqi_optimization
from qrisp  import multi_measurement
import numpy as np
import numpy as np


def test_dqi_algorithm():


    def random_binary_matrix(n, seed=None):
        """
        generate random adjacency matrix for a connected graph
        """
        rng = np.random.default_rng(seed)
        adj_matrix = np.zeros((n, n), dtype=int)
        nodes = list(range(n))
        rng.shuffle(nodes)
        for i in range(1, n):
            a = nodes[i]
            b = nodes[rng.integers(0, i)]
            adj_matrix[a, b] = 1
            adj_matrix[b, a] = 1  # undirected
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i, j] == 0 and rng.random() < 0.4:  # ~40% chance to add an edge
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        return adj_matrix


    from itertools import product
    def brute_force_max(B: np.ndarray, v: np.ndarray):
        """
        Wrapper that scores by the (-1)^(B·x + v) sum metric.
        """
        m, n = B.shape
        
        def score(bits):
            x = np.array(bits, dtype=int)
            return int(np.sum((-1)**(B.dot(x)+v)))
        results = []
        for bits in product([0,1], repeat=n):
            results.append((bits, score(bits)))
        results.sort(key=lambda x: x[1], reverse=True)
        raw = results
        # return sorted ascending by bitstring integer (you can re-sort here if you like)
        # or leave descending by score—choose whichever API you prefer
        return [("".join(map(str,bits)), sc) for bits, sc in raw]


    n = 12
    # create instance 
    B = random_binary_matrix(n, #seed= seed
                            )
    v = np.ones((n,)) 

    # run algorithm
    qvin, resqf = dqi_optimization(B, v)
    res_dict = multi_measurement([qvin,  resqf])

    # post select best sols on 0-state in error register
    best_sols_dqi = []
    max_val = max(res_dict.values())
    for key,val in res_dict.items():
        cor_val = True
        for i in range(len(key[0])):
            if key[0][i]!= 0:
                cor_val = False
        if cor_val: 
            if val == max_val:
                best_sols_dqi.append(key[1])

    # create brute force solution
    brute_full = brute_force_max(B,v)
    import heapq

    # get two highest cost values
    max_v_two = heapq.nlargest(2, set([brute_full[i][1] for i in range(len(brute_full)) ]))  
    brute_best = [item for item in brute_full if (item[1] in max_v_two)]
    # and associated binary solutions
    brute_best_names = [[int(binary) for binary in item[0]] for item in brute_best]
    
    # check if all highest dqi solutions are also in above list
    is_in = [(item in brute_best_names) for item in best_sols_dqi]

    assert is_in

