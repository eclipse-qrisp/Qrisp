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

import numpy as np

def find_max(single_cor, double_cor, res, solutions):
    """
    General subroutine for finding the values with maximal correlation in the QIRO algorithm

    Parameters
    ----------
    single_cor : List
        Single qubits correlations to check.
    double_cor : List[Tuple]
        Multi qubits correlations to check.
    res : dict
        Result dictionary of initial QAOA optimization procedure
    solutions : List
        Qubits which have been found to be positive correlated, i.e. part of the problem solution  

    Returns
    -------
    max_item , sign
        The item with maximal correlation and the sign of the correlation.

    """
    
    max = 0

    for item2 in double_cor:
        if abs(item2[0]) == abs(item2[1]):
            continue
        summe = 0

        # calc correlation expectation
        for key, val in res.items():
            summe += pow(val, 2) * pow(-1, int(key[int(abs(item2[0]))])) * pow(-1, int(key[int(abs(item2[1]))]))

        #find max
        if abs(summe) > abs(max):
            max, max_item = summe, item2
            sign = np.sign(summe)

    for node in single_cor:
        if node in solutions:
            continue
        summe = 0

        for key, val in res.items():
            summe += val * pow(-1, int(key[int(node)])) 
            
        if abs(summe) > abs(max):
            max, max_item = summe, node
            sign = np.sign(summe)
    
    return max_item, sign
