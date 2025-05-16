"""
********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from scipy.sparse import random as random_sparse
from scipy.sparse.linalg import norm
from qrisp.operators import QubitOperator

def test_from_matrix_converter():

    for k in range(100):

        matrix = random_sparse(8, 8, density=0.3, format='csr')
        operator = QubitOperator.from_matrix(matrix)
        delta = norm(operator.to_sparse_matrix()-matrix)
        
        if not delta<1e-5:
            print(matrix.todense())
            print(operator)
        assert delta<1e-5