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
import pytest
from qrisp import *
from qrisp.block_encodings import BlockEncoding
from qrisp.gqsp import dalzell_inversion


@pytest.mark.parametrize("A, b", [
    (np.array([[ 0.78, -0.01, -0.16, -0.1 ],
    [-0.01,  0.57, -0.03,  0.08],
    [-0.16, -0.03,  0.69, -0.15],
    [-0.1 ,  0.08, -0.15,  0.88]]), 
    np.array([1, 1, 1, 1])),
])
def test_pseudo_inversion(A, b):

    BA = BlockEncoding.from_array(A)

    # All singular values of (A / alpha) must lie in [1 / kappa, 1]
    _, S, _ = np.linalg.svd(A, full_matrices=False)
    kappa = 1 / (np.min(S) / BA.alpha)

    # Find optimal t = |x|
    t = np.linalg.norm(np.linalg.inv(A / BA.alpha) @ b / np.linalg.norm(b))

    def prep_b(operand):
        prepare(operand, b)

    BA_inv = dalzell_inversion(BA, prep_b, t=t, eps=0.01, kappa=kappa)

    n = (A.shape[0] - 1).bit_length()

    @terminal_sampling
    def main():
        # Prepare operand in state |0>
        operand = QuantumFloat(n)
        ancillas = BA_inv.apply(operand)
        return operand, *ancillas

    res_dict = main()

    # Post-selection on ancillas being in |0> state
    filtered_dict = {k[0]: p for k, p in res_dict.items() \
                    if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    # Verify constant success probability for t = |x| 
    # Empirical threshold value 0.5
    assert(success_prob > 0.5)

    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
    amps = np.sqrt([filtered_dict.get(i, 0) for i in range(len(b))])

    # Compare to target values
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.allclose(amps, np.abs(c), atol=1e-2)
