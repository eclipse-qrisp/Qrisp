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

import numpy as np
import jax.numpy as jnp
import pytest
from qrisp import *
from qrisp.operators import X, Y, Z


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_addition(H1, H2):

    BE1 = H1.pauli_block_encoding()
    BE2 = H2.pauli_block_encoding()

    H3 = H1 + H2
    BE3 = H3.pauli_block_encoding()
    BE_addition = BE1 + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @RUS
    def main(BE):
        qv = QuantumVariable(n)
        ancillas = BE.apply(qv)
        bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
        success_bool = jnp.all(bools)

        # garbage collection
        [reset(anc) for anc in ancillas]
        [anc.delete() for anc in ancillas]
        return success_bool, qv

    @terminal_sampling
    def run_main(BE):
        qv = main(BE)
        return qv
    
    res_be3 = run_main(BE3)
    res_be_add = run_main(BE_addition)

    for k in range(2 ** n):
        val_be3 = res_be3.get(k, 0)
        val_be_add = res_be_add.get(k, 0)
        assert np.isclose(val_be3, val_be_add), f"Mismatch at state |{k}>: {val_be3} vs {val_be_add}"