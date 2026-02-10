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
import pytest
from qrisp import *
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import X, Y, Z


def test_block_encoding_from_array():
    A = np.array([[0,1,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0]])
    B = BlockEncoding.from_array(A)

    @terminal_sampling
    def main():
        return B.apply_rus(lambda: QuantumFloat(2))()

    res = main()
    assert res == {1.0: 0.5, 3.0: 0.5}


def test_block_encoding_from_lcu():
    def f0(x): x-=1
    def f1(x): x+=1
    BE = BlockEncoding.from_lcu(np.array([1., 1.]), [f0, f1])

    @terminal_sampling
    def main():
        return BE.apply_rus(lambda : QuantumFloat(2))()

    res = main()
    assert res == {1.0: 0.5, 3.0: 0.5}


def test_block_encoding_from_operator():
    H = X(0)*X(1) + 0.2*Y(0)*Y(1)
    B = BlockEncoding.from_operator(H)

    @terminal_sampling
    def main():
        return B.apply_rus(lambda: QuantumFloat(2))()

    res = main()
    assert res == {3.0: 1.0}


def test_block_encoding_resources():
    H = X(0)*X(1) + 0.5*Z(0)*Z(1)
    BE = BlockEncoding.from_operator(H)

    res_dict = BE.resources(QuantumFloat(2))
    # {'gate counts': {'x': 3, 'cz': 2, 'u3': 2, 'cx': 4, 'gphase': 2}, 
    # 'depth': 12}
    assert isinstance(res_dict['gate counts'], dict)
    assert isinstance(res_dict['depth'], int)


def test_block_encoding_alpha_dynamic():

    @terminal_sampling
    def main():

        a = QuantumFloat(2)
        x(a)
        # b is dynamic
        b = measure(a)

        def U(qv):
            x(qv)

        BE1 = BlockEncoding(b,[],U)
        BE2 = BlockEncoding(1,[],U)
        BE = BE1 + BE2

        return BE.apply_rus(lambda: QuantumVariable(2))()

    res = main()
    assert res == {3: 1.0}


    @terminal_sampling
    def main():

        a = QuantumFloat(1)
        x(a)
        # b is dynamic
        b = measure(a)

        def U1(qv):
            x(qv)

        def U2(qv):
            pass

        BE1 = BlockEncoding(b,[],U1)
        BE2 = BlockEncoding(1,[],U2)
        BE = BE1 + BE2

        return BE.apply_rus(lambda: QuantumVariable(2))()
    
    res = main()
    assert res == {0: 0.5, 3: 0.5}