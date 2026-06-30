"""********************************************************************************
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

import pytest

from qrisp import QuantumEnum

@pytest.mark.parametrize(
    "encoding", ["OneHot", "Binary"]
)
def test_encoding_variants(encoding):
    """Test that the correct encoding is used for QuantumEnum"""
    if encoding == "OneHot":
        from qrisp import QuantumEnum, x
        from enum import auto
        class Color(QuantumEnum.OneHot):
                    RED = auto()
                    YELLOW = auto()
                    GREEN = auto()
                    BLUE = auto()

        @QuantumEnum.auto(Color)
        class QuantumColor(QuantumEnum):
                    pass
                
        q_color = QuantumColor()
        q_color[:] = Color.RED
        x(q_color[0])
        x(q_color[1])
        result = q_color.get_measurement()
        assert Color.BLUE not in result
        assert Color.YELLOW in result

    elif encoding == "Binary":
        from qrisp import QuantumEnum, x
        from enum import auto
        class Color(QuantumEnum.Binary):
                    RED = auto()
                    YELLOW = auto()
                    GREEN = auto()
                    BLUE = auto()

        @QuantumEnum.auto(Color)
        class QuantumColor(QuantumEnum):
                    pass
                
        q_color = QuantumColor()
        q_color[:] = Color.RED
        x(q_color[0])
        x(q_color[1])
        result = q_color.get_measurement()
        assert Color.BLUE in result
        assert Color.YELLOW not in result

    else:
        pass


def test_OneHot_decoding():
    r"""Test that decoding OneHot encoded enums fail when state is $\ket{000}$"""
    from qrisp import QuantumEnum, x
    from enum import auto
    class Color(QuantumEnum.OneHot):
                RED = auto()
                GREEN = auto()
                BLUE = auto()

    @QuantumEnum.auto(Color)
    class QuantumColor(QuantumEnum):
                pass
            
    q_color = QuantumColor()
    with pytest.raises(ValueError, match="Can not decode value"):
        result = q_color.decoder(0)


def test_Binary_decoding():
    r"""Test that Binary encoded enums fail to decode when $\ket{N}$ is outside of used range"""
    from qrisp import QuantumEnum, x
    from enum import auto
    class Color(QuantumEnum.Binary):
                RED = auto()
                GREEN = auto()
                BLUE = auto()

    @QuantumEnum.auto(Color)
    class QuantumColor(QuantumEnum):
                pass
            
    q_color = QuantumColor()
    with pytest.raises(ValueError, match="Can not decode value outside of range"):
        result = q_color.decoder(3)