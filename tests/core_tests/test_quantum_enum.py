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


@pytest.mark.parametrize("encoding", ["OneHot", "Binary"])
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


def test_OneHot_apply_phase_if_eq():
    r"""Test that phases are applied according to the enum encoding if and only if the enum variants are equal"""
    from qrisp import QuantumEnum
    from enum import auto

    class Color(QuantumEnum.OneHot):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    @QuantumEnum.auto(Color)
    class QuantumColor(QuantumEnum):
        pass

    # No phase if not equal
    q_color_a = QuantumColor()
    q_color_a[:] = Color.RED

    q_color_b = QuantumColor()
    q_color_b[:] = Color.GREEN

    q_color_a.qs.statevector_array()
    q_color_a.apply_phase_if_eq(q_color_b, 0.5)

    res = q_color_a.qs.statevector_array()

    expected_res = np.zeros(64, dtype=complex)
    expected_res[34] = 1

    assert np.allclose(expected_res, res)

    # Phase if equal
    q_color_c = QuantumColor()
    q_color_c[:] = Color.RED

    q_color_d = QuantumColor()
    q_color_d[:] = Color.RED

    q_color_c.apply_phase_if_eq(q_color_d, 0.5)

    res = q_color_c.qs.statevector_array()

    expected_res = np.zeros(64, dtype=complex)
    expected_res[36] = 0.87758255 + 0.47942555j

    assert np.allclose(expected_res, res)


def test_Binary_apply_phase_if_eq():
    r"""Test that phases are applied according to the enum encoding if and only if the enum variants are equal"""
    from qrisp import QuantumEnum
    from enum import auto

    class Color(QuantumEnum.Binary):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    @QuantumEnum.auto(Color)
    class QuantumColor(QuantumEnum):
        pass

    q_color_a = QuantumColor()
    q_color_a[:] = Color.RED

    q_color_b = QuantumColor()
    q_color_b[:] = Color.GREEN

    q_color_a.qs.statevector_array()
    q_color_a.apply_phase_if_eq(q_color_b, 0.5)

    res = q_color_a.qs.statevector_array()

    expected_res = np.zeros(32, dtype=complex)
    expected_res[4] = 1

    assert np.allclose(expected_res, res)

    q_color_c = QuantumColor()
    q_color_c[:] = Color.RED

    q_color_d = QuantumColor()
    q_color_d[:] = Color.RED

    q_color_c.apply_phase_if_eq(q_color_d, 0.5)

    res = q_color_c.qs.statevector_array()

    expected_res = np.zeros(32, dtype=complex)
    expected_res[0] = 0.87758255 + 0.47942555j

    assert np.allclose(expected_res, res)
