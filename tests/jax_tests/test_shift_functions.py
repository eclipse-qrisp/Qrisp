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

Comprehensive tests for singular_shift and cyclic_shift in both static
(non-JASP) and dynamic (JASP/boolean_simulation) modes.
"""

import numpy as np


# =============================================================================
# singular_shift tests
# =============================================================================


def test_singular_shift_static_power_of_2():
    """Static mode: singular_shift on power-of-2 sized QuantumFloat."""
    from qrisp import QuantumFloat, singular_shift

    for n_bits in [2, 3, 4]:
        for val in range(2**n_bits):
            qf = QuantumFloat(n_bits)
            qf[:] = val
            singular_shift(qf)
            result = list(qf.get_measurement().keys())[0]

            # singular_shift acts on qubits: bit at position i moves to i+1,
            # last bit wraps to position 0 (cyclic shift by 1 on the qubit register)
            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Static singular_shift({n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_singular_shift_static_non_power_of_2():
    """Static mode: singular_shift on non-power-of-2 sized QuantumFloat."""
    from qrisp import QuantumFloat, singular_shift

    for n_bits in [3, 5, 6, 7]:
        for val in [0, 1, 2**(n_bits - 1), 2**n_bits - 1]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            singular_shift(qf)
            result = list(qf.get_measurement().keys())[0]

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Static singular_shift({n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_singular_shift_dynamic_power_of_2():
    """Dynamic mode (boolean_simulation): singular_shift on power-of-2 sizes."""
    from qrisp import boolean_simulation, QuantumFloat, singular_shift, measure

    @boolean_simulation
    def run_singular_shift(val, n_bits):
        qf = QuantumFloat(n_bits)
        qf[:] = val
        singular_shift(qf)
        return measure(qf)

    for n_bits in [2, 3, 4]:
        for val in range(2**n_bits):
            result = run_singular_shift(val, n_bits)

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Dynamic singular_shift({n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_singular_shift_dynamic_non_power_of_2():
    """Dynamic mode (boolean_simulation): singular_shift on non-power-of-2 sizes."""
    from qrisp import boolean_simulation, QuantumFloat, singular_shift, measure

    @boolean_simulation
    def run_singular_shift(val, n_bits):
        qf = QuantumFloat(n_bits)
        qf[:] = val
        singular_shift(qf)
        return measure(qf)

    for n_bits in [3, 5, 6, 7]:
        for val in [0, 1, 2**(n_bits - 1), 2**n_bits - 1]:
            result = run_singular_shift(val, n_bits)

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Dynamic singular_shift({n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_singular_shift_saeedi_static():
    """Static mode: singular_shift with use_saeedi=True."""
    from qrisp import QuantumFloat, singular_shift

    for n_bits in [3, 4, 5]:
        for val in [0, 1, 2**(n_bits - 1), 2**n_bits - 1]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            singular_shift(qf, use_saeedi=True)
            result = list(qf.get_measurement().keys())[0]

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Static singular_shift(saeedi, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_singular_shift_saeedi_dynamic():
    """Dynamic mode (boolean_simulation): singular_shift with use_saeedi=True."""
    from qrisp import boolean_simulation, QuantumFloat, singular_shift, measure

    @boolean_simulation
    def run_singular_shift_saeedi(val, n_bits):
        qf = QuantumFloat(n_bits)
        qf[:] = val
        singular_shift(qf, use_saeedi=True)
        return measure(qf)

    for n_bits in [3, 4, 5]:
        for val in [0, 1, 2**(n_bits - 1), 2**n_bits - 1]:
            result = run_singular_shift_saeedi(val, n_bits)

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Dynamic singular_shift(saeedi, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_singular_shift_static_vs_dynamic_consistency():
    """Verify static and dynamic singular_shift produce identical results."""
    from qrisp import QuantumFloat, singular_shift, boolean_simulation, measure

    @boolean_simulation
    def run_dynamic(val, n_bits):
        qf = QuantumFloat(n_bits)
        qf[:] = val
        singular_shift(qf)
        return measure(qf)

    for n_bits in [3, 4, 5, 6]:
        for val in range(min(2**n_bits, 16)):
            # Static
            qf_static = QuantumFloat(n_bits)
            qf_static[:] = val
            singular_shift(qf_static)
            static_result = list(qf_static.get_measurement().keys())[0]

            # Dynamic
            dynamic_result = run_dynamic(val, n_bits)

            assert static_result == dynamic_result, (
                f"Mismatch ({n_bits}-bit, val={val}): "
                f"static={static_result}, dynamic={dynamic_result}"
            )


# =============================================================================
# cyclic_shift tests
# =============================================================================


def test_cyclic_shift_static_shift_1():
    """Static mode: cyclic_shift with shift_amount=1."""
    from qrisp import QuantumFloat, cyclic_shift

    for n_bits in [3, 4, 5, 6]:
        for val in [0, 1, 2**(n_bits - 1), 2**n_bits - 1]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=1)
            result = list(qf.get_measurement().keys())[0]

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Static cyclic_shift(1, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_cyclic_shift_dynamic_shift_1():
    """Dynamic mode (boolean_simulation): cyclic_shift with shift_amount=1."""
    from qrisp import boolean_simulation, QuantumFloat, cyclic_shift, measure

    @boolean_simulation
    def run_cyclic_shift(val, n_bits):
        qf = QuantumFloat(n_bits)
        qf[:] = val
        cyclic_shift(qf, shift_amount=1)
        return measure(qf)

    for n_bits in [3, 4, 5, 6]:
        for val in [0, 1, 2**(n_bits - 1), 2**n_bits - 1]:
            result = run_cyclic_shift(val, n_bits)

            expected_bits = [(val >> i) & 1 for i in range(n_bits)]
            shifted_bits = [expected_bits[-1]] + expected_bits[:-1]
            expected = sum(b << i for i, b in enumerate(shifted_bits))
            assert result == expected, (
                f"Dynamic cyclic_shift(1, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_cyclic_shift_static_positive_shifts():
    """Static mode: cyclic_shift with various positive shift amounts."""
    from qrisp import QuantumFloat, cyclic_shift

    n_bits = 5
    for val in [0, 1, 5, 16, 31]:
        for shift in [1, 2, 3, 4]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            result = list(qf.get_measurement().keys())[0]

            # Apply cyclic right shift `shift` times at qubit level
            bits = [(val >> i) & 1 for i in range(n_bits)]
            for _ in range(shift):
                bits = [bits[-1]] + bits[:-1]
            expected = sum(b << i for i, b in enumerate(bits))

            assert result == expected, (
                f"Static cyclic_shift({shift}, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_cyclic_shift_dynamic_positive_shifts():
    """Dynamic mode (boolean_simulation): cyclic_shift with various positive shift amounts."""
    from qrisp import boolean_simulation, QuantumFloat, cyclic_shift, measure

    n_bits = 5
    for shift in [1, 2, 3, 4]:

        @boolean_simulation
        def run_cyclic_shift(val):
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            return measure(qf)

        for val in [0, 1, 5, 16, 31]:
            result = run_cyclic_shift(val)

            bits = [(val >> i) & 1 for i in range(n_bits)]
            for _ in range(shift):
                bits = [bits[-1]] + bits[:-1]
            expected = sum(b << i for i, b in enumerate(bits))

            assert result == expected, (
                f"Dynamic cyclic_shift({shift}, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_cyclic_shift_static_negative_shifts():
    """Static mode: cyclic_shift with negative shift amounts (left shift)."""
    from qrisp import QuantumFloat, cyclic_shift

    n_bits = 5
    for val in [0, 1, 5, 16, 31]:
        for shift in [-1, -2, -3]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            result = list(qf.get_measurement().keys())[0]

            # Negative shift = cyclic left shift at qubit level
            bits = [(val >> i) & 1 for i in range(n_bits)]
            for _ in range(-shift):
                bits = bits[1:] + [bits[0]]
            expected = sum(b << i for i, b in enumerate(bits))

            assert result == expected, (
                f"Static cyclic_shift({shift}, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_cyclic_shift_dynamic_negative_shifts():
    """Dynamic mode (boolean_simulation): cyclic_shift with negative shift amounts."""
    from qrisp import boolean_simulation, QuantumFloat, cyclic_shift, measure

    n_bits = 5
    for shift in [-1, -2, -3]:

        @boolean_simulation
        def run_cyclic_shift(val):
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            return measure(qf)

        for val in [0, 1, 5, 16, 31]:
            result = run_cyclic_shift(val)

            bits = [(val >> i) & 1 for i in range(n_bits)]
            for _ in range(-shift):
                bits = bits[1:] + [bits[0]]
            expected = sum(b << i for i, b in enumerate(bits))

            assert result == expected, (
                f"Dynamic cyclic_shift({shift}, {n_bits}-bit, val={val}): "
                f"got {result}, expected {expected}"
            )


def test_cyclic_shift_static_vs_dynamic_consistency():
    """Verify static and dynamic cyclic_shift produce identical results."""
    from qrisp import QuantumFloat, cyclic_shift, boolean_simulation, measure

    n_bits = 5
    for shift in [-2, -1, 1, 2, 3]:

        @boolean_simulation
        def run_dynamic(val):
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            return measure(qf)

        for val in [0, 1, 5, 16, 31]:
            # Static
            qf_static = QuantumFloat(n_bits)
            qf_static[:] = val
            cyclic_shift(qf_static, shift_amount=shift)
            static_result = list(qf_static.get_measurement().keys())[0]

            # Dynamic
            dynamic_result = run_dynamic(val)

            assert static_result == dynamic_result, (
                f"Mismatch (shift={shift}, val={val}): "
                f"static={static_result}, dynamic={dynamic_result}"
            )


def test_cyclic_shift_roundtrip_static():
    """Static mode: shift by k then by -k should return to original."""
    from qrisp import QuantumFloat, cyclic_shift

    n_bits = 5
    for val in [0, 1, 7, 16, 31]:
        for shift in [1, 2, 3]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            cyclic_shift(qf, shift_amount=-shift)
            result = list(qf.get_measurement().keys())[0]
            assert result == val, (
                f"Static roundtrip (shift={shift}, val={val}): "
                f"got {result}, expected {val}"
            )


def test_cyclic_shift_roundtrip_dynamic():
    """Dynamic mode: shift by k then by -k should return to original."""
    from qrisp import boolean_simulation, QuantumFloat, cyclic_shift, measure

    n_bits = 5
    for shift in [1, 2, 3]:

        @boolean_simulation
        def run_roundtrip(val):
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=shift)
            cyclic_shift(qf, shift_amount=-shift)
            return measure(qf)

        for val in [0, 1, 7, 16, 31]:
            result = run_roundtrip(val)
            assert result == val, (
                f"Dynamic roundtrip (shift={shift}, val={val}): "
                f"got {result}, expected {val}"
            )


def test_cyclic_shift_full_rotation_static():
    """Static mode: shifting by N should be identity (N = number of qubits)."""
    from qrisp import QuantumFloat, cyclic_shift

    for n_bits in [3, 4, 5]:
        for val in [0, 1, 2**n_bits - 1]:
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=n_bits)
            result = list(qf.get_measurement().keys())[0]
            assert result == val, (
                f"Static full rotation ({n_bits}-bit, val={val}): "
                f"got {result}, expected {val}"
            )


def test_cyclic_shift_full_rotation_dynamic():
    """Dynamic mode: shifting by N should be identity (N = number of qubits)."""
    from qrisp import boolean_simulation, QuantumFloat, cyclic_shift, measure

    for n_bits in [3, 4, 5]:

        @boolean_simulation
        def run_full_rotation(val):
            qf = QuantumFloat(n_bits)
            qf[:] = val
            cyclic_shift(qf, shift_amount=n_bits)
            return measure(qf)

        for val in [0, 1, 2**n_bits - 1]:
            result = run_full_rotation(val)
            assert result == val, (
                f"Dynamic full rotation ({n_bits}-bit, val={val}): "
                f"got {result}, expected {val}"
            )


def test_cyclic_shift_quantum_array_static():
    """Static mode: cyclic_shift on a QuantumArray."""
    from qrisp import QuantumFloat, QuantumArray, cyclic_shift

    qa = QuantumArray(QuantumFloat(3), 4)
    qa[:] = [0, 1, 2, 3]
    cyclic_shift(qa, shift_amount=1)
    result = list(qa.get_measurement().keys())[0]
    expected = np.array([3, 0, 1, 2])
    assert np.array_equal(result, expected), (
        f"Static QuantumArray cyclic_shift: got {result}, expected {expected}"
    )


def test_cyclic_shift_quantum_array_shift_2_static():
    """Static mode: cyclic_shift on QuantumArray with shift_amount=2."""
    from qrisp import QuantumFloat, QuantumArray, cyclic_shift

    qa = QuantumArray(QuantumFloat(3), 8)
    qa[:] = list(range(8))
    cyclic_shift(qa, shift_amount=2)
    result = list(qa.get_measurement().keys())[0]
    expected = np.array([6, 7, 0, 1, 2, 3, 4, 5])
    assert np.array_equal(result, expected), (
        f"Static QuantumArray cyclic_shift(2): got {result}, expected {expected}"
    )
