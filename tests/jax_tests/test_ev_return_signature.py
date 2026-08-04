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

r"""Comprehensive tests for the pytree-aware return signature of
:func:`~qrisp.jasp.expectation_value`.

.. note::

    Only **top-level** :class:`~qrisp.QuantumVariable` instances are
    automatically decoded.  QVs nested inside containers must be
    measured explicitly (``measure(qv)``) to appear as classical leaves.

    Because ``expectation_value`` *accumulates* (sums) values, ``bool``
    leaves promote to ``int64`` or ``float64`` in the result.
"""

import jax.numpy as jnp
import pytest

from qrisp import (
    QuantumArray,
    QuantumBool,
    QuantumFloat,
    control,
    cx,
    h,
    measure,
    x,
)
from qrisp.jasp import expectation_value, jaspify

SHOTS = 500


def _double(*args):
    if len(args) == 1:
        return 2 * args[0]
    return tuple(2 * x for x in args)


def _sum(*args):
    return sum(args)


def _to_tuple(x):
    return (x, x * 2)


# =============================================================================
# 1. Scalar returns
# =============================================================================


class TestScalarReturns:
    def test_scalar_quantum_jit(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return qf

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert abs(float(main()) - 0.5) < 0.2

    def test_scalar_quantum_terminal(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return qf

        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert abs(float(main()) - 0.5) < 0.1

    def test_scalar_classical_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            h(qf[1])
            return measure(qf)

        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert abs(float(main()) - 1.5) < 0.3

    def test_scalar_classical_terminal_rejected(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            h(qf[1])
            return measure(qf)

        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()


# =============================================================================
# 2. Flat tuple returns
# =============================================================================


class TestFlatTupleReturns:
    def test_two_elements_jit(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            x(b[0])
            return a, b

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert abs(float(res[0]) - 0.5) < 0.2
        assert abs(float(res[1]) - 1.0) < 0.2

    def test_three_elements_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            c = QuantumFloat(3)
            h(a[0])
            x(b[0])
            return a, b, c

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 3

    def test_two_elements_terminal(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2

    def test_classical_tuple_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            cx(a[0], b[0])
            return measure(a), measure(b)

        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert isinstance(main(), tuple) and len(main()) == 2


# =============================================================================
# 3. Nested / container returns (classical leaves only)
# =============================================================================


class TestNestedReturns:
    def test_tuple_of_measured_leaves_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            x(b[0])
            return a, (measure(b),)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert isinstance(res[1], tuple) and len(res[1]) == 1

    def test_nested_undecoded_qv_raises_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return a, (b,)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()

    def test_list_of_undecoded_qv_raises_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return [a, b]

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()

    def test_dict_of_undecoded_qv_raises_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return {"val": a, "aux": b}

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()

    def test_mixed_qv_classical_nested_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            h(b[1])
            return a, (measure(b),)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert isinstance(res[1], tuple) and len(res[1]) == 1


# =============================================================================
# 4. Dtype preservation
# =============================================================================


class TestDtypePreservation:
    def test_bool_in_flat_tuple_jit(self):
        def kernel():
            qf = QuantumFloat(4)
            qb = QuantumBool()
            h(qf[0])
            h(qb)
            return qf, qb

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert res[0].dtype == jnp.float64
        assert res[1].dtype in (jnp.int64, jnp.float64)  # bool promotes under sum


# =============================================================================
# 5. Array-valued leaves
# =============================================================================


class TestArrayValuedLeaves:
    def test_array_and_scalar_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return jnp.array([1.0, 2.0, 3.0]), qf

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert res[0].shape == (3,)
        assert jnp.all(res[0] == jnp.array([1.0, 2.0, 3.0]))

    def test_array_inside_tuple_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return (jnp.ones(4), qf)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (4,)
        assert jnp.allclose(res[0], jnp.ones(4))

    def test_single_non_scalar_array_jit(self):
        """Kernel returns a single non-scalar JAX array.  The mean should
        preserve the array shape (not broadcast into a scalar)."""
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return jnp.array([[1.0, 2.0], [3.0, 4.0]])

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        res = main()
        assert res.shape == (2, 2)
        assert jnp.allclose(res, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


# =============================================================================
# 6. Post-processor
# =============================================================================


class TestPostProcessor:
    def test_structure_preserving_jit(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS, post_processor=_double)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2

    def test_structure_preserving_terminal(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(kernel, shots=SHOTS, post_processor=_double)()

        assert isinstance(main(), tuple) and len(main()) == 2

    def test_tuple_to_scalar_jit(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            x(b[0])
            return a, b

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS, post_processor=_sum)()

        assert not isinstance(main(), tuple)

    def test_scalar_to_tuple_jit(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return qf

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS, post_processor=_to_tuple)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2

    def test_tuple_to_scalar_terminal(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(kernel, shots=SHOTS, post_processor=_sum)()

        assert not isinstance(main(), tuple)


# =============================================================================
# 7. Edge cases
# =============================================================================


class TestEdgeCases:
    def test_user_defined_pytree_raises(self):
        from dataclasses import dataclass
        import jax.tree_util as jtu

        @dataclass
        class MyPytree:
            x: object
            y: object

        jtu.register_pytree_node(MyPytree, lambda obj: ((obj.x, obj.y), None), lambda _, c: MyPytree(*c))

        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return MyPytree(a, b)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        with pytest.raises(TypeError, match="Unsupported return type"):
            main()

    def test_single_element_tuple_becomes_scalar(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return (qf,)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert not isinstance(main(), tuple)

    def test_pure_classical_multiple_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            h(b[1])
            return measure(a), measure(b), measure(a)

        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert isinstance(main(), tuple) and len(main()) == 3

    def test_mixed_quantum_classical_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            mes = measure(qf[1])
            return qf, mes

        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert isinstance(main(), tuple) and len(main()) == 2

    def test_dynamic_kernel_arg_jit(self):
        def kernel(k):
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            with control(a[0]):
                x(b[k])
                return a, b

        @jaspify
        def main(k):
            return expectation_value(kernel, shots=SHOTS)(k)

        assert isinstance(main(2), tuple) and len(main(2)) == 2


# =============================================================================
# 8. Stress
# =============================================================================


class TestStress:
    def test_many_return_values_jit(self):
        N = 6

        def kernel():
            qvs = QuantumArray(qtype=QuantumFloat(3), shape=(N,))
            for qv in qvs:
                h(qv[0])
            return tuple(qv for qv in qvs)

        @jaspify
        def main():
            return expectation_value(kernel, shots=SHOTS)()

        assert isinstance(main(), tuple) and len(main()) == N
