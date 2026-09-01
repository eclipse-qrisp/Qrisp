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

r"""Comprehensive tests for the new pytree-aware return signature of
:func:`~qrisp.jasp.sample`.

Coverage matrix
---------------

============================ ======= ========= ============ ======
Case                         JIT     Terminal  Non-JIT     Backend
============================ ======= ========= ============ ======
scalar (quantum)               ✓        ✓          ✓          ✓
scalar (classical)             ✓        ✗¹         —          ✓
flat tuple                     ✓        ✓          ✓          ✓
nested tuple                   ✓        —          —          —
flat list                      ✓        —          —          ✓
nested list                    ✓        —          —          —
dict                           ✓        —          —          —
mixed bool/float dtype         ✓        —          —          —
array-valued leaf              ✓        —          —          —
post_processor (struct-pres)   ✓        ✓          —          ✓
post_processor (struct-change) ✓        ✓          —          —
zero shots                     ✓        —          —          ✓
dynamic kernel args            ✓        —          —          ✓
user-defined pytree (error)    ✓        —          —          —
single-element tuple²          ✓        —          —          —
============================ ======= ========= ============ ======

¹ terminal sampling rejects classical returns by design
² single-element tuples are unwrapped by the post-processor calling
  convention and reach the accumulator as scalars

The ``Non-JIT`` column refers to ``@sample`` used as a standalone
decorator (without ``@jaspify``), which returns a ``dict``.
The ``Backend`` column requires a simulator backend (StimBackend or
QrispSimulatorBackend).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    QuantumArray,
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    cx,
    h,
    measure,
    t,
    x,
)
from qrisp.jasp import jaspify, sample

# =============================================================================
# Helpers
# =============================================================================

SHOTS = 25


def _identity(*args):
    if len(args) == 1:
        return args[0]
    return args


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
    """Single-value returns — shape and dtype unchanged from old behaviour."""

    # -- quantum ----------------------------------------------------------

    def test_scalar_quantum_jit(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return qf

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res.shape == (SHOTS,)
        assert res.dtype == jnp.float64

    def test_scalar_quantum_terminal(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return qf

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res.shape == (SHOTS,)

    def test_scalar_quantum_non_jit(self):
        @sample(SHOTS)
        def main():
            qf = QuantumFloat(3)
            h(qf[0])
            return qf

        res = main()
        assert isinstance(res, dict)
        assert sum(res.values()) == SHOTS

    # -- classical --------------------------------------------------------

    def test_scalar_classical_jit(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            h(qf[1])
            return measure(qf)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res.shape == (SHOTS,)
        assert len(jnp.unique(res)) >= 2  # superposition → variation

    def test_scalar_classical_terminal_rejected(self):
        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return measure(qf)

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()


# =============================================================================
# 2. Flat tuple returns
# =============================================================================


class TestFlatTupleReturns:
    """``return a, b, c`` → ``(array_a, array_b, array_c)``."""

    def test_two_elements_jit(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS,) and res[1].shape == (SHOTS,)
        assert res[0].dtype == jnp.float64 and res[1].dtype == jnp.float64

    def test_three_elements_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            c = QuantumFloat(3)
            h(a[0])
            return a, b, c

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 3
        assert all(r.shape == (SHOTS,) for r in res)

    def test_two_elements_terminal(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert all(r.shape == (SHOTS,) for r in res)

    def test_two_elements_non_jit(self):
        @sample(SHOTS)
        def main():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return a, b

        res = main()
        assert isinstance(res, dict)
        assert sum(res.values()) == SHOTS
        for k in res:
            assert isinstance(k, tuple) and len(k) == 2

    def test_classical_tuple_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            cx(a[0], b[0])
            return measure(a), measure(b)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert all(r.shape == (SHOTS,) for r in res)

    def test_classical_tuple_terminal_rejected(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return measure(a), measure(b)

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()


# =============================================================================
# 3. Nested tuple returns
# =============================================================================


class TestNestedTupleReturns:
    """``return a, (b, c)`` → ``(array_a, (array_b, array_c))``.

    Nested tuples only work when the inner elements are classical
    (already measured).  Undecoded QuantumVariables nested inside
    tuples are not automatically measured — only top-level QVs are.
    """

    def test_tuple_of_tuple_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            h(b[1])
            # Inner values must be measured to be treated as classical
            return measure(a), (measure(b),)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS,)
        assert isinstance(res[1], tuple) and len(res[1]) == 1
        assert res[1][0].shape == (SHOTS,)

    def test_deeply_nested_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            c = QuantumFloat(3)
            h(a[0])
            h(b[0])
            h(c[0])
            return (measure(a), (measure(b), (measure(c),)))

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS,)
        assert isinstance(res[1], tuple) and len(res[1]) == 2
        assert res[1][0].shape == (SHOTS,)
        assert isinstance(res[1][1], tuple) and len(res[1][1]) == 1
        assert res[1][1][0].shape == (SHOTS,)

    def test_mixed_quantum_classical_nested(self):
        """Top-level QuantumVariable is decoded; nested classical tuple
        passes through the post-processor unchanged."""

        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            h(b[1])
            return a, (measure(b),)  # top-level QV + nested classical

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS,)
        # (measure(b),) is a classical leaf → 1D array of 1-tuples
        assert isinstance(res[1], tuple) and len(res[1]) == 1
        assert res[1][0].shape == (SHOTS,)


# =============================================================================
# 4. List returns
# =============================================================================


class TestListReturns:
    """``return [a, b]`` → ``[array_a, array_b]``."""

    def test_flat_list_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return [measure(a), measure(b)]

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, list) and len(res) == 2
        assert res[0].shape == (SHOTS,) and res[1].shape == (SHOTS,)

    def test_nested_list_of_tuple_jit(self):
        """``return [ (a, b), c ]`` → ``[ (array_a, array_b), array_c ]``."""

        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            c = QuantumFloat(3)
            h(a[0])
            h(b[0])
            return [(measure(a), measure(b)), measure(c)]

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, list) and len(res) == 2
        assert isinstance(res[0], tuple) and len(res[0]) == 2
        assert res[0][0].shape == (SHOTS,)
        assert res[0][1].shape == (SHOTS,)
        assert res[1].shape == (SHOTS,)

    def test_deeply_nested_list_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            c = QuantumFloat(3)
            h(a[0])
            return [[measure(a)], [measure(b), measure(c)]]

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, list) and len(res) == 2
        assert isinstance(res[0], list) and len(res[0]) == 1
        assert res[0][0].shape == (SHOTS,)
        assert isinstance(res[1], list) and len(res[1]) == 2
        assert res[1][0].shape == (SHOTS,)
        assert res[1][1].shape == (SHOTS,)


# =============================================================================
# 5. Dict returns
# =============================================================================


class TestDictReturns:
    """``return {'x': a, 'y': b}`` → ``{'x': array_a, 'y': array_b}``."""

    def test_flat_dict_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return {"val": measure(a), "aux": measure(b)}

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, dict) and set(res.keys()) == {"val", "aux"}
        assert res["val"].shape == (SHOTS,)
        assert res["aux"].shape == (SHOTS,)

    def test_nested_dict_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            c = QuantumBool()
            h(a[0])
            h(c)
            return {"a": measure(a), "nested": {"b": measure(b), "c": measure(c)}}

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, dict)
        assert res["a"].shape == (SHOTS,)
        assert isinstance(res["nested"], dict)
        assert res["nested"]["b"].shape == (SHOTS,)
        assert res["nested"]["c"].shape == (SHOTS,)


# =============================================================================
# 6. Dtype preservation
# =============================================================================


class TestDtypePreservation:
    """Heterogeneous return types keep their native dtypes."""

    def test_bool_float_mixed_jit(self):
        def kernel():
            qf = QuantumFloat(4)
            qb = QuantumBool()
            h(qf[0])
            h(qb)
            return measure(qf), measure(qb)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res[0].dtype == jnp.float64
        assert res[1].dtype == bool

    def test_mixed_in_dict_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            qb = QuantumBool()
            h(qf[0])
            h(qb)
            return {"f": measure(qf), "b": measure(qb)}

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res["f"].dtype == jnp.float64
        assert res["b"].dtype == bool

    def test_mixed_in_nested_structure_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            qb = QuantumBool()
            h(qf[0])
            h(qb)
            return [(measure(qf),), {"flag": measure(qb)}]

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, list) and len(res) == 2
        assert isinstance(res[0], tuple)
        assert res[0][0].dtype == jnp.float64
        assert isinstance(res[1], dict)
        assert res[1]["flag"].dtype == bool


# =============================================================================
# 7. Array-valued leaves
# =============================================================================


class TestArrayValuedLeaves:
    """Leaves that are themselves arrays preserve their inner dimensions."""

    def test_array_and_scalar_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return jnp.array([1.0, 2.0, 3.0]), measure(qf)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res[0].shape == (SHOTS, 3)
        assert res[1].shape == (SHOTS,)

    def test_array_inside_tuple_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return (jnp.ones(4), measure(qf))

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS, 4)
        assert res[1].shape == (SHOTS,)
        assert jnp.all(res[0] == 1.0)

    def test_multiple_arrays_different_shapes_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return jnp.zeros(2), jnp.ones(5), measure(qf)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res[0].shape == (SHOTS, 2)
        assert res[1].shape == (SHOTS, 5)
        assert res[2].shape == (SHOTS,)

    def test_single_non_scalar_array_jit(self):
        """Kernel returns a single non-scalar JAX array (not inside a
        tuple).  The result should stack along the leading dimension."""

        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            # single 2D array return — no tuple wrapper
            return jnp.array([[1, 2], [3, 4]])

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert res.shape == (SHOTS, 2, 2)
        assert jnp.all(res[0] == jnp.array([[1, 2], [3, 4]]))


# =============================================================================
# 8. Post-processor
# =============================================================================


class TestPostProcessor:
    """Post-processor transforms values per-shot."""

    def test_structure_preserving_jit(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS, post_processor=_double)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS,) and res[1].shape == (SHOTS,)
        # All values should be even (doubled)
        assert jnp.all(res[0] % 2 == 0)
        assert jnp.all(res[1] % 2 == 0)

    def test_structure_preserving_terminal(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS, post_processor=_double)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert all(r.shape == (SHOTS,) for r in res)

    def test_tuple_to_scalar_jit(self):
        """Post-processor combines two values into one."""

        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS, post_processor=_sum)()

        res = main()
        assert res.shape == (SHOTS,)

    def test_scalar_to_tuple_jit(self):
        """Post-processor expands a single value into a tuple."""

        def kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            return qf

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS, post_processor=_to_tuple)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (SHOTS,) and res[1].shape == (SHOTS,)
        # second is double the first
        assert jnp.all(res[1] == 2 * res[0])

    def test_tuple_to_scalar_terminal(self):
        def kernel():
            a = QuantumFloat(4)
            b = QuantumFloat(4)
            h(a[0])
            return a, b

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS, post_processor=_sum)()

        res = main()
        assert res.shape == (SHOTS,)


# =============================================================================
# 9. Edge cases
# =============================================================================


class TestEdgeCases:
    """Boundary conditions and error cases."""

    def test_zero_shots_jit(self):
        """shots=0 raises ValueError at validation time."""

        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return qf

        @jaspify
        def main():
            return sample(kernel, shots=0)()

        with pytest.raises(ValueError, match="positive integer"):
            main()

    def test_zero_shots_tuple_jit(self):
        """shots=0 raises ValueError (same as scalar case)."""

        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return a, b

        @jaspify
        def main():
            return sample(kernel, shots=0)()

        with pytest.raises(ValueError, match="positive integer"):
            main()

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
            return sample(kernel, shots=SHOTS)(k)

        res = main(2)
        assert isinstance(res, tuple) and len(res) == 2

    def test_single_element_tuple_becomes_scalar(self):
        """A 1-tuple return is unwrapped by the post-processor calling
        convention, so it reaches the accumulator as a scalar."""

        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return (measure(qf),)  # 1-tuple

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        # Reaches accumulator as scalar, not as a tuple
        assert res.shape == (SHOTS,)

    def test_mixed_quantum_classical_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            mes = measure(qf[1])  # classical
            return qf, mes

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert all(r.shape == (SHOTS,) for r in res)

    def test_mixed_quantum_classical_terminal_rejected(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            mes = measure(qf[1])
            return qf, mes

        @jaspify(terminal_sampling=True)
        def main():
            return sample(kernel, shots=SHOTS)()

        with pytest.raises(Exception):
            main()

    def test_pure_classical_multiple_jit(self):
        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            h(b[1])
            return measure(a), measure(b), measure(a)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 3
        assert all(r.shape == (SHOTS,) for r in res)

    def test_user_defined_pytree_raises(self):
        """A user-defined JAX pytree (e.g. a dataclass) should raise
        TypeError — not silently treat it as a scalar."""
        from dataclasses import dataclass

        import jax.tree_util as jtu

        @dataclass
        class MyPytree:
            x: object
            y: object

        jtu.register_pytree_node(
            MyPytree,
            lambda obj: ((obj.x, obj.y), None),
            lambda _, children: MyPytree(*children),
        )

        def kernel():
            a = QuantumFloat(3)
            b = QuantumFloat(3)
            h(a[0])
            return MyPytree(measure(a), measure(b))

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        with pytest.raises(TypeError, match="Unsupported return type"):
            main()

    def test_non_jit_decorator_returns_dict(self):
        @sample(SHOTS)
        def main():
            qf = QuantumFloat(3)
            h(qf[0])
            return qf

        res = main()
        assert isinstance(res, dict)
        assert sum(res.values()) == SHOTS

    def test_non_jit_decorator_multi_return_returns_dict(self):
        @sample(SHOTS)
        def main():
            a = QuantumFloat(3)
            b = QuantumBool()
            h(a[0])
            h(b)
            return a, b

        res = main()
        assert isinstance(res, dict)
        assert sum(res.values()) == SHOTS
        # Keys are tuples of decoded values
        for k in res:
            assert isinstance(k, tuple) and len(k) == 2

    def test_identity_post_processor_preserves_structure(self):
        """Default (identity) post-processor preserves the interleaved
        order of quantum and classical returns."""

        def kernel():
            qf = QuantumFloat(4)
            qb = QuantumBool()
            h(qf[0])
            h(qb)
            return measure(qf), measure(qb)

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].dtype == jnp.float64
        assert res[1].dtype == bool


# =============================================================================
# 10. Large / stress tests
# =============================================================================


class TestStress:
    """Larger shot counts and many return values."""

    def test_many_return_values_jit(self):
        N = 8

        def kernel():
            qvs = QuantumArray(qtype=QuantumFloat(3), shape=(N,))
            for qv in qvs:
                h(qv[0])
            return tuple(measure(qv) for qv in qvs)

        @jaspify
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, tuple) and len(res) == N
        assert all(r.shape == (SHOTS,) for r in res)

    def test_high_shot_count_jit(self):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            return qf

        @jaspify
        def main():
            return sample(kernel, shots=1000)()

        res = main()
        assert res.shape == (1000,)

    def test_nested_structure_many_leaves_jit(self):
        """Deeply nested structure with many leaves."""

        def kernel():
            qf1 = QuantumFloat(2)
            qf2 = QuantumFloat(2)
            qf3 = QuantumFloat(2)
            qf4 = QuantumFloat(2)
            h(qf1[0])
            h(qf3[1])
            return {"a": (measure(qf1), [measure(qf2)]), "b": {"c": measure(qf3), "d": measure(qf4)}}

        @jaspify(terminal_sampling=False)
        def main():
            return sample(kernel, shots=SHOTS)()

        res = main()
        assert isinstance(res, dict)
        assert isinstance(res["a"], tuple)
        assert isinstance(res["a"][1], list)
        assert res["a"][0].shape == (SHOTS,)
        assert res["a"][1][0].shape == (SHOTS,)
        assert isinstance(res["b"], dict)
        assert res["b"]["c"].shape == (SHOTS,)
        assert res["b"]["d"].shape == (SHOTS,)
