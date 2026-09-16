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


# IR-level tests for the ranked-tensor -> CC array lowering pass.
#
# These assert on the MLIR text produced by the Quake lowering pipeline rather
# than on CUDA-Q execution results, so they run in well under a second and
# pinpoint which construct leaks a tensor operation. The end-to-end behaviour of
# the same programs is covered by test_cudaq_arrays.py.

import re

import jax.numpy as jnp

from qrisp import QuantumVariable, jrange, measure, rz
from qrisp.jasp import make_jaspr, qache
from qrisp.jasp.cudaq_interface.quake_lowering.jaspr_to_quake import _jaspr_to_quake_mlir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(func, *args) -> str:
    """Lower *func* through the Quake pipeline and return the MLIR text."""
    return str(_jaspr_to_quake_mlir(make_jaspr(func)(*args), execution_mode="run"))


def _assert_no_tensors(mlir: str) -> None:
    """Assert that no tensor operation or tensor type survived the lowering.

    CUDA-Q 0.16 no longer registers the upstream ``tensor`` dialect in its MLIR
    context, so any residual tensor operation makes ingestion fail at parse time.
    """
    residual_ops = sorted(set(re.findall(r"\btensor\.[a-z_]+", mlir)))
    assert not residual_ops, f"residual tensor operations: {residual_ops}\n\n{mlir}"

    residual_types = sorted(set(re.findall(r"tensor<[^>]*>", mlir)))
    assert not residual_types, f"residual tensor types: {residual_types}\n\n{mlir}"


# ---------------------------------------------------------------------------
# Constant materialization
# ---------------------------------------------------------------------------


def test_static_index_materializes_cc_array():
    """A local array constant becomes a cc.alloca filled by element stores."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    def main():
        arr = jnp.array(values)
        qv = QuantumVariable(5)
        rz(arr[0], qv[0])
        return measure(qv[0])

    mlir = _lower(main)

    _assert_no_tensors(mlir)
    assert f"cc.alloca !cc.array<f64 x {len(values)}>" in mlir
    assert mlir.count("cc.store") == len(values)
    assert "cc.load" in mlir


def test_element_zero_store_is_addressed_by_cc_cast():
    """Stores to element 0 go through cc.cast, matching what native CUDA-Q emits.

    Stores to element 1 and above use cc.compute_ptr. ``cc.compute_ptr %a[0]``
    would be an equally valid way to address element 0, but the materialization
    deliberately mirrors native CUDA-Q output here.

    Note that the asymmetry is confined to the store side: element *loads* use
    cc.compute_ptr for every index, element 0 included.
    """

    def main():
        arr = jnp.array([1.0, 2.0, 3.0])
        qv = QuantumVariable(1)
        rz(arr[0], qv[0])
        return measure(qv[0])

    mlir = _lower(main)

    cast = "cc.cast %0 : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>"
    assert cast in mlir
    # The cast result is what element 0 is stored through.
    cast_result = re.search(rf"%(\d+) = {re.escape(cast)}", mlir).group(1)
    assert f"cc.store %1, %{cast_result}" in mlir

    assert "cc.compute_ptr %0[1]" in mlir
    assert "cc.compute_ptr %0[2]" in mlir


# ---------------------------------------------------------------------------
# Element access
# ---------------------------------------------------------------------------


def test_dynamic_index_lowers_to_computed_pointer():
    """A runtime index becomes a dynamic cc.compute_ptr, not a tensor slice."""

    def main():
        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        qv = QuantumVariable(5)
        ind = jnp.int32(measure(qv[0]))
        rz(arr[ind], qv[1])
        return measure(qv[1])

    mlir = _lower(main)

    _assert_no_tensors(mlir)
    assert re.search(r"cc\.compute_ptr %\d+\[%\d+\]", mlir), mlir


# ---------------------------------------------------------------------------
# Function boundaries
# ---------------------------------------------------------------------------


def test_array_argument_becomes_array_pointer():
    """An array passed to a @qache function is passed as a CC array pointer."""

    @qache
    def apply_angle(arr, qv):
        rz(arr[0], qv[0])
        return measure(qv[0])

    def main():
        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        qv = QuantumVariable(5)
        return apply_angle(arr, qv)

    mlir = _lower(main)

    _assert_no_tensors(mlir)
    assert "%arg0: !cc.ptr<!cc.array<f64 x 5>>" in mlir


def test_dynamic_index_inside_qache():
    """A runtime index into an array argument lowers inside the callee."""

    @qache
    def apply_angle(arr, qv):
        ind = jnp.int32(measure(qv[0]))
        rz(arr[ind], qv[1])
        return measure(qv[1])

    def main():
        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        qv = QuantumVariable(2)
        return apply_angle(arr, qv)

    mlir = _lower(main)

    _assert_no_tensors(mlir)
    assert "%arg0: !cc.ptr<!cc.array<f64 x 5>>" in mlir


# ---------------------------------------------------------------------------
# Loop-carried arrays
# ---------------------------------------------------------------------------


def test_loop_carried_array_becomes_pointer():
    """An array carried through a cc.loop is carried as a CC array pointer."""

    def main():
        arr = jnp.array([0.1, 0.2, 0.3])
        qv = QuantumVariable(3)
        for i in jrange(3):
            rz(arr[i], qv[i])
        return measure(qv[0])

    mlir = _lower(main)

    _assert_no_tensors(mlir)
    assert "cc.loop" in mlir
    assert "!cc.ptr<!cc.array<f64 x 3>>" in mlir


# ---------------------------------------------------------------------------
# Arrays returned across function boundaries
# ---------------------------------------------------------------------------


def test_array_returned_from_qache_becomes_array_pointer():
    """An array returned from a @qache function crosses the boundary as a pointer.

    While the pass converted function argument types but not result types, the
    callee kept its tensor<NxT> result and the caller rebuilt a CC array from it
    with tensor.extract - operations that CUDA-Q 0.16 can no longer parse, since
    it no longer registers the upstream tensor dialect.
    """

    @qache
    def make_angles():
        return jnp.array([0.0, 3.14159265])

    @qache
    def apply_angle(arr, qv):
        rz(arr[1], qv[0])
        return measure(qv[0])

    def main():
        arr = make_angles()
        qv = QuantumVariable(1)
        return apply_angle(arr, qv)

    mlir = _lower(main)

    _assert_no_tensors(mlir)
    assert "@make_angles() -> (!cc.ptr<!cc.array<f64 x 2>>)" in mlir
