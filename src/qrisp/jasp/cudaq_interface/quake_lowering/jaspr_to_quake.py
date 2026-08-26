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

# Entry point for the Jasp → Quake (memory-semantics) lowering pipeline.
# =========================================================================
#
# Pipeline
# --------
# The lowering consists of the following passes:
#
# 0. Emission (mlir_emission) – Translate the Jaspr to an initial xDSL
#    builtin.ModuleOp via jaspr_to_mlir.
# 0a. Safeguard (safeguard_no_ranked_tensor_linalg) – Reject any module
#     that contains linalg.generic operations on ranked tensors before lowering begins.
# 1. JASP → Quake (jasp_to_quake) – Replace jasp.* operations with
#    Quake equivalents and eliminate !jasp.QuantumState threading.
# 2. SCF → CC (scf_to_cc) – Replace structured control flow with
#    cc.if and cc.loop operations.
# 3. Scalar tensor unwrapping (scalar_tensor_unwrap) – Fold trivial
#    rank-0 tensor constants, extracts, and wrappers into scalars.
# 4. Static register allocation (static_veq_alloca) – Rewrite
#    constant-sized !quake.veq<?> allocations as !quake.veq<N>.
# 5. Ranked tensor → CC array (ranked_tensor_to_array) – Lower ranked
#    tensor constants, accesses, signatures, and calls to CC arrays.
# 6. Array → stdvec (array_to_stdvec) – Rewrite entrypoint array
#    pointers to !cc.stdvec<T> for CUDA-Q runtime compatibility.
#
# The returned ModuleOp contains only the dialects and operations supported by
# the CUDA-Q ingestion layer; no !jasp.* types or tensor operations remain.

from xdsl.dialects.builtin import ModuleOp

from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.array_to_stdvec import (
    _lower_array_to_stdvec,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.jasp_to_quake.jasp_to_quake import (
    _jasp_to_quake,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.ranked_tensor_to_array import (
    _lower_ranked_tensors,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.safeguard_no_ranked_tensor_linalg import (
    _verify_no_ranked_tensor_linalg,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.scalar_tensor_unwrap import (
    _unwrap_scalar_tensors,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.scf_to_cc import _lower_scf_to_cc
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.static_veq_alloca import (
    _staticize_veq_alloca,
)
from qrisp.jasp.cudaq_interface.quake_lowering.pass_manager import (
    _LoweringPass,
    _run_pass_pipeline,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir


def _jaspr_to_quake_mlir(jaspr: Jaspr, execution_mode: str = "run") -> ModuleOp:
    """Lower a :class:`~qrisp.jasp.Jaspr` to a Quake+CC ``builtin.ModuleOp``.

    Parameters
    ----------
    jaspr: Jaspr
        A :class:`~qrisp.jasp.Jaspr` (closed-form JAX trace) to lower.
    execution_mode:
        Controls how quantum measurements are lowered and how the function
        signature is generated.  Two values are accepted:

        ``"run"`` *(default)*
            Targets ``cudaq.run``.  Array measurements are lowered to a
            ``cc.loop`` that extracts each qubit, calls ``quake.mz`` +
            ``quake.discriminate``, and packs the resulting bits into an
            ``i64`` accumulator.  Single-qubit measurements are lowered to
            ``quake.mz`` + ``quake.discriminate`` returning ``tensor<i1>``.
            Classical return values are preserved in the function signature.

        ``"sample"``
            Targets ``cudaq.sample``.  Every ``quake.mz`` is emitted on the
            full operand (``!quake.ref`` or ``!quake.veq<?>``), leaving the
            ``!quake.measure`` / ``!cc.stdvec<!quake.measure>`` result for the
            CUDAQ runtime to collect across shots.  To keep SSA valid through
            all intermediate passes, a zero dummy constant (``tensor<i1>``
            for single qubits, ``tensor<i64>`` for arrays) is substituted
            wherever the classical measurement result would otherwise be used.
            All classical return values are then stripped from ``func.return``
            and the function signature so that the kernel has a ``void``
            return type, as required by ``cudaq.sample``.

    Returns
    -------
    xdsl.dialects.builtin.ModuleOp
        An xDSL module representing the quantum computation in Quake and CC dialects.

    Raises
    ------
    ImportError
        If the ``xdsl`` package is not installed.
    CudaqUnsupportedArrayOperationError
        If the emitted module contains an unsupported array operation.

    """
    # Step 0 – Produce the initial xDSL module with Jasp IR.
    module: ModuleOp = jaspr_to_mlir(jaspr, lower_stableHLO=True)

    _run_pass_pipeline(
        module,
        (
            _LoweringPass("verify-no-ranked-tensor-linalg", _verify_no_ranked_tensor_linalg),
            _LoweringPass(
                "jasp-to-quake",
                lambda current_module: _jasp_to_quake(current_module, execution_mode),
            ),
            _LoweringPass("scf-to-cc", _lower_scf_to_cc),
            _LoweringPass("scalar-tensor-unwrap", _unwrap_scalar_tensors),
            _LoweringPass("staticize-veq-alloca", _staticize_veq_alloca),
            _LoweringPass("ranked-tensor-to-array", _lower_ranked_tensors),
            _LoweringPass("array-to-stdvec", _lower_array_to_stdvec),
        ),
    )

    return module
