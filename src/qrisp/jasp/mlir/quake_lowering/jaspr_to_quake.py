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

"""
Entry point for the Jasp → Quake (memory-semantics) lowering pipeline.

Usage example::

    from qrisp import QuantumVariable, h, cx, measure
    from qrisp.jasp import make_jaspr
    from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake

    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    jaspr = make_jaspr(bell)()
    mlir_str = jaspr_to_quake_mlir(jaspr)
    print(mlir_str)

Pipeline
--------
The lowering consists of the following passes:

0. **Emission** (:mod:`.mlir_emission`) – Translate the :class:`~qrisp.jasp.Jaspr`
   to an initial xDSL ``builtin.ModuleOp`` via ``jaspr_to_mlir``.
0a. **Safeguard** (:mod:`.safeguard_no_ranked_tensor_linalg`) – Reject any module
    that contains ``linalg.generic`` ops on ranked tensors before lowering begins.
1. **PASS 1** (:mod:`.pass1_jasp_to_quake`) – Replace every ``jasp.*`` op by
   its Quake equivalent, eliminate the ``!jasp.QuantumState`` threading, and
   perform QuantumState elimination.
2. **PASS 2** (:mod:`.pass2_scf_to_cc`) – Replace ``scf.if`` / ``scf.while``
   (where they have no SSA results) with ``cc.if`` / ``cc.loop``.
3. **PASS 3** (:mod:`.pass3_scalar_tensor_unwrap`) – Fold trivial rank-0 tensor
   constants / extracts into scalars.
4. **PASS 4** (:mod:`.pass4_ranked_tensor_to_array`) – Lower ranked tensor constants
    and accesses to CC array operations.
5. **PASS 5** (:mod:`.pass5_array_to_stdvec`) – Rewrite static array pointer
   parameters to ``!cc.stdvec<T>`` for CUDA-Q runtime compatibility.

The returned string contains only Quake + CC + arith, math, func ops;
no ``!jasp.*`` types or tensor ops remain.
"""


from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from qrisp.jasp.jasp_expression import Jaspr

# from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.pass1_jasp_to_quake import lower_jasp_to_quake
from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.pass1_jasp_to_quake import jasp_to_quake
from qrisp.jasp.mlir.quake_lowering.pass2_scf_to_cc import lower_scf_to_cc
from qrisp.jasp.mlir.quake_lowering.pass3_scalar_tensor_unwrap import (
    unwrap_scalar_tensors,
)
from qrisp.jasp.mlir.quake_lowering.pass4_ranked_tensor_to_array import (
    lower_ranked_tensors,
)
from qrisp.jasp.mlir.quake_lowering.pass5_array_to_stdvec import (
    lower_array_to_stdvec,
)
from qrisp.jasp.mlir.quake_lowering.safeguard_no_ranked_tensor_linalg import (
    verify_no_ranked_tensor_linalg,
)


def jaspr_to_quake_mlir(jaspr: Jaspr, execution_mode: str = "run") -> str:
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
    str
        A string representation of an xDSL module containing Quake (memory-semantics) + CC ops.

    Raises
    ------
    ImportError
        If the ``xdsl`` package is not installed.
    LinalgRankedTensorError
        If the emitted module contains ``linalg.generic`` on ranked tensors.
    """
    # Step 0 – Produce the initial xDSL module with Jasp IR.
    from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir

    module: ModuleOp = jaspr_to_mlir(jaspr, lower_stableHLO=True)

    # Step 0a – Safeguard: reject ranked-tensor linalg.generic early.
    verify_no_ranked_tensor_linalg(module)

    # Step 1 – PASS 1: QuantumState elimination + Jasp → Quake rewriting.
    jasp_to_quake(module, execution_mode=execution_mode)

    # Step 2 – PASS 2: SCF → CC lowering.
    lower_scf_to_cc(module)

    # Step 3 – PASS 3: scalar tensor unwrapping + scalar constant folding.
    unwrap_scalar_tensors(module)

    # Step 4 – PASS 4: ranked tensor → CC array lowering.
    lower_ranked_tensors(module)

    # Step 5 – PASS 5: array ptr params → stdvec (CUDA-Q runtime compatibility).
    lower_array_to_stdvec(module)

    return str(module)
