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
    module = jaspr_to_quake(jaspr)
    print(module)

Pipeline
--------
The lowering consists of three passes:

1. **PASS 1** (:mod:`.pass1_jasp_to_quake`) – Replace every ``jasp.*`` op by
   its Quake equivalent and eliminate the ``!jasp.QuantumState`` threading.
2. **PASS 2** (:mod:`.pass2_scf_to_cc`) – Replace ``scf.if`` / ``scf.while``
   (where they have no SSA results) with ``cc.if`` / ``cc.loop``.
3. **PASS 3** (:mod:`.pass3_tensor_unwrap`) – Fold trivial rank-0 tensor
   constants / extracts into scalars.

The returned ``builtin.ModuleOp`` contains only Quake + CC + arith/tensor/func
ops; no ``!jasp.*`` types remain.
"""


from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from qrisp.jasp.mlir.quake_lowering.pass1_jasp_to_quake import lower_jasp_to_quake
from qrisp.jasp.mlir.quake_lowering.pass2_scf_to_cc_v3 import lower_scf_to_cc
from qrisp.jasp.mlir.quake_lowering.pass3a_ranked_tensor import lower_ranked_tensors
from qrisp.jasp.mlir.quake_lowering.pass3_tensor_unwrap import unwrap_tensors
from qrisp.jasp.mlir.mlir_rewrites.scalar_tensor_folding import scalar_tensor_folding


def jaspr_to_quake(jaspr, lower_stableHLO: bool = True) -> ModuleOp:
    """Lower a :class:`~qrisp.jasp.Jaspr` to a Quake+CC ``builtin.ModuleOp``.

    Parameters
    ----------
    jaspr:
        A :class:`~qrisp.jasp.Jaspr` (closed-form JAX trace) to lower.
    lower_stableHLO:
        When *True* (default) the intermediate step uses
        ``jaxpr_to_xdsl(jaxpr, lower_stableHLO=True)`` so that classical
        arithmetic is already in ``arith``/``tensor``/``linalg`` form before
        the quantum lowering begins.

    Returns
    -------
    builtin.ModuleOp
        An xDSL module containing Quake (memory-semantics) + CC ops.

    Raises
    ------
    ImportError
        If the ``xdsl`` package is not installed.
    """
    # Step 1 – Produce the initial xDSL module with Jasp IR.
    from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir
    module: ModuleOp = jaspr_to_mlir(jaspr, lower_stableHLO=lower_stableHLO)

    # Step 2 – PASS 1: QuantumState elimination + Jasp→Quake rewriting.
    lower_jasp_to_quake(module)

    # Step 3 – PASS 2: SCF→CC lowering.
    lower_scf_to_cc(module)

    # Step 4 – PASS 3a: ranked tensor lowering.
    lower_ranked_tensors(module)

    # Step 5 – PASS 3b: tensor unwrapping + scalar constant folding.
    unwrap_tensors(module)

    scalar_tensor_folding(Context(), module)

    return module
