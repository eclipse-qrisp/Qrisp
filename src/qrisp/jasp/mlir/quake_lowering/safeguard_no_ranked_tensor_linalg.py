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

"""Early safeguard for ranked-tensor ``linalg.generic`` patterns.

This module provides a small verifier that can be run on freshly emitted MLIR
to fail early with a clear error if unsupported ranked-tensor linalg patterns
are present before Quake lowering starts.
"""

from xdsl.dialects import linalg
from xdsl.dialects.builtin import ModuleOp, TensorType


class LinalgRankedTensorError(RuntimeError):
    """Raised when ranked-tensor ``linalg.generic`` survives emission."""


def _is_ranked_tensor_type(t) -> bool:
    """Return ``True`` iff *t* is a ranked TensorType (rank > 0)."""
    return isinstance(t, TensorType) and len(t.get_shape()) > 0


def verify_no_ranked_tensor_linalg(module: ModuleOp) -> None:
    """Fail early if any ``linalg.generic`` operates on ranked tensors."""
    for current_op in module.walk():
        if not isinstance(current_op, linalg.GenericOp):
            continue

        operand_has_ranked_tensor = any(_is_ranked_tensor_type(operand.type) for operand in current_op.operands)
        result_has_ranked_tensor = any(_is_ranked_tensor_type(result.type) for result in current_op.results)

        if operand_has_ranked_tensor or result_has_ranked_tensor:
            location = getattr(current_op, "location", None)
            loc_info = str(location) if location is not None else "unknown location"
            raise LinalgRankedTensorError(
                "Cannot lower this program to CUDA-Q because it still contains a "
                "ranked-tensor linalg.generic operation.\n\n"
                "What this usually means:\n"
                "- You performed arithmetic on traced jax.numpy arrays (for example "
                "array expressions inside the traced function), which introduced "
                "ranked tensor linalg ops.\n\n"
                "How to fix it:\n"
                "- Prefer scalar values inside the traced kernel body.\n"
                "- Move array arithmetic outside tracing and pass only the needed "
                "elements/scalars into the kernel.\n"
                "- If you pass arrays as kernel parameters, stick to simple indexed "
                "access patterns.\n\n"
                f"Location: {loc_info}\n"
                f"Offending operation: {current_op}"
            )
