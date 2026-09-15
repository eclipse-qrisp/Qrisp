# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Assembles the BlockEncoding class by attaching its constructor and transformation methods."""

from .block_encoding_base import BlockEncoding
from .block_encoding_combination import (
    LinearCombinationBlockEncoding,
    apply_add,
    apply_kron,
    apply_matmul,
    apply_mul,
    apply_neg,
    apply_radd,
    apply_sub,
    build_from_lcu_terms,
    build_linear_combination,
)
from .constructors import (
    build_from_array,
    build_from_eye,
    build_from_foqcs_lcu_operator,
    build_from_foqcs_lcu_prep,
    build_from_lcu,
    build_from_operator,
    build_from_projector,
)
from .transformations import apply_inv, apply_poly, apply_pseudo_inv, apply_sim, apply_svt

BlockEncoding.from_array = classmethod(build_from_array)
BlockEncoding.from_eye = classmethod(build_from_eye)
BlockEncoding.from_lcu = classmethod(build_from_lcu)
BlockEncoding.from_foqcs_lcu_prep = classmethod(build_from_foqcs_lcu_prep)
BlockEncoding.from_foqcs_lcu_operator = classmethod(build_from_foqcs_lcu_operator)
BlockEncoding.from_operator = classmethod(build_from_operator)
BlockEncoding.from_projector = classmethod(build_from_projector)

BlockEncoding.linear_combination = classmethod(build_linear_combination)
BlockEncoding._from_lcu_terms = classmethod(build_from_lcu_terms)

# Special methods are looked up on the type, so assigning them here is equivalent
# to defining them in the class body.
BlockEncoding.__add__ = apply_add
BlockEncoding.__radd__ = apply_radd
BlockEncoding.__sub__ = apply_sub
BlockEncoding.__mul__ = apply_mul
BlockEncoding.__rmul__ = apply_mul
BlockEncoding.__matmul__ = apply_matmul
BlockEncoding.__neg__ = apply_neg
BlockEncoding.kron = apply_kron

BlockEncoding.inv = apply_inv
BlockEncoding.poly = apply_poly
BlockEncoding.pseudo_inv = apply_pseudo_inv
BlockEncoding.sim = apply_sim
BlockEncoding.svt = apply_svt

__all__ = ["BlockEncoding", "LinearCombinationBlockEncoding"]
