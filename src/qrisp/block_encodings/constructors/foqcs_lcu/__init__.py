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

from .from_foqcs_lcu_prep import build_from_foqcs_lcu_prep
from .from_foqcs_lcu_operator import build_from_foqcs_lcu_operator
from .foqcs_preps import foqcs_prep_heisenberg, foqcs_prep_spin_glass
from .foqcs_analysis import (
    is_operator_foqcs_compatible,
    foqcs_analyze_operator_spin_glass,
    foqcs_analyze_operator_heisenberg,
)

__all__ = [
    "build_from_foqcs_lcu_prep",
    "build_from_foqcs_lcu_operator",
    "foqcs_prep_heisenberg",
    "is_operator_foqcs_compatible",
    "foqcs_analyze_operator_spin_glass",
    "foqcs_analyze_operator_heisenberg",
    "foqcs_prep_spin_glass",
]
