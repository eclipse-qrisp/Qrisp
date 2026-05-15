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

from qrisp.jasp.mlir.mlir_rewrites.scalar_tensor_folding import scalar_tensor_folding
from qrisp.jasp.mlir.mlir_rewrites.scalar_linalg_folding import scalar_linalg_folding
from qrisp.jasp.mlir.mlir_rewrites.cmpi_extui_folding import cmpi_extui_folding
