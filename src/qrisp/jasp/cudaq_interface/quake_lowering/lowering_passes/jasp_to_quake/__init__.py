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

# Jasp → Quake lowering (QuantumState elimination + op rewriting).
# ====================================================================
#
# jasp_to_quake
#     Orchestrator, running 1a then 1b.
# lower_jasp_to_quake
#     Rewrites Jasp quantum ops into their Quake dialect equivalents.
# strip_qst
#     Eliminates the QuantumState-threading value, replacing it with direct
#     Quake qubit-reference semantics.
# gate_mapping
#     Mapping from Jasp gate names to Quake gate descriptors.
# helper_functions
#     Shared xDSL-construction helpers used by 1a/1b.
