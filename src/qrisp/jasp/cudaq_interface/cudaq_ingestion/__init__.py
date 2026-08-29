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

"""Provide utilities for ingesting lowered modules into CUDA-Q."""

# CUDA-Q runtime ingestion: turning a Quake/CC xDSL module into a native,
# callable CUDA-Q kernel.
# =========================================================================
#
# This package picks up where qrisp.jasp.cudaq_interface.quake_lowering
# leaves off. While quake_lowering is responsible for producing a valid
# Quake/CC xDSL module from a Jasp representation, this package handles the
# CUDA-Q-runtime-specific packaging required to actually execute that module,
# namely:
#
# Sub-modules
# -----------
# host_attributes
#     Stage 1 -- "Target Extraction": detect the host's LLVM data layout and
#     target triple, required by CUDA-Q's C++ backend to allocate memory.
# cudaq_prep
#     Stage 2 -- "Interface Adaptation": restructure the module to match what
#     CUDA-Q's Module.parse expects (entrypoint renaming/attributes,
#     .run/.run.entry variants, cc.log_output synthesis, etc.).
# xdsl_ingestion
#     Stage 3 -- "Re-Compilation": serialize the adapted module, normalize it
#     from xDSL's to CUDA-Q's MLIR printing conventions, and parse it back
#     into a native PyKernelDecorator via cudaq_kernel_from_xdsl_module.

from qrisp.jasp.cudaq_interface.cudaq_ingestion.xdsl_ingestion import (
    _cudaq_kernel_from_xdsl_module,
)
