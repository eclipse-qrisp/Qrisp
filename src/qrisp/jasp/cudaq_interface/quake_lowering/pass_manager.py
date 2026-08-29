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

# Utilities for running the Quake lowering stages.

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp


@dataclass(frozen=True)
class _LoweringPass:
    """A named transformation that mutates an xDSL module in place."""

    name: str
    run: Callable[[ModuleOp], None]


def _run_pass_pipeline(module: ModuleOp, passes: Iterable[_LoweringPass]) -> None:
    """Run each named lowering pass against *module* in sequence."""
    for lowering_pass in passes:
        lowering_pass.run(module)
