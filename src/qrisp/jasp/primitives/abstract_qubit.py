# """
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
# """

"""Defines :class:`AbstractQubit`, the JAX abstract value representing a single traced qubit."""

from jax.core import AbstractValue


class AbstractQubit(AbstractValue):
    """JAX abstract value representing a single traced qubit."""

    def __repr__(self):
        return "Qubit"

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        if not isinstance(other, AbstractQubit):
            return False
        return isinstance(other, AbstractQubit)

    def _add(self, a, b):
        # Deferred import: qrisp.jasp.tracing_logic is loaded after
        # qrisp.jasp.primitives, so this can't be a top-level import.
        from qrisp.jasp import DynamicQubitArray, fuse_qb_array

        if isinstance(b, DynamicQubitArray):
            b = b.tracer
        return DynamicQubitArray(fuse_qb_array(a, b))
