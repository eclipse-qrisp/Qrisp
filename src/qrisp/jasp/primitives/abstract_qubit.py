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

from jax.core import AbstractValue


class AbstractQubit(AbstractValue):

    def __repr__(self):
        return "Qubit"

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        if not isinstance(other, AbstractQubit):
            return False
        return isinstance(other, AbstractQubit)

    def _add(self, a, b):
        from qrisp.jasp import fuse_qb_array, DynamicQubitArray

        if isinstance(b, DynamicQubitArray):
            b = b.tracer
        return DynamicQubitArray(fuse_qb_array(a, b))
