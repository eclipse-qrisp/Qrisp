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

from jax.extend.core import Primitive


class QuantumPrimitive(Primitive):
    """Base class for Jasp quantum primitives."""

    def __init__(self, name: str) -> None:
        """Initialize a new quantum primitive with the given name."""

        Primitive.__init__(self, "jasp." + name)
        self.qrisp_name = name

    def abstract_eval(self, *args, **params):
        """Abstract evaluation of the primitive."""
        raise NotImplementedError("Subclasses must implement abstract_eval")

    def impl(self, *args, **params):
        """Implementation of the primitive."""
        raise NotImplementedError("Subclasses must implement impl")
