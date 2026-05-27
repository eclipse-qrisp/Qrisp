# ********************************************************************************
# * Copyright (c) 2026 the Qrisp Authors
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

"""
Backward-compatibility shim.

:class:`DefaultBackend`, :class:`DefaultJob`, and :data:`def_backend` have
moved to :mod:`qrisp.interface.simulators.default_backend`.  This module
re-exports them so that existing code using
``from qrisp.default_backend import DefaultBackend`` (or equivalent) continues
to work without changes.

.. deprecated::
    Import directly from :mod:`qrisp.interface.simulators.default_backend`
    in new code.
"""

from qrisp.interface.simulators.default_backend import (
    DefaultBackend,
    DefaultJob,
    def_backend,
)
