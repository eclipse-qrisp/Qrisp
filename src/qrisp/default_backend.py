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

"""
Default backend configuration.

:data:`def_backend` is the module-level singleton used as the default backend
throughout Qrisp when no explicit backend is provided. It is an instance of
:class:`~qrisp.interface.simulators.qrisp_simulator_backend.QrispSimulatorBackend`.

To change the global default, replace *def_backend* with a different backend
instance, e.g.::

    import qrisp.default_backend as db
    from my_custom_backend import MyBackend
    db.def_backend = MyBackend()

The implementing classes live in
:mod:`qrisp.interface.simulators.qrisp_simulator_backend` and are re-exported here
for convenience.
"""

from qrisp.interface.simulators.qrisp_simulator_backend import (
    QrispSimulatorBackend,
)

def_backend = QrispSimulatorBackend()
