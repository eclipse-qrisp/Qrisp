"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from qrisp.interface.backend import Backend
from qrisp.simulator.simulator import run as default_run


class DefaultBackend(Backend):
    """A default backend that uses the built-in simulator."""

    def run(self, qc, shots=None, token=""):
        return default_run(qc, shots, token)

    @classmethod
    def _default_options(cls):
        return {}

    @property
    def max_circuits(self):
        return None


def_backend = DefaultBackend()
