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

"""Shared helpers for all the Qrisp adder implementations."""

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable
from qrisp.jasp import DynamicQubitArray


def _is_quantum_register(obj):
    """Return True if ``obj`` is a quantum register.

    A quantum register is a QuantumVariable (or subclass thereof), a
    DynamicQubitArray or a list of Qubits.
    """
    if isinstance(obj, (QuantumVariable, DynamicQubitArray)):
        return True
    if isinstance(obj, list):
        return all(isinstance(qb, Qubit) for qb in obj)
    return False
