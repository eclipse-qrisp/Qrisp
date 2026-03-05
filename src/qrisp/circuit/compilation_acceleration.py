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

import numpy as np
import threading

from qrisp.circuit.quantum_circuit import QuantumCircuit


class CompilationAccelerator:

    def __init__(self, xla_mode=2):
        self.xla_mode = xla_mode

    def __enter__(self):

        self.original_xla_mode = QuantumCircuit.xla_mode

        if threading.current_thread() is threading.main_thread():
            QuantumCircuit.xla_mode = self.xla_mode

    def __exit__(self, exception_type, exception_value, traceback):
        if threading.current_thread() is threading.main_thread():
            QuantumCircuit.xla_mode = self.original_xla_mode


fast_append = CompilationAccelerator
