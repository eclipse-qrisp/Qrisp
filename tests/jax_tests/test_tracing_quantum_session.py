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

from qrisp.jasp.tracing_logic.tracing_quantum_session import TracingQuantumSession
from qrisp.jasp.primitives.abstract_quantum_state import AbstractQuantumState


def test_tracing_quantum_session_is_singleton():
    """Test that TracingQuantumSession is a singleton."""
    session1 = TracingQuantumSession()
    session2 = TracingQuantumSession()
    assert session1 is session2


def test_get_instance():

    instance = TracingQuantumSession.get_instance()

    assert instance.abs_qst is None
    assert instance.qubit_cache == {}
    assert instance.qv_list == []
    assert instance.deleted_qv_list == []

    abs_qst = AbstractQuantumState()

    instance.start_tracing(abs_qst)

    assert instance.abs_qst is abs_qst
    assert instance.qubit_cache == {}
    assert instance.qv_list == []
    assert instance.deleted_qv_list == []
