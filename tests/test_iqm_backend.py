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

# Created by manzanillo
import pytest
from unittest.mock import patch
from qrisp.interface import IQMBackend, VirtualBackend

import pytest
from unittest.mock import patch, MagicMock
from qrisp.interface import IQMBackend

# Mock the IQMProvider and transpile_to_IQM functions from qiskit_iqm
def test_IQMBackend():
    # TO DO: set up proper mock backend
    return
    # Mock the server_url and token
    api_token = "mock_api_token"

    # Create a mock device instance
    device_instance = "garnet:fake"

    from iqm.iqm_client import CircuitCompilationOptions
    compilation_options = CircuitCompilationOptions()
    # Create an instance of IQMBackend
    backend = IQMBackend(api_token, device_instance, compilation_options = compilation_options)

    # Check that the backend is an instance of VirtualBackend
    assert isinstance(backend, VirtualBackend)

    # Create a mock QASM string
    from qrisp import QuantumCircuit
    qc = QuantumCircuit(4, name = "test")

    # Call the run method
    result = backend.run(qc, shots=1000, token=api_token)


# Test that the API token is a string
def test_IQMBackend_api_token_string_is_missing():
    with pytest.raises(TypeError):
        IQMBackend(123, "garnet")

# Test that the device instance is a string
def test_IQMBackend_device_instance_string_is_missing():
    with pytest.raises(TypeError):
        IQMBackend("mock_api_token", 123)