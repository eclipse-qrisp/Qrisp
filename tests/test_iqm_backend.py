# Created by manzanillo
import pytest
from unittest.mock import patch
from qrisp.interface import IQMBackend, VirtualBackend

import pytest
from unittest.mock import patch, MagicMock
from qrisp.interface import IQMBackend

# Mock the IQMProvider and transpile_to_IQM functions from qiskit_iqm
@patch("iqm.qiskit_iqm.IQMProvider")
@patch("iqm.qiskit_iqm.transpile_to_IQM")
def test_IQMBackend(mock_transpile, mock_iqmprovider):
    # Mock the server_url and token
    server_url = "https://cocos.resonance.meetiqm.com/garnet"
    api_token = "mock_api_token"

    # Create a mock device instance
    device_instance = "garnet"

    # Create an instance of IQMBackend
    backend = IQMBackend(api_token, device_instance)

    # Check that the backend is an instance of VirtualBackend
    assert isinstance(backend, VirtualBackend)

    # Create a mock QASM string
    from qrisp import QuantumCircuit
    qc = QuantumCircuit(4, name = "test")

    # Mock the result of the run method
    mock_result = {"mock_key": 1}

    # Use a side effect to return the mock result
    mock_iqmprovider.return_value.get_backend.return_value.run.return_value.result.return_value.get_counts.side_effect = [mock_result]

    # Call the run method
    result = backend.run(qc, shots=1000, token=api_token)

    # Check that the result is as expected
    assert result == mock_result

    # Check that the IQMProvider and transpile_to_IQM functions are called correctly
    mock_iqmprovider.assert_called_once_with(server_url, token=api_token)
    mock_transpile.assert_called_once()

# Test that the API token is a string
def test_IQMBackend_api_token_string_is_missing():
    with pytest.raises(TypeError):
        IQMBackend(123, "garnet")

# Test that the device instance is a string
def test_IQMBackend_device_instance_string_is_missing():
    with pytest.raises(TypeError):
        IQMBackend("mock_api_token", 123)