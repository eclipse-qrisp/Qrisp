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

import pytest

from qrisp.interface.backend import Backend
from qrisp.interface.qunicorn.backend_client import BackendClient
from qrisp.interface.qunicorn.backend_server import BackendServer
from qrisp.interface.virtual_backend import VirtualBackend
from qrisp.misc.exceptions import QrispDeprecationWarning

###############################################
### Tests for the new Backend infrastructure
###############################################


class DummyBackend(Backend):
    """Minimal subclass of Backend for testing."""

    @classmethod
    def _default_options(cls):
        return {"shots": 1000, "flag": False}

    def run(self, circuit, **kwargs):
        return {"circuit": circuit, "options": {**self._options, **kwargs}}


class BackendNoOptionsNoDefaultsOptions(Backend):
    """
    Dummy Backend for testing the following scenario:

    - The backend is instantiated without passing any options.

    - Therefore, it must use the default options provided by Backend._default_options().

    - The update_options method should only allow modifying existing keys.
    """

    def run(self, circuit, **kwargs):
        pass


class BackendWithExplicitOptions(Backend):
    """
    Dummy Backend for testing the following scenario:

    - The backend is instantiated with explicit options.

    - Therefore, these options must override the default options provided by
      Backend._default_options() entirely.

    """

    def __init__(self, options=None):
        super().__init__(options=options)

    def run(self, circuit, **kwargs):
        pass


class BackendWithChildDefaultOptions(Backend):
    """
    Dummy Backend for testing the following scenario:

    - The backend is instantiated without passing any options, but the child class
      defines its own _default_options().

    - Therefore, it must use the child class's default options.
    """

    @classmethod
    def _default_options(cls):
        return {"shots": 1024, "custom_default": 42}

    def run(self, circuit, **kwargs):
        pass


class TestDummyBackend:
    """Test suite for DummyBackend implementation."""

    def test_backend_is_abstract(self):
        """Ensure Backend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_dummy_backend_instantiation_with_name(self):
        """Ensure a subclass can be instantiated."""
        b = DummyBackend(name="test_backend")
        assert b.name == "test_backend"

    def test_dummy_backend_instantiation_without_name(self):
        """Ensure a subclass can be instantiated without a name."""
        b = DummyBackend()
        assert b.name == "DummyBackend"

    def test_dummy_backend_run_method(self):
        """Test the run method of DummyBackend."""
        b = DummyBackend()
        result = b.run(circuit=None, param=True)
        assert b.options == {"shots": 1000, "flag": False}
        assert result["circuit"] is None
        assert result["options"]["shots"] == 1000
        assert result["options"]["flag"] is False
        assert result["options"]["param"] is True


class TestBackendOptions:
    """Test suite for Backend options handling."""

    def test_no_options_no_defaults_options(self):
        """Test backend instantiated without options uses default options."""

        backend = BackendNoOptionsNoDefaultsOptions()
        assert backend.options == {"shots": 1024}

        backend.update_options(shots=2048)
        assert backend.options["shots"] == 2048

        with pytest.raises(AttributeError):
            backend.update_options(custom_option="new_value")

    def test_backend_with_explicit_options(self):
        """Test backend instantiated with explicit options."""

        options = {"shots": 1024, "custom_option": "old_value"}
        backend = BackendWithExplicitOptions(options=options)

        assert backend.options == {"shots": 1024, "custom_option": "old_value"}
        backend.update_options(custom_option="new_value")
        assert backend.options["custom_option"] == "new_value"

        options["custom_option"] = "modified_externally"
        assert backend.options["custom_option"] == "new_value"

        with pytest.raises(AttributeError):
            backend.update_options(unknown_field=123)

    def test_backend_with_child_default_options(self):
        """Test backend instantiated without options uses child class default options."""

        backend = BackendWithChildDefaultOptions()
        assert backend.options == {"shots": 1024, "custom_default": 42}

        backend.update_options(shots=2048, custom_default="updated")
        assert backend.options["shots"] == 2048
        assert backend.options["custom_default"] == "updated"

        with pytest.raises(AttributeError):
            backend.update_options(new_param=999)


###########################################################
### Deprecation tests for the old Backend infrastructure
###########################################################


def test_backend_client_deprecation_warning():
    """Test that BackendClient raises a deprecation warning upon instantiation."""

    with pytest.warns(QrispDeprecationWarning):
        _ = BackendClient(api_endpoint="not_used")


def test_backend_server_deprecation_warning():
    """Test that BackendServer raises a deprecation warning upon instantiation."""

    def dummy_run():
        pass

    with pytest.warns(QrispDeprecationWarning):
        _ = BackendServer(run_func=dummy_run)


def test_virtual_backend_deprecation_warning():
    """Test that VirtualBackend raises a deprecation warning upon instantiation."""

    def dummy_run():
        pass

    with pytest.warns(QrispDeprecationWarning):
        _ = VirtualBackend(run_func=dummy_run)
