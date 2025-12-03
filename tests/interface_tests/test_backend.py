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


class DummyBackend(Backend):
    """Minimal subclass of Backend for testing."""

    @classmethod
    def _default_options(cls):
        return {"shots": 1000, "flag": False}

    @property
    def max_circuits(self):
        return 1

    def run(self, run_input, **kwargs):
        # just return what was sent, to test passthrough
        return {"input": run_input, "options": {**self._options, **kwargs}}


class TestBackend:
    """Test suite for Backend abstract class and DummyBackend implementation."""

    def test_backend_is_abstract(self):
        """Ensure Backend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_dummy_backend_instantiation(self):
        """Ensure a subclass can be instantiated."""
        b = DummyBackend(name="test_backend")
        assert b.name == "test_backend"
        assert b.options == {"shots": 1000, "flag": False}

    def test_override_default_options_at_init(self):
        """Ensure constructor options override defaults."""
        b = DummyBackend(options={"shots": 500, "flag": True})
        assert b.options["shots"] == 500
        assert b.options["flag"] is True

    def test_set_options_updates_existing_values(self):
        """Ensure set_options modifies only allowed keys."""
        b = DummyBackend()
        b.set_options(shots=2000, flag=True)
        assert b.options["shots"] == 2000
        assert b.options["flag"] is True

    def test_set_options_rejects_unknown_keys(self):
        """Ensure unknown options raise an error."""
        b = DummyBackend()
        with pytest.raises(AttributeError):
            b.set_options(unknown_option=123)

    def test_dummy_backend_run_behavior(self):
        """Ensure run() behaves as expected in subclass."""
        b = DummyBackend()
        result = b.run("test_qasm", shots=250)
        assert result["input"] == "test_qasm"
        assert result["options"]["shots"] == 250
        assert result["options"]["flag"] is False

    def test_dummy_backend_default_max_circuits(self):
        """Ensure subclass exposes max_circuits property."""
        b = DummyBackend()
        assert b.max_circuits == 1
