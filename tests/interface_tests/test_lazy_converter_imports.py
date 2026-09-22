"""
Tests for lazy loading of optional circuit converters.

********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.eclipse.org/legal/epl-2.0/.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import subprocess
import sys

CONVERTER_MODULES = (
    "qrisp.interface.converter.qiskit_converter",
    "qrisp.interface.converter.pytket_converter",
    "qrisp.interface.converter.pennylane_converter",
    "qrisp.interface.converter.qulacs_converter",
    "qrisp.interface.converter.stim_converter",
    "qrisp.interface.converter.cirq_converter",
)


def _run_isolated(code):
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_qrisp_import_does_not_load_converters():
    """Importing Qrisp does not import any converter implementation."""
    modules = repr(CONVERTER_MODULES)
    _run_isolated(
        f"""
import sys
from qrisp import *

for module in {modules}:
    assert module not in sys.modules, module
"""
    )


def test_interface_reexports_load_converters_on_demand():
    """Public converter re-exports load only when their symbols are requested."""
    converters = {
        "convert_to_qiskit": "qrisp.interface.converter.qiskit_converter",
        "convert_from_qiskit": "qrisp.interface.converter.qiskit_converter",
        "pytket_converter": "qrisp.interface.converter.pytket_converter",
        "qml_converter": "qrisp.interface.converter.pennylane_converter",
        "qulacs_converter": "qrisp.interface.converter.qulacs_converter",
        "qrisp_to_stim": "qrisp.interface.converter.stim_converter",
        "convert_to_cirq": "qrisp.interface.converter.cirq_converter",
        "convert_from_cirq": "qrisp.interface.converter.cirq_converter",
    }
    _run_isolated(
        f"""
import sys
import qrisp

converters = {converters!r}
loaded_modules = set()
for name, module in converters.items():
    if module not in loaded_modules:
        assert module not in sys.modules, module
    namespace = {{}}
    exec(f"from qrisp.interface import {{name}}", namespace)
    assert callable(namespace[name])
    assert module in sys.modules, module
    loaded_modules.add(module)
"""
    )


def test_converter_package_reexports_load_converters_on_demand():
    """The converter package also resolves its public symbols lazily."""
    _run_isolated(
        """
import sys
import qrisp

assert "qrisp.interface.converter.pennylane_converter" not in sys.modules
from qrisp.interface.converter import qml_converter
assert callable(qml_converter)
assert "qrisp.interface.converter.pennylane_converter" in sys.modules
"""
    )


def test_interface_star_import_exposes_converter_names():
    """Star imports preserve the public converter API."""
    _run_isolated(
        """
from qrisp.interface import *

for name in (
    "convert_to_qiskit",
    "convert_from_qiskit",
    "pytket_converter",
    "qml_converter",
    "qulacs_converter",
    "qrisp_to_stim",
    "convert_to_cirq",
    "convert_from_cirq",
):
    assert callable(globals()[name])
"""
    )


def test_missing_optional_converter_dependencies_raise_import_error():
    """Converters report missing optional dependencies when they are used."""
    _run_isolated(
        """
import importlib.abc
import sys

import qrisp


class BlockedImport(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.package or fullname.startswith(self.package + "."):
            raise ModuleNotFoundError(f"blocked test dependency: {self.package}")
        return None


def clear_modules(prefix):
    for module in list(sys.modules):
        if module == prefix or module.startswith(prefix + "."):
            del sys.modules[module]


missing_dependencies = (
    ("pennylane", "qml_converter"),
    ("pytket", "pytket_converter"),
    ("qulacs", "qulacs_converter"),
    ("stim", "qrisp_to_stim"),
    ("cirq", "convert_to_cirq"),
)

for package, converter_name in missing_dependencies:
    clear_modules(package)
    blocker = BlockedImport(package)
    sys.meta_path.insert(0, blocker)
    try:
        namespace = {}
        exec(f"from qrisp.interface import {converter_name}", namespace)
        try:
            namespace[converter_name](None)
        except ImportError as exc:
            if converter_name == "qml_converter":
                assert "requires pennylane" in str(exc)
        else:
            raise AssertionError(f"{converter_name} did not report a missing dependency")
    finally:
        sys.meta_path.remove(blocker)
        clear_modules(package)
"""
    )
