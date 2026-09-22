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

from importlib import import_module


_CONVERTER_EXPORTS = {
    "convert_to_qiskit": "qiskit_converter",
    "create_qiskit_instruction": "qiskit_converter",
    "convert_from_qiskit": "qiskit_converter",
    "create_tket_instruction": "pytket_converter",
    "pytket_converter": "pytket_converter",
    "qml_converter": "pennylane_converter",
    "qulacs_converter": "qulacs_converter",
    "qrisp_to_stim": "stim_converter",
    "convert_to_cirq": "cirq_converter",
    "convert_from_cirq": "cirq_converter",
}

__all__ = list(_CONVERTER_EXPORTS)


def __getattr__(name):
    """Load a converter only when one of its public symbols is requested."""
    module_name = _CONVERTER_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
