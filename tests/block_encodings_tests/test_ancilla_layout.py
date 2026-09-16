"""********************************************************************************
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

from qrisp import QuantumBool, QuantumFloat, QuantumVariable, jaspify, measure, x
from qrisp.block_encodings.ancilla_layout import _AncillaLayout, _maximum_layout_size


def test_ancilla_layout_constructs_typed_views():
    """Test that _AncillaLayout constructs typed views correctly."""
    layout = _AncillaLayout.from_templates([QuantumFloat(2).template(), QuantumBool().template()])
    shared_ancilla = QuantumVariable(layout.total_size)

    views = layout.construct_views(shared_ancilla)

    assert [view.size for view in views] == [2, 1]
    assert all(view_qubit is shared_qubit for view_qubit, shared_qubit in zip(views[0].reg, shared_ancilla.reg[:2]))
    assert views[1].reg[0] is shared_ancilla.reg[2]


def test_ancilla_layout_uses_maximum_total_size():
    """Test that _maximum_layout_size returns the correct maximum size."""
    layouts = [
        _AncillaLayout.from_templates([QuantumFloat(2).template(), QuantumBool().template()]),
        _AncillaLayout.from_templates([QuantumFloat(1).template()]),
    ]

    assert _maximum_layout_size(layouts) == 3


def test_ancilla_layout_constructs_jasp_views():
    """Test that _AncillaLayout constructs typed views correctly in Jasp execution."""

    @jaspify
    def main():
        layout = _AncillaLayout.from_templates([QuantumFloat(2).template(), QuantumBool().template()])
        shared_ancilla = QuantumVariable(_maximum_layout_size([layout]))
        views = layout.construct_views(shared_ancilla)
        x(views[0])
        x(views[1])
        return measure(shared_ancilla)

    assert main() == 7
