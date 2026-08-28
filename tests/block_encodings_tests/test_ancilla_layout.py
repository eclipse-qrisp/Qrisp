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
