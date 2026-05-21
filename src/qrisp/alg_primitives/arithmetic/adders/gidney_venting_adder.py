from __future__ import annotations
import qrisp
from qrisp.jasp import check_for_tracing_mode


def _require_dynamic_mode(func_name: str) -> None:
    if not check_for_tracing_mode():
        raise RuntimeError(
            f"{func_name} requires JASP dynamic mode because downstream "
            "streaming-adder blocks rely on mid-circuit measurement and classical feedforward."
        )
 
 
def bit_inverted_mcx(
    parity_check_ctrl: qrisp.Qubit | qrisp.QuantumVariable,
    simple_ctrl: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
    b: bool,
) -> None:
    """Figure 1 left: Toffoli with two controls, one inverted by classical bit b.

    parity_check_ctrl has the Z⊕b box and is implemented as an explicit X^b sandwich:
    apply X to parity_check_ctrl iff b is True, do a standard MCX, then undo X.
    simple_ctrl is a normal control (fires on |1⟩).
    """
    _require_dynamic_mode("bit_inverted_mcx")
    if b:
        qrisp.x(parity_check_ctrl)

    qrisp.mcx([parity_check_ctrl, simple_ctrl], target)

    if b:
        qrisp.x(parity_check_ctrl)


 
 
def zz_mcx(
    z0_left: qrisp.Qubit | qrisp.QuantumVariable,
    z1_left: qrisp.Qubit | qrisp.QuantumVariable,
    control: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
) -> None:
    """Figure 1 middle: Toffoli with one normal control and one ZZ parity control.

    z0_left, z1_left are the two Z-box qubits whose parity acts as one control.
    control is a normal control (fires on |1⟩).
    """
    _require_dynamic_mode("zz_mcx")
    qrisp.cx(z0_left, z1_left)
    qrisp.mcx([z1_left, control], target)
    qrisp.cx(z0_left, z1_left)

 
 
def zz_zz_mcx(
    z_left: qrisp.Qubit | qrisp.QuantumVariable,
    z_left_right: qrisp.Qubit | qrisp.QuantumVariable,
    z_right: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
) -> None:
    """Figure 1 right: Toffoli controlled on AND of two ZZ parity checks.

    Left pair:  (z_left, z_left_right) — Z box on left side.
    Right pair: (z_left_right, z_right) — Z box on right side.
    z_left_right is shared between both pairs.
    """
    _require_dynamic_mode("zz_zz_mcx")
    qrisp.cx(z_left_right, z_left)    # z_left now holds z_left ⊕ z_left_right (left parity)
    qrisp.cx(z_left_right, z_right)    # z_right now holds z_left_right ⊕ z_right (right parity)
    qrisp.mcx([z_left, z_right], target)
    qrisp.cx(z_left_right, z_right)    # restore z_right
    qrisp.cx(z_left_right, z_left)    # restore z_left

