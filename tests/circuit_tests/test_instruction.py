# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Regression tests for the Instruction class."""

from qrisp.circuit import Clbit, Instruction, Qubit, XGate


def test_instruction_default_arguments_are_not_shared():
    # Regression test: Instruction used mutable default arguments, so
    # modifying the qubit/clbit list of one instruction leaked into every
    # other instruction created without explicit qubits/clbits.
    first = Instruction(XGate())
    first.qubits.append(Qubit("q0"))
    first.clbits.append(Clbit("c0"))

    second = Instruction(XGate())
    assert second.qubits == []
    assert second.clbits == []
    assert first.qubits is not second.qubits
    assert first.clbits is not second.clbits
