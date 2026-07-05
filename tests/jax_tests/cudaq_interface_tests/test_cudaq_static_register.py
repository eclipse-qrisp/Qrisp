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

import pytest

import cudaq

from qrisp import QuantumFloat, QuantumVariable, measure, x
from qrisp.jasp import jaspr_to_static_register_jaspr, jrange, make_jaspr
from qrisp.jasp.cudaq_interface import cudaq_kernel, cudaq_kernel_from_xdsl_module
from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake_mlir


def test_cudaq_static_register():
    """Test static register lowering of a simple quantum program."""

    def main():
        a = QuantumVariable(10)
        x(a[0])
        x(a[0])
        a.delete()

        b = QuantumVariable(10)
        x(b[0])
        x(b[0])
        b.delete()

        c = QuantumVariable(10)
        x(c)

        return measure(c)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)
    xdsl_module = jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 1023
    assert static_reg_jaspr() == 1023

    kernel = cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 1023


def test_cudaq_static_register_quantum_float_addition():
    """Test static register lowering of a simple QuantumFloat addition program with dynamic loop."""

    def main():
        a = QuantumFloat(8)

        for _ in jrange(10):
            a += 10

        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)
    xdsl_module = jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 100
    assert static_reg_jaspr() == 100

    kernel = cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 100
