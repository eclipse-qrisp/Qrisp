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

# TODO: I am still adding tests here, many more to come (they will also be nicely organized later)

import jax
from jax import numpy as jnp

from qrisp import *
from qrisp.jasp import *


def test_depth_one_qubit_1():

    @depth(meas_behavior="0")
    def main():
        qf = QuantumFloat(1)
        h(qf[0])
        return measure(qf[0])

    assert main() == 1


def test_depth_one_qubit_1_dyn():

    @depth(meas_behavior="0")
    def main(num_qubits):
        qf = QuantumFloat(num_qubits)
        h(qf[0])
        return measure(qf[0])

    assert main(1) == 1


def test_depth_one_qubit_2():

    @depth(meas_behavior="0")
    def main():
        qf = QuantumFloat(1)
        h(qf[0])
        h(qf[0])
        return measure(qf[0])

    assert main() == 2


def test_depth_one_qubit_2_dyn():

    @depth(meas_behavior="0")
    def main(num_qubits):
        qf = QuantumFloat(num_qubits)
        h(qf[0])
        h(qf[0])
        return measure(qf[0])

    assert main(1) == 2


def test_depth_one_qubit_3():

    @depth(meas_behavior="0")
    def main():
        qf = QuantumFloat(2)
        h(qf[0])
        h(qf[1])
        return measure(qf[0])

    assert main() == 1


def test_depth_one_qubit_3_dyn():

    @depth(meas_behavior="0")
    def main(num_qubits):
        qf = QuantumFloat(num_qubits)
        h(qf[0])
        h(qf[1])
        return measure(qf[0])

    assert main(2) == 1


def test_depth_one_qubit_4():

    @depth(meas_behavior="0")
    def main():
        qf = QuantumFloat(2)
        h(qf[1])
        h(qf[1])
        return measure(qf[1])

    assert main() == 2


def test_depth_one_qubit_4_dyn():

    @depth(meas_behavior="0")
    def main(num_qubits):
        qf = QuantumFloat(num_qubits)
        h(qf[1])
        h(qf[1])
        return measure(qf[1])

    assert main(2) == 2
