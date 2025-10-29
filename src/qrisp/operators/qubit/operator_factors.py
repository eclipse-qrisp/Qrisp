"""
********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp.circuit.qubit import Qubit
from qrisp.operators.qubit.bound_qubit_operator import BoundQubitOperator
from qrisp.operators.qubit.bound_qubit_term import BoundQubitTerm
from qrisp.operators.qubit.qubit_operator import QubitOperator
from qrisp.operators.qubit.qubit_term import QubitTerm


def X(arg):
    if isinstance(arg, int):
        return QubitOperator({QubitTerm({arg: "X"}): 1})
    elif isinstance(arg, Qubit):
        return BoundQubitOperator({BoundQubitTerm({arg: "X"}): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))


def Y(arg):
    if isinstance(arg, int):
        return QubitOperator({QubitTerm({arg: "Y"}): 1})
    elif isinstance(arg, Qubit):
        return BoundQubitOperator({BoundQubitTerm({arg: "Y"}): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))


def Z(arg):
    if isinstance(arg, int):
        return QubitOperator({QubitTerm({arg: "Z"}): 1})
    elif isinstance(arg, Qubit):
        return BoundQubitOperator({BoundQubitTerm({arg: "Z"}): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))


def A(arg):
    return QubitOperator({QubitTerm({arg: "A"}): 1})


def C(arg):
    return QubitOperator({QubitTerm({arg: "C"}): 1})


def P0(arg):
    return QubitOperator({QubitTerm({arg: "P0"}): 1})


def P1(arg):
    return QubitOperator({QubitTerm({arg: "P1"}): 1})
