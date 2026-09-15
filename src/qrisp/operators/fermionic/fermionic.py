# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
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

"""Defines the a() and c() factory functions for fermionic annihilation/creation operators."""

import warnings
from qrisp.operators.fermionic.fermionic_operator import FermionicOperator
from qrisp.operators.fermionic.fermionic_term import FermionicTerm


def a_f(arg: int):
    if isinstance(arg, int):
        return FermionicOperator({FermionicTerm([(arg, False)]): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))


def c_f(arg: int):
    if isinstance(arg, int):
        return FermionicOperator({FermionicTerm([(arg, True)]): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))


def a(arg: int):
    warnings.warn(
        "Using 'a' for the fermionic annihilation operator is deprecated; use 'a_f' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return a_f(arg)


def c(arg: int):
    warnings.warn(
        "Using 'c' for the fermionic annihilation operator is deprecated; use 'c_f' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return c_f(arg)
