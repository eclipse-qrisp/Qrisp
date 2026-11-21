# ********************************************************************************
# * Copyright (c) 2023-2026 Contributors to the Eclipse Foundation
# *
# * See the NOTICE file(s) distributed with this work for additional
# * information regarding copyright ownership.
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

from qrisp.operators.fermionic.fermionic_operator import FermionicOperator
from qrisp.operators.fermionic.fermionic_term import FermionicTerm


def a(arg):
    if isinstance(arg, int):
        return FermionicOperator({FermionicTerm([(arg, False)]): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))


def c(arg):
    if isinstance(arg, int):
        return FermionicOperator({FermionicTerm([(arg, True)]): 1})
    else:
        raise Exception("Cannot initialize operator from type " + str(type(arg)))
