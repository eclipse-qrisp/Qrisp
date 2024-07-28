"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp.operators import PauliOperator

#
# Pauli symbols
#  

class X(PauliOperator):

    def __init__(self, index):
        super().__init__({((index,'X'),):1})

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliOperator({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

class Y(PauliOperator):

    def __init__(self, index):
        super().__init__({((index,'Y'),):1})

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliOperator({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

class Z(PauliOperator):

    def __init__(self, index):
        super().__init__({((index,'Z'),):1})

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliOperator({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))


