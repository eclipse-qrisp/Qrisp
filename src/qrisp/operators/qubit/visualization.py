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

#
# ONLY USED FOR LATEX PRINTING
#

from sympy import Symbol

#
# Pauli symbols (only used for visualization, i.e., LateX printing with SymPy)
#  

class X_(Symbol):

    __slots__ = ("axis","index")

    def __new__(cls, index):
        obj = Symbol.__new__(cls, "%s(%s)" %("X",index), commutative=False, hermitian=True)
        obj.index = index
        return obj

class Y_(Symbol):

    __slots__ = ("axis","index")

    def __new__(cls, index):
        obj = Symbol.__new__(cls, "%s(%s)" %("Y",index), commutative=False, hermitian=True)
        obj.index = index
        return obj
       
class Z_(Symbol):

    __slots__ = ("axis","index")

    def __new__(cls, index):
        obj = Symbol.__new__(cls, "%s(%s)" %("Z",index), commutative=False, hermitian=True)
        obj.index = index
        return obj
    







