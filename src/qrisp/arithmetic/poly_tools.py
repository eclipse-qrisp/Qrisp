"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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


import sympy as sp


# Transforms a given symbol into a qb object
# For this we require that there is some quantum variable
# in qv_list which has the same name as the symbol
# ie. if the symbol has the name g3
# qv_list needs to contain a quantum variable with name g
# then qubit with index 3 is returned
def symbol_to_qbit(symbol, qv_list):
    if isinstance(symbol, sp.core.power.Pow):
        symbol = symbol.args[0]

    var_name = symbol.name.split("%")

    for qv in qv_list:
        if var_name[0] == qv.name:
            return qv.reg[int(var_name[1])]

    raise Exception("Could not find matching qubit for symbol " + symbol.name)


def monomial_to_list(monom):
    if not isinstance(
        monom,
        (
            sp.core.mul.Mul,
            sp.core.power.Pow,
            sp.core.numbers.Number,
            sp.core.symbol.Symbol,
        ),
    ):
        raise Exception("Tried to convert invalid monomial to list")

    from sympy import preorder_traversal

    result_list = []
    while True:
        for x in preorder_traversal(monom):
            if isinstance(x, sp.core.symbol.Symbol):
                monom = monom / x
                result_list.append(x)
                break
        else:
            break

    return [monom] + result_list


# Retrieves a list of the symbols appearing in the
# given sympy expression in alphabetical order
def get_ordered_symbol_list(expr):
    symbol_list = list(expr.free_symbols)
    symbol_list.sort(key=lambda x: x.name)
    return symbol_list


# Filters powers out of polynomial
def filter_pow(expr):
    pow_dic = {}
    for sub_expr in sp.preorder_traversal(expr):
        if isinstance(sub_expr, sp.core.power.Pow):
            pow_dic.update({sub_expr: get_ordered_symbol_list(sub_expr)[0]})
    return expr.subs(pow_dic)


# Turns a polynomial expression into a list of lists where
# example: x0*x1*x3 + x2*x4 --> [[x0,x1,x3],[x2,x4]]
def expr_to_list(expr):
    if not check_for_polynomial(expr):
        raise Exception("Tried to turn a non-polynomial expression into a list")

    result_list = []

    expr = expr.expand()
    if isinstance(expr, sp.core.add.Add):
        # Go through each summand
        for e1 in expr.args:
            result_list.append(monomial_to_list(e1))
    else:
        result_list.append(monomial_to_list(expr))

    return result_list


# Check if the given expression is a polynomial
def check_for_polynomial(expr):
    for x in sp.preorder_traversal(expr):
        if not isinstance(
            x,
            (
                sp.core.mul.Mul,
                sp.core.add.Add,
                sp.core.symbol.Symbol,
                sp.core.numbers.Number,
                sp.core.power.Pow,
            ),
        ):
            return False
    return True
