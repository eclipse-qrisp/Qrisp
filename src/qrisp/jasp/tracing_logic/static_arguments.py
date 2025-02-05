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

import types

from jax import tree_util

# This class is used to wrap arguments that are supposed to be static and therefore
# prevents Jax from turning them dynamic. This is helpful because some types (such as functions)
# will never be dynamic and can therefore automatically be denoted as static.
class StaticArg:
    
    # The idea is to wrap the value in a class that has a custom (un)flattening procedure
    # to hide the object from jax
    def __init__(self, val):
        self.val = val
    
    def __hash__(self):
        return hash(self.val)
    
    def __eq__(self, other):
        return self.val == other



def unflatten_qv(aux_data, children):
    # return the tracers and auxiliary data (structure of the object)
    return aux_data.val

import types
from jax import tree_util
def unflatten_function(aux_data, children):
    return aux_data
def flatten_function(arg):
    # return the tracers and auxiliary data (structure of the object)
    return tuple(), arg

tree_util.register_pytree_node(types.FunctionType, flatten_function, unflatten_function)


# tree_util.register_pytree_node(StaticArg, flatten_static_arg, unflatten_qv)
# Turns a list or tuple of arguments into static values and therefore prevents
# jax from "tracerinzing"
def auto_static(args):
    return args
    res = []
    for i in range(len(args)):
        if isinstance(args[i], types.FunctionType):
            res.append(StaticArg(args[i]))
        else:
            print("----")
            print(type(args[i]))
            res.append(args[i])
    return tuple(res)
    
    