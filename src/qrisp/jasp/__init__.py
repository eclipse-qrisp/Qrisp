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

from qrisp.jasp.primitives import*
from qrisp.jasp.tracing_quantum_session import *
from qrisp.jasp.qaching import qache
from qrisp.jasp.interpreter_tools import *
from qrisp.jasp.jasp_expression import *
from qrisp.jasp.testing_utils import *
from qrisp.jasp.control_flow import *

def compare_jaxpr(jaxpr, primitive_name_list):
    assert len(jaxpr.eqns) == len(primitive_name_list)
    for i in range(len(primitive_name_list)):
        assert jaxpr.eqns[i].primitive.name == primitive_name_list[i]
    


