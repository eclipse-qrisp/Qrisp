"""
\********************************************************************************
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
********************************************************************************/
"""

import pytest
import time

from qrisp import *

def test_amy_mcx():
    compilation_kwargs = {"compile_mcm" : True}
    for i in range(4):
        
        ctrl = QuantumFloat(2)
        target = QuantumBool()
        
        ctrl[:] = i
        mcx(ctrl, target, method = "amy")
        with invert():
            mcx(ctrl, target, method = "amy")
        
        assert ctrl.get_measurement(compilation_kwargs = compilation_kwargs) == {i : 1}
        
        
        ctrl = QuantumFloat(2)
        target = QuantumBool()
        ctrl[:] = i
        
        h(ctrl)
        mcx(ctrl, target, method = "amy")
        with invert():
            mcx(ctrl, target, method = "amy")
        h(ctrl)
        
        assert ctrl.get_measurement(compilation_kwargs = compilation_kwargs) == {i : 1}
        
        ctrl = QuantumFloat(2)
        target = QuantumBool()
        
        ctrl[:] = i
        mcx(ctrl, target, method = "amy")
        target.uncompute()
        
        assert ctrl.get_measurement(compilation_kwargs = compilation_kwargs) == {i : 1}
        
        
        ctrl = QuantumFloat(2)
        target = QuantumBool()
        ctrl[:] = i
        
        h(ctrl)
        mcx(ctrl, target, method = "amy")
        target.uncompute()
        h(ctrl)
        
        assert ctrl.get_measurement(compilation_kwargs = compilation_kwargs) == {i : 1}
        
    print(target.qs.compile(**compilation_kwargs).transpile())
        
