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

from jax import make_jaxpr
from qrisp.jax import jaxpr_to_qc


def test_qfunction(func):
    
    def testing_function(*args, **kwargs):
        
        qv = func(*args, **kwargs)
        from qrisp.core import QuantumVariable
        qv.__class__ = QuantumVariable
        
        old_counts_dic = qv.get_measurement()
        
        jaxpr = make_jaxpr(func)(*args, **kwargs)
        
        qc, qv_qubits = jaxpr_to_qc(jaxpr)(*args, **kwargs)
        
        clbit_list = []
        for qb in qv_qubits:
            clbit_list.append(qc.measure(qb))
            
        
        
        counts = qc.run(shots = 100000)
        
        # Remove other measurements outcomes from counts dic
        new_counts_dic = {}
        for key in counts.keys():
            # Remove possible whitespaces
            new_key = key.replace(" ", "")
            # Remove other measurements
            new_key = new_key[:len(clbit_list)][::-1]

            # new_key = int(new_key, base=2)
            try:
                new_counts_dic[new_key] += counts[key]/100000
            except KeyError:
                new_counts_dic[new_key] = counts[key]/100000
        
        dynamic_counts = new_counts_dic
        
        
        for k in old_counts_dic.keys():
            if abs(old_counts_dic[k] - new_counts_dic[k]) > 1E-4:
                return False
        return True
        
    return testing_function
        
        
        
        
            
            
        
        
        
