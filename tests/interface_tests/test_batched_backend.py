"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import pytest
import time

from qrisp import *

def test_batched_backend():
    
    def run_func_batch(batch):
        """
        Parameters
        ----------
        batch : list[tuple[QuantumCircuit, int]]
            The circuit and shot batch indicating the backend queries.
    
        Returns
        -------
        results : list[dict[string, int]]
            The list of results.
    
        """
        
        results = []
        for i in range(len(batch)):
            qc = batch[i][0]
            shots = batch[i][1]
            results.append(qc.run(shots = shots))
            
        return results
    
    from qrisp.interface import BatchedBackend
    # Create some backend calls        
    a = QuantumFloat(4)
    b = QuantumFloat(3)
    a[:] = 1
    b[:] = 2
    c = a + b
    
    d = QuantumFloat(4)
    e = QuantumFloat(3)
    d[:] = 2
    e[:] = 2
    f = d + e
    
    
    # Set up batched backend
    bb = BatchedBackend(run_func_batch)
    
    # Create threads
    import threading
    
    results = []
    def eval_measurement(qv):
        results.append(qv.get_measurement(backend = bb))
    
    thread_0 = threading.Thread(target = eval_measurement, args = (c,))
    thread_1 = threading.Thread(target = eval_measurement, args = (f,))
    
    # Start the threads
    thread_0.start()
    thread_1.start()
    
    # Call the dispatch routine
    # The min_calls keyword will make it wait 
    # until the batch has a size of 2
    bb.dispatch(min_calls = 2)
    
    # Wait for the threads to join
    thread_0.join()
    thread_1.join()
    
    # Inspect the results
    assert {3 : 1.0} in results
    assert {4 : 1.0} in results
    
    
    
    
    
    
            
