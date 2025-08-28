"""
********************************************************************************
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
********************************************************************************
"""

from qrisp import *

def test_quantum_array():

    qf = QuantumFloat(5)
    qa = QuantumArray(shape = 5, qtype = qf)
    for i in range(len(qa)):
        qa[i][:] = i

    assert qa.get_measurement() == {OutcomeArray([0, 1, 2, 3, 4]): 1.0}
    
    qf = QuantumFloat(5)
    qa = QuantumArray(shape = (2,5), qtype = qf)
    qa[0,:] = np.arange(5)
    
    assert qa.most_likely()[0,:] == OutcomeArray([0, 1, 2, 3, 4])
    
    # test dynamic indexing
    
    @jaspify
    def main(k):
        
        qtype = QuantumFloat(8)
        q_array = QuantumArray(qtype, 10)
        q_array[:] = np.arange(10)
        
        for i in jrange(1, k):
            q_array[i] += q_array[i-1]

        return measure(q_array)

    assert np.all(main(8) == np.array([ 0.,  1.,  3.,  6., 10., 15., 21., 28.,  8.,  9.]))
    
    # Test the snippets from the documentation
    
    qtype = QuantumFloat(5, -2)
    q_array = QuantumArray(qtype = qtype, shape = (2, 2, 2))

    from qrisp import h
    qv = q_array[0,0,1]
    h(qv[0])
    
    assert q_array.get_measurement() == {OutcomeArray([[[0., 0.],[0., 0.]],[[0., 0.],[0., 0.]]]): 0.5, OutcomeArray([[[0.  , 0.25],[0.  , 0.  ]],[[0.  , 0.  ],[0.  , 0.  ]]]): 0.5}
    
    q_array = q_array.reshape(2,4)

    q_array_swap = np.swapaxes(q_array, 0, 1)
    
    assert q_array_swap.get_measurement() == {OutcomeArray([[0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.]]): 0.5, OutcomeArray([[0.  , 0.  ],
                  [0.25, 0.  ],
                  [0.  , 0.  ],
                  [0.  , 0.  ]]): 0.5}

    q_array[1:,:] = 2*np.ones((1,4))

    assert q_array.get_measurement() ==     {OutcomeArray([[0., 0., 0., 0.],
                       [2., 2., 2., 2.]]): 0.5,
         OutcomeArray([[0.  , 0.25, 0.  , 0.  ],
                       [2.  , 2.  , 2.  , 2.  ]]): 0.5}

                                                 
    # quantum indexing
    
    q_array = QuantumArray(QuantumFloat(1), shape = (4,4))
    index_0 = QuantumFloat(2)
    index_1 = QuantumFloat(2)


    index_0[:] = 2
    index_1[:] = 1

    h(index_0[0])

    with q_array[index_0, index_1] as entry:
        x(entry)

    assert multi_measurement([index_0, index_1, q_array]) == {(2, 1, OutcomeArray([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]])): 0.5, (3, 1, OutcomeArray([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0]])): 0.5}
    
    qtype = QuantumFloat(5, -2)
    q_array_1 = QuantumArray(qtype, (2,2))
    q_array_2 = QuantumArray(qtype, (2,2))
    q_array_1[:] = 2*np.eye(2)
    q_array_2[:] = [[1,2],[3,4]]
    
    assert (q_array_1 @ q_array_2).get_measurement() ==     {OutcomeArray([[2., 4.],
                       [6., 0.]]): 1.0}
                                                      

    q_array = QuantumArray(qtype, (2,2))
    q_array[:] = 3*np.eye(2)
    cl_array = np.array([[1,2],[3,4]])
    assert (q_array @ cl_array).get_measurement() ==     {OutcomeArray([[3., 6.],
                       [1., 4.]]): 1.0}                                        
    
    
    # test duplicate    
    q_array_0 = QuantumArray(qtype, (2,2))
    q_array_0[:,:] = np.eye(2)
    q_array_1 = q_array_0.duplicate()
    q_array_2 = q_array_0.duplicate(init = True)
    
    assert q_array_1.get_measurement() == {OutcomeArray([[0., 0.],
                  [0., 0.]]): 1.0}
    
    assert q_array_2.get_measurement() == {OutcomeArray([[1., 0.],
                                                         [0., 1.]]): 1.0}
    
    
    
    