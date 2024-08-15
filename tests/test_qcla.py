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

import numpy as np

from qrisp import QuantumFloat, QuantumVariable, QuantumBool, qq_calc_carry, cq_calc_carry, h, multi_measurement, qcla, bin_rep, control, x


def test_qq_carry_generation():
    from qrisp.permeability import verify
    verify[0] = 1
    from qrisp.arithmetic.adders.qcla.quantum_quantum.qq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 1
    
    
    # Test functions
    # Function to classically calculate the carry value
    def full_adder_carry(bitstring1, bitstring2):
        if len(bitstring1) != len(bitstring2):
            raise ValueError("Bitstrings must be of equal length")

        n = len(bitstring1)
        carry = 0
        carry_values = []

        for i in range(n):
            a = int(bitstring1[i])
            b = int(bitstring2[i])

            # Calculate the carry-out (Cout) for this bit
            sum_ab = a + b + carry
            carry_out = sum_ab // 2

            carry_values.append(carry_out)

            # Update the carry for the next iteration
            carry = carry_out

        return ''.join(str(c) for c in carry_values)

        
    n = 4
    for n in range(3, 8):
        for r in range(2,n):
            a = QuantumVariable(n)
            b = QuantumVariable(n)
            
            h(a)
            h(b)
                    
            c = qq_calc_carry(a, b, r)
            
            mes_res = multi_measurement([a, b, c])
            
            for k in mes_res.keys():
                if k[2] != full_adder_carry(k[0], k[1])[:-1]:
                    print(k[0])
                    print(k[1])
                    print(k[2])
                    print(full_adder_carry(k[0], k[1])[:-1])
                    
                    assert False
                    
    from qrisp.permeability import verify
    verify[0] = 0
    from qrisp.arithmetic.adders.qcla.quantum_quantum.qq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 0

def test_qq_qcla_adder():
    
    from qrisp.permeability import verify
    verify[0] = 1
    from qrisp.arithmetic.adders.qcla.quantum_quantum.qq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 1
    
    import time
    
    t0 = time.time()
    for radix_base in [2,3]:
        for radix_exponent in [1,0,2]:
            for m in range(1, 7):        
                for n in range(1, m):
                    
                    a = QuantumFloat(n)
                    b = QuantumFloat(m)
                    c = QuantumFloat(m)
                    
                    h(a)
                    h(b)
                    c[:] = b
                    
                    qcla(a, b, radix_base = radix_base, radix_exponent = radix_exponent)
                    mes_res = multi_measurement([a, c, b])
                    
                    for k in mes_res.keys():
                        if k[2] != (k[0] + k[1])%2**m:
                            print(k[0])
                            print(k[1])
                            print(k[2])
                            print((k[0]+k[1])%2**m)
                            
                            assert False
                    
                    statevector_arr = a.qs.compile().statevector_array()
                    angles = np.angle(
                        statevector_arr[
                            np.abs(statevector_arr) > 1 / 2 ** ((a.size + b.size) / 2 + 1)
                        ]
                    )
        
                    # Test correct phase behavior
                    assert np.sum(np.abs(angles)) < 0.1
                    
                    
    print(time.time()-t0)
    from qrisp.permeability import verify
    verify[0] = 0
    from qrisp.arithmetic.adders.qcla.quantum_quantum.qq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 0
    
    a = QuantumFloat(8)
    b = QuantumFloat(8)
    a[:] = 4
    b[:] = 15
    qcla(a, b)

    from qrisp import t_depth_indicator
    gate_speed = lambda x : t_depth_indicator(x, epsilon = 2**-10)
    qc = b.qs.compile(gate_speed = gate_speed, compile_mcm = True)
    assert qc.t_depth() < 20

    qc = b.qs.compile(workspace = 10, gate_speed = gate_speed, compile_mcm = True)
    assert qc.t_depth() < 9

    a = QuantumFloat(40)
    b = QuantumFloat(40)
    x(a)
    x(b)
    qcla(a, b)
    qc = b.qs.compile(workspace = 50, gate_speed = gate_speed, compile_mcm = True)
    assert qc.t_depth() < 22


def test_cq_carry_generation():
    from qrisp.permeability import verify
    verify[0] = 1
    from qrisp.arithmetic.adders.qcla.classical_quantum.cq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 1
    
    # Test functions
    # Function to classically calculate the carry value
    def full_adder_carry(bitstring1, bitstring2):
        if len(bitstring1) != len(bitstring2):
            raise ValueError("Bitstrings must be of equal length")

        n = len(bitstring1)
        carry = 0
        carry_values = []

        for i in range(n):
            a = int(bitstring1[i])
            b = int(bitstring2[i])

            # Calculate the carry-out (Cout) for this bit
            sum_ab = a + b + carry
            carry_out = sum_ab // 2

            carry_values.append(carry_out)

            # Update the carry for the next iteration
            carry = carry_out

        return ''.join(str(c) for c in carry_values)

        
    n = 4
    for n in [3, 4, 7]:
        for r in range(2,n):
            for a in range(2**n):
                b = QuantumVariable(n)
                ctrl = QuantumBool()
                
                h(ctrl)
                h(b)
                        
                c = cq_calc_carry(a, b, r, ctrl = ctrl[0])
                
                mes_res = multi_measurement([b, c, ctrl])
                
                for k in mes_res.keys():
                    
                    if k[2]:
                        if k[1] != full_adder_carry(k[0], bin_rep(a, n)[::-1])[:-1]:
                            print(k[0])
                            print(k[1])
                            print(k[2])
                            print(bin_rep(a, n)[::-1])
                            print(full_adder_carry(k[0], bin_rep(a, n)[::-1])[:-1])
                            
                            assert False
                    else:
                        assert k[1] == (n-1)*"0"
                        
                b = QuantumVariable(n)
                
                h(b)
                        
                c = cq_calc_carry(a, b, r)
                
                mes_res = multi_measurement([b, c])
                
                for k in mes_res.keys():
                    if k[1] != full_adder_carry(k[0], bin_rep(a, n)[::-1])[:-1]:
                        print(k[0])
                        print(k[1])
                        print(k[2])
                        print(full_adder_carry(k[0], k[1])[:-1])
                        
                        assert False
                    
    from qrisp.permeability import verify
    verify[0] = 0
    from qrisp.arithmetic.adders.qcla.classical_quantum.cq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 0
    


def test_cq_qcla_adder():
    
    from qrisp.permeability import verify
    verify[0] = 1
    from qrisp.arithmetic.adders.qcla.classical_quantum.cq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 1
    
    import time
    
    t0 = time.time()
    for radix_base in [2,3]:
        for radix_exponent in [1,0,2]:
            for m in range(1, 6):        
                for a in range(2**m):
                    
                    b = QuantumFloat(m)
                    c = QuantumFloat(m)
                    ctrl = QuantumBool()
                    
                    h(ctrl)
                    h(b)
                    c[:] = b
                    
                    qcla(a, b, radix_base = radix_base, radix_exponent = radix_exponent, ctrl = ctrl[0])
                    mes_res = multi_measurement([c, b, ctrl])

                    statevector_arr = b.qs.compile().statevector_array()
                    angles = np.angle(
                        statevector_arr[
                            np.abs(statevector_arr) > 1 / 2 ** ((c.size + b.size) / 2 + 1)
                        ]
                    )
        
                    # Test correct phase behavior
                    assert np.sum(np.abs(angles)) < 0.1

                    for c, b, ctrl in mes_res.keys():
                        
                        if ctrl:
                            if b != (c + a)%2**m:
                                print(b)
                                print(c)
                                print(a)
                                assert False
                        else:
                            assert b == c
                            
                    
                    # Test non-controlled version
                    
                    b = QuantumFloat(m)
                    c = QuantumFloat(m)
                    
                    h(b)
                    c[:] = b

                    qcla(a, b, radix_base = radix_base, radix_exponent = radix_exponent)
                    mes_res = multi_measurement([c, b])

                    statevector_arr = b.qs.compile().statevector_array()
                    angles = np.angle(
                        statevector_arr[
                            np.abs(statevector_arr) > 1 / 2 ** ((c.size + b.size) / 2 + 1)
                        ]
                    )
        
                    # Test correct phase behavior
                    assert np.sum(np.abs(angles)) < 0.1
                    
                    
                    
                    for c, b in mes_res.keys():
                        if b != (c + a)%2**m:
                            print(b)
                            print(c)
                            print(a)
                            assert False
                    
                    
    from qrisp.permeability import verify
    verify[0] = 0
    from qrisp.arithmetic.adders.qcla.classical_quantum.cq_qcla_adder import verify_manual_uncomputations
    verify_manual_uncomputations[0] = 0
    
    b = QuantumFloat(10)
    b += 4
    qbl = QuantumBool()
    h(qbl)

    with control(qbl):
        
        qcla(16, b)
        
    assert multi_measurement([qbl, b]) == {(True, 20) : 0.5, (False, 4) : 0.5}
                    
    b = QuantumFloat(8)
    a = 4
    b[:] = 15
    qcla(a, b)
    
    from qrisp import t_depth_indicator
    gate_speed = lambda x : t_depth_indicator(x, epsilon = 2**-10)
    qc = b.qs.compile(gate_speed = gate_speed, compile_mcm = True)
    assert qc.t_depth() < 11
    
    qc = b.qs.compile(workspace = 10, gate_speed = gate_speed, compile_mcm = True)
    assert qc.t_depth() < 7
    
    a = QuantumFloat(40)
    b = QuantumFloat(40)
    qcla(a, b)
    qc = b.qs.compile(workspace = 50, gate_speed = gate_speed, compile_mcm = True)
    assert qc.t_depth() < 21
