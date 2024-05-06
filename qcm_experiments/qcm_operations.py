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

from qrisp import *
def U(op, ra):
    
    if op == "h":
        h(ra)
    elif op == "x":
        x(ra)
    else:
        raise Exception(f"Don't know operation {op}")
        
from qrisp import swap

def get_bit(ra, rb, rc):
    demux(rc, ra, rb)

def add(ra, rb):
    if ra is rb:
        ra.quantum_bit_shift(1)
    else:
        ra += rb

def multiply(ra, rb, verify_uncomputation = False):
    
    res_type = QuantumFloat(ra.size + rb.size)
    
    comp_q_dic = QuantumDictionary(return_type = res_type)
    
    if ra is rb:
        
        for a in [ra.decoder(i) for i in range(2**ra.size)]:
                comp_q_dic[a] = a**2
                
        rres = comp_q_dic[ra]
        
    else:
        
        for a in [ra.decoder(i) for i in range(2**ra.size)]:
            for b in [rb.decoder(i) for i in range(2**rb.size)]:
                comp_q_dic[a, b] = a*b
                
        rres = comp_q_dic[ra, rb]
        
        
        
        
    def uncomputation_wrapper(rres, rb):
        
        uncomp_q_dic = QuantumDictionary(return_type = ra)
        
        for b in [rb.decoder(i) for i in range(2**rb.size)]:
            for res in [rres.decoder(i) for i in range(2**rres.size)]:
                
                uncomp_q_dic[res, b] = 0
                
                if b == 0:
                    continue
                
                if res%b == 0:
                    if res//b < 2**ra.size:
                        
                        if ra is rb:
                            uncomp_q_dic[res] = res//b
                        else:
                            uncomp_q_dic[res, b] = res//b
        
        if ra is rb:
            return uncomp_q_dic[rres]
        else:
            return uncomp_q_dic[rres, rb]
            
    with invert():
        redirect_qfunction(uncomputation_wrapper)(rres, rb, target = ra)

    ra.extend(rb.size)
    
    swap(ra, rres)
    
    rres.delete(verify = verify_uncomputation)

"""    
# Test multiplication
ra = QuantumFloat(3)
rb = QuantumFloat(3)

ra[:] = 3
rb[:] = 2
multiply(ra, rb, verify_uncomputation = True)
print(multi_measurement([ra, rb]))

# Test squaring
ra = QuantumFloat(3)

ra[:] = 3
multiply(ra, ra, verify_uncomputation = True)
print(multi_measurement([ra, rb]))
"""
    
def jump(br, p):
    br += p%(2**br.size)

def conditional_jump(br, ra, p):
    with ra == 0:
        jump(br, p)
        
def indirect_jump(br, ra):
    br += ra
    

def execute_operation(operation, args, br):
    
    if operation == "nop":
        return
    elif operation == "h":
        h(args[0])
    elif operation == "x":
        x(args[0])
    elif operation == "swap":
        swap(args[0], args[1])
    elif operation == "get":
        get_bit(args[0], args[1], args[2])
    elif operation == "add":
        add(args[0], args[1])
    elif operation == "mul":
        mul(args[0], args[1])
    elif operation == "jmp":
        jump(br, args[1])
    elif operation == "jz":
        conditional_jump(br, args[0], args[1])
    elif operation == "jmp*":
        indirect_jump(br, args[0])
    else:
        raise Exception(f"Don't know operation {operation}")
