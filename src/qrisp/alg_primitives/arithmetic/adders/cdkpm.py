from qrisp import QuantumCircuit, QuantumFloat, QuantumBool, QuantumModulus, cx, mcx, x
import jax.lax as lax
import jax.numpy as jnp
from qrisp import *

ALLOWED_CDKPM_ADDER_QUANTUM_TYPES = (QuantumFloat, QuantumBool, QuantumModulus)

def qq_cdkpm_adder(a, b, c_in = None, c_out = None):
    """Static implementation of the CDKPM adder from https://arxiv.org/abs/quant-ph/0410184 with both 
    quantum inputs.

    This function works for inputs of equal and unequal lengths. 
    
    """
    # verify inputs are both quantum
    if not (isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES) and isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)):
        raise ValueError("Attempted to call the static quantum-quantum CDKPM adder on non-quantum inputs.""")

    if len(a) != len(b):
        # if the inputs are of unequal length
        # extend the size of the smaller input
        diff =  abs(len(a) - len(b))
        if len(a) < len(b):
            a.extend(diff)
        if len(a) > len(b):
            b.extend(diff)

    b_qubits = list(b)
    a_qubits = list(a)
        

    # carry bit is initialized to 0
    if c_in is None:
        ancilla = QuantumFloat(1)
        ancilla[:] = 0
    else:
        ancilla = [c_in]

    if c_out is None:
        ancilla2 = QuantumFloat(1)
    else:
        ancilla2 = [c_out]

    carry_qubits = list(ancilla) + a_qubits[:-1]

    # maj gate application
    for i in range(len(carry_qubits)):
        cx(a_qubits[i], b_qubits[i])
        cx(a_qubits[i], carry_qubits[i])
        mcx((carry_qubits[i], b_qubits[i]), a_qubits[i])

    cx(a_qubits[-1], ancilla2)

    # uma gate application
    for i in reversed(range(len(carry_qubits))):
        x(b_qubits[i])
        cx(carry_qubits[i], b_qubits[i])
        mcx((carry_qubits[i], b_qubits[i]), a_qubits[i])
        x(b_qubits[i])
        cx(a_qubits[i], carry_qubits[i])
        cx(a_qubits[i], b_qubits[i])

    if c_in is None:
        ancilla.delete()

    if c_out is None:
        ancilla2.delete()

def cq_cdkpm_adder(a, b, c_in = None, c_out = None):
    """Static implementation of the CDKPM adder from https://arxiv.org/abs/quant-ph/0410184 for one classical
    input and one quantum input. 

    This function works for inputs of equal and unequal lengths. 
    
    """
    # verify one input is classical and the other is quantum
    if not (isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES) ^ isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)):
        raise ValueError("Attempted to call the static quantum-classical CDKPM adder on invalid inputs.")
        
    # convert the classical input to a quanutm input
    if not isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES):
        q_a = QuantumFloat(len(b))
        q_a[:] = a
        a = q_a
    elif not isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES):
        q_b = QuantumFloat(len(a))
        q_b[:] = b
        b = q_b
    
    # now apply the adder to both quantum inputs of the same size
    qq_cdkpm_adder(a, b, c_in, c_out)
        


def jasp_qq_cdkpm_adder(a, b, c_in = None, c_out = None):        
    """Dynamic implementation of the CDKPM adder from https://arxiv.org/abs/quant-ph/0410184 with both 
    quantum inputs.

    This function works for inputs of equal and unequal lengths. 
    
    """
    # pad the size of the input with the smaller size
    dim_a = a.size
    dim_b = b.size

    max_size = jnp.maximum(dim_a, dim_b)
    a.extend(abs(max_size-dim_a))
    b.extend(abs(max_size-dim_b))


    # carry bit is initialized to 0
    if c_in is None:
        ancilla = QuantumFloat(1)
        ancilla[:] = 0
    else:
        ancilla = [c_in]

    if c_out is None:
        ancilla2 = QuantumFloat(1)
    else:
        ancilla2 = [c_out]


    # verify inputs are both quantum
    if not (isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES) and isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)):
        raise ValueError("Attempted to call the static quantum-quantum CDKPM adder on non-quantum inputs.""")


    # first maj gate application
    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    # iterator maj gate application
    for i in jrange(1, a.size):
        cx(a[i], b[i])
        cx(a[i], a[i-1])
        mcx([a[i-1], b[i]], a[i])


    # cnot
    cx(a[-1], ancilla2[-1])

    # iterator uma gate application
    for j in jrange(a.size-1):
        # reverse the iteration
        i = a.size - j -1
        
        x(b[i])
        cx(a[i-1], b[i])
        mcx([a[i-1], b[i]], a[i])
        x(b[i])
        cx(a[i], a[i-1])
        cx(a[i], b[i])

    # last uma gate application
    x(b[0])
    cx(ancilla[0], b[0])
    mcx([ancilla[0], b[0]], a[0])
    x(b[0])
    cx(a[0], ancilla[0])
    cx(a[0], b[0])

    if c_in is None:
        ancilla.delete()

    if c_out is None:
        ancilla2.delete()

def jasp_cq_cdkpm_adder(a, b, c_in = None, c_out = None):        
    """Dynamic implementation of the CDKPM adder from https://arxiv.org/abs/quant-ph/0410184 for one
    classical input and one quantum input. 

    This function works for inputs of equal and unequal lengths. 
    
    """
     # verify one input is classical and the other is quantum
    if not (isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES) ^ isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)):
        raise ValueError("Attempted to call the static quantum-classical CDKPM adder on invalid inputs.")
        
    # convert the classical input to a quanutm input
    if not isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES):
        q_a = QuantumFloat(b.size)
        q_a[:] = a
        a = q_a
    elif not isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES):
        q_b = QuantumFloat(a.size)
        q_b[:] = b
        b = q_b
    
    # now apply the adder to both quantum inputs of the same size
    jasp_qq_cdkpm_adder(a, b, c_in, c_out)
    