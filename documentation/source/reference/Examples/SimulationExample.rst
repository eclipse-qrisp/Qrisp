.. _SimulationExample:

Simulating a QC on a QC
===================================================

In this example we will showcase how data structures like the :ref:`QuantumArray` in combination with the :ref:`QuantumFloat` can be used to encode the arithmetic for simulating a quantum computer into a quantum circuit itself.

We use the :meth:`tensordot` function to contract quantum tensors (ie. a :ref:`QuantumArray` with multiple indices).

>>> import numpy as np
>>> from qrisp import QuantumFloat, QuantumArray, tensordot

Initiate the :ref:`QuantumArray` holding the statevector. We initate the state of uniform
superposition

.. math::

    \ket{+} = \frac{1}{\sqrt{2^n}} \sum_{i = 0}^{2^n - 1} \ket{i}

>>> qfloat_type = QuantumFloat(3, -2, signed = True)
>>> num_qubits = 4
>>> statevector = QuantumArray(shape = 2**num_qubits, qtype = qfloat_type)
>>> statevector[:] = [1/(2**num_qubits)**0.5]*2**num_qubits
>>> print(statevector)
{OutcomeArray([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
               0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]): 1.0}

Initiate the :ref:`QuantumArray` holding the unitary of a Z-gate

>>> z_gate = QuantumArray(shape = (2,2), qtype = qfloat_type)
>>> z_gate[:] = [[1,0], [0,-1]]
>>> print(z_gate)
{OutcomeArray([[ 1.,  0.],
              [ 0., -1.]]): 1.0}

Perform the contraction

>>> statevector = statevector.reshape(num_qubits*[2])
>>> target_qubit = 3
>>> new_statevector = tensordot(z_gate, statevector, (1, target_qubit))
>>> new_statevector = new_statevector.reshape(2**num_qubits)
>>> print(new_statevector)
{OutcomeArray([ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
               -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]): 1.0}

We perform a similar contraction using numpy arrays

>>> from numpy import tensordot
>>> statevector = 0.25*np.ones(2**num_qubits)
>>> print(statevector)
[0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
 0.25 0.25]
>>> statevector = statevector.reshape(num_qubits*[2])
>>> z_gate = np.zeros((2,2))
>>> z_gate[:] = [[1,0], [0,-1]]
>>> new_statevector = tensordot(z_gate, statevector, (1, target_qubit))
>>> print(new_statevector.reshape(2**num_qubits))
[ 0.25  0.25  0.25  0.25  0.25  0.25  0.25  0.25 -0.25 -0.25 -0.25 -0.25
 -0.25 -0.25 -0.25 -0.25]
