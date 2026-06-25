.. _AbstractParameters:

Abstract Parameters
===================
In this example we will show how to create quantum circuits with abstract parameters, ie. parameters that can be specified after the circuit has already been constructed.

>>> from qrisp import QuantumCircuit, PGate
>>> from sympy import Symbol
>>> import numpy as np

Create a QuantumCircuit to operate on

>>> qc = QuantumCircuit(3)

Create a controlled phase gate with a SymPy Symbol $\phi$  as a parameter.

>>> phi = Symbol("phi")
>>> cp_gate = PGate(phi).control(2, ctrl_state = "00")
>>> qc.h(-1)
>>> qc.append(cp_gate, qc.qubits)

To evaluate the circuit we can now substitute using the :meth:`bind_parameters <qrisp.QuantumCircuit.bind_parameters>` method:

>>> subs_dict = {phi : np.pi/2}
>>> bound_qc = qc.bind_parameters(subs_dict)
>>> print(bound_qc)

.. code-block:: none

    qb_34: ──────o───────
                 │       
    qb_35: ──────o───────
           ┌───┐ │P(π/2) 
    qb_36: ┤ H ├─■───────
           └───┘         
       
       
Using the :meth:`statevector_array <qrisp.QuantumCircuit.statevector_array>` method you can retrieve an array with symbolic entries:

>>> print(qc.statevector_array())
[0.707106769084930 0.70710676908493*exp(1.0*I*phi) 0 0 0 0 0 0]

Even the unitary is available as a symbolic array using the :meth:`get_unitary <qrisp.QuantumCircuit.get_unitary>` method.

Abstract parameters also work for standalone gate application functions like :meth:`rx <qrisp.p>`.

>>> from qrisp import QuantumVariable, rx
>>> qv = QuantumVariable(4)
>>> rx(phi, qv[0])
>>> theta = Symbol("theta")
>>> rx(theta, qv[1])

Using the :meth:`statevector<qrisp.QuantumSession.statevector>` method of the associated :ref:`QuantumSession`, we can retrieve a symbolic ket expression:

>>> qv.qs.statevector()
- sin(phi/2)*sin(theta/2)*|1100> - I*sin(phi/2)*cos(theta/2)*|1000> - I*sin(theta/2)*cos(phi/2)*|0100> + cos(phi/2)*cos(theta/2)*|0000>