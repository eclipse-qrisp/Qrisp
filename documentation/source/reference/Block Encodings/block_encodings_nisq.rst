.. _block_encodings_nisq:

How to run on NISQ
==================

Since current quantum hardware lacks support for repeat-until-success protocols, this example demonstrates how to implement block encodings on NISQ devices using post-selection.

Define a block encoding for a Heisenberg Hamiltonian and apply it to an initial system state.

::

    from qrisp import *
    from qrisp.operators import X, Y, Z

    H = sum(X(i)*X(i+1) + Y(i)*Y(i+1) + Z(i)*Z(i+1) for i in range(3))
    BE = H.pauli_block_encoding()

    # Prepare initial system state
    operand = QuantumFloat(4)
    h(operand[0])

    # Apply the operator to an initial system state
    ancillas = BE.apply(operand)

Utilize the Qrisp :ref:`BackendInterface` to define a backend. While this example uses the Qiskit Aer simulator, the interface also supports physical quantum backends.

::

    from qrisp.interface import QiskitBackend
    from qiskit_aer import AerSimulator
    example_backend = QiskitBackend(backend = AerSimulator())

    # Use backend keyword to specify quantum backend
    res_dict = multi_measurement([operand] + ancillas, shots=1000, backend=example_backend)

    # Post-selection on ancillas being in |0> state
    new_dict = dict()
    success_prob = 0

    for key, prob in res_dict.items():
        if all(k == 0 for k in key[1:]):
            new_dict[key[0]] = prob
            success_prob += prob

    for key in new_dict.keys():
        new_dict[key] = new_dict[key] / success_prob

    new_dict

Note that the limited number of shots leads to significant statistical variance in the output distribution.