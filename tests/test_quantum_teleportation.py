"""
/********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2 
* or later with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0
********************************************************************************/
"""

# Created by ann81984 at 22.07.2022
from qrisp.quantum_network import QuantumNetwork
import random
import numpy as np
import time


def test_quantum_teleportation():
    # In this example we will demonstrate Qrisps quantum network simulator
    # by performing a teleportation

    # Create a Server - this server will keep track of the participants
    # and simulate the state of the network
    qn = QuantumNetwork()

    # Register clients
    qn.register_client("alice")
    qn.register_client("bob")

    # Create Bell-Pair function
    def share_bell_pair(qn, recipient_0, recipient_1):
        client_name = "bell_pair_distributor_" + str(random.randint(0, int(1e6)))
        qn.register_client(client_name)

        qn.request_qubits(2, client_name)

        telamon_qc = qn.get_clear_qc(client_name)

        telamon_qc.h(0)
        telamon_qc.cx(0, 1)

        qn.run(telamon_qc, client_name)

        qn.send_qubits(client_name, recipient_0, [telamon_qc.qubits[0]], "")
        qn.send_qubits(client_name, recipient_1, [telamon_qc.qubits[1]], "")

        return telamon_qc.qubits[0], telamon_qc.qubits[1]

    def teleport_qubit(qn, alice, bob, teleported_qubit):
        # Share Bell-Pair
        bell_pair_0, bell_pair_1 = share_bell_pair(qn, alice, bob)

        # Get the updated circuit for alice (now contains one half of the bell-pair)
        alice_qc = qn.get_clear_qc(alice)

        # Perform the Alice's steps of the quantum teleportation protocol
        alice_qc.cx(teleported_qubit, bell_pair_0)
        alice_qc.h(teleported_qubit)
        alice_qc.measure(bell_pair_0)
        alice_qc.measure(teleported_qubit)

        # Execute the circuit
        alice_res = qn.run(alice_qc, alice)

        # Perform Bob's steps of the quantum teleportation protocol
        bob_qc = qn.get_clear_qc(bob)

        # The information about the outcome alice_res is transfered via a classical channel
        if list(alice_res.keys())[0][0] == "1":
            bob_qc.x(bell_pair_1)

        if list(alice_res.keys())[0][1] == "1":
            bob_qc.z(bell_pair_1)

        return bell_pair_1

    qn.request_qubits(2, "alice")

    # Get clear circuit for alice
    alice_qc = qn.get_clear_qc("alice")

    # Apply some arbitrary transformation (the resulting state is teleported to bob)
    # alice_qc.rx(np.pi/2, 0)

    qn.run(alice_qc, "alice")

    recieved_qubit = teleport_qubit(qn, "alice", "bob", alice_qc.qubits[0])

    # Check the teleported_qubit
    bob_qc = qn.get_clear_qc("bob")
    # Measure the teleported qubit
    bob_qc.measure(recieved_qubit)

    qn.run(bob_qc, "bob")

    def gen_random_bitstring(size):
        res = np.zeros(size)
        for i in range(size):
            if random.random() > 0.5:
                res[i] = 1
        return res

    n = 50

    start_time = time.time()
    qn = QuantumNetwork()

    qn.register_client("alice")
    qn.request_qubits(n, "alice")
    alice_qc = qn.get_clear_qc("alice")

    qn.register_client("bob")
    qn.request_qubits(0, "bob")
    bob_qc = qn.get_clear_qc("bob")
    a = gen_random_bitstring(n)
    b = gen_random_bitstring(n)

    for i in range(n):
        if a[i] == 1:
            alice_qc.x(i)
        if b[i] == 1:
            alice_qc.h(i)

    qn.run(alice_qc, "alice")

    tp_qubits = []
    for i in range(n):
        tp_qubit = teleport_qubit(qn, "alice", "bob", alice_qc.qubits[i])
        tp_qubits.append(tp_qubit)
        # qn.send_qubit("alice", "bob", alice_qc.qubits[i])

    bob_qc = qn.get_clear_qc("bob")

    b_prime = gen_random_bitstring(n)

    for i in range(n):
        if b_prime[i] == 1:
            bob_qc.h(tp_qubits[i])
        bob_qc.measure(tp_qubits[i])

    res = qn.run(bob_qc, "bob")
    res = list(res.keys())[0]
    res = np.array([int(c) for c in res])

    congruent_b = b == b_prime

    temp_key_bob = res[congruent_b]
    temp_key_alice = a[congruent_b]
    k = len(temp_key_bob)

    if np.all(temp_key_bob[: k // 2] == temp_key_alice[: k // 2]):
        raise Exception("Evesdropper detected!")

    shared_key = temp_key_bob[k // 2 :]

    print(shared_key)
    assert all(i == 0 or i == 1 for i in shared_key)

    print(len(shared_key))
    assert len(shared_key) < 20

    print("Transmission rate: ", len(shared_key) / (time.time() - start_time))
    assert len(shared_key) / (time.time() - start_time) < 50
    overall_qc = qn.get_overall_qc()
    print(len(overall_qc.qubits))
    assert len(overall_qc.qubits) == 150
