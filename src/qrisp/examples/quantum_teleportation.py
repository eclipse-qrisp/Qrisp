"""
\********************************************************************************
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
********************************************************************************/
"""

import random
import numpy as np
import time
from qrisp import QuantumCircuit
from qrisp.quantum_network import QuantumNetworkClient, QuantumNetworkServer

# In this example we will demonstrate Qrisps quantum network simulator
# by performing a teleportation


# Create Bell-Pair function
def share_bell_pair(recipient_0, recipient_1):
    # Create Bell-Pair distribution client
    import random

    client_name = "bell_pair_distributor_" + str(random.randint(0, int(1e6)))
    telamon = QuantumNetworkClient(client_name, recipient_0.socket_ip)

    # Create Bell-pair QuantumCircuit
    bell_pair_qc = QuantumCircuit(2)

    bell_pair_qc.h(0)
    bell_pair_qc.cx(0, 1)

    # Run it
    telamon.run(bell_pair_qc)

    # Distribute the qubits
    telamon.send_qubits(recipient_0.name, [bell_pair_qc.qubits[0]], "")
    telamon.send_qubits(recipient_1.name, [bell_pair_qc.qubits[1]], "")

    messages_0 = recipient_0.inbox()
    messages_1 = recipient_1.inbox()

    # Retrieve Qubit objects

    qubit_0 = messages_0[-1][0][0]
    qubit_1 = messages_1[-1][0][0]

    return qubit_0, qubit_1


def teleport_qubit(alice, bob, teleported_qubit):
    # Share Bell-Pair
    bell_pair_0, bell_pair_1 = share_bell_pair(alice, bob)

    # Get the updated circuit for alice (now contains one half of the bell-pair)
    alice_qc = alice.get_clear_qc()

    # Perform Alice's steps of the quantum teleportation protocol
    alice_qc.cx(teleported_qubit, bell_pair_0)
    alice_qc.h(teleported_qubit)
    alice_qc.measure(bell_pair_0)
    alice_qc.measure(teleported_qubit)

    # Execute the circuit
    alice_res = alice.run(alice_qc)

    # Perform Bob's steps of the quantum teleportation protocol
    bob_qc = bob.get_clear_qc()
    # The information about the outcome alice_res is transfered via a classical channel
    if list(alice_res.keys())[0][1] == "1":
        bob_qc.x(bell_pair_1)

    if list(alice_res.keys())[0][0] == "1":
        bob_qc.z(bell_pair_1)

    bob.run(bob_qc)

    return bell_pair_1


# Create a Server - this server will keep track of the participants
# and simulate the state of the network
qns = QuantumNetworkServer(socket_ip="127.0.0.1", background=True)
qns.start()


alice = QuantumNetworkClient("alice", socket_ip="127.0.0.1")
bob = QuantumNetworkClient("bob", socket_ip="127.0.0.1")


alice.request_qubits(1)

# Get clear circuit for alice
alice_qc = alice.get_clear_qc()

# Apply some arbitrary transformation (the resulting state is teleported to bob)
# alice_qc.rx(np.pi/4, 0)
alice_qc.h(0)
alice.run(alice_qc)

received_qubit = teleport_qubit(alice, bob, alice_qc.qubits[0])


bob_qc = bob.get_clear_qc()
# Measure the teleported qubit
bob_qc.measure(received_qubit)

print(bob.run(bob_qc))


# %%

# This example displays quantum key distribution using the BB84 protocoll
# https://en.wikipedia.org/wiki/BB84


def gen_random_bitstring(size):
    res = np.zeros(size)
    for i in range(size):
        if random.random() > 0.5:
            res[i] = 1
    return res


n = 100

start_time = time.time()
qns = QuantumNetworkServer(socket_ip="127.0.0.1", background=True)
qns.start()

alice = QuantumNetworkClient("alice", socket_ip="127.0.0.1")


alice.request_qubits(n)
alice_qc = alice.get_clear_qc()

bob = QuantumNetworkClient("bob", socket_ip="127.0.0.1")
bob_qc = bob.get_clear_qc()


a = gen_random_bitstring(n)
b = gen_random_bitstring(n)


for i in range(n):
    if a[i] == 1:
        alice_qc.x(i)
    if b[i] == 1:
        alice_qc.h(i)

alice.run(alice_qc)

tp_qubits = []
for i in range(n):
    tp_qubit = teleport_qubit(alice, bob, alice_qc.qubits[i])
    tp_qubits.append(tp_qubit)

bob_qc = bob.get_clear_qc()

b_prime = gen_random_bitstring(n)

for i in range(n):
    if b_prime[i] == 1:
        bob_qc.h(tp_qubits[i])

    bob_qc.measure(tp_qubits[i])

res = bob.run(bob_qc)
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

print(len(shared_key))

print("Transmission rate: ", len(shared_key) / (time.time() - start_time))
overall_qc = alice.get_overall_qc()
