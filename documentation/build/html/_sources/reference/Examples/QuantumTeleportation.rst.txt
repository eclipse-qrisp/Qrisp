.. _QuantumTeleportation:

Quantum Teleportation
=====================

In this example we will showcase how to use Qrisps quantum network module to realize a quantum teleportation.

For this, we first create a Server - this server will keep track of the participants and simulate the state of the network.
Subsequently, we connect two clients ::

>>> from qrisp.quantum_network import *
>>> qns = QuantumNetworkServer(socket_ip = "127.0.0.1", background = True)
>>> qns.start()
>>> alice = QuantumNetworkClient("alice", socket_ip = "127.0.0.1")
>>> bob = QuantumNetworkClient("bob", socket_ip = "127.0.0.1")


We now define the functions for executing the teleportation algorithm: ::

    #Define Bell-Pair function
    def share_bell_pair(recipient_0, recipient_1):
       
        #Create Bell-Pair distribution client
        import random
        client_name = "bell_pair_distributor_" + str(random.randint(0, int(1E6)))
        telamon = QuantumNetworkClient(client_name, recipient_0.socket_ip)

        #Create Bell-pair QuantumCircuit    
        bell_pair_qc = QuantumCircuit(2)
       
        bell_pair_qc.h(0)
        bell_pair_qc.cx(0,1)
       
        #Run it
        telamon.run(bell_pair_qc)

        #Distribute the qubits
        telamon.send_qubits(recipient_0.name, [bell_pair_qc.qubits[0]], "")
        telamon.send_qubits(recipient_1.name, [bell_pair_qc.qubits[1]], "")


        messages_0 = recipient_0.inbox()
        messages_1 = recipient_1.inbox()
       
        #Retrieve Qubit objects
       
        qubit_0 = messages_0[-1][0][0]
        qubit_1 = messages_1[-1][0][0]

        return qubit_0, qubit_1


   #Define quantum teleportation function
    def teleport_qubit(alice, bob, teleported_qubit):

        #Share Bell-Pair
        bell_pair_0, bell_pair_1 = share_bell_pair(alice, bob)

        #Get the updated circuit for alice (now contains one half of the bell-pair)
        alice_qc = alice.get_clear_qc()
       
        #Perform Alice's steps of the quantum teleportation protocol
        alice_qc.cx(teleported_qubit, bell_pair_0)
        alice_qc.h(teleported_qubit)
        alice_qc.measure(bell_pair_0)
        alice_qc.measure(teleported_qubit)
       
        #Execute the circuit
        alice_res = alice.run(alice_qc)
       
        #Perform Bob's steps of the quantum teleportation protocol
        bob_qc = bob.get_clear_qc()
        #The information about the outcome alice_res is transfered via a classical channel
        if list(alice_res.keys())[0][1] == "1":
            bob_qc.x(bell_pair_1)
              
        if list(alice_res.keys())[0][0] == "1":
            bob_qc.z(bell_pair_1)
       
        bob.run(bob_qc)
       
        return bell_pair_1

Evaluate the defined functions:

>>> alice.request_qubits(1)

Get clear circuit for alice


>>> alice_qc = alice.get_clear_qc()

Apply some arbitrary transformation (the resulting state is teleported to bob)

>>> alice_qc.h(0)
>>> alice.run(alice_qc)
>>> received_qubit = teleport_qubit(alice, bob, alice_qc.qubits[0])
>>> bob_qc = bob.get_clear_qc()

Measure the teleported qubit

>>> bob_qc.measure(received_qubit)
>>> print(bob.run(bob_qc))
{'1': 1}

Not that quantum network simulations are always performed by the single shot simulator. To get a probability distribution of the measurement outcomes of the teleported qubit, we would have to run the above script multiple times.
