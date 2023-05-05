Quantum Networks
================

This module allows the simulation of a network of quantum computers, being able to send and receive qubits. The simulation can be performed on an actual IP-network, which allows testing the behavior of parallel classical interfaces which amend the information transfer.

.. image:: ./structure_overview.png
  :width: 900
  :alt: Alternative text


The architecture of this module involves a server (:doc:`QuantumNetworkServer`), which simulates quantum operations and measurements and manages the participants. The quantum network itself is then formed by a number of participants connecting to this server via a client object (:doc:`QuantumNetworkClient`).

The client object allows basic low-level operations such as quantum circuit execution and qubit sending. In order to make use of Qrisps high-level programming infrastructure, we provide the :doc:`QuantumNetworkSession` class, which behaves in many ways as a regular QuantumSession (ie. it supports QuantumVariable creation, environments etc.) but also allows sending QuantumVariables to other network participants.


.. toctree::
   :maxdepth: 2
   :hidden:

   
   QuantumNetworkServer
   QuantumNetworkClient
   QuantumNetworkSession
