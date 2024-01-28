.. _DockerSimulators:

Docker Simulators
=================

The Qrisp network interface enables convenient access to a variety of simulators through a docker container. You can simply download the docker container and obtain access to simulation without having to fight through installation and/or conversion issues. For this execute

.. code-block:: console
    
    docker pull qrisp/qrisp_sim_collection:latest
    
To start the docker container your run

.. code-block:: console

    docker run -p 8083:8083 -p 8084:8084 -p 8085:8085 -p 8086:8086 -p 8087:8087 -p 8088:8088 -p 8089:8089 -p 8090:8090 qrisp/qrisp_sim_collection
    
The ``-p`` commands open the ports of the docker container such that Qrisp can send the simulation requests.

Once you have done so, you can start using the following backends on your machine:


.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Simulator Name
     - Description
   * - ``CirqSim()``
     - `"A sparse matrix state vector simulator that uses numpy."  <https://quantumai.google/reference/python/cirq/Simulator>`_
   * - ``PennylaneSim()``
     - `"The default.qubit device is PennyLane’s standard qubit-based device." <https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_qubit.html>`_
   * - ``MQTSim()``
     - `"A quantum circuit simulator based on decision diagrams written in C++." <https://mqt.readthedocs.io/projects/ddsim/en/latest/>`_ 
   * - ``PennylaneRigettiSim()``
     - `Simulator for the Pennylane-Rigetti plugin <https://docs.pennylane.ai/projects/rigetti/en/latest/code.html>`_
   * - ``PyTketStimSim()``
     - `"Stim is a fast simulator for quantum stabilizer circuits." <https://github.com/quantumlib/stim>`_
   * - ``QulacsSim()``
     - `"Qulacs is a fast quantum circuit simulator for simulating large, noisy, or parametric quantum circuits." <https://docs.qulacs.org/en/latest/>`_
   * - ``QSimCirq()``
     - `"qsim is a Schrödinger full state-vector simulator." <https://github.com/quantumlib/qsim/tree/master>`_
   * - ``QiboSim()``
     - `"qsim is a Schrödinger full state-vector simulator." <https://github.com/quantumlib/qsim/tree/master>`_
     


To utilize these simulator you can import the corresponding backend in Python

>>> from qrisp import QuantumFloat
>>> a = QuantumFloat(3)
>>> a[:] = 3
>>> b = QuantumFloat(3)
>>> b[:] = 4
>>> c = a*b
>>> from qrisp.interface import MQTSim
>>> c.get_measurement(backend = MQTSim())
{12: 1.0}