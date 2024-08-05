.. _DockerSimulators:

Docker Simulators
=================

The Qrisp network interface enables convenient access to a variety of simulators through a docker container. You can simply download the docker container and obtain access to simulation without having to fight through installation and/or conversion issues. For this you need `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_. This software allows you to install and execute the most complex software environments without the hassle of compatibility issues regarding your platform. It is therefore a perfect fit for hosting quantum simulators, which can be tricky to get running. After you are done installing, please execute:

.. code-block:: console
    
    docker pull qrisp/qrisp_sim_collection:x86-version
    
If you have an ARM based CPU (as many Macs tend to), please replace the ``x86-version`` with ``arm-version``. To start the docker container your run:

.. code-block:: console

    docker run -p 8083:8083 -p 8084:8084 -p 8085:8085 -p 8086:8086 -p 8087:8087 -p 8088:8088 -p 8089:8089 -p 8090:8090 qrisp/qrisp_sim_collection:x86-version
    
If you are on ARM, make sure to also put ``arm-version`` at the end of the command. The ``-p`` commands open the ports of the docker container such that Qrisp can send the simulation requests. Once you have run this command, the container should appear in the Docker GUI, so you can simply press start if you need it again.

Once the container is running, you can start using the following backends on your machine:


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
     - `The simulator of the Qibo framework <https://qibo.science/qibo/stable/index.html>`_
    * - ``QiboSim()``
 - `The simulator of the Qibo framework <https://qibo.science/qibo/stable/index.html>`_
     


To utilize these simulators you can import the corresponding backend in Python

>>> from qrisp import QuantumFloat
>>> a = QuantumFloat(3)
>>> a[:] = 3
>>> b = QuantumFloat(3)
>>> b[:] = 4
>>> c = a*b
>>> from qrisp.interface import MQTSim
>>> c.get_measurement(backend = MQTSim())
{12: 1.0}