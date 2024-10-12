Backend Interface
=================

.. toctree::
   :maxdepth: 2
   :hidden:

   BackendServer
   BackendClient
   VirtualBackend
   DockerSimulators
   QiskitBackend
   IQMBackend
   QiskitRuntimeBackend
   
The backend interface contains a minimal set of features that apply to every gate-based quantum computer.
The main motivation in designing the interface, is to provide a convenient setup for both clients and providers of physical quantum
backends. To ensure compatibility to existing network architectures, the interface is realized as REST-based interface.
Also available is a Thrift implementation of the same interface.

:ref:`BackendServer`
--------------------

This class is a wrapper for convenient setup of a server, capable of processing quantum circuits. Backend providers
simply have to create a Python function called ``run_func`` which receives a :ref:`QuantumCircuit`, an integer called ``shots`` and
optionally a string called ``token``. The return value should be the measurement results as a dictionary of bitstrings.

.. code-block:: python

   from qrisp.interface import BackendServer
   from backend_provider import run_func, ping_func
   
   server = BackendServer( run_func, port = 8080)
                           
   server.start()


:ref:`BackendClient`
--------------------

This class can be used to connect to :ref:`BackendServers <BackendServer>` in order to send out requests for processing quantum circuits.

.. code-block:: python

   from qrisp.interface import BackendClient
   backend = BackendClient(server_ip, port)
   
   from some_library import some_quantum_algorithm
   
   qv = some_quantum_algorithm()
   
   res = qv.get_measurement(backend = backend)

In this code snippet, we create connect a BackendClient, to the BackendServer running under the ip ``server_ip`` and ``port``. Subsequently, we run some quantum algorithm returing a :ref:`QuantumVariable` and call the :meth:`qrisp.QuantumVariable.get_measurement` method, to query the remote backend.


:ref:`Docker Backend <DockerSimulators>`
----------------------------------------

This module describes the inbuilt Docker container to enable utilization of alternative simulation frameworks.  
It supports: Cirq (cirq.Simulator), MQT (ddsim qasm_simulator), Qiskit (Aer Backend), PyTket (AerBackend), Rigetti (numpy wavefunction simulator), Qulacs (sampler).  

This functionality is meant to be accessed via a Docker container. Appropriate setup should be conducted.
This setup includes: Installation of Docker, building of the image provided in this repo, and starting the container built with the image. Specific steps/ info for utilization in this context is given below.


:ref:`VirtualBackend`
---------------------

The VirtualBackend class allows to run external circuit dispatching code locally and
having adherence to the Qrisp interface at the same time. Using this class it is possible
to use the (Python) infrastructure of any backend provider as a Qrisp backend.

:ref:`QiskitBackend`
---------------------------

This class is a wrapper for the VirtualBackend to quickly integrate Qiskit backend instances.

::

   from qiskit import Aer
   from qrisp.interface import QiskitBackend
   qiskit_backend = Aer.get_backend('qasm_simulator')
   vrtl_qasm_sim = QiskitBackend(qiskit_backend)

Naturally, this also works for non-simulator Qiskit backends.


:ref:`IQMBackend`
---------------------

The IQMBackend class allows to run Qrisp programs on IQM quantum computers available via 
`IQM Resonance <https://resonance.meetiqm.com/>`. 
For an up-to-date list of device instance names check the IQM Resonance Dashboard. 
Devices available via IQM Resonance currently support up to 20 000 shots. 

.. code-block:: python

   from qrisp.interface import IQMBackend
   qrisp_garnet = IQMBackend(
      api_token = "YOUR_IQM_RESONANCE_TOKEN", 
      device_instance = "garnet" # check the website for an up-to-date list of devices
   )
