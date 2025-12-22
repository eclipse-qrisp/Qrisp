.. _BackendInterface:

Backend Interface
=================

.. toctree::
   :maxdepth: 2
   :hidden:

   Backend
   BackendServer
   BackendClient
   VirtualBackend
   BatchedBackend
   DockerSimulators
   QiskitBackend
   IQMBackend
   AQTBackend
   QiskitRuntimeBackend
   
The backend interface provides a minimal, hardware-agnostic abstraction for executing gate-based quantum circuits.

The primary goal of the interface is to support both clients and providers of physical quantum hardware 
while remaining flexible enough to accommodate simulators, emulators, and experimental backends.

Rather than enforcing a universal hardware model, the interface focuses on:

- a minimal execution contract,

- a lightweight mechanism for runtime configuration,

- optional, backend-defined hardware metadata.

This design allows Qrisp to interoperate with a wide range of existing hardware APIs and network architectures 
without duplicating or reimplementing vendor-specific data structures


:ref:`Backend`
--------------

The :class:`Backend` class is the abstract base class for all Qrisp backends.
It defines the minimal interface required to execute quantum circuits and optionally expose hardware metadata.

Concrete backends may represent simulators, local emulators, or remote quantum hardware.

All backends must implement the :meth:`Backend.run` method, which executes a quantum circuit on the backend.

The exact execution semantics are backend-specific. A simulator may return a statevector, 
while a hardware backend may return measurement counts or a job handle. 
Users should treat the return value of run as backend-specific.

For example, we can define a simple default backend implementation that uses the built-in simulator in Qrisp:

.. code-block:: python

   from qrisp.interface.backend import Backend
   from qrisp.simulator.simulator import run as default_run


   class DefaultBackend(Backend):
      """A default backend that uses the built-in simulator."""

      @classmethod
      def _default_options(cls):
         # shots=None means analytic (exact) execution for the simulator
         return {"shots": None, "token": ""}

      def run(self, circuit, shots: int | None = None):
         shots = shots if shots is not None else self.options.get("shots", None)
         token = self.options.get("token", "")
         return default_run(circuit, shots, token)


As clarified in the comment, `shots=None` means analytic (exact) execution for the simulator.

We can then use this backend to run a simple quantum circuit.
Let's start by creating a quantum circuit that applies a Hadamard gate to a single qubit and measures it.

.. code-block:: python

   from qrisp import QuantumCircuit

   circuit = QuantumCircuit(1)
   circuit.h(0)
   circuit.measure(0)


Now, we can create an instance of our `DefaultBackend` and execute the circuit.

We have multiple ways to specify the number of shots for the execution. 
We can either set it as a runtime option when creating the backend instance, or we can pass it directly to the :meth:`Backend.run` method.


.. code-block:: python

   >>> backend = DefaultBackend()
   >>> result = backend.run(circuit)
   >>> print(result)
   {'0': 0.5, '1': 0.5}

   >>> print(backend.options)
   {'shots': None, 'token': ''}


We can also set different runtime options when creating the backend instance.


.. code-block:: python

   >>> backend = DefaultBackend(options={"shots": 1024, "token": "fake_token"})
   >>> result = backend.run(circuit)
   >>> print(result)
   {'0': 510, '1': 514}   # Note that the actual counts may vary due to randomness

   >>> print(backend.options)
   {'shots': 1024, 'token': 'fake_token'}


Finally, we can update the backend options after instantiation using the :meth:`Backend.update_options` method.


.. code-block:: python

   >>> backend.update_options(shots=2048)
   >>> result = backend.run(circuit)
   >>> print(result)
   {'0': 1029, '1': 1019}   # Note that the actual counts may vary due to randomness


:ref:`BackendServer`
--------------------

.. note::

   The `BackendServer` class is deprecated and will be removed in future releases. Please use the :ref:`Backend` class instead. 


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

.. note::

   The `BackendClient` class is deprecated and will be removed in future releases. Please use the :ref:`Backend` class instead.


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

.. note::

   The `Docker Backend` module is deprecated and will be removed in future releases.

This module describes the inbuilt Docker container to enable utilization of alternative simulation frameworks.  
It supports: Cirq (cirq.Simulator), MQT (ddsim qasm_simulator), Qiskit (Aer Backend), PyTket (AerBackend), Rigetti (numpy wavefunction simulator), Qulacs (sampler).  

This functionality is meant to be accessed via a Docker container. Appropriate setup should be conducted.
This setup includes: Installation of Docker, building of the image provided in this repo, and starting the container built with the image. Specific steps/ info for utilization in this context is given below.


:ref:`VirtualBackend`
---------------------

.. note::

   The `VirtualBackend` class is deprecated and will be removed in future releases. Please use the :ref:`Backend` class instead.

The VirtualBackend class allows to run external circuit dispatching code locally and
having adherence to the Qrisp interface at the same time. Using this class it is possible
to use the (Python) infrastructure of any backend provider as a Qrisp backend.

:ref:`QiskitBackend`
---------------------------

This class is a wrapper for the VirtualBackend to quickly integrate Qiskit backend instances.

.. code-block:: python

   from qiskit_aer import AerSimulator
   from qrisp.interface import QiskitBackend
   qiskit_backend = AerSimulator()
   vrtl_aer_sim = QiskitBackend(qiskit_backend)

Naturally, this also works for non-simulator Qiskit backends (e.g. IBM quantum computers).


:ref:`IQMBackend`
---------------------

The IQMBackend class allows to run Qrisp programs on IQM quantum computers available via 
`IQM Resonance <https://resonance.meetiqm.com/>`_. 
For an up-to-date list of device instance names check the IQM Resonance Dashboard. 
Devices available via IQM Resonance currently support up to 20 000 shots. 

.. code-block:: python

   from qrisp.interface import IQMBackend
   qrisp_garnet = IQMBackend(
      api_token = "YOUR_IQM_RESONANCE_TOKEN", 
      device_instance = "garnet" # check the website for an up-to-date list of devices
   )


:ref:`AQTBackend`
---------------------

The AQTBackend class allows to run Qrisp programs on AQT quantum computers available via 
`AQT Cloud <https://www.aqt.eu/products/arnica/>`_. 
Devices available via AQT Cloud currently support up to 2000 shots. 

.. code-block:: python

   from qrisp.interface import AQTBackend
   qrisp_ibex = AQTBackend(
      api_token = "YOUR_AQT_ARNICA_TOKEN", 
      device_instance = "ibex", 
      workspace = "YOUR_COMPANY_OR_PROJECT_NAME"
   )
