.. _BackendInterface:

Backend Interface
=================

.. toctree::
   :maxdepth: 2
   :hidden:

   Backend
   Job
   JobStatus
   JobResult
   BackendServer
   BackendClient
   VirtualBackend
   BatchedBackend
   DockerSimulators
   QiskitBackend
   IQMBackend
   AQTBackend
   QiskitRuntimeBackend
   StimBackend
   
The backend interface provides a minimal, hardware-agnostic abstraction for executing quantum circuits.

The primary goal of the interface is to support both clients and providers of physical quantum hardware
while remaining flexible enough to accommodate simulators and experimental backends.

Rather than enforcing a universal hardware model, the interface focuses on:

- a minimal execution contract,
- a lightweight mechanism for runtime configuration,
- optional, backend-defined hardware metadata.

This design allows Qrisp to interoperate with a wide range of existing hardware APIs
without duplicating or reimplementing vendor-specific data structures.


:ref:`Backend`
--------------

The :ref:`Backend` class is the abstract base class for all Qrisp backends.
It defines the minimal interface required to submit quantum circuits for execution
and optionally expose hardware metadata.

Concrete backends may represent local simulators or remote quantum hardware clients.
All backends must implement the ``Backend.run`` method, which submits one or more circuits
for execution and returns a :ref:`Job` handle immediately.


Synchronous Backends
~~~~~~~~~~~~~~~~~~~~

For example, we can define a simple backend that wraps the built-in Qrisp 
``run`` function for synchronous circuit simulation:

.. code-block:: python

   from qrisp.interface.backend import Backend
   from qrisp.interface.job import Job, JobResult, JobStatus
   from qrisp.simulator.simulator import run as default_run
   from typing import cast


   class SimulatorJob(Job):
      """A simple synchronous Job implementation for the default Qrisp simulator."""

      def __init__(self, backend, circuits, shots):
         super().__init__(backend=backend)
         self._circuits = circuits
         self._shots = shots
         self._status = JobStatus.INITIALIZING
         self._result_data = None
         self._error = None

      def submit(self):
         self._status = JobStatus.RUNNING
         try:
               counts = [default_run(c, self._shots) for c in self._circuits]
               self._result_data = JobResult(counts)
               self._status = JobStatus.DONE
         except Exception as exc:
               self._error = exc
               self._status = JobStatus.ERROR

      def result(self):
         if self._status == JobStatus.ERROR:
            raise RuntimeError(self._error)
         return cast(JobResult, self._result_data)

      def cancel(self):
         return False  # synchronous jobs cannot be cancelled

      def status(self):
         return self._status


   class DefaultBackend(Backend):
      """A default backend that uses the built-in Qrisp simulator."""

      @classmethod
      def _default_options(cls):
         # shots=None means analytic (exact) execution for the simulator
         return {"shots": None, "token": ""}

      def run(self, circuits, shots=None):
         circuits = [circuits] if not isinstance(circuits, list) else circuits
         n_shots = shots if shots is not None else self.options.get("shots")
         job = SimulatorJob(backend=self, circuits=circuits, shots=n_shots)
         job.submit()
         return job


Let's create a quantum circuit that applies a Hadamard gate to a single qubit and measures it:

.. code-block:: python

   from qrisp import QuantumCircuit

   circuit = QuantumCircuit(1)
   circuit.h(0)
   circuit.measure(0)

We can now create an instance of ``DefaultBackend`` and execute the circuit.
The ``Backend.run`` method returns a :ref:`Job` immediately.
Results are retrieved by calling ``Job.result``:

.. code-block:: python

   >>> backend = DefaultBackend()
   >>> job = backend.run(circuit)
   >>> print(job.status())
   JobStatus.DONE

   >>> result = job.result()
   >>> print(result.get_counts())
   {'0': 0.5, '1': 0.5}

   >>> print(backend.options)
   {'shots': None, 'token': ''}

We can also pass explicit runtime options at construction time:

.. code-block:: python

   >>> backend = DefaultBackend(options={"shots": 1024, "token": "fake_token"})
   >>> result = backend.run(circuit).result()
   >>> print(result.get_counts())
   {'0': 510, '1': 514}   # Note: actual counts may vary due to randomness

Runtime options can be updated after instantiation via ``Backend.update_options``.
Only keys that were present at construction time may be modified:

.. code-block:: python

   >>> backend.update_options(shots=2048)
   >>> result = backend.run(circuit).result()
   >>> print(result.get_counts())
   {'0': 1029, '1': 1019}   # Note: actual counts may vary due to randomness

It is also possible to submit a batch of circuits in a single call.
The backend decides internally whether to run them sequentially or in parallel:

.. code-block:: python

   >>> circuit_b = QuantumCircuit(2)
   >>> circuit_b.h(0)
   >>> circuit_b.cx(0, 1)
   >>> circuit_b.measure([0, 1])

   >>> job = backend.run([circuit, circuit_b], shots=512)
   >>> result = job.result()
   >>> print(result.get_counts(0))   # first circuit
   {'0': 254, '1': 258}
   >>> print(result.get_counts(1))   # second circuit
   {'00': 259, '11': 253}
   >>> print(result.all_counts)      # all circuits
   [{'0': 254, '1': 258}, {'00': 259, '11': 253}]


Asynchronous Backends
~~~~~~~~~~~~~~~~~~~~~

For hardware backends, circuit execution is typically asynchronous: the job is submitted
to a remote queue, and the caller does not block waiting for it to complete.
We can simulate this behaviour locally by running circuits in parallel background threads,
using Python's ``threading`` module.

The following example wraps the same built-in simulator, but submits each circuit to a
separate thread so they all execute concurrently:

.. code-block:: python

   import threading
   from qrisp.interface.backend import Backend
   from qrisp.interface.job import Job, JobResult, JobStatus
   from qrisp.simulator.simulator import run as default_run
   from typing import cast


   class AsyncSimulatorJob(Job):
      """An asynchronous Job that runs each circuit in a separate background thread."""

      def __init__(self, backend, circuits, shots):
         super().__init__(backend=backend)
         self._circuits = circuits
         self._shots = shots
         self._status = JobStatus.INITIALIZING
         self._result_data = None
         self._error = None
         # threading.Event is the synchronisation primitive that allows
         # result() to block cheaply until all threads have finished.
         self._done_event = threading.Event()

      def submit(self):
         self._status = JobStatus.QUEUED
         # Start a single coordinator thread that manages all circuit threads.
         # daemon=True means the thread will not prevent the process from exiting.
         threading.Thread(target=self._execute, daemon=True).start()

      def _execute(self):
         """Run all circuits in parallel and collect results."""
         self._status = JobStatus.RUNNING
         try:
               # Pre-allocate a results list so each thread can write to its own slot
               # without race conditions (list indexing is thread-safe in CPython).
               counts = [None] * len(self._circuits)

               def run_one(index, circuit):
                  counts[index] = default_run(circuit, self._shots)

               # Create one thread per circuit.
               threads = [
                  threading.Thread(target=run_one, args=(i, c))
                  for i, c in enumerate(self._circuits)
               ]
               # Start all threads at once.
               for t in threads:
                  t.start()
               # Wait for every thread to finish before proceeding.
               for t in threads:
                  t.join()

               self._result_data = JobResult(counts)  # type: ignore
               self._status = JobStatus.DONE

         except Exception as exc:
               self._error = exc
               self._status = JobStatus.ERROR

         finally:
               # Signal the event so that any caller blocked in result() wakes up.
               self._done_event.set()

      def result(self):
         # Block here until _done_event is set by the coordinator thread.
         self._done_event.wait()
         if self._status == JobStatus.ERROR:
               raise RuntimeError(self._error)
         return cast(JobResult, self._result_data)

      def cancel(self):
         # Python threads cannot be forcibly stopped, so cancellation is not supported.
         return False

      def status(self):
         return self._status


   class AsyncBackend(Backend):
      """A backend that executes each circuit in a separate background thread."""

      @classmethod
      def _default_options(cls):
         return {"shots": 1024}

      def run(self, circuits, shots=None):
         circuits = [circuits] if not isinstance(circuits, list) else circuits
         n_shots = shots if shots is not None else self.options.get("shots")
         job = AsyncSimulatorJob(backend=self, circuits=circuits, shots=n_shots)
         # submit() returns immediately after starting the background thread.
         job.submit()
         return job


The key difference from ``DefaultBackend`` is that ``run()`` returns to the caller
before any circuit has finished executing.
The coordinator thread is already running in the background:

.. code-block:: python

   from qrisp import QuantumCircuit

   circuit_a = QuantumCircuit(1)
   circuit_a.h(0)
   circuit_a.measure(0)

   circuit_b = QuantumCircuit(2)
   circuit_b.h(0)
   circuit_b.cx(0, 1)
   circuit_b.measure([0, 1])

   backend = AsyncBackend(options={"shots": 1024})
   job = backend.run([circuit_a, circuit_b])

   # At this point run() has already returned. The job is not yet done.
   print(job.status())    # JobStatus.QUEUED or JobStatus.RUNNING


We can check the status of the job without blocking, and then call ``result()`` 
to block until all circuits have finished executing:


.. code-block:: python

   >>> result = job.result()   # result() blocks here until all threads have finished.
   >>> print(job.status())    
   JobStatus.DONE

   >>> print(result.get_counts(0))   
   {'0': 517, '1': 507}   # Note: actual counts may vary due to randomness

   >>> print(result.get_counts(1))   
   {'00': 537, '11': 487}   # Note: actual counts may vary due to randomness

   >>> print(result.all_counts)  
   [{'0': 517, '1': 507}, {'00': 537, '11': 487}]


In this example, all circuits start running at the same time, so the total
wall-clock time is roughly equal to the time of the slowest circuit rather than the sum
of all circuit times. For a batch of many circuits, this can be significantly faster
than sequential execution.

Note that this example uses threads rather than processes. For CPU-bound workloads such
as statevector simulation, the `Python GIL <https://docs.python.org/3/library/threading.html#gil-and-performance-considerations>`_ 
limits true parallelism on a single machine. In practice, real hardware backends achieve 
genuine parallelism because the work happens on remote QPU hardware, not locally in Python. 
The threading model here is therefore a faithful simulation of the asynchronous contract 
(the caller submits and returns immediately) even if the local speedup is modest.

These two examples cover the most important local execution patterns: blocking simulation
and parallelised asynchronous simulation. A third important case is the remote hardware
backend that submits circuits to a vendor API, receives a job identifier, and polls for
results over the network. This is architecturally distinct and will be covered in a dedicated
example in a future release.


:ref:`Job`
----------

The :ref:`Job` class is an abstract handle for a (potentially asynchronous) backend execution.
It is returned by ``Backend.run`` immediately after submission, regardless of whether
the execution is synchronous or asynchronous.

This follows the *Future* (or *Promise*) pattern: execution happens independently of the
caller, and the caller decides when to block for the result.

The base ``Job`` class defines *only the observable contract*: four abstract methods that
every concrete implementation must provide.
It deliberately prescribes no internal synchronisation mechanism.
A synchronous simulator may resolve the job inline; an asynchronous hardware backend
may poll a remote queue in a background thread or coroutine.
From the caller's perspective, both are used identically:

.. code-block:: python

   job = backend.run(circuit, shots=1024)

   # Non-blocking: inspect the current state without waiting.
   print(job.status())    # e.g. JobStatus.QUEUED or JobStatus.RUNNING

   # Blocking: wait until the job finishes and retrieve the result.
   result = job.result()

   # Attempt cancellation (best-effort, may return False if already done).
   accepted = job.cancel()

Concrete subclasses must implement the four abstract methods:

- ``Job.submit``: trigger the actual execution.
- ``Job.result``: block until the result is available and return it.
- ``Job.cancel``: attempt to cancel the job.
- ``Job.status``: return the current :ref:`JobStatus` without blocking.

Several non-blocking convenience helpers are provided by the base class
and derived from ``Job.status``:

.. code-block:: python

   job.done()            # True only if the job completed successfully (JobStatus.DONE)
   job.in_final_state()  # True if the job has reached any terminal state (DONE, CANCELLED, or ERROR)


Additional helpers ``running()``, ``queued()``, and ``cancelled()`` are also available 
for polling-style workflows. If ``result()`` is called on a job that has 
failed or been cancelled, a :exc:`RuntimeError` is raised.


:ref:`JobStatus`
----------------

The :ref:`JobStatus` enumeration defines the possible states of a :ref:`Job`
during its lifecycle.

The six states are:

- ``INITIALIZING``: the job has been created but not yet submitted to the backend.
- ``QUEUED``: the job has been submitted and is waiting for execution resources.
- ``RUNNING``: the job is currently being executed.
- ``DONE``: the job completed successfully. Results are available via ``Job.result``.
- ``CANCELLED``: the job was cancelled before or during execution.
- ``ERROR``: the job failed due to an error during execution.

The three terminal states (``DONE``, ``CANCELLED``, ``ERROR``) are collected in
the module-level constant ``JOB_FINAL_STATES``.
Once a job reaches any of these states, its outcome is final:

.. code-block:: python

   from qrisp.interface.job import JOB_FINAL_STATES, JobStatus

   assert JobStatus.DONE      in JOB_FINAL_STATES
   assert JobStatus.CANCELLED in JOB_FINAL_STATES
   assert JobStatus.ERROR     in JOB_FINAL_STATES


:ref:`JobResult`
----------------

The :ref:`JobResult` class wraps the outcome of one or more circuit executions.

For more details, see the :ref:`JobResult` documentation.

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


:ref:`StimBackend`
---------------------

The StimBackend function returns a :ref:`BatchedBackend` that uses `Stim <https://github.com/quantumlib/Stim>`_ 
for fast Clifford circuit simulation. Stim is particularly well-suited for simulating quantum error correction circuits.

::

   from qrisp.interface import StimBackend
   
   backend = StimBackend()
