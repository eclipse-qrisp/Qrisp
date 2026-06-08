.. _BackendInterface:

Backend Interface
=================

.. toctree::
   :maxdepth: 2
   :hidden:

   Hardware Backends/index
   Simulator Backends/index
   Abstract Backend Interface/index
   BatchedBackend
   VirtualBackend
   
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
All backends must implement the :meth:`~qrisp.interface.Backend.run_async` method, which submits one or more
circuits for execution and returns a :ref:`Job` handle immediately.

A synchronous convenience method :meth:`~qrisp.interface.Backend.run` is also provided by the base class.
It calls :meth:`~qrisp.interface.Backend.run_async` internally, blocks until the job finishes, and returns the
results as :class:`~qrisp.interface.MeasurementResult` objects (a single object for a
single circuit, or a list of them for a sequence). :class:`~qrisp.interface.MeasurementResult`
is a :class:`~collections.abc.Mapping`, so all dict-style access works unchanged.
New code that needs the full :ref:`Job` interface (status polling, cancellation, or
concurrent execution) should call :meth:`~qrisp.interface.Backend.run_async` instead.


Example: Building a Synchronous Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, we can define a simple backend that wraps the built-in Qrisp
``run`` function for synchronous circuit simulation:

.. code-block:: python

   from qrisp import QuantumCircuit
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
         self._result_data = None

      def submit(self):
         self._last_known_status = JobStatus.RUNNING
         try:
               counts = [default_run(c, self._shots) for c in self._circuits]
               self._result_data = JobResult(counts)
               self._last_known_status = JobStatus.DONE
         except Exception as exc:
               self._failure_cause = exc
               self._last_known_status = JobStatus.ERROR

      def result(self, timeout=None):
         # This job is synchronous and completes inside submit(); timeout is ignored.
         self._raise_for_status(self._last_known_status)
         return cast(JobResult, self._result_data)

      def cancel(self):
         return False  # synchronous jobs cannot be cancelled

      def status(self):
         return self._last_known_status


   class QrispSimulatorBackend(Backend):
      """A default backend that uses the built-in Qrisp simulator."""

      @classmethod
      def _default_options(cls):
         # shots=None means analytic (exact) execution for the simulator
         return {"shots": None, "token": ""}

      def run_async(self, circuits, shots=None):
         if not isinstance(circuits, QuantumCircuit):
            circuits = list(circuits)
         else:
            circuits = [circuits]
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

We can now create an instance of ``QrispSimulatorBackend`` and execute the circuit.
Calling ``run_async`` returns a :ref:`Job` immediately, and because the simulator is
synchronous the job is already ``DONE`` before ``run_async`` returns. Results are
retrieved by calling ``job.result()``:

.. code-block:: python

   >>> backend = QrispSimulatorBackend()
   >>> job = backend.run_async(circuit)
   >>> print(job.status())
   done

   >>> result = job.result()
   >>> print(result.get_counts())
   {'0': 0.5, '1': 0.5}

   >>> print(backend.options)
   {'shots': None, 'token': ''}

If all you need is the measurement result and do not require the :ref:`Job` handle,
the inherited ``run`` method provides a shorter path. It calls ``run_async`` internally,
waits for completion, and returns a :class:`~qrisp.interface.MeasurementResult` directly:

.. code-block:: python

   >>> counts = backend.run(circuit)
   >>> print(counts)
   {'0': 0.5, '1': 0.5}

We can also pass explicit runtime options at construction time:

.. code-block:: python

   >>> backend = QrispSimulatorBackend(options={"shots": 1024, "token": "fake_token"})
   >>> result = backend.run_async(circuit).result()
   >>> print(result.get_counts())
   {'0': 510, '1': 514}   # Note: actual counts may vary due to randomness

Runtime options can be updated after instantiation via :meth:`~qrisp.interface.Backend.update_options`.
Only keys that were present at construction time may be modified:

.. code-block:: python

   >>> backend.update_options(shots=2048)
   >>> result = backend.run_async(circuit).result()
   >>> print(result.get_counts())
   {'0': 1029, '1': 1019}   # Note: actual counts may vary due to randomness

It is also possible to submit a batch of circuits in a single call.
The backend decides internally whether to run them sequentially or in parallel:

.. code-block:: python

   >>> circuit_b = QuantumCircuit(2)
   >>> circuit_b.h(0)
   >>> circuit_b.cx(0, 1)
   >>> circuit_b.measure([0, 1])

   >>> job = backend.run_async([circuit, circuit_b], shots=512)
   >>> result = job.result()
   >>> print(result.get_counts(0))   # first circuit
   {'0': 254, '1': 258}
   >>> print(result.get_counts(1))   # second circuit
   {'00': 259, '11': 253}
   >>> print(result.all_counts)      # all circuits
   [{'0': 254, '1': 258}, {'00': 259, '11': 253}]

On a synchronous backend, we can also use ``run`` with a batch of circuits.
When a list is passed, the result is a list of :class:`~qrisp.interface.MeasurementResult`
objects (one per circuit). Each supports the same dict-style access as above:

.. code-block:: python

   >>> result = backend.run([circuit, circuit_b], shots=10)
   >>> print(result)
   [{'0': 5, '1': 5}, {'00': 6, '11': 4}]   # Note: actual counts may vary


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
   from qrisp import QuantumCircuit
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
         self._result_data = None
         # threading.Event is the synchronisation primitive that allows
         # result() to block cheaply until all threads have finished.
         self._done_event = threading.Event()

      def submit(self):
         self._last_known_status = JobStatus.QUEUED
         # Start a single coordinator thread that manages all circuit threads.
         # daemon=True means the thread will not prevent the process from exiting.
         threading.Thread(target=self._execute, daemon=True).start()

      def _execute(self):
         """Run all circuits in parallel and collect results."""
         self._last_known_status = JobStatus.RUNNING
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

               self._result_data = JobResult(counts)
               self._last_known_status = JobStatus.DONE

         except Exception as exc:
               self._failure_cause = exc
               self._last_known_status = JobStatus.ERROR

         finally:
               # Signal the event so that any caller blocked in result() wakes up.
               self._done_event.set()

      def result(self, timeout=None):
         # Block here until _done_event is set by the coordinator thread.
         if not self._done_event.wait(timeout=timeout):
               raise TimeoutError(f"Job did not complete within {timeout}s.")
         self._raise_for_status(self._last_known_status)
         return cast(JobResult, self._result_data)

      def cancel(self):
         # Python threads cannot be forcibly stopped, so cancellation is not supported.
         return False

      def status(self):
         return self._last_known_status


   class AsyncBackend(Backend):
      """A backend that executes each circuit in a separate background thread."""

      @classmethod
      def _default_options(cls):
         return {"shots": 1024}

      def run_async(self, circuits, shots=None):
         if not isinstance(circuits, QuantumCircuit):
            circuits = list(circuits)
         else:
            circuits = [circuits]
         n_shots = shots if shots is not None else self.options.get("shots")
         job = AsyncSimulatorJob(backend=self, circuits=circuits, shots=n_shots)
         # submit() returns immediately after starting the background thread.
         job.submit()
         return job


The key difference from ``QrispSimulatorBackend`` is that ``run_async`` returns to the caller
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
   job = backend.run_async([circuit_a, circuit_b])

   # At this point run_async() has already returned. The job is not yet done.
   print(job.status())    # queued or running


We can check the status of the job without blocking, and then call ``result()``
to block until all circuits have finished executing:

.. code-block:: python

   >>> result = job.result()   # result() blocks here until all threads have finished.
   >>> print(job.status())
   done

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
It is returned by ``Backend.run_async`` immediately after submission, regardless of whether
the execution is synchronous or asynchronous.

This follows the *Future* (or *Promise*) pattern: execution happens independently of the
caller, and the caller decides when to block for the result.

The base ``Job`` class defines *only the observable contract*: three abstract methods that
every concrete implementation must provide.
It deliberately prescribes no internal synchronisation mechanism.
A synchronous simulator may resolve the job inline; an asynchronous hardware backend
may poll a remote queue in a background thread or coroutine.
From the caller's perspective, both are used identically:

.. code-block:: python

   job = backend.run_async(circuit, shots=1024)

   # Non-blocking: inspect the current state without waiting.
   print(job.status())    # e.g. JobStatus.QUEUED or JobStatus.RUNNING

   # Blocking: wait until the job finishes and retrieve the result.
   result = job.result()

   # Attempt cancellation (best-effort, may return False if already done).
   accepted = job.cancel()

Concrete subclasses must implement three abstract methods:

- :meth:`~qrisp.interface.Job.result`: block until the result is available and return it.
- :meth:`~qrisp.interface.Job.cancel`: attempt to cancel the job.
- :meth:`~qrisp.interface.Job.status`: return the current :ref:`JobStatus` without blocking.

.. note::

   *Submission responsibility:* ``Job`` does not mandate how or when a job transitions
   out of :attr:`~qrisp.interface.JobStatus.INITIALIZING`. That responsibility belongs
   entirely to the concrete backend author. 
   
   The only hard contract is that the job returned by
   :meth:`~qrisp.interface.Backend.run_async` must not still be in
   :attr:`~qrisp.interface.JobStatus.INITIALIZING` state.

   For real-world examples, see
   :meth:`QiskitBackend.run_async <qrisp.interface.QiskitBackend.run_async>` and
   :meth:`AQTBackend.run_async <qrisp.interface.AQTBackend.run_async>`.

Several non-blocking convenience helpers are provided by the base class
and derived from :meth:`~qrisp.interface.Job.status`:

.. code-block:: python

   job.done()            # True only if the job completed successfully (JobStatus.DONE)
   job.in_final_state()  # True if the job has reached any terminal state (DONE, CANCELLED, or ERROR)


Additional helpers ``running()``, ``queued()``, and ``cancelled()`` are also available
for polling-style workflows. If :meth:`~qrisp.interface.Job.result` is called on a job that has failed, a
:exc:`~qrisp.interface.JobFailureError` is raised; if the job was cancelled, a
:exc:`~qrisp.interface.JobCancelledError` is raised. Both are subclasses of
:exc:`RuntimeError`.


:ref:`JobStatus`
----------------

The :ref:`JobStatus` enumeration defines the possible states of a :ref:`Job`
during its lifecycle.

The six states are:

- ``INITIALIZING``: the job object has been created but execution has not yet been
  handed off to the backend. This is a transient state: a correctly implemented backend
  must ensure that every job exits ``INITIALIZING`` before :meth:`~qrisp.interface.Backend.run_async`
  returns.
- ``QUEUED``: the job has been submitted and is waiting for execution resources.
- ``RUNNING``: the job is currently being executed.
- ``DONE``: the job completed successfully. Results are available via :meth:`~qrisp.interface.Job.result`.
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


:ref:`MeasurementResult` and :ref:`DecodedMeasurementResult`
-------------------------------------------------------------

:class:`~qrisp.interface.MeasurementResult` is the return type of
:meth:`~qrisp.interface.Backend.run`.  It is a
:class:`~collections.abc.Mapping` of raw bitstring keys to counts, so all
dict-style access (``result["0"]``, ``.items()``, ``len()``, ``==`` with a
plain dict) works unchanged.

For standard backends the object arrives pre-populated.  For
:class:`~qrisp.interface.BatchedBackend` it starts empty and is filled when
:meth:`~qrisp.interface.BatchedBackend.dispatch` is called (accessing it
before that raises :exc:`RuntimeError`).

:class:`~qrisp.interface.DecodedMeasurementResult` is what
:meth:`QuantumVariable.get_measurement <qrisp.QuantumVariable.get_measurement>`
returns to the user.  It wraps a raw result together with the variable's
decoder and translates raw bitstring indices into user-facing labels (e.g.
integers for :class:`~qrisp.QuantumFloat`).  Decoding is deferred: the
plain dict representation is built the first time any key is accessed.

Both classes inherit from :class:`~qrisp.interface.LazyDict`, an abstract
:class:`~collections.abc.Mapping` whose data is computed exactly once on
first access and cached for all subsequent reads.

For more details, see the :ref:`MeasurementResult` 
and :ref:`DecodedMeasurementResult` documentation.


:ref:`BatchedBackend`
---------------------

:class:`~qrisp.interface.BatchedBackend` wraps any :ref:`Backend` and changes
the execution model: instead of submitting one circuit per
:meth:`~qrisp.QuantumVariable.get_measurement` call, all circuits are
collected and then submitted together in a single
:meth:`~qrisp.interface.Backend.run_async` call when
:meth:`~qrisp.interface.BatchedBackend.dispatch` is invoked.

.. note::

   ``BatchedBackend`` is not a subclass of :ref:`Backend`.  It is a
   *wrapper* (Virtual Proxy pattern): it holds a reference to a concrete
   backend and intercepts :meth:`~qrisp.interface.Backend.run` calls.

Obtain a ``BatchedBackend`` from any existing backend via
:meth:`~qrisp.interface.Backend.batched`:

.. code-block:: python

   from qrisp.default_backend import QrispSimulatorBackend

   backend = QrispSimulatorBackend()
   batched_backend = backend.batched()

The key benefit is visible when measuring multiple quantum variables.
Below, a ``CountingBackend`` wrapper records every ``run_async``
invocation so we can observe exactly how many hardware calls are made.

**Standard backend** (one hardware call per circuit):

.. code-block:: python

   from qrisp import QuantumFloat, h
   from qrisp.default_backend import QrispSimulatorBackend

   class CountingBackend(QrispSimulatorBackend):
       def __init__(self):
           super().__init__()
           self.run_async_calls = 0

       def run_async(self, circuits, shots=None):
           n = len(circuits) if hasattr(circuits, '__len__') else 1
           self.run_async_calls += 1
           print(f"run_async called (call #{self.run_async_calls}) "
                 f"with {n} circuit(s)")
           return super().run_async(circuits, shots)

   backend = CountingBackend()

   qf1 = QuantumFloat(3); qf1[:] = 2
   qf2 = QuantumFloat(3); qf2[:] = 5
   qf3 = QuantumFloat(3); h(qf3[0])

   r1 = qf1.get_measurement(backend=backend)
   r2 = qf2.get_measurement(backend=backend)
   r3 = qf3.get_measurement(backend=backend)

   # Output:
   # run_async called (call #1) with 1 circuit(s)
   # run_async called (call #2) with 1 circuit(s)
   # run_async called (call #3) with 1 circuit(s)

   print(backend.run_async_calls)   # 3 (one per circuit)

**BatchedBackend** (one hardware call for all circuits):

.. code-block:: python

   backend = CountingBackend()
   batched_backend = backend.batched()

   qf1 = QuantumFloat(3); qf1[:] = 2
   qf2 = QuantumFloat(3); qf2[:] = 5
   qf3 = QuantumFloat(3); h(qf3[0])

   # Each call returns immediately (no hardware submission yet).
   r1 = qf1.get_measurement(backend=batched_backend)
   r2 = qf2.get_measurement(backend=batched_backend)
   r3 = qf3.get_measurement(backend=batched_backend)

   print(backend.run_async_calls)   # 0 (nothing submitted yet)
   print(r1._populated)             # False (result is still empty)

   # One call submits all circuits in a single job.
   batched_backend.dispatch()

   # Output:
   # run_async called (call #1) with 3 circuit(s)

   print(backend.run_async_calls)   # 1 (one call for all circuits)
   print(r1, r2, r3)               # {2: 1.0} {5: 1.0} {0: 0.5, 1: 0.5}

The lazy :class:`~qrisp.interface.DecodedMeasurementResult` objects
returned by :meth:`~qrisp.QuantumVariable.get_measurement` are what
make this possible: they act as empty placeholders that
:meth:`~qrisp.interface.BatchedBackend.dispatch` populates all at once.
Decoding (e.g. from bitstring indices to ``QuantumFloat`` values) is
further deferred and happens on first access.

For more details, see the :ref:`BatchedBackend` documentation.

:ref:`QiskitBackend`
---------------------------

:class:`~qrisp.interface.QiskitBackend` is a :ref:`Backend` that wraps any
Qiskit-compatible backend (simulators or real hardware). Circuits are converted
directly to Qiskit ``QuantumCircuit`` objects, transpiled for the target device,
and submitted through Qiskit's ``SamplerV2`` primitive.

.. code-block:: python

   from qiskit_aer import AerSimulator
   from qrisp.interface import QiskitBackend

   backend = QiskitBackend(backend=AerSimulator())

The same interface works for real IBM Quantum hardware via
:class:`~qrisp.interface.QiskitRuntimeBackend`, which authenticates with the
Qiskit Runtime service and supports both single-job and session execution modes:

.. code-block:: python

   from qrisp.interface import QiskitRuntimeBackend

   backend = QiskitRuntimeBackend(
       api_token="YOUR_IBM_CLOUD_TOKEN",
       backend="ibm_brisbane",
       channel="ibm_cloud",
   )


:ref:`IQMBackend`
---------------------

.. note::

   ``IQMBackend`` is deprecated and will be removed in a future release.

``IQMBackend`` is a factory function (not a class) that returns a :ref:`BatchedBackend`
for executing circuits on IQM quantum computers available via
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

:class:`~qrisp.interface.StimBackend` is a :ref:`Backend` that simulates Clifford circuits
via `Stim <https://github.com/quantumlib/Stim>`_. Stim is particularly well-suited for
simulating quantum error correction circuits with thousands of qubits.

For synchronous use, call :meth:`~qrisp.interface.Backend.run` directly:

::

   from qrisp.interface import StimBackend

   backend = StimBackend()

For lazy, buffered execution, obtain a :ref:`BatchedBackend` via :meth:`~qrisp.interface.Backend.batched`:

::

   from qrisp import QuantumVariable
   from qrisp.interface import StimBackend

   bb = StimBackend().batched()

   qv = QuantumVariable(2)
   qv[:] = "10"

   res = qv.get_measurement(backend=bb)  # lazy (returns immediately)
   bb.dispatch()
   print(res)   # {'10': 1.0}
