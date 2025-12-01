"""
********************************************************************************
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
********************************************************************************
"""

import threading
import time


class BatchedBackend:
    """
    This class tackles the problem that many physical backends have a high-overhead
    regarding individual circuit execution. This overhead typically comes
    from finite network latency, authentication procedures, compilation steps etc.
    Typically this overhead is remedied through supporting the execution of
    batches of circuits, which however doesn't really fit that well into the Qrisp
    programming model, which shields the user from handling individual circuits
    and automatically decodes the measurement results into human readable labels.

    In order to bridge these worlds and still allow automatic decoding, the
    ``BatchedBackend`` allows Qrisp users to evaluate measurements from a
    multi-threading perspective. The idea is here that the circuit
    batch is collected through several threads, which each execute Qrisp code
    until a individual backend call is required. This backend call is then saved
    until the batch is complete. The batch can then be sent through the ``.dispatch``
    method, which resumes each thread to execute the post-processing logic.

    .. note::

        Calling the ``.run`` method of a BatchedBackend from the
        main thread will automatically dispatch all queries
        (including the query set up by the main thread).


    Parameters
    ----------
    batch_run_func : function
        A function that recieves a list of tuples in the form
        list[tuple[QuantumCircuit, int]], which represents the quantum circuits
        and the corresponding shots to execute on the backend. It should return
        a list of dictionaries, where each dictionary corresponds to the measurement
        results of the appropriate backend call.

    Examples
    --------

    We set up a BatchedBackend, which sequentially executes the QuantumCircuits
    on the Qrisp simulator.

    ::

        from qrisp import *
        from qrisp.interface import BatchedBackend

        def run_func_batch(batch):
            # Parameters
            # ----------
            # batch : list[tuple[QuantumCircuit, int]]
            #     The circuit and shot batch indicating the backend queries.

            # Returns
            # -------
            # results : list[dict[str, int]]
            #     The list of results.

            results = []
            for qc, shots in batch:
                results.append(qc.run(shots=shots))

            return results

        # Set up batched backend
        bb = BatchedBackend(run_func_batch)

    Create some backend calls

    ::

        a = QuantumFloat(4)
        b = QuantumFloat(3)
        a[:] = 1
        b[:] = 2
        c = a + b

        d = QuantumFloat(4)
        e = QuantumFloat(3)
        d[:] = 2
        e[:] = 3
        f = d + e

    Create threads

    ::

        import threading

        results = []
        def eval_measurement(qv):
            results.append(qv.get_measurement(backend = bb))

        thread_0 = threading.Thread(target = eval_measurement, args = (c,))
        thread_1 = threading.Thread(target = eval_measurement, args = (f,))

    Start the threads and subsequently dispatch the batch.

    ::

        # Start the threads
        thread_0.start()
        thread_1.start()

        # Call the dispatch routine
        # The min_calls keyword will make it wait
        # until the batch has a size of 2
        bb.dispatch(min_calls = 2)

        # Wait for the threads to join
        thread_0.join()
        thread_1.join()

        # Inspect the results
        print(results)

    This is automated by the :meth:`batched_measurement <qrisp.batched_measurement>`:

    >>> batched_measurement([c,f], backend=bb)
    [{3: 1.0}, {5: 1.0}]

    """

    def __init__(self, batch_run_func):

        # The function to call the backend
        self.batch_run_func = batch_run_func

        # A list[tuple[QuantumCircuit, int]] representing the quantum circuits and
        # the shots of the batch
        self.batch = []

        # This attribute tracks if the backend evaluation concluded. Having
        # this attribute is important because it facilitates the communication
        # of threaded execution model
        self.results_available = False

        # A dictionary of the form dict[QuantumCircuit,dict[str, int]] indicating
        # which QuantumCircuit gave which results
        self.results = {}

        # This attributes stores any potential exception that might have occured
        # during the backed evaluation and transmits them to the main thread
        # to be properly raised.
        self.backend_exception = None

    def run(self, qc, shots):

        # Appends the circuit-shot tuple
        self.batch.append((qc, shots))

        # If the run function is called from the main thread, the backend is evaluated
        # immediately. This makes sure that users who are not interested in batched
        # execution can still use the backend like an unbatched backend.
        if threading.current_thread() is threading.main_thread():
            dispatching_thread = threading.Thread(target=self.dispatch)
            dispatching_thread.start()

        # Wait for the results to be available
        while not self.results_available:
            time.sleep(0.01)

        # Raise any potential execption
        if self.backend_exception is not None:
            temp = self.backend_exception
            self.backend_exception = None
            raise temp

        result = self.results[id(qc)]
        del self.results[id(qc)]

        if threading.current_thread() is threading.main_thread():
            dispatching_thread.join()

        return result

    def dispatch(self, min_calls=0):
        """
        This method dispatches all collected queries and
        subsequently resumes their threads.

        Parameters
        ----------
        min_calls : int, optional
            If specified, the dispatch will be delayed until that
            many queries have been collected. The default is 0.

        """

        while len(self.batch) < min_calls:
            time.sleep(0.01)

        # We now perform the backend call and catch potential exceptions
        try:
            run_func_results = self.batch_run_func(self.batch)
            self.results = {
                id(self.batch[i][0]): run_func_results[i]
                for i in range(len(self.batch))
            }
        except Exception as e:
            if threading.current_thread() is threading.main_thread():
                raise e
            self.backend_exception = e

        self.batch = []
        self.results_available = True

        while len(self.results) or not self.backend_exception is None:
            time.sleep(0.01)

        self.results_available = False
