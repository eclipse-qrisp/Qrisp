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

import jax.numpy as jnp
import numpy as np

from qrisp.jasp.evaluation_tools.buffered_quantum_state import BufferedQuantumState
from qrisp.jasp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues
from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.tracing_logic import qache


def terminal_sampling(func=None, shots=0):
    """
    The ``terminal_sampling`` decorator performs a hybrid simulation and afterwards
    samples from the resulting quantum state.
    The idea behind this function is that it is very cheap for a classical simulator
    to sample from a given quantum state without simulating the whole state from
    scratch. For quantum simulators that simulate pure quantum computations
    (i.e. no classical steps) this is very established and usually achieved through
    a "shots" keyword. For hybrid simulators (like Jasp) it is not so straightforward
    because mid-circuit measurements can alter the classical computation.

    In general, generating N samples from a hybrid program requires N executions
    of said programm. If it is however known that the quantum state is the same
    regardless of mid-circuit measurement outcomes, we can use the terminal sampling
    function. If this condition is not met, the ``terminal_sampling`` function
    will not return a valid distribution. A demonstration for this is given in the
    examples section.

    To use the terminal sampling decorator, a Jasp-compatible function returning
    some QuantumVariables has to be given as a parameter.

    Parameters
    ----------
    func : callable
        A Jasp compatible function returning QuantumVariables.
    shots : int, optional
        An integer specifying the amount of shots. The default is None, which
        will result in probabilities being returned.

    Returns
    -------
    callable
        A function that returns a dictionary of measurement results similar to
        :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`.

    Examples
    --------

    We sample from a :ref:`QuantumFloat` that has been brought in a superposition.

    ::

        from qrisp import QuantumFloat, QuantumBool, h, cx
        from qrisp.jasp import terminal_sampling

        @terminal_sampling(shots = 1000)
        def main(i):
            qf = QuantumFloat(8)
            qbl = QuantumBool()
            h(qf[i])
            cx(qf[i], qbl[0])
            return qf, qbl

        sampling_function = terminal_sampling(main, shots = 1000)

        print(main(0))
        print(main(1))
        print(main(2))

        # Yields:
        {(1.0, True): 526, (0.0, False): 474}
        {(2.0, True): 503, (0.0, False): 497}
        {(4.0, True): 502, (0.0, False): 498}

    **Example of invalid use**

    In this example we demonstrate a hybrid program that can not be properly sample
    via ``terminal_sampling``. The key ingredient here is a realtime component.

    ::

        from qrisp import QuantumBool, measure, control

        @terminal_sampling
        def main():

            qbl = QuantumBool()
            qf = QuantumFloat(4)

            # Bring qbl into superposition
            h(qbl)

            # Perform a measure
            cl_bl = measure(qbl)

            # Perform a conditional operation based on the measurement outcome
            with control(cl_bl):
                qf[:] = 1
                h(qf[2])

            return qf

        print(main())
        # Yields either {0.0: 1.0} or {1.0: 0.5, 5.0: 0.5} (with a 50/50 probability)

    The problem here is the fact that the distribution of the returned QuantumFloat
    is depending on the measurement outcome of the :ref:`QuantumBool`. The
    ``terminal_sampling`` function performs this simulation (including the measurement)
    only once and simply samples from the final distribution.
    """

    if isinstance(func, int):
        shots = func
        func = None

    if func is None:
        return lambda x: terminal_sampling(x, shots)

    def tracing_function(*args):
        from qrisp.jasp.program_control import expectation_value

        return expectation_value(func, shots, return_dict=True)(*args)

    def return_function(*args):
        from qrisp.jasp import simulate_jaspr

        jaspr = make_jaspr(tracing_function, garbage_collection=True)(*args)
        return simulate_jaspr(jaspr, *args, terminal_sampling=True)

    return return_function
