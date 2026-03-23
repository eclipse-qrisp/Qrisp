"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from jax.extend.core import ClosedJaxpr

from qrisp.jasp.primitives import (
    create_quantum_kernel_p,
    consume_quantum_kernel_p,
    AbstractQubit,
    AbstractQubitArray,
)
from qrisp.jasp.tracing_logic import TracingQuantumSession, qache, get_last_equation


def quantum_kernel(func):
    """
    This decorator allows you to annotate a subroutine as a "quantum kernel".
    Quantum kernels are functions that are restricted in the sense that they
    can not have quantum inputs or outputs, yet their inner working can be quantum.
    What is the benefit in that? The underlying idea why this
    can be helpful is that future execution environments might host several
    QPUs that can operate in parallel, much like many of todays HPC
    environments can access multiple GPUs.

    Annotating a function as a quantum kernel therefore allows the execution
    environment to identify the subroutine as a separate quantum state and can
    assign it to a dedicated QPU.


    .. note::

        While not many quantum algorithms exist that directly allow such a
        parallelization, any sampling task can be performed in this manner:
        If you need to execute a 1000 shots of a certain quantum circuit, but
        you have 4 QPUs available, you can execute the task 4 times faster by
        assigning 250 shots to each QPU.
        As such the :ref:`sample <sample>` and :ref:`expectation_value <expectation_value>` function
        automatically wraps the state preparation and measurement into a
        dedicated quantum kernel.


    Parameters
    ----------
    func : callable
        A function that receives only classical values as inputs and returns
        classical values as outpus. The function's body can however perform
        quantum logic.

    Returns
    -------
    quantum_kernel : callable
        A function that performs the task of the input but the compiler can
        identify it as a closed quantum procedure without any external entaglement.

    Examples
    --------

    We demonstrate a naive implementation of an expectation value please use
    :ref:`expectation_value` if you required this functionality. For this
    we define a state preparation procedure and call it from a kernelized
    function.

    ::

        from qrisp import *
        from qrisp.jasp import *

        def state_prep(k):
            qf = QuantumFloat(5)
            h(qf[k])
            return qf

        @quantum_kernel
        def sampling_kernel(k):
            # Receives a classical (!) integer k

            qf = state_prep(k)

            # Returns a classical integer
            return measure(qf)

    We now call the kernel within a purely classical Jax script.

    ::

        @jaspify
        def main(k):

            shots = 100

            res = 0
            for i in range(shots):
                res += sampling_kernel(k)

            return res/shots

    Perform some experiments:

    ::

        print(main(3))
        # Yields: 3.92
        # Expected: 2**3/2 = 4
        print(main(4))
        # Yields: 8.96
        # Expected: 2**4/2 = 8

    """

    func = qache(
        func,
    )

    def return_function(*args, **kwargs):

        from qrisp.jasp.jasp_expression.centerclass import Jaspr, collect_environments

        qs = TracingQuantumSession.get_instance()

        qs.start_tracing(create_quantum_kernel_p.bind())

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            qs.conclude_tracing()
            raise e

        eqn = get_last_equation()

        flattened_jaspr = Jaspr.from_cache(
            collect_environments(eqn.params["jaxpr"])
        ).flatten_environments()
        for var in flattened_jaspr.invars:
            if isinstance(var.aval, (AbstractQubitArray, AbstractQubit)):
                raise Exception("Tried to construct quantum kernel with quantum input")
        for var in flattened_jaspr.outvars:
            if isinstance(var.aval, (AbstractQubitArray, AbstractQubit)):
                raise Exception("Tried to construct quantum kernel with quantum output")

        eqn.params["jaxpr"] = flattened_jaspr

        abs_qst = qs.conclude_tracing()

        consume_quantum_kernel_p.bind(abs_qst)

        return res

    return return_function
