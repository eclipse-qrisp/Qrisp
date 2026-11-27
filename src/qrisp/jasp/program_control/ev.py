# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Implements expectation_value, estimating expectation values via repeated quantum kernel sampling."""

from qrisp.jasp.program_control._sampling_core import _sampling_procedure


def expectation_value(sampling_kernel, shots, post_processor=None):
    r"""Estimates the expectation value from a sampling kernel.

    The ``expectation_value`` function allows to estimate the expectation value
    from a *sampling kernel* — a Python function that receives only classical
    arguments and returns arbitrary values.  Any
    :ref:`QuantumVariables <QuantumVariable>` and
    :ref:`QuantumArrays <QuantumArray>` in the return are automatically
    measured and decoded; classical values are interleaved in-place.

    .. note::

        When used inside :func:`~qrisp.jasp.jaspify` with
        ``terminal_sampling=True``, the same restrictions apply as for
        :func:`~qrisp.jasp.sample`: kernels that return classical values are
        rejected, and kernels whose quantum state depends on mid-circuit
        measurement outcomes may produce invalid results.  Use
        ``terminal_sampling=False`` (the default) for those cases.  See
        :func:`~qrisp.jasp.terminal_sampling` for details.

    Parameters
    ----------
    sampling_kernel : callable
        A sampling kernel — a function receiving only classical arguments and
        returning one or more :ref:`QuantumVariables <QuantumVariable>`,
        :ref:`QuantumArrays <QuantumArray>`, classical measurement results,
        or a mixture of these, also nested inside ``tuple``, ``list`` and
        ``dict`` containers.
        The function must **not** receive quantum arguments because a quantum
        value would need to be copied for each sampling iteration, which is
        prohibited by the no-cloning theorem.
    shots : int or jax.core.Tracer
        The amount of samples to take to compute the expectation value.
    post_processor : callable, optional
        A classical Jax traceable function to apply to the results
        directly after measuring. By default no post processing is applied.

    Raises
    ------
    Exception
        Tried to sample from sampling kernel taking a quantum value

    Returns
    -------
    callable
        A function returning the expectation value.  For a kernel returning a
        single value the result is a scalar ``jax.Array``.  For a kernel
        returning a container (``tuple``, ``list``, ``dict``, or nested
        combinations thereof) the container structure is preserved and every
        leaf is replaced by its mean.  Since an expectation value is a mean,
        each leaf is floating point regardless of the kernel's own dtype
        (complex leaves stay complex) -- unlike :func:`~qrisp.jasp.sample`,
        which preserves the native dtype of every leaf.  A returned
        :ref:`QuantumArray` gives an array of element-wise means with the
        QuantumArray's shape.

    Examples
    --------
    We prepare the state

    .. math::

        \ket{\psi_k} = \frac{1}{\sqrt{2}} \left(\ket{0}\ket{0}\ket{\text{False}} + \ket{k}\ket{k}\ket{\text{True}}\right)

    ::

        from qrisp import *
        from qrisp.jasp import *


        def sampling_kernel(k):
            a = QuantumFloat(4)
            b = QuantumFloat(4)

            qbl = QuantumBool()
            h(qbl)

            with control(qbl[0]):
                a[:] = k

            cx(a, b)

            return a, b

    And compute the expectation value of the QuantumFloats

    ::

        @jaspify
        def main(k):

            ev_function = expectation_value(sampling_kernel, shots = 50)

            return ev_function(k)

        print(main(3))
        # Yields e.g.
        # (Array(1.44, dtype=float64), Array(1.44, dtype=float64))

    The true value 1.5 is not reached exactly because of `shot noise <https://en.wikipedia.org/wiki/Shot_noise>`_ —
    the printed value fluctuates around 1.5 from run to run. To improve the
    approximation, feel free to increase the shots!

    To demonstrate the ``post_processor`` keyword we define a simple post processing
    function

    ::

        def post_processor(x, y):
            return x*y

        @jaspify
        def main(k):

            ev_function = expectation_value(sampling_kernel, shots = 50,
                                             post_processor = post_processor)

            return ev_function(k)

        print(main(3))
        # Yields e.g.
        # 4.86

    This result is expected because the inputs of ``post_processor`` are
    either (0,0) or (3,3) with 50% probability, so the expectation value is

    .. math::

        4.5 = \frac{3\cdot 3 + 0\cdot 0}{2}

    As with the previous example, the printed value fluctuates around this
    expectation from run to run because of shot noise.

    """
    from qrisp.jasp import make_tracer

    if isinstance(shots, int):
        shots = make_tracer(shots)

    expectation_value_eval_function = _sampling_procedure(sampling_kernel, post_processor, "ev")

    def return_function(*args):
        return expectation_value_eval_function(*args, shots=shots)

    return return_function
