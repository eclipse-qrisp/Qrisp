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

"""Implements sample, drawing measurement samples from a quantum sampling kernel."""

import jax

from qrisp.jasp.program_control._sampling_core import _sampling_procedure
from qrisp.jasp.tracing_logic import check_for_tracing_mode


def sample(sampling_kernel=None, shots=0, post_processor=None):
    r"""Takes samples from a quantum computation specified by a sampling kernel.

    The ``sample`` function allows to take samples from a quantum computation
    specified by a *sampling kernel* — a Python function that receives only
    classical arguments and returns arbitrary values.  Any
    :ref:`QuantumVariables <QuantumVariable>` and
    :ref:`QuantumArrays <QuantumArray>` in the return are automatically
    measured and decoded.

    The samples are returned in the form of a
    `Jax Array <https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html>`_
    which is shaped according to the ``shots`` parameter. Because of this, shots
    can only be a **static integer** (no dynamic values!). If you want to sample
    with a dynamic shot amount, look into :ref:`expectation_value`.

    Sample calls can be efficiently simulated via terminal sampling by setting the
    corresponding keyword within :func:`~qrisp.jasp.jaspify` to ``True``.

    .. note::

        **Terminal sampling** (``terminal_sampling=True`` inside
        :func:`~qrisp.jasp.jaspify`) does **not** support sampling kernels
        that return classical values alongside quantum variables, or kernels
        that return *only* classical values.  Use ``terminal_sampling=False``
        for those cases.

        Even when the kernel returns only
        :ref:`QuantumVariables <QuantumVariable>`, terminal sampling relies on
        the quantum state being **independent** of mid-circuit measurement
        outcomes.  If a classical measurement result influences the quantum
        circuit (e.g. via :func:`control <qrisp.control>`), terminal sampling
        may produce an invalid distribution because it simulates the quantum
        part only once.  See :func:`~qrisp.jasp.terminal_sampling` for details
        and an example.

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
    shots : int
        The amounts of samples to take.
    post_processor : callable, optional
        A function to apply to the samples directly after measuring. By default
        no post processing is applied.

    Raises
    ------
    Exception
        Tried to sample with dynamic shots value (static integer required)
    Exception
        Tried to sample from sampling kernel taking a quantum value
    Exception
        Tried to use terminal sampling with a kernel that returns classical
        values (use ``terminal_sampling=False`` instead)

    Returns
    -------
    callable
        A classical, Jax traceable function.  The result mirrors the
        structure of the kernel's return value, with every leaf replaced by
        an array of shape ``(shots, *leaf_shape)`` that keeps the leaf's
        native dtype.  A scalar leaf therefore gives a 1D array of length
        ``shots``, and a ``(3,)`` leaf gives a ``(shots, 3)`` array.
        Containers (``tuple``, ``list``, ``dict``, and nested combinations
        thereof) are preserved — e.g. ``{'a': bool_array, 'b': float_array}``.
        A returned :ref:`QuantumArray` is decoded into an array of its own
        shape, so it gives a leaf of shape ``(shots, *qa.shape)``.

    Examples
    --------
    We prepare the state

    .. math::

        \ket{\psi} = \frac{1}{\sqrt{2}} \left(\ket{0}\ket{0}\ket{\text{True}} + \ket{k}\ket{k}\ket{\text{True}})\right)

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

    And subsequently sample from the QuantumFloats:

    ::

        @jaspify
        def main(k):

            sampling_function = sample(sampling_kernel,
                                       shots = 10)

            return sampling_function(k)

        print(main(3))

        # Yields e.g. (a tuple of two 1D arrays)
        # (Array([3., 0., 0., 3., 0., 0., 3., 3., 0., 0.], dtype=float64),
        #  Array([3., 0., 0., 3., 0., 0., 3., 3., 0., 0.], dtype=float64))
        #
        # Both arrays agree entry by entry: every shot gives either 0 or 3
        # with 50% probability each. The exact order of the entries is random
        # and varies between runs.

    To demonstrate the post processing feature, we write a simple post
    processing function:

    ::

        def post_processor(x, y):
            return 2*x + y//2

        @jaspify
        def main(k):

            sampling_function = sample(sampling_kernel,
                                       shots = 10,
                                       post_processor = post_processor)

            return sampling_function(k)

        print(main(4))
        # Yields e.g.
        # [10. 10.  0.  0.  0.  0.  0.  0. 10. 10.]
        #
        # Each entry is either 0 or 10 with 50% probability each. The exact
        # order of entries is random and varies between runs.

    **Sampling kernels returning QuantumArrays**

    A :ref:`QuantumArray` is measured element-wise.  Each sample is an array
    of the QuantumArray's shape, and a post-processor receives it as such:

    ::

        def array_kernel():
            qa = QuantumArray(QuantumFloat(2), shape=(2,))
            h(qa)
            return qa

        @jaspify
        def main():
            return sample(array_kernel, shots=5)()

        print(main())
        # Yields e.g. (an array of shape (5, 2))
        # [[1. 3.]
        #  [0. 2.]
        #  [3. 3.]
        #  [2. 0.]
        #  [1. 1.]]

    **Sampling kernels returning classical values**

    A sampling kernel may also return classical values from mid-circuit
    measurements alongside (or instead of) quantum variables:

    ::

        def mixed_kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            h(qf[1])
            mes = measure(qf[1])      # classical measurement result
            return qf, mes            # mixed: quantum + classical

        @jaspify
        def main():
            return sample(mixed_kernel, shots=20)()

        print(main())
        # Yields e.g. (a tuple of two 1D arrays):
        # (Array([0., 0., 1., ...], dtype=float64),
        #  Array([0., 0., 0., ...], dtype=float64))

    .. note::

        The above example uses ``@jaspify`` (which defaults to
        ``terminal_sampling=False``).  Using ``@jaspify(terminal_sampling=True)``
        with a mixed-returns kernel will raise an error.

    """
    if isinstance(sampling_kernel, int):
        shots = sampling_kernel
        sampling_kernel = None

    if sampling_kernel is None:
        return lambda x: sample(x, shots, post_processor=post_processor)

    if isinstance(shots, jax.core.Tracer):
        raise Exception("Tried to sample with dynamic shots value (static integer required)")
    elif not isinstance(shots, int):
        raise Exception(f"Tried to sample with shots value of non-integer type {type(shots)}")

    sampling_eval_function = _sampling_procedure(sampling_kernel, post_processor, "sample", static_shots=shots)

    def return_function(*args):

        if check_for_tracing_mode():
            # Under tracing the sampling loop writes one row per shot, so a
            # non-positive count has no meaningful result and would otherwise
            # fail obscurely while indexing the zero-length accumulator.
            # Outside tracing mode shots=0 is legitimate -- it selects the exact
            # probabilities via terminal sampling -- so the branch below is
            # deliberately left unguarded.
            if shots < 1:
                raise ValueError(
                    f"Tried to sample with a shots value of {shots} inside a traced function "
                    "(at least one shot is required). A shot count of 0 returns exact "
                    "probabilities, which is only available outside of tracing mode."
                )
            return sampling_eval_function(*args, shots=shots)
        else:
            from qrisp.jasp.evaluation_tools.terminal_sampling import _terminal_sampling

            return _terminal_sampling(sampling_kernel, shots, post_processor)(*args)

    return return_function
