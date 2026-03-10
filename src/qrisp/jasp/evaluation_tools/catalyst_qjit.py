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

from jax.tree_util import tree_unflatten
from qrisp.jasp.jasp_expression import make_jaspr


def qjit(function=None, device=None):
    """
    Decorator to leverage the jasp + Catalyst infrastructure to compile the given
    function to QIR and run it on the Catalyst QIR runtime.

    Parameters
    ----------
    function : callable
        A function performing Qrisp code.
    device : object
        The `PennyLane device <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/devices.html>`_ to execute the function.
        The default device is `"lightning.qubit" <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`_,
        a fast state-vector qubit simulator.

    Returns
    -------
    callable
        A function executing the compiled code.

    Notes
    -----

    Lightning-GPU is compatible with systems featuring NVIDIA Volta (SM 7.0) GPUs or newer.
    It is specifically optimized for Linux environments on X86-64 or ARM64 architectures running CUDA-12.

    To install Lightning-GPU with NVIDIA CUDA support, the following packages need to be installed

    ::

        pip install custatevec_cu12
        pip install pennylane-lightning-gpu


    Pre-built wheels for Lightning-AMDGPU are available for AMD MI300 series GPUs and systems running ROCm 7.0 or newer.

    ::

        pip install pennylane-lightning-amdgpu

    If the setup uses an older version of ROCm or a different AMD GPU series, Lightning-AMDGPU must be built manually from source.

    Installation instructions for different platforms are available at `pennylane.ai/install <https://pennylane.ai/install#high-performance-computing-and-gpus>`_.

    Examples
    --------

    We write a simple function using the QuantumFloat quantum type and execute
    via ``qjit``:

    ::

        from qrisp import *
        from qrisp.jasp import qjit

        @qjit
        def test_fun(i):
            qv = QuantumFloat(i, -2)
            with invert():
                cx(qv[0], qv[qv.size-1])
                h(qv[0])
            meas_res = measure(qv)
            return meas_res + 3


    We execute the function a couple of times to demonstrate the randomness

    >>> test_fun(4)
    [array(5.25, dtype=float64)]
    >>> test_fun(5)
    [array(3., dtype=float64)]
    >>> test_fun(5)
    [array(7.25, dtype=float64)]


    For executing on "lightning.gpu" we specify the device:

    ::

        import pennylane as qml
        from qrisp import *
        from qrisp.jasp import qjit

        dev = qml.device("lightning.gpu", wires=0)

        @qjit(device=dev)
        def test_fun(i):
            qv = QuantumFloat(i, -2)
            with invert():
                cx(qv[0], qv[qv.size-1])
                h(qv[0])
            meas_res = measure(qv)
            return meas_res + 3

    """

    if function is None:
        return lambda x: qjit(x, device=device)

    def jitted_function(*args):

        if not hasattr(function, "jaspr_dict"):
            function.jaspr_dict = {}

        args = list(args)

        signature = tuple([type(arg) for arg in args])
        if not signature in function.jaspr_dict:
            # Use return_shape=True to capture the output PyTree structure
            jaspr, out_tree = make_jaspr(function, return_shape=True)(*args)
            function.jaspr_dict[signature] = (jaspr, out_tree)

        jaspr, out_tree = function.jaspr_dict[signature]
        result = jaspr.qjit(*args, function_name=function.__name__, device=device)

        # Reconstruct the PyTree structure from flat results
        if isinstance(result, (tuple, list)):
            return tree_unflatten(out_tree, result)
        elif result is not None:
            # Single value case
            return tree_unflatten(out_tree, [result])
        else:
            return None

    return jitted_function
