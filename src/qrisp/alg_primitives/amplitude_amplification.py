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

from collections.abc import Callable, Sequence
from typing import Any

from qrisp.alg_primitives.reflection import reflection
from qrisp.core import QuantumVariable, QuantumArray, merge, recursive_qs_search
from qrisp.environments import IterationEnvironment
from qrisp.jasp import check_for_tracing_mode, jrange


def amplitude_amplification(
    args: QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray],
    state_function: Callable,
    oracle_function: Callable,
    kwargs_oracle: dict[str, Any] | None = None,
    iter: int = 1,
    reflection_indices: list[int] | None = None,
) -> None:
    r"""
    This method performs `quantum amplitude amplification <https://arxiv.org/abs/quant-ph/0005055>`_.

    The problem of quantum amplitude amplification is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}`.
    * Write :math:`\ket{\Psi}=\ket{\Psi_1}+\ket{\Psi_0}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Enhance the probability :math:`a=\langle\Psi_1|\Psi_1\rangle` that a measurement of :math:`\ket{\Psi}` yields a good state.

    Let :math:`\theta_a\in [0,\pi/2]` such that :math:`\sin^2(\theta_a)=a`. Then the amplitude amplification operator :math:`\mathcal Q` acts as

    .. math::

        \mathcal Q^j\ket{\Psi}=\frac{1}{\sqrt{a}}\sin((2j+1)\theta_a)\ket{\Psi_1}+\frac{1}{\sqrt{1-a}}\cos((2j+1)\theta_a)\ket{\Psi_0}.

    Therefore, after :math:`m` iterations the probability of measuring a good state is :math:`\sin^2((2m+1)\theta_a)`.

    Parameters
    ----------
    args : QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray]
        The quantum variable, array, or collection thereof on which amplitude amplification 
        is performed. These variables must already be prepared in the initial state 
        :math:`\ket{\Psi}` before calling this method (i.e., the user is responsible 
        for applying the ``state_function`` to the zero state prior to execution).
    state_function : Callable
        A Python function preparing the state :math:`\ket{\Psi}` from the zero state.
        The required signature of this function depends on the input ``args``: 

        - if ``args`` is a single variable or array, it receives that single object. 
        - if ``args`` is a list, the elements are unpacked and passed as separate 
          positional arguments (e.g., for ``args=[qv1, qv2]``, the signature 
          must be ``state_function(qv1, qv2)``).

        Although ``args`` must already be in the state :math:`\ket{\Psi}` upon input, 
        this function is strictly required internally to construct the amplitude 
        amplification operator :math:`\mathcal{Q}` (specifically to perform the 
        reflection about the initial state).
    oracle_function : Callable
        A Python function tagging the good state :math:`\ket{\Psi_1}`.
        Like ``state_function``, its required signature matches the structure of ``args``:
        it takes a single argument if ``args`` is a single object, or unpacked 
        positional arguments if ``args`` is a list.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is None.
    iter : int, optional
        The exact amount of amplitude amplification iterations to perform. The default is 1.
    reflection_indices : list[int], optional
        A list of indices indicating with respect to which variables the reflection is performed, i.e.,
        `oblivious amplitude amplification <https://arxiv.org/pdf/1312.1414>`_ is performed.
        Indices correspond to the flattened ``args`` (e.g., if ``args = QuantumArray(QuantumFloat(3), (6,))``,
        ``reflection_indices=[0,1,2,3]`` corresponds to the first four variables in the array).
        By default, the reflection is performed with respect to all variables in ``args``
        (standard amplitude amplification).

    Examples
    --------

    We define a function that prepares the state :math:`\ket{\Psi}=\cos(\frac{\pi}{16})\ket{0}+\sin(\frac{\pi}{16})\ket{1}`
    and an oracle that tags the good state :math:`\ket{1}`. In this case, we have :math:`a=\sin^2(\frac{\pi}{16})\approx 0.19509`.

    ::

        from qrisp import z, ry, QuantumBool, amplitude_amplification
        import numpy as np

        def state_function(qb):
            ry(np.pi/8,qb)

        def oracle_function(qb):
            z(qb)

        qb = QuantumBool()

        state_function(qb)

    >>> qb.qs.statevector(decimals=5)
    0.98079∣False⟩+0.19509∣True⟩

    We can enhance the probability of measuring the good state with amplitude amplification:

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.83147*|False> + 0.55557*|True>

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.55557*|False> + 0.83147*|True>

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.19509*|False> + 0.98079*|True>

    """


    if kwargs_oracle is None:
        kwargs_oracle = {}

    if isinstance(args, (QuantumVariable, QuantumArray)):
        args = [args]

    if check_for_tracing_mode():
        for _ in jrange(iter):
            oracle_function(*args, **kwargs_oracle)
            reflection(
                args,
                state_function=state_function,
                reflection_indices=reflection_indices,
            )
    else:
        merge(args)
        qs = recursive_qs_search(args)[0]
        if iter > 0:
            with IterationEnvironment(qs, iter):
                oracle_function(*args, **kwargs_oracle)
                reflection(
                    args,
                    state_function=state_function,
                    reflection_indices=reflection_indices,
                )
