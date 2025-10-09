"""
********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp import QuantumArray, mcp, conjugate, invert
from qrisp.jasp import q_fori_loop, q_cond, check_for_tracing_mode
from jax import lax
import sympy as sp
import numpy as np
import jax.numpy as jnp


def QITE(qarg, U_0, exp_H, s, k, method="GC"):
    r"""
    Performs `Double-Braket Quantum Imaginary-Time Evolution (DB-QITE) <https://arxiv.org/abs/2412.04554>`_.
    Given a Hamiltonian :ref:`Operator <Operators>` $H$, this method implements the unitary $U_k$ that is recursively defined by either of

    * Group commutator (GQ) approximation:

    .. math::

        U_{k+1} = e^{i\sqrt{s_k}H}e^{i\sqrt{s_k}\omega_k}e^{-i\sqrt{s_k}H}U_k

    * Higher-order product formula (HOPF) approximation:

    .. math::

        U_{k+1} = e^{i\phi\sqrt{s_k}H}e^{i\phi\sqrt{s_k}\omega_k}e^{-i\sqrt{s_k}H}e^{-i(1+\phi)\sqrt{s_k}\omega_k}e^{i(1-\phi)\sqrt{s_k}H}U_k

    where $e^{it\omega_k}=U_ke^{it\ket{0}\bra{0}}U_k^{\dagger}$ is the refection around the state $\ket{\omega_k}=U_k\ket{0}$.



    Parameters
    ----------
    qarg : :ref:`QuantumVariable` or :ref:`QuantumArray`
        The quantum argument on which quantum imaginary time evolution is performed.
    U_0 : function
        A Python function that takes a QuantumVariable or QuantumArray ``qarg`` as input, and prepares the initial state.
    exp_H : function
        A Python function that takes a QuantumVariable or QuantumArray ``qarg`` and time ``t`` as input, and performs forward evolution $e^{-itH}$.
    s : list[float] or list[Sympy.Symbol]
        A list of evolution times for each step.
    k : int
        The number of steps.
    method : str, optional
        The method for approximating the double-bracket flow (DBF). Available are ``GC`` and ``HOPF``.
        The default is ``GC``.


    Examples
    --------

    We utilize QITE to approximate the ground state energy of a Heisenberg chain. We start by defining the lattice graph $G$:

    ::

        import networkx as nx

        # Create a graph
        N = 4
        G = nx.Graph()
        G.add_edges_from([(k,k+1) for k in range(N-1)])

    Next, we set up the Heisenberg Hamiltonian and calculate the ground state energy classically:

    ::

        from qrisp.operators import X, Y, Z

        def create_heisenberg_hamiltonian(G):
            H = sum(X(i)*X(j)+Y(i)*Y(j)+Z(i)*Z(j) for (i,j) in G.edges())
            return H

        H = create_heisenberg_hamiltonian(G)
        print(H)
        print(H.ground_state_energy())


    As explained :ref:`in this example <VQEHeisenberg>`, a suitable initial approximation for the ground state is given by a tensor product of singlet states $\frac{1}{\sqrt{2}}(\ket{10}-\ket{01})$ corresponding to a maximal matching of the graph $G$.
    Accordingly, we define the function ``U_0``:

    ::

        from qrisp import QuantumVariable
        from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function

        M = nx.maximal_matching(G)
        U_0 = create_heisenberg_init_function(M)

        def state_prep():
            qv = QuantumVariable(N)
            U_0(qv)
            return qv

        E_0 = H.expectation_value(state_prep)()
        print(E_0)

    For the function ``exp_H`` that performs forward evolution $e^{-itH}$, we use the :meth:`trotterization <qrisp.operators.qubit.QubitOperator.trotterization>` method with 5 Trotter steps:

    ::

        def exp_H(qv, t):
            H.trotterization(method='commuting')(qv,t,5)

    With all the necessary ingredients, we use QITE to approximate the ground state:

    ::

        import numpy as np
        import sympy as sp
        from qrisp.qite import QITE

        steps = 4
        s_values = np.linspace(.01,.3,10)

        theta = sp.Symbol('theta')
        optimal_s = [theta]
        optimal_energies = [E_0]

        for k in range(1,steps+1):

            # Perform k steps of QITE
            def state_prep():
                qv = QuantumVariable(N)
                QITE(qv, U_0, exp_H, optimal_s, k)
                return qv

            qv = state_prep()
            qc = qv.qs.compile()

            # Find optimal evolution time
            # Use "precompliled_qc" keyword argument to avoid repeated compilation of the QITE circuit
            energies = [H.expectation_value(state_prep, diagonalisation_method='commuting', subs_dic={theta:s_}, precompiled_qc=qc)() for s_ in s_values]

            index = np.argmin(energies)
            s_min = s_values[index]

            optimal_s.insert(-1,s_min)
            optimal_energies.append(energies[index])

        print(optimal_energies)

    Finally, we visualize the results:

    ::

        import matplotlib.pyplot as plt

        evolution_times = [sum(optimal_s[i] for i in range(k)) for k in range(steps+1)]

        plt.xlabel('Evolution time', fontsize=15, color='#444444')
        plt.ylabel('Energy', fontsize=15, color='#444444')
        plt.axhline(y=H.ground_state_energy(), color='#6929C4', linestyle='--', linewidth=2, label='Exact energy')
        plt.plot(evolution_times, optimal_energies, c='#20306f', marker="o", linestyle='solid', linewidth=3, zorder=3, label='DB-QITE')
        plt.legend(fontsize=12, labelcolor='linecolor')
        plt.tick_params(axis='both', labelsize=12)
        plt.grid()
        plt.show()

    .. figure:: /_static/heisenberg_qite.png
        :scale: 80%
        :align: center

    """
    if not check_for_tracing_mode():

        if k == 0:
            U_0(qarg)
        else:
            s_ = sp.sqrt(s[k - 1])

            def conjugator(qarg):
                with invert():
                    QITE(qarg, U_0, exp_H, s, k - 1, method=method)

            def reflection(qarg, t_):
                with conjugate(conjugator)(qarg):
                    if isinstance(qarg,QuantumArray):
                        qubits = sum([qv.reg for qv in qarg.flatten()], [])
                        mcp(t_, qubits, ctrl_state=0, method="khattar")
                    else:
                        mcp(t_, qarg, ctrl_state=0, method="khattar")

            if method == "GC":

                QITE(qarg, U_0, exp_H, s, k - 1, method=method)

                with conjugate(exp_H)(qarg, s_):
                    reflection(qarg, s_)

            if method == "HOPF":

                phi = (sp.sqrt(5) - 1) / 2

                QITE(qarg, U_0, exp_H, s, k - 1, method=method)

                # exp_H performs forward evolution $e^{-itH}
                exp_H(qarg, -(1 - phi) * s_)
                reflection(qarg, -(1 + phi) * s_)
                exp_H(qarg, s_)
                reflection(qarg, phi * s_)
                exp_H(qarg, -phi * s_)

    else:

        """
        To create a jasp-compatible implementation of QITE, we need to remove the recursive structure.
        We achieve this by fully expanding the recursive formula for $U_k$ down to the $k=0$ level.
        From there, we find a tree structure with branching factor 3 (GC) or 5 (HOPF) where
        some branches are inverted due to the presence of conjugate operators $U_i^\dagger$.
        We traverse the tree depth-first using up-, down-, bounce-, and leaf-operations that we obtain from
        inspecting the formula for $U_k$.
        """

        def int_to_base(n, base=3, max_digits=10):
            """
            Get the array representation of an integer `n` with base `base`.
            The array has length `max_digits` and the least significant digit is at index `0`.
            """

            def cond_fun(state):
                n, digits, i = state
                return jnp.logical_and(n > 0, i < max_digits)

            def body_fun(state):
                n, digits, i = state
                digits = digits.at[i].set(n % base)
                n = n // base
                return n, digits, i + 1

            init_digits = jnp.zeros((max_digits,), dtype=jnp.int32)
            _, digits, _ = lax.while_loop(cond_fun, body_fun, (n, init_digits, 0))
            return digits

        # Define basic operations
        def U_0_dag(q_arg):
            with invert():
                U_0(q_arg)

        def exp_00(q_arg, time):
            mcp(time, q_arg, ctrl_state=0)

        if method == "GC":

            def body_fun(i, val):
                qarg = val

                # Obtain old and new position
                old_pos = int_to_base(i, 3)
                new_pos = int_to_base(i + 1, 3)
                # Obtain largest changed index + 1
                num_changes = jnp.count_nonzero(new_pos != old_pos)

                # Compute which operations must be inverted
                inv_mode_leaf = jnp.equal(jnp.count_nonzero(old_pos == 1) % 2, 0)
                inv_mode_up = inv_mode_leaf
                inv_mode_bounce = jnp.logical_xor(
                    inv_mode_up, old_pos[num_changes - 1] == 1
                )
                inv_mode_down = jnp.equal(jnp.count_nonzero(new_pos == 1) % 2, 0)

                # Apply U_0
                q_cond(inv_mode_leaf, U_0, U_0_dag, qarg)

                # Go up the branch
                time = q_fori_loop(
                    0, num_changes - 1, lambda j, time: time + jnp.sqrt(s[j]), 0
                )
                q_cond(inv_mode_up, exp_H, lambda a, b: None, qarg, -time)

                # Bounce to next branch
                q_cond(
                    jnp.logical_and(old_pos[num_changes - 1] == 0, inv_mode_bounce),
                    exp_H,
                    lambda a, b: None,
                    qarg,
                    jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(old_pos[num_changes - 1] == 1, inv_mode_bounce),
                    exp_00,
                    lambda a, b: None,
                    qarg,
                    jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(
                        old_pos[num_changes - 1] == 0, jnp.logical_not(inv_mode_bounce)
                    ),
                    exp_00,
                    lambda a, b: None,
                    qarg,
                    -jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(
                        old_pos[num_changes - 1] == 1, jnp.logical_not(inv_mode_bounce)
                    ),
                    exp_H,
                    lambda a, b: None,
                    qarg,
                    -jnp.sqrt(s[num_changes - 1]),
                )

                # Go down to leaf
                q_cond(inv_mode_down, lambda a, b: None, exp_H, qarg, time)

                return qarg

            # Iterate all leafs except last
            q_fori_loop(0, 3**k - 1, body_fun, qarg)

            # Do last leaf
            U_0(qarg)
            time = lax.fori_loop(0, k, lambda j, time: time + jnp.sqrt(s[j]), 0)
            exp_H(qarg, -time)

        if method == "HOPF":

            phi = (jnp.sqrt(5) - 1) / 2

            def body_fun(i, val):
                qarg = val

                # Obtain old and new position
                old_pos = int_to_base(i, 5)
                new_pos = int_to_base(i + 1, 5)
                # Obtain largest changed index + 1
                num_changes = jnp.count_nonzero(new_pos != old_pos)

                # Compute which operations must be inverted
                inv_mode_leaf = (
                    jnp.count_nonzero(old_pos == 1) + jnp.count_nonzero(old_pos == 3)
                ) % 2 == 0
                inv_mode_up = inv_mode_leaf
                inv_mode_bounce = jnp.logical_xor(
                    inv_mode_up,
                    jnp.logical_or(
                        old_pos[num_changes - 1] == 1, old_pos[num_changes - 1] == 3
                    ),
                )
                inv_mode_down = (
                    jnp.count_nonzero(new_pos == 1) + jnp.count_nonzero(new_pos == 3)
                ) % 2 == 0

                # Apply U_0
                q_cond(inv_mode_leaf, U_0, U_0_dag, qarg)

                # Go up the branch
                time = (
                    q_fori_loop(
                        0, num_changes - 1, lambda j, time: time + jnp.sqrt(s[j]), 0
                    )
                    * phi
                )
                q_cond(inv_mode_up, exp_H, lambda a, b: None, qarg, -time)

                # Bounce to next branch
                q_cond(
                    jnp.logical_and(old_pos[num_changes - 1] == 0, inv_mode_bounce),
                    exp_H,
                    lambda a, b: None,
                    qarg,
                    -(1 - phi) * jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(old_pos[num_changes - 1] == 1, inv_mode_bounce),
                    exp_00,
                    lambda a, b: None,
                    qarg,
                    -(1 + phi) * jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(old_pos[num_changes - 1] == 2, inv_mode_bounce),
                    exp_H,
                    lambda a, b: None,
                    qarg,
                    jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(old_pos[num_changes - 1] == 3, inv_mode_bounce),
                    exp_00,
                    lambda a, b: None,
                    qarg,
                    phi * jnp.sqrt(s[num_changes - 1]),
                )

                q_cond(
                    jnp.logical_and(
                        old_pos[num_changes - 1] == 3, jnp.logical_not(inv_mode_bounce)
                    ),
                    exp_H,
                    lambda a, b: None,
                    qarg,
                    (1 - phi) * jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(
                        old_pos[num_changes - 1] == 2, jnp.logical_not(inv_mode_bounce)
                    ),
                    exp_00,
                    lambda a, b: None,
                    qarg,
                    (1 + phi) * jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(
                        old_pos[num_changes - 1] == 1, jnp.logical_not(inv_mode_bounce)
                    ),
                    exp_H,
                    lambda a, b: None,
                    qarg,
                    -jnp.sqrt(s[num_changes - 1]),
                )
                q_cond(
                    jnp.logical_and(
                        old_pos[num_changes - 1] == 0, jnp.logical_not(inv_mode_bounce)
                    ),
                    exp_00,
                    lambda a, b: None,
                    qarg,
                    -phi * jnp.sqrt(s[num_changes - 1]),
                )

                # Go down to leaf
                q_cond(inv_mode_down, lambda a, b: None, exp_H, qarg, time)

                return qarg

            # Iterate all leafs except last
            q_fori_loop(0, 5**k - 1, body_fun, qarg)

            # Do last leaf
            U_0(qarg)
            time = -phi * lax.fori_loop(0, k, lambda j, time: time + jnp.sqrt(s[j]), 0)
            exp_H(qarg, time)
