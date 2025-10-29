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

from qrisp import auto_uncompute, control, rx, rz, x


def qiro_rx_mixer(problem_updated):
    """
    RX-Mixer for QIRO algorithm. Works analogously to the normal RX-Mixer, but respects solutions and exclusions that have been found in the QIRO reduction steps.

    Parameters
    ----------
    solutions: List
        Solutions that have been found in the QIRO reduction steps.
    exclusions: List
        Solutions that have been found in the QIRO reduction steps.

    Returns
    -------
    RX_mixer: function
        The RX-mixer, according to the update steps that have been undertaken

    """

    solutions = problem_updated[1]
    exclusions = problem_updated[2]
    union = solutions + exclusions

    def RX_mixer(qv, beta):

        for i in range(len(qv)):
            if not i in union:
                rx(2 * beta, qv[i])

    return RX_mixer


def qiro_rz_mixer(problem_updated):
    """
    This function applies an RZ gate with a negative phase shift to a given quantum variable.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable to which the RZ gate is applied.
    beta : float or sympy.Symbol
        The phase shift value for the RZ gate.

    """

    solutions = problem_updated[1]
    exclusions = problem_updated[2]
    union = solutions + exclusions

    def RZ_mixer(qv, beta):

        for i in range(len(qv)):
            if not i in union:
                rz(2 * beta, qv[i])

    return RZ_mixer


def qiro_controlled_RX_mixer_gen(predicate, union):
    r"""
    For a QIRO MIS instances, generate a controlled RX mixer for a given predicate function.

    Parameters
    ----------
    predicate : function
        A function receiving a ``QuantumVariable`` and an index $i$.
        This function returns a ``QuantumBool`` indicating if the predicate is satisfied for ``qv[i]``,
        that is, if the element ``qv[i]`` should be swapped in.
    union : List
        List of Qubits which were found to be positively or negatively correlated, i.e. they should not be mixed again.


    Returns
    -------
    controlled_RX_mixer : function
        A function receiving a ``QuantumVariable`` and a real parameter $\beta$.
        This function performs the application of the mixing operator.

    Examples
    --------

    We define the predicate function for the :ref:`MaxIndepSet <maxIndepSetQAOA>` problem. It returns ``True`` for the index (node) $i$ if
    all neighbors $j$ of the node $i$ in the graph $G$ are not selected, and ``False`` otherwise.

    ::

        from qrisp import QuantumVariable, QuantumBool, h, mcx, auto_uncompute, multi_measurement
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        neighbors_dict = {node: list(G.adj[node]) for node in G.nodes()}

        def predicate(qv,i):
            qbl = QuantumBool()
            if len(neighbors_dict[i])==0:
                x(qbl)
            else:
                mcx([qv[j] for j in neighbors_dict[i]],qbl,ctrl_state='0'*len(neighbors_dict[i]))
            return qbl

        qv = QuantumVariable(3)
        h(qv)
        qbl = predicate(qv,0)
        multi_measurement([qv,qbl])
        # Yields: {('000', True): 0.125,('100', True): 0.125,('010', False): 0.125,('110', False): 0.125,('001', False): 0.125,('101', False): 0.125,('011', False): 0.125,('111', False): 0.125}

    The resulting ``controlled_RX_mixer`` then only swaps the node $i$ in if all neighbors $j$ in the graph $G$ are not selected.

    """

    @auto_uncompute
    def controlled_RX_mixer(qv, beta):
        m = qv.size
        for i in range(m):
            if i not in union:
                with control(predicate(qv, i)):
                    rx(beta, qv[i])

    return controlled_RX_mixer


def qiro_init_function(solutions=[], exclusions=[]):
    """
    State initialization function for QIRO algorithm. Works analogously to the normal initialization function, but respects solutions and exclusions that have been found in the QIRO reduction steps.

    Parameters
    ----------
    solutions: List
        Solutions that have been found in the QIRO reduction steps.
    exclusions: List
        Solutions that have been found in the QIRO reduction steps.

    Returns
    -------
    init_state: function
        The state initiation function, according to the update steps that have been undertaken

    """
    union = solutions + exclusions

    def init_state(qv):
        from qrisp import h

        # for i in problem.nodes:
        for i in range(len(qv)):
            if not i in union:
                h(qv[i])
        for i in solutions:
            x(qv[i])

    return init_state
