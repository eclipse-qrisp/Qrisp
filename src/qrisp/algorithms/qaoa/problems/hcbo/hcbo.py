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

import numpy as np
from collections import defaultdict
from qrisp import *
from qrisp.algorithms.qaoa import QAOAProblem
from qrisp.algorithms.qaoa.problems.hcbo.state_preparation import (
    W,
    build_predecessor_graph,
    prepare_constraints_state,
)
from qrisp.algorithms.quantum_backtracking import OHQInt


class HCBO:
    r"""
    Higher-Order Constrained Binary Optimization problem:.

    This class represents combinatorial optimization problems of the following form:

    Let $[M]=\{0,1,\dotsc,M-1\}$, and $[N]=\{0,1,\dotsc,N-1\}$. Find a mapping $f\colon [M]\rightarrow [N]$,
    i.e., assigning $N$ items into $M$ slots (with putting them back). In genreral, there are $N^M$ combinations.

    The binaray variables $x_{i,j}$ for $0\leq i<M$, $0\leq j<N$ represent one-hot encoed intergers $0\leq x_i<M$.

    The objective funtion is given by a polynomial

    .. math::

        O(x) = \sum_{k\in K}c_km_k(x)

    where c_k are real coefficients and m_k(x) are monomials $m_k(x)=x_{i_1,j_1}x_{i_2,j_2}\dotsc$.
    Constraints ensuring that $\sum_jx_{i,j}=1$ for all $i$, i.e., ensuring that $x_{i,j}$ represent
    a proper one-hot encoded integer are built into the QAOA ansatz, and must not be part of the objective function.

    The objective function is represented as dictionary ``objective_dict`` where keys are tuples of tuples of integers,
    e.g., ((1, 2), (3, 4)) representing the monomial $x_{1,2}x_{3,4}$, and values are floats.

    Parameters
    ----------
    objective_dict : dict
        A dictionary representing the objective function.
        Keys are tuples of tuples, e.g., ((1, 2), (3, 4)). Values are floats.
    constraints_list: list of tuples
        A list of tuples representing "all different" constraints.
        E.g., (1,3,4) means that the variables (1,3,4) must be mapped to different values.
        Also supports nested tuples, e.g., (1,2,(3,4)) means f(1), f(2), f(3) must be different,
        and f(3) and f(4) must be different. **Each index may only appear once.**
        Such constraints must not be part of the objective function.

    Examples
    --------

    ::

        from qrisp.algorithms.qaoa import HCBO

        objective_dict = {
            ((0, 1),): 2.5,                             # Linear: Cost of mapping 0->1
            ((1, 2),): 3.0,                             # Linear: Cost of mapping 1->2
            ((0, 1), (1, 2)): -1.5,                     # Quadratic interaction
            ((1, 2), (0, 1)): -1.5,                     # Duplicate quadratic (will combine)
            ((0, 0), (1, 1), (0, 2)): 5.0               # Cubic interaction
        }

        constraints_list = [(0,1)] # Enforce that f(0)!=f(1)

        hcbo_instance = HCBO(objective_dict, constraints_list)
        print("HCBO objective")
        print(hcbo_instance.objective_dict)

        print("Solutions")
        solutions = hcbo_instance.solve()
        print(solutions[:3])

        # HCBO objective
        # {((0, 1),): 2.5, ((1, 2),): 3.0, ((0, 1), (1, 2)): -3.0, ((0, 0), (0, 2), (1, 1)): 5.0}
        # Solutions
        # [(array([2, 1]), 0.0, 0.5272), (array([2, 0]), 0.0, 0.188), (array([0, 1]), 0.0, 0.1458),
        # (array([1, 0]), 2.5, 0.0946), (array([0, 2]), 3.0, 0.0402), (array([1, 2]), 2.5, 0.0042)]

    """

    def __init__(self, objective_dict, constraints_list=[]):
        """
        Parameters
        ----------
        objective_dict: dict
            A dictionary representing the objective function.
            Keys are tuples of tuples, e.g., ((1, 2), (3, 4)). Values are floats.
        constraints_list: list of tuples, optional
            A list of tuples representing "all different" constraints.
            E.g., (1,3,4) means that the variables (1,3,4) must be mapped to different values.
            Also supports nested tuples, e.g., (1,2,(3,4)) means f(1), f(2), f(3) must be different,
            and f(3) and f(4) must be different. **Each index may only appear once.**
            Such constraints must not be part of the objective function.

        """
        # 1. Clean, sort, and consolidate the dictionary using original tuples
        self.objective_dict = self._clean_and_sort(objective_dict)
        self.max_degree = max((len(k) for k in self.objective_dict.keys()), default=0)

        self.num_nodes = 0
        self.num_slots = 0
        # 2. Discover all unique tuple variables and create a mapping
        self._build_variable_mapping(objective_dict)

        # 3. Create coistraints forest
        self.constraints_list = constraints_list
        self.constraints_forest = []
        self.constraints_nodes = []
        for item in self.constraints_list:
            G, root = build_predecessor_graph(item)
            self.constraints_forest.append((G, root))
            self.constraints_nodes.extend(G.nodes)

        self.remaining_nodes = [
            i for i in range(self.num_nodes) if i not in self.constraints_nodes
        ]

    def _build_variable_mapping(self, objective_dict):
        """Discovers all unique variables and maps them to contiguous integers."""
        unique_vars = set()
        for term in objective_dict.keys():
            for var in term:
                unique_vars.add(var)

        # Sort to ensure deterministic ordering across different runs
        # self.id_to_var = list(sorted(unique_vars))
        # self.var_to_id = {var: idx for idx, var in enumerate(self.id_to_var)}

        # self.num_vars = len(self.id_to_var)
        # self.dummy_index = self.num_vars # Placed at the end of the array

        # Extract domain dimensions for mappings like f: [M] -> [N]
        if unique_vars:
            max_i = max(var[0] for var in unique_vars)
            max_j = max(var[1] for var in unique_vars)

            # Assuming 0-indexed variables, total count is max + 1
            self.num_nodes = max_i + 1
            self.num_slots = max_j + 1
        else:
            self.num_nodes = 0
            self.num_slots = 0

    def _clean_and_sort(self, objective_dict):
        """Sorts terms to prevent duplicates (e.g., A*B becomes B*A)."""
        clean_dict = defaultdict(float)
        for variables, weight in objective_dict.items():
            # Sort the tuples lexicographically
            sorted_vars = tuple(sorted(variables))
            clean_dict[sorted_vars] += weight

        # Strip out terms that sum exactly to 0.0
        return {k: v for k, v in clean_dict.items() if v != 0.0}

    def create_state_prep(self):
        """
        Creates the state preparation function for the HCBO problem instance.
        """

        N = self.num_slots

        def state_prep(q_array):

            for i in self.remaining_nodes:
                W(q_array[i], N)

            for item in self.constraints_forest:
                G, root = item
                prepare_constraints_state(G, root, N, q_array)

        return state_prep

    def create_cost_operator(self):
        """
        Creates the cost operator for the HCBO problem instance.
        """

        def cost_operator(q_array, gamma):
            terms_dict = self.objective_dict
            for term, weight in terms_dict.items():
                qubits = []
                for var in term:
                    i = var[0]
                    j = var[1]
                    qubits.append(q_array[i][j])

                mcp(gamma * weight, qubits)

        return cost_operator

    def create_mixer_operator(self):
        """
        Creates the mixer operator for the HCBO problem instance.
        """

        state_prep = self.create_state_prep()

        def inv_state_prep(q_array):
            with invert():
                state_prep(q_array)

        def mixer_operator(q_array, beta):
            with conjugate(inv_state_prep)(q_array):
                for i in range(len(q_array)):
                    mcp(beta, q_array[i], ctrl_state=0)

        return mixer_operator

    def create_cl_cost_function(self):
        """
        Creates the classical cost function for the HCBO problem instance.
        """

        def _evaluate(meas, term):

            for var in term:
                i = var[0]
                j = var[1]
                if meas[i] != j:
                    return 0

            return 1

        def cl_cost_function(res_dict):

            energy = 0
            for term, weight in self.objective_dict.items():
                for meas, prob in res_dict.items():
                    energy += prob * weight * _evaluate(meas, term)

            return energy

        return cl_cost_function

    def create_qaoa_problem(self):
        """
        Creates a QAOA Problem instance for the HCBO problem instance.
        """

        cl_cost_function = self.create_cl_cost_function()
        cost_operator = self.create_cost_operator()
        mixer_operator = self.create_mixer_operator()
        state_prep = self.create_state_prep()

        qaoa_problem = QAOAProblem(
            cost_operator=cost_operator,
            mixer=mixer_operator,
            cl_cost_function=cl_cost_function,
        )

        qaoa_problem.set_init_function(state_prep)

        return qaoa_problem

    def solve(self, depth=3, mes_kwargs={}, max_iter=100, optimizer="COBYLA"):
        """
        Solves the HCBO problem instance using QAOA.

        Parameters
        ----------
        depth : int
            The number of layers.
        mes_kwargs : dict
            The keyword arguments for the measurement function. Default is an empty dictionary.
        max_iter : int
            The maximum number of iterations for the optimization method. Default is 100.
        optimizer : str, optional
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            Available are, e.g., ``COBYLA``, ``COBYQA``, ``Nelder-Mead``. The Default is ``COBYLA``.

        Returns
        -------
        list of tuples
            A list containing the results, their cost values, and their probabilities.
        """

        qaoa_problem = self.create_qaoa_problem()

        qtype = OHQInt(self.num_slots)
        q_array = QuantumArray(qtype=qtype, shape=(self.num_nodes))

        res_dict = qaoa_problem.run(
            q_array,
            depth=depth,
            mes_kwargs=mes_kwargs,
            max_iter=max_iter,
            optimizer=optimizer,
        )

        cl_cost_function = self.create_cl_cost_function()

        solutions = [
            (np.array(meas, dtype=int), cl_cost_function({meas: 1.0}), prob)
            for meas, prob in res_dict.items()
        ]

        return solutions
