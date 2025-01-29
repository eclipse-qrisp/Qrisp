.. _QUBOQAOA:

QUBO as a QAOAProblem Instance
==============================

In this tutorial you’ll be guided through the process of defining a new phase separator to be used within the scope of the :ref:`Alternating Operator Ansatz <AOA>` focussed on solving various QUBO problems with only needing the QUBO matrix $Q$ as an input.
QUBO, or `Quadratic Unconstrained Binary Optimization <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_, is a type of problem that involves optimizing a quadratic function of binary variables.

After first translating the QUBO Hamiltonian $H_C$ from the binary basis $x_i$​ to the state basis with the translation $x_i\rightarrow \frac{\mathbb{I}-Z_i}{2}$, we'll construct the phase separator unitary and run QAOA to solve the set partitioning problem.

Not to get too ahead of ourselves with how to do it, let's first show how to solve a QUBO problem using Qrisp. To obtain the optimal solutions to a problem, we only need to know the QUBO matrix $Q$. In order to get familiar with that, we propose having a look at the `Tutorial on Formulating and Using QUBO Models <https://arxiv.org/abs/1811.11538>`_, which explains how to derive such a matrix $Q$. 

One can then simply minimize the cost function $\min C(x)=\min x^TQx$ for binary variables $x_i\in\{0,1\}$. This is done either by sending the QUBO matrix $Q$ to the annealer/quantum computer, which calculates and returns the bitstrings with corresponding values of $C(x)$. The lower the $C(x)$, the better the solution. In other words, we want to find the Hamiltonian $H_C|x\rangle=C(x)|x\rangle$.

Let's borrow the QUBO matrix (explained in depth in the above mentioned `tutorial <https://arxiv.org/abs/1811.11538>`_) for the set partitioning problem. The QUBO matrix in that case is 

$$Q = \\begin{pmatrix}-17&10&10&10&0&20\\\\10&-18&10&10&10&20\\\\10&10&-29&10&20&20\\\\10&10&10&-19&10&10\\\\0&10&20&10&-17&10\\\\20&20&20&10&10&-28\\end{pmatrix}$$

Usually, QUBO matrices are upper triangular (by convention) - this means that only elements above the diagonal (with the diagonal included) are not equal to zero. This is because the variables in QUBO problems are binary, which results in the identity $q_iq_j=q_jq_i$ for binary variables. Because of this, the elements above the diagonal are doubled compared to the symmetrized version of $Q$ we wrote down above.

Of course it's easy to go from the conventional QUBO $Q_{\text{up}\Delta}$ formulation to the symmetrized $Q_\text{sym}$ QUBO: $Q_\text{sym}=\frac{1}{2}\big(Q_{\text{up}\Delta}+Q_{\text{up}\Delta}^T\big)$, and back:
::

    import numpy as np
    Q_up_triang = 2 * np.triu(Q_sym) - np.diag(np.diag(Q_sym))


Our implementation of solving QUBO problems using QAOA works for both the upper triangular, as well as symmetrized matrix conventions. As promised, we can now immediately solve the QUBO by calling the :meth:`solve_QUBO <qrisp.qaoa.problems.QUBO.solve_QUBO>` method:
::

    from qrisp.qaoa import *
    import numpy as np

    Q = np.array(
        [
            [-17,  10,  10,  10,   0,  20],
            [ 10, -18,  10,  10,  10,  20],
            [ 10,  10, -29,  10,  20,  20],
            [ 10,  10,  10, -19,  10,  10],
            [ 0,   10,  20,  10, -17,  10],
            [ 20,  20,  20,  10,  10, -28],
        ]
    )

    solve_QUBO(Q, depth = 1, shots=5000, print_res=True)

That's it! You can try running this block of code on the website using the Thebe interactivity integration, or run it on your own Qrispy environment. 
We see that we obtain the 5 best solutions with the corresponding bitsring of the binary variables. 
You can test the statements above, and convert the symmetrized matrix into the upper triangular one, and run :meth:`solve_QUBO <qrisp.qaoa.problems.QUBO.solve_QUBO>` again.

We see, that we only needed to provide the matrix $Q$, specify the depth of the QAOA circuit that is running in the background, and the backend you want to run the algorithm on. Clean and qrispy!

From QUBO to QAOA
-----------------

Of course it's beneficial to not only know the "how", but also understand the "why". So let's dig in!

To construct such a circuit with quantum gates and run it on a quantum computer, one has to translate between the basis of $x_i$ to Pauli gate operators.

$$Z_i|x\\rangle=(-1)^{x_i}\\rightarrow\\frac{\\mathbb{1}-Z_i}{2}|x\\rangle=x_i$$

To find $H_C$ one can calculate

$$\\begin{align}\\begin{split}H_C\&=\\sum_{i,j=1}^nQ_{ij}\\frac{\\mathbb{1}-Z_i}{2}\\frac{\\mathbb{1}-Z_j}{2}\\\\&=\\sum_{i,j=1}^n\\frac{Q_{ij}}{4}+\\sum_{i,j}^n\\frac{Q_{ij}}{4}Z_iZ_j\\\\&\\space\\space\\space\\space-\\sum_{i=1}^n\\bigg(\\sum_{j=1}^n\\frac{Q_{ij}}{4}\\bigg)Z_i-\\sum_{j=1}^n\\bigg(\\sum_{i=1}^n\\frac{Q_{ij}}{4}\\bigg)Z_j\\\\&\\space\\space\\space\\space\\end{split}\\end{align}$$

Swapping the indices $i$ and $j$ in the last sum, and using the identity $Z_iZ_i=\mathbb{1}$, we get 

$$\\begin{align}\\begin{split}H_C&=\\frac{1}{4}\\sum_{i\\neq j}Q_{ij}Z_iZ_j-\\frac{1}{4}\\sum_{i=1}^n\\sum_{j=1}^n(Q_{ij}+Q_{ji})Z_i&+\\frac{1}{4}\\sum_{i,j=1}^nQ_{ij}+\\frac{1}{4}\\sum_{i=1}^nQ_{ii}\\end{split}\\end{align}$$

Note that for each single $Z_i$ we sum the $i$-th row and the $i$-th column of the matrix $Q$. 


For the cost operator $U_C$, which we feed into ``QAOAProblem``, we get

$$\\begin{align}\\begin{split}U_C=e^{-i\\gamma H_C}=\&\\prod_{i,j=1}^nR_{Z_iZ_j}\\Bigg(\\frac{\\gamma}{2}Q_{ij}\\Bigg)\\times\\prod_{i=1}^nR_{Z_i}\\Bigg(-\\frac{\\gamma}{2}\\bigg(\\sum_{j=1}^n(Q_{ij}+Q_{ji})\\bigg)\\Bigg)\\\\&\\times\\exp\\Bigg(-\\frac{i\\gamma}{4}\\sum_{i,j=1}^nQ_{ij}-\\frac{i\\gamma}{4}\\sum_{i=1}^nQ_{ii}\\Bigg)\\end{split}\\end{align}$$

Here, the last factor correspods to a global phase.

Let's translate this into a function:
::

    def create_QUBO_cost_operator(Q):

        def QUBO_cost_operator(qv, gamma):

            # Rescaling for enhancing the performance of the QAOA
            gamma = gamma/np.linalg.norm(Q)

            gphase(-gamma/4*(np.sum(Q)+np.trace(Q)),qv[0])
            for i in range(len(Q)):
                rz(-gamma/2*(sum(Q[i])+sum(Q[:,i])), qv[i])
                for j in range(len(Q)):
                    if i != j and Q[i][j] != 0:
                        rzz(gamma/2*Q[i][j], qv[i], qv[j])
        return QUBO_cost_operator

Like we did for :ref:`MaxCut <MaxCutQAOA>` and :ref:`M$\\kappa$CS <MkCSQAOA>` we also define the general QUBO objective function, the classical cost function, as well as construct the ``QUBOProblem`` blueprint bringing everything together.
::

    from qrisp import rzz, rz, gphase
    import numpy as np

    def QUBO_obj(bitstring, Q):
        x = np.array(list(bitstring), dtype=int)
        cost = x.T @ Q @ x
        return cost

    def create_QUBO_cl_cost_function(Q):

        def cl_cost_function(counts):
        
            def QUBO_obj(bitstring, Q):
                x = np.array(list(bitstring), dtype=int)
                cost = x.T @ Q @ x
                return cost
        
            energy = 0
            for meas, meas_count in counts.items():
                obj_for_meas = QUBO_obj(meas,Q)
                energy += obj_for_meas * meas_count
            return energy
    
        return cl_cost_function

    def QUBO_problem(Q,init_type='random'):

        from qrisp.qaoa import QAOAProblem, RX_mixer
    
        return QAOAProblem(create_QUBO_cost_operator(Q), RX_mixer, create_QUBO_cl_cost_function(Q),init_type=init_type)

That's it for the necessary ingredients you learned about in the :ref:`QAOA theory 101 section <QAOA101>`! Let's solve the set partitioning problem from above using this newly acquired information, and combine with how we already ran the QAOA algorithm using the :meth:`run <qrisp.qaoa.QAOAProblem.run>` method:

- define the QUBO matrix $Q$,
- define the quantum argument ``qarg`` as a :ref:`QuantumArray <QuantumArray>` of :ref:`QuantumVariables <QuantumVariable>`,
- create the QUBO instance using ``QUBO_problem`` we defined above,
- run the algorithm using the :meth:`run <qrisp.qaoa.QAOAProblem.run>` method, and last but not least,
- examine the QAOA solutions and perform for classical post processing: compute the cost functions, sort the solutions by their cost in ascending order, and print the solutions with their costs.


.. warning::

    For small QUBO instance the number of ``shots`` typically exceeds the number of possible solutions.
    In this case, even QAOA with ``depth=0``, i.e., sampling from a uniform superposition, may yield the optimal solution as the classical post-processing amounts to brute force search!
    Performance of :meth:`solve_QUBO <qrisp.qaoa.problems.QUBO.solve_QUBO>` for small instance may not be indicative of performance for large instances. 


These are exactly the pieces in the mosaic of code that :meth:`solve_QUBO <qrisp.qaoa.problems.QUBO.solve_QUBO>` consists of and performs: 
::
    
    from qrisp.default_backend import def_backend
    from qrisp import QuantumVariable, QuantumArray
    from qrisp.qaoa import QUBO_problem, QUBO_obj
    from operator import itemgetter

    Q = np.array(
        [
            [-17,  20,  20,  20,   0,  40],
            [  0, -18,  20,  20,  20,  40],
            [  0,   0, -29,  20,  40,  40],
            [  0,   0,   0, -19,  20,  20],
            [  0,   0,   0,   0, -17,  20],
            [  0,   0,   0,   0,   0, -28],
        ]
    )

    qarg = QuantumArray(qtype=QuantumVariable(1), shape=len(Q))

    QUBO_instance = QUBO_problem(Q)

    depth = 1
    res = QUBO_instance.run(qarg, depth, mes_kwargs={"backend" : def_backend}, max_iter=50)

    costs_and_solutions = [(QUBO_obj(bitstring, Q), bitstring) for bitstring in res.keys()]

    sorted_costs_and_solutions = sorted(costs_and_solutions, key=itemgetter(0))

    for i in range(5):
        print(f"Solution {i+1}: {sorted_costs_and_solutions[i][1]} with cost: {sorted_costs_and_solutions[i][0]} and probability: {res[sorted_costs_and_solutions[i][1]]}")


Now you are prepared to solve all QUBOs you derive and want to solve. On the other hand, if you would just like to play around instead, try out some QUBOs from this `list of QUBO formulations <https://blog.xa0.de/post/List-of-QUBO-formulations>`_.



