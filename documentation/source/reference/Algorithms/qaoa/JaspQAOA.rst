.. _JaspQAOA:

How to use QAOA in Jasp
=======================

In :ref:`How to think in Jasp <jasp_tutorial>` we learned how Jasp allows to future-proof Qrisp code for practically relevant problems.
For variational quantum algorithms like QAOA and VQE, hybrid quantum-classical workflows can be seamlessly compliled, optimized and executed.

We demonstrate how to use QAOA in Jasp for MaxCut problem:

For a graph $G$ with $n$ nodes, a bipartition $S$, $V\setminus S$ of the set of vertices $V$ can be encoded with a :ref:`QuantumVariable` with $n$ qubits: 
we measure the $i$-th qubit in 0 if the node $i$ is in the set $S$, and 1 if the node $i$ is in the set $V\setminus S$.
The cut value is the number of edges $e=(i,j)$ in $G$ such that $i\in S$ and $j\in V\setminus S$.

In Jasp, varibales are decoded to integers (i.e. jax.numpy.int) and not to binrary strings. In this case, the binary representation of an integer encodes a bipartition of the graph $G$.
Therefore, repeated sampling from a QuantumVariable in a superposition state will result in an array of integers representing bipartitions of the graph $G$. 
Within QAOA, we require a post processing function to compute the average cut value for an array of samples. 
More details on how to build an efficient post processing function are provided in the :ref:`JaspQAOAtutorial` tutorial. 

Apart from this, running :ref:`QAOA for MaxCut <maxCutQAOAdoc>` in Jasp is as easy as wrapping the code in a ``main`` function:

::
    
    from qrisp import QuantumVariable, jaspify
    from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxcut_cost_operator, create_maxcut_sample_array_post_processor
    import networkx as nx

    def main():

        G = nx.erdos_renyi_graph(6, 0.7, seed = 133)

        cl_cost = create_maxcut_sample_array_post_processor(G)

        qarg = QuantumVariable(G.number_of_nodes())

        qaoa_maxcut = QAOAProblem(cost_operator=create_maxcut_cost_operator(G),
                            mixer=RX_mixer,
                            cl_cost_function=cl_cost)
        res_sample = qaoa_maxcut.run(qarg, depth=5, max_iter=50, optimizer="SPSA")

        return res_sample

The :ref:`jaspify <jaspify>` method allows for running Jasp-traceable functions using the integrated Qrisp simulator. 
For hybrid algorithms like QAOA and VQE that rely on calculating expectation values based on sampling, the ``terminal_sampling`` feature significantly 
speeds up the simulation: samples are drawn from the state vector instead of performing repeated simulation and measurement of the quantum circuits.

::

    jaspify(main, terminal_sampling=True)()
    #Yields: array([39., 40., 56., ..., 56.,  5., 20.])

You can also create the :ref:`jaspr` object and compile to `QIR <https://www.qir-alliance.org>`_ using `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_.

::

    jaspr = make_jaspr(main)()
    qir_str = jaspr.to_qir()




