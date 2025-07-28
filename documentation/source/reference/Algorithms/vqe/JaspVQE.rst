.. _JaspVQE:

How to use VQE in Jasp
======================

In :ref:`How to think in Jasp <tutorial>` we learned how Jasp allows to future proof Qrisp code for practically relevant problems.
For variational quantum algorithms like QAOA and VQE, hybrid quantum-classical workflows can be seamlessly compliled, optimized and executed.

We again use the example of calculating the ground state energy of the $H_2$ molecule with VQE.
Running this example in Jasp is as easy as wrapping the code in a ``main`` function:

::
    
    from qrisp import *
    from qrisp.vqe.problems.electronic_structure import *
    from pyscf import gto

    def main():

        mol = gto.M(
            atom = '''H 0 0 0; H 0 0 0.74''',
            basis = 'sto-3g')

        vqe = electronic_structure_problem(mol)

        energy = vqe.run(lambda : QuantumFloat(4), depth=1, max_iter=100, optimizer="SPSA")
        
        return energy

The :ref:`jaspify <jaspify>` method allows for running Jasp-traceable functions using the integrated Qrisp simulator. 
For hybrid algorithms like QAOA and VQE that rely on calculating expectation values based on sampling, the ``terminal_sampling`` feature significantly 
speeds up the simulation: samples are drawn from the state vector instead of performing repeated simulation and measurement of the quantum circuits.

::

    jaspify(main, terminal_sampling=True)()
    #Yields: Array(-1.84467526, dtype=float64)

You can also create the :ref:`jaspr` object and compile to `QIR <https://www.qir-alliance.org>`_ using `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_.

::

    jaspr = make_jaspr(main)()
    qir_str = jaspr.to_qir()




