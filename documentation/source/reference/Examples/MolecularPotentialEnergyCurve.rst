.. _MolecularPotentialEnergyCurveExample:

Molecular Potential Energy Curves
=================================

.. currentmodule:: qrisp.vqe

Example Hydrogen
================

We caluclate the Potetial Energy Curve for the Hydrogen molecule for varying interatomic distance of the atoms.


We implement a function ``problem_data`` that sets up a molecule and 
utilizes the `PySCF <https://pyscf.org>`_ quantum chemistry library to compute the :meth:`electronic data <qrisp.vqe.problems.electronic_structure.electronic_structure_problem>` (one- and two-electron integrals, number of orbitals, number of electroins, nuclear repulsion energy, Hartree-Fock energy)
for a given molecular geometry. In this example, we vary the **atomic distance** between the two Hydrogen atoms.

::
    
    from pyscf import gto
    from qrisp.vqe.problems.electronic_structure import *
    from qrisp import QuantumVariable

    def problem_data(r):
        mol = gto.M(
            atom = f'''H 0 0 0; H  0 0 {r}''',
            basis = 'sto-3g')
        return electronic_data(mol)

Next, we define an array of atomic ``distances``. For each distance ``r`` we compute the electronic ``data`` including the Hartree-Fock energy. 
Then we utilize the :meth:`electronic_structure_problem <qrisp.vqe.problems.electronic_structure.electronic_structure_problem>` method to create a :ref:`VQEProblem` for the given electronic ``data``. By default, the ``QCCSD`` ansatz is utilized. 

VQE is a probabilistic algorithm and may not yield the optimal solution every time. 
Therefore, we run VQE five times for each instance and select the minimal energy that was found.
For more acurate results we adjust the measurement precision ``mes_kwargs={'precision':0.005}``. 

::

    distances = np.arange(0.2, 4.0, 0.1)
    y_hf = []
    y_qccsd = []

    for r in distances:
        data = problem_data(r)
        y_hf.append(data['energy_hf'])

        vqe = electronic_structure_problem(data)
        results = []
        for i in range(5):
            res = vqe.run(QuantumVariable(data['num_orb']),depth=1,max_iter=50,optimizer='COBYLA',mes_kwargs={'precision':0.005})
            results.append(res+data['energy_nuc'])
        y_qccsd.append(min(results))

Finally, we visualize the results.

::

    import matplotlib.pyplot as plt
    plt.scatter(x, y_hf, color='#7d7d7d',marker="o", linestyle='solid',s=10, label='HF energy')
    plt.scatter(x, y_qccsd, color='#6929C4',marker="o", linestyle='solid',s=10, label='VQE (QCCSD) energy')
    plt.xlabel("Atomic distance", fontsize=15, color="#444444")
    plt.ylabel("Energy", fontsize=15, color="#444444")
    plt.legend(fontsize=12, labelcolor="#444444")
    plt.tick_params(axis='both', labelsize=12)
    plt.grid()
    plt.show()


.. figure:: /_static/H2_PEC.png
   :alt: Hydrogen Potential Energy Curve
   :scale: 80%
   :align: center


Example Beryllium hydride
=========================

We caluclate the Potetial Energy Curve for the Beryllium hydride molecule for varying interatomic distance of the atoms.






