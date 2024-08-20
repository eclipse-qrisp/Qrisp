.. _QAOA:

Quantum Approximate Optimization Algorithm
==========================================

.. currentmodule:: qrisp.qaoa

.. toctree::
   :hidden:
   
   QAOAProblem
   QAOABenchmark
   QAOAImplemens

This modules facilitates the execution of The `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/abs/1411.4028>`_  and related techniques called the `Quantum Alternating Operator Ansatz <https://arxiv.org/abs/1709.03489>`_. 

The end goal in QAOA is to optimize the variational parameters $\gamma_p$ and $\beta_p$ in order to minimize the expectation value of the cost function with respect to the final state $\ket{\psi_p}$. You can read more about the theoretical fundamentals behind QAOA in the tutorial :ref:`Theoretical Overview <TheoryQAOA>`. You can also find efficient implementations of this algorithm for :ref:`MaxCut <MaxCutQAOA>` and :ref:`MkCSQAOA` in an easy to read, insightful tutorials in which we describe the recipe to implement QAOA for other problem instances in a modular way, and independent of the encoding used. 

The central data structure of the QAOA module is the :ref:`QAOAProblem` class.

:ref:`QAOAProblem`
------------------

The :ref:`QAOAProblem` class is like a blueprint for a implementing QAOA for a specific problem instance we're trying to solve. When we create an instance of this class, we need to provide three things:

- Cost operator aka phase separator: a function that represents the problem we’re trying to solve, defining what makes a good or bad solution.
- Mixer operator: a function that drives transitions between different states. In the Quantum Alternating Operator Ansatz, mixers can be chosen from a broader set of unitaries, also specified below.
- Classical cost function: a function that takes a potential solution and calculates its cost.

Apart from the basic three ingredients mentioned above, some problems require the specification of the initial state. This can be achieved using the :meth:`.set_init_function <qrisp.qaoa.QAOAProblem.set_init_function>` method.

The :meth:`.run <qrisp.qaoa.QAOAProblem.run>` method prepares the initial state, applies $p$ layers of phase separators and mixers, and compiles a quantum circuit with intended measurements. Subsequently, the optimization algorithm is executed and the measurement results of the optimized circuit are returned.

For benchmarking, we provide the :meth:`.benchmark <qrisp.qaoa.QAOAProblem.benchmark>` method, which allows you to collect performance data about your implementation.

Additionally, a circuit can be pretrained with the method :meth:`.train_function <qrisp.qaoa.QAOAProblem.train_function>` . This allows preparing a new QuantumVariable with already optimized parameters, such that no new optimization is conducted. The results will therefore be the same. 
   
:ref:`QAOABenchmark`
--------------------

As an approximation algorithm, benchmarking various aspects such as solution quality and execution cost is a central question for QAOA. The results of the benchmarks of :ref:`QAOAProblem` are represented in a class called :ref:`QAOABenchmark`. This enables convenient evaluation of several important metrics as well as visualization of the results.

.. _MIXers:

Collection of mixers and implemented problem instances
------------------------------------------------------
Qrisp comes with a variety of predefined mixers to tackle various types of problem instances:


.. autosummary::
   :toctree: generated/
   
   RX_mixer
   RZ_mixer
   XY_mixer
   grover_mixer
   constrained_mixer_gen
  

   
The following problem instances have already been successfully implemented using the Qrisp framework:

.. list-table::
   :widths: 45 45 10
   :header-rows: 1

   * - PROBLEM INSTANCE
     - MIXER TYPE
     - IMPLEMENTED IN QRISP
   * - :ref:`MaxCut <QAOAMaxCut>`
     - X mixer
     -    ✅
   * - :ref:`Max-$\\ell$-SAT <maxsatQAOA>`
     - X mixer
     -    ✅
   * - :ref:`QUBO (NEW since 0.4!) <QUBOQAOA>`
     - X mixer
     -    ✅ 
   * - :ref:`MaxIndependentSet <QAOAMaxIndependentSet>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MaxClique <maxcliqueQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MaxSetPacking <maxSetPackQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MinSetCover <minsetcoverQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`Max-$\\kappa$-Colorable Subgraph <QAOAMkCS>`
     - XY mixer
     -    ✅ 
     

