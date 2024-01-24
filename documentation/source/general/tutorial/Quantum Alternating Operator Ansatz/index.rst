.. role:: red
.. role:: orange
.. role:: yellow
.. role:: green
.. role:: blue
.. role:: indigo
.. role:: violet

.. _QAOA101:

QAOA implementation and QAOAProblem
===================================

Welcome to our :ref:`Quantum Approximate Optimization Algorithm (QAOA) <QAOA>` module! This module equips you with the essential theoretical fundamentals of QAOA, a promising algorithm for tackling combinatorial optimization problems. We’ll highlight why qrisp is the ideal framework for implementing QAOA, thanks to its unique features.

Once you’ve grasped the basics including the structure, operation, and the functions of phase separator and mixer Hamiltonians in QAOA, we’ll transition into its practical application. Our focus will be on two specific problem instances: :ref:`MaxCut <MaxCutQAOA>` and :ref:`Max-$\\kappa$-Colorable Subgraph <MkCSQAOA>`. In the latter we will see that in the quantum realm of Qrisp, we’re not just seeing in black and white, we’re coding in :red:`Q`:orange:`U`:yellow:`A`:green:`N`:blue:`T`:indigo:`U`:violet:`M`:red:`C`:orange:`O`:yellow:`L`:green:`O`:blue:`R`:indigo:`!`


QAOA in a nutshell
------------------

The `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/abs/1411.4028>`_ is a hybrid quantum-classical variational algorithm designed for solving combinatorial optimization problems.

The quantum and classical part work together iteratively: the quantum computer prepares a quantum state and measures it, producing a classical output; this output is then fed into a classical optimization routine which produces new parameters for the quantum part. This process is repeated until the algorithm converges to an optimal or near-optimal solution.

Before even running the algorithm there is a need to define some initial state $|\psi_0\rangle$, often chosen to be equal superposition state 
$$|s\\rangle=\\frac{1}{\\sqrt{2^n}}\\sum_z|z\\rangle, $$ where $n$ is the number of qubits.

The QAOA operates in a sequence of layers, each consisting of a problem-specific operator and a mixing operator. To be a little more exact, the state $|\psi_0\rangle$ is then evolved under the action of $p$ layers of QAOA, where one layer consists of applying the unitary phase separating operator
$$U_P(C,\\gamma)=e^{-i\\gamma C}=\\prod_{\\alpha=1}^me^{-i\\gamma C _{\\alpha}}, $$ which applies phase to each computational basis state based on its cost function value; 
and the unitary mixer operator 
$$U_M(B,\\beta)=e^{-i\\beta B}, $$ 
where $B$ represents a specific mixer operator that drives the transitions between different states. Because of properties of the unitaries' eigenvalues, the QAOA parameters are bound to hold values $\gamma\in\{0, 2\pi\},$ and $\beta\in\{0,\pi\}$ which are then optimized classically. 

After $p$ layers of QAOA, we can define the angle dependent quantum state
$$|\\psi_p\\rangle=|\\boldsymbol\\gamma,\\boldsymbol\\beta\\rangle=U_M(B,\\beta_p)U_P(C,\\gamma_p)\\cdots U_M(B,\\beta_1)U_P(C,\\gamma_1)|s\\rangle.$$

The end goal in QAOA is now to optimize the variational parameters $\gamma_p$ and $\beta_p$ in order to minimize the expectation value of the cost function with respect to the final state $∣\psi_p\rangle$. This is done using classical optimization techniques.

It's important to remember that QAOA provides an approximate solution, and its performance depends on factors like problem size and structure, choice of initial state, and number of layers $p$. Increasing $p$ generally leads to better solutions but also increases computational cost.

Alternating Operator Ansatz
---------------------------

QAOA on its own is an efficient tool for exploring complex combinatorial optimization problems. However, as with any tool, there is always room for improvement and expansion. Its potential can be further expanded by introducing a broader range of operators, not just those derived from the time evolution under a fixed local Hamiltonian proposed in the original paper.

Exactly this concept is well illustrated in the paper `“From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz” <https://arxiv.org/abs/1709.03489>`_ by Stuart Hadfield and his team. They have taken the foundational principles of the QAOA and extended them, creating an upgraded version that is both more adaptable and easier to implement. The main ideas we use in our implementations are inspired by this work.

Similarly to the original QAOA, we have several key components when implementing the Quantum Alternating Operator Ansatz:

- **COST FUNCTION:** The cost function is problem-specific and defines the optimization landscape. In the Quantum Alternating Operator Ansatz, cost functions can be represented by more general families of operators.
- **INITIAL STATE:** initial state can be any state over all computational basis states. It is in most cases chosen to be superposition.
- **PHASE SEPARATOR:** this applies a phase to each computational basis state based on its cost function value. In the Quantum Alternating Operator Ansatz, we can use a wider range of operators tailored to each problem instance.
- **MIXER:** drives transitions between different states. In the Quantum Alternating Operator Ansatz, mixers can be chosen from a broader set of unitaries, which allows for more efficient implementation and potentially better exploration of the solution space.

In their paper, Hadfield and his colleagues give us some really useful examples of how to formulate different problem instances using these building blocks. The appendix section, in particular, stands out as it provides detailed problem formulations, most of which are now implemented in Qrisp using the :ref:`QAOAProblem class <QAOAProblem>`. The following table provides a detailed overview of the problem instances already implemented within our framework, along with their :ref:`corresponding mixer type <MIXers>`:

.. list-table::
   :widths: 45 45 10
   :header-rows: 1

   * - PROBLEM INSTANCE
     - MIXER TYPE
     - 
   * - :ref:`MaxCut <MaxCutQAOA>`
     - X mixer
     -    ✅
   * - Max-$\ell$-SAT
     - X mixer
     -    ✅
   * - :ref:`QUBO (NEW since 0.4!) <QUBOQAOA>`
     - X mixer
     -    ✅ 
   * - MaxIndependentSet
     - Controlled X mixer
     -    ✅
   * - MaxClique
     - Controlled X mixer
     -    ✅
   * - MaxSetPacking
     - Controlled X mixer
     -    ✅
   * - MinSetCover
     - Controlled X mixer
     -    ✅
   * - :ref:`Max-$\\kappa$-Colorable Subgraph <MkCSQAOA>`
     - XY mixer
     -    ✅ 

Our QAOA journey doesn’t stop here. In the next tutorials we’re going to tackle two fascinating problems: :ref:`MaxCut <MaxCutQAOA>` and :ref:`Max-$\\kappa$-Colorable Subgraph <MkCSQAOA>`, showcasing multiple unique features of Qrisp, including the functionality of creating custom QuantumVariable types - get ready to add a splash of :red:`Q`:orange:`u`:yellow:`a`:green:`n`:blue:`t`:indigo:`u`:violet:`m`:red:`C`:orange:`o`:yellow:`l`:green:`o`:blue:`r` to your code.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   Theoretical
   MaxCut
   MkCS
   ConstrainedMixers
   QUBO
