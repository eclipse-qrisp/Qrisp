.. _tutorial:

Tutorials
---------

Welcome to the tutorial of Qrisp! This page will help you get started by providing step-by-step instructions and examples. 
Whether you're a beginner or an experienced quantum programmer, these tutorials provide a helping hand exploring the fundamentals of the framework and make you familiar with its syntax and features.

Along the way, we'll introduce you to important concepts and techniques that you'll need to know in order to write highly efficient algorithms in Qrisp.
We encourage you to follow along with the examples and try out the code for yourself. Don't worry if you make mistakes or don't understand something right away - programming is a process of learning and experimentation, and it takes time to become proficient.

By the end of this tutorial, you'll have a solid foundation of our high-level framework and be ready to tackle more complex projects. So let's get **started**!


.. grid::

    .. grid-item-card:: :ref:`Getting familiar with Qrisp <Qrisp101>`

        | You will, well, get familiar with Qrisp. After being stimulated to replace thinking with quantum ciruits with thinking with **QuantumVariables**, you'll learn to solve a quadratic equation using **Grover's algorithm** and get acquainted with **Quantum Phase Estimation**.

.. grid::
  
    .. grid-item-card:: :ref:`Combinatorial optimization with QAOA <QAOA101>`

        | Learn the theory behind this variational algorithm before implementing it for the **MaxCut** problem and the **graph coloring** problem. Discover a new **constrained mixer** letting you reduce the search space! We also include tutorials on **how to solve QUBOs** and **portfolio rebalancing** with QAOA. Disclaimer: the tutorial might or might not involve **crayons**.

.. grid::

    .. grid-item-card:: :ref:`Quantum-Informed Recursive Optimization <Qiro_tutorial>`

        | Learn about a variational algorithm, that aims to adjust the given problem after each round of optimization. This tutorial will show you how to apply the theory to implement the algorithm and solve a **MaxIndepentSet** problem with it!

.. grid::

    .. grid-item-card:: :ref:`Solving the Traveling Salesman Problem <tsp>`

        | is again pretty self explanatory - you'll set up a **superposition** of all routes, **evaluate** route distance, and create a **Grover's oracle** to eventually evaluate it.

.. grid::

    .. grid-item-card:: :ref:`Quantum Monte Carlo Integration <QMCItutorial>`

        | Learn how to evaluate integrals numerically on a quantum computuer using **Iterative Quantum Amplitude Estimation**. 
      
.. grid::

    .. grid-item-card:: :ref:`Solving Sudoku with quantum Backtracking <Sudoku>`

        | explains to you how to efficiently implement the Sudoku problem specific quantum oracles, and how to use the general **Quantum Backtracking** implementation within Qrisp.

.. grid::

    .. grid-item-card:: :ref:`Simulating the dynamics of the $H_2$ molecule <H2>`

        | will show you how to leverage Qrisp’s advanced capabilities to perform **molecular simulations** on quantum computers.
 
.. grid::

    .. grid-item-card:: :ref:`Implementing Shor's algorithm <shor_tutorial>`

        | will guide you through our state-of-the-art implementation of  **Shor's algorithm**, allowing you to factor numbers and fiddle around encrypting and decrypting hidden messages.

.. grid::

    .. grid-item-card:: :ref:`Fault-Tolerant compilation of Shor's algorithm <ft_compilation_shor>`

        | delves into the realm of **compiling for fault-tolerant quantum devices**, exploring the specialized techniques and considerations that set this stage apart from the compilation challenges encountered in NISQ environments. At the end you will also optimize the implementation of Shor's algorithm from the previous tutorial.

.. grid::

    .. grid-item-card:: :ref:`How to think in Jasp <jasp_tutorial>`

        | explains how this **new compilation pipeline** accelerates quantum algorithm compilation, and allows for **real-time computations**.


.. toctree::
   :maxdepth: 2
   :hidden:
   
   tutorial
   Quantum Alternating Operator Ansatz/index
   QIROtutorial
   TSP
   QMCItutorial
   Sudoku
   H2
   Shor
   FT_compilation
   Jasp