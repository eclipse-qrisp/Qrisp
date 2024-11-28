.. _tutorial:

Tutorial
--------

Welcome to the tutorial of Qrisp! This page will help you get started by providing step-by-step instructions and examples. Whether you're a beginner or an experienced quantum programmer, these tutorials provide a helping hand exploring the fundamentals of the framework and make you familiar with its syntax and features.

To gradually qrispify your programming game you will start with the basics and gradually build your qrispertoire to more advanced features like automatic :ref:`recomputation <recomputation>`. We've structured the tutorial in a way that is easy to follow:

- in :ref:`Getting familiar with Qrisp <Qrisp101>` you will, well, get familiar with Qrisp. After being stimulated to replace thinking with quantum cirtuits with thinking with **QuantumVariables**, you'll learn to solve a quadratic equation using **Grover's algorithm** and get acquainted with **Quantum Phase Estimation**.

- :ref:`QAOA implementation and QAOAProblem <QAOA101>` will break down the **theory** behind this promising algorithm before implementing it for the **MaxCut** problem, the **graph coloring** problem, as well as providing a new **constrained mixer type** letting you reduce the search space! Since the 0.4 update we also include a tutorial on how to solve **QUBO problems** with QAOA. Disclaimer: the tutorial might or might not involve **crayons**.

- :ref:`Quantum-Informed recursive optimization <Qiro_tutorial>` covers the theory of a variational algorithm, that aims to adjust the given problem after each round of optimization. Additionally, this tutorial will show you how to apply said theory to implement the algorithm in **Qrisp** and solve a **MaxIndepentSet** problem with it!

- :ref:`Solving the Traveling Salesman Problem (TSP) <tsp>` is again pretty self explanatory - you'll set up a **superposition** of all routes, **evaluate** route distance, and create a **Grover's oracle** to eventually evaluate it.

- :ref:`The Quantum Monte Carlo Integration tutorial <QMCItutorial>` will show you how to integrate functions with **Quantum Monte Carlo methods**, using **Iterative Quantum Amplitude Estimation**. 

- :ref:`Solving Sudoku with quantum Backtracking <Sudoku>` explains to you how to efficiently implement the Sudoku problem specific quantum oracles, and how to use the general **Quantum Backtracking** implementation within Qrisp.

- :ref:`Simulating the dynamics of the $H_2$ molecule <H2>` will show you how to leverage Qrisp’s advanced capabilities to perform **molecular simulations** on quantum computers.

- :ref:`Implementing Shor's algorithm <shor_tutorial>` will guide you through our state-of-the-art implementation of  **Shor's algorithm**, allowing you to factor numbers and fiddle around encrypting and decrypting hidden messages.

- :ref:`Fault-Tolerant compilation of Shor's algorithm <ft_compilation_shor>` delves into the realm of **compiling for fault-tolerant quantum devices**, exploring the specialized techniques and considerations that set this stage apart from the compilation challenges encountered in NISQ environments. At the end you will also optimize the implementation of Shor's from the tutorial above.


Along the way, we'll introduce you to important concepts and techniques that you'll need to know in order to write highly efficient algorithms in Qrisp.
We encourage you to follow along with the examples and try out the code for yourself. Don't worry if you make mistakes or don't understand something right away - programming is a process of learning and experimentation, and it takes time to become proficient.

By the end of this tutorial, you'll have a solid foundation of our high-level framork and be ready to tackle more complex projects. So let's get **started**!

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
   
   