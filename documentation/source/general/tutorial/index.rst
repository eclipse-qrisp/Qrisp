.. _tutorial:

Tutorial
--------

Welcome to the tutorial of Qrisp! This page will help you get started by providing step-by-step instructions and examples. Whether you're a beginner or an experienced quantum programmer, these tutorials provide a helping hand exploring the fundamentals of the framework and make you familiar with its syntax and features.

To gradually qrispify your programming game you will start with the basics and gradually build your qrispertoire to more advanced features like automatic :ref:`recomputation <recomputation>`. We've structured the tutorial in a way that is easy to follow:

- in :ref:`Getting familiar with Qrisp <Qrisp101>` you will, well, get familiar with Qrisp. After being stimulated to replace thinking with quantum cirtuits with thinking with **QuantumVariables**, you'll learn to solve a quadratic equation using **Grover's algorithm** and get acquainted with **Quantum Phase Estimation**.

- :ref:`QAOA implementation and QAOAProblem <QAOA101>` will break down the **theory** behind this promising algorithm before implementing it for the **MaxCut** problem, the **graph coloring** problem, as well as providing a new **constrained mixer type** letting you reduce the search space! Since the 0.4 update we also include a tutorial on how to solve **QUBO problems** with QAOA. Disclaimer: the tutorial might or might not involve **crayons**.

- :ref:`Quantum Informed Recursive Optimization <_qiro_tutorial>` will show you problem specific implementations for solving optimization problems. With this, we establishes a blueprint for developing alogrithms, that update the problem instance recursively based on the correlations in results obtained by QAOA.

- :ref:`Solving the Traveling Salesman Problem (TSP) <tsp>` is again pretty self explanatory - you'll set up a **superposition** of all routes, **evaluate** route distance, and create a **Grover's oracle** to eventually evaluate it.

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
   QIRO
   TSP
   Sudoku
   Shor
   FT_compilation

   
   