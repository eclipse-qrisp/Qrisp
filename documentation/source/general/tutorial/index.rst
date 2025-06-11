.. _tutorial:

Tutorials
=========

Welcome to the tutorial of Qrisp! This page will help you get started by providing step-by-step instructions and examples. Whether you're a beginner or an experienced quantum programmer, these tutorials provide a helping hand exploring the fundamentals of the framework and make you familiar with its syntax and features.

.. grid:: 1 1 2 2


    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Getting familiar with Qrisp</h4>

        .. image:: ../../_static/hello_world.png
            :align: center
            :target: ./tutorial.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            You will, well, get familiar with Qrisp. After being stimulated to replace thinking with quantum cirtuits with thinking with **QuantumVariables**, you'll learn to solve a quadratic equation using **Grover's algorithm** and get acquainted with **Quantum Phase Estimation**.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Simulate the dynamics of the $H_2$ molecule</h4>

        .. image:: ../../_static/hydrogen.png
            :align: center
            :target: ./H2.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will show you how to leverage Qrisp’s advanced capabilities to perform **molecular simulations** on quantum computers.

        .. raw:: html

            </div>


.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Hybrid real-time algorithm conrol with Jasp</h4>

        .. image:: ../../_static/hybrid_realtime.png
            :width: 180
            :align: center
            :target: ./Jasp.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            You will learn how this new compilation pipeline accelerates quantum algorithm compilation, and allows for hybrid **real-time computations**.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Factoring integers using Shor's algorithm</h4>

        .. image:: ../../_static/period_finding.svg
            :align: center
            :target: ./Shor.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will guide you through our state-of-the-art implementation of Shor's algorithm, allowing you to factor numbers and fiddle around encrypting and decrypting hidden messages.

        .. raw:: html

            </div>


.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Solving systems of linear equations via HHL</h4>

        .. image:: ../../_static/HHL.png
            :align: center
            :target: ./HHL.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            The Harrow-Hassidim-Lloyd (HHL) quantum algorithm offers an exponential speed-up over classical methods for solving linear system problems $Ax=b$ for certain sparse matrices $A$.
            The tutorial demonstrates hybrid quantum-classical workflows using the Catalyst framework and highlights how Qrisp and Catalyst work together for implementing and compiling advanced quantum algorithms.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Fault-Tolerant compilation</h4>

        .. image:: ../../_static/torus.png
            :align: center
            :target: ./FT_compilation.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial delves into the realm of **compiling for fault-tolerant quantum devices**, exploring the specialized techniques and considerations that set this stage apart from the compilation challenges encountered in NISQ environments. At the end you will also optimize the implementation of Shor's from the tutorial above.

        .. raw:: html

            </div>


.. grid:: 1 1 2 2
            
    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Solving the Traveling Salesman Problem</h4>

        .. image:: ../../_static/tsp.svg
            :align: center
            :target: ./TSP.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            You will set up a **superposition** of all routes, evaluate route distance, and create a **Grover's oracle** to eventually evaluate it.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Solving Sudoku using Quantum Backtracking</h4>

        .. image:: ../../_static/backtracking.svg
            :align: center
            :target: ./Sudoku.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            You will learn how to efficiently implement the Sudoku problem specific quantum oracles, and how to use the general **Quantum Backtracking** implementation within Qrisp.

        .. raw:: html

            </div>


.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Solving combinatorial optimization problems with QAOA</h4>

        .. image:: ../../_static/maxcut_tutorial.png
            :align: center
            :target: ./QAOAtutorial/index.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will break down the **theory** behind this promising algorithm before implementing it for the **MaxCut** problem, the **graph coloring** problem, as well as providing a new **constrained mixer type** letting you reduce the search space! We also include tutorials on how to solve **QUBO problems** and **portfolio rebalancing** with QAOA. Disclaimer: the tutorial might or might not involve **crayons**.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Building a QAOA implementation in Jasp</h4>

        .. image:: ../../_static/maxcut_jasp.png
            :align: center
            :target: ./JaspQAOAtutorial.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial details how to build an efficient custom QAOA implementation in Jasp using a Jasp-traceable objective function for the example of the MaxCut problem.

        .. raw:: html

            </div>


.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Quantum-Informed Recursive Optimization</h4>

        .. image:: ../../_static/maxIndepSet.png
            :align: center
            :target: ./QIROtutorial.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            Learn about a variational algorithm, that aims to adjust the given problem after each round of optimization. This tutorial will show you how to apply the theory to implement the algorithm and solve a **Maximal Independent Set** problem with it!

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">Quantum Monte Carlo Integration</h4>

        .. image:: ../../_static/qmci.svg
            :align: center
            :target: ./QMCItutorial.html

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will show you how to numerically calculate integrals with Quantum Monte Carlo Integration using Iterative Quantum Amplitude Estimation.

        .. raw:: html

            </div>


Along the way, we'll introduce you to important concepts and techniques that you'll need to know in order to write highly efficient algorithms in Qrisp.
We encourage you to follow along with the examples and try out the code for yourself. Don't worry if you make mistakes or don't understand something right away - programming is a process of learning and experimentation, and it takes time to become proficient.

By the end of this tutorial, you'll have a solid foundation of our high-level framork and be ready to tackle more complex projects. So let's get **started**!

.. toctree::
   :maxdepth: 2
   :hidden:
   
   tutorial
   H2
   Jasp
   HHL
   LCU
   Shor
   FT_compilation
   TSP
   Sudoku
   QAOAtutorial/index
   JaspQAOAtutorial
   QIROtutorial
   QMCItutorial


   
   