.. _tutorial:

Tutorials
=========
 
Welcome to the Qrisp tutorials! This gallery is designed to take you from a curious beginner to a proficient developer of utility-scale quantum algorithms. By shifting the focus from low-level gate gymnastics to high-level programming with :ref:`QuantumVariables <QuantumVariable>`, Qrisp allows you to express complex logic with ease.

The tutorials are organized into four topics, each containing multiple hands-on examples that build on each other:

- **Foundations and first Steps**: Start here to master the core syntax of Qrisp, where you will learn to replace manual circuit building with high-level variables and implement textbook algorithms like Shor's and Grover's.

- **Optimization and Numerics**: Explore how to solve combinatorial optimization problems in logistics and scheduling using hybrid approaches like QAOA, QIRO, and COLD, as well as learning about our qrispy Quantum Backtracking implementation.

- **High-Performance Compilation and Execution with Jasp**: Learn how to scale your code, while estimating your quantum resources, for the real world using the Jasp pipeline for hybrid real-time control. You will also learn about optimizing for Fault-Tolerant compilation for next-generation hardware.

- **Scientific Computing and Quantum Numerical Linear Algebra**: Dive into high-level abstractions for quantum chemistry, physics and linear systems (HHL, CKS, QSP) using our new :ref:`BlockEncoding` class.

If you’re the type who learns best by breaking things (and then fixing them), you can download any of these tutorials as a Jupyter notebook. Just look for the download box on the right side of the page within each specific tutorial to grab the code and run it in your own local environment.

By the end of these tutorial, you'll have a solid foundation of our high-level framork and be ready to tackle more complex projects. So let's get **started**!


Foundations and first steps
---------------------------
Kickstart your quantum programming journey here! This section is designed to familiarize you with Qrisp's core philosophy: shifting away from low-level circuit manipulation and toward intuitive, high-level programming using QuantumVariables. You will cover the absolute essentials, starting from your very first quantum script to implementing and understanding textbook algorithms like Grover's, Quantum Phase Estimation, and Shor's algorithm.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./tutorial.html" style="text-decoration: none; color: inherit;">Getting familiar with Qrisp</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./tutorial.html">
                    <img src="../../_static/hello_world.png" alt="Qrisp Tutorial">
                </a>
            </div>

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

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./Shor.html" style="text-decoration: none; color: inherit;">Factoring integers using Shor's algorithm</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./Shor.html">
                    <img src="../../_static/period_finding.svg" alt="Shor Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will guide you through our state-of-the-art implementation of Shor's algorithm, allowing you to factor numbers and fiddle around encrypting and decrypting hidden messages.

        .. raw:: html

            </div>


Optimization and Numerics
-------------------------
Discover how quantum algorithms can tackle complex search and optimization problems.
In this section, you will dive into solving famous computational challenges (such as the Traveling Salesman Problem and Sudoku) using a powerful suite of quantum tools. 
Whether you are applying Quantum Backtracking, tuning variational models with QAOA and QIRO, or exploring Counterdiabatic Driving Protocols and Monte Carlo Integration, 
you will learn how to optimize solutions across a variety of real-world domains.

.. grid:: 1 1 2 2
            
    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./TSP.html" style="text-decoration: none; color: inherit;">Solving the Traveling Salesman Problem</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./TSP.html">
                    <img src="../../_static/tsp.svg" alt="Traveling Salesperson Problem Tutorial">
                </a>
            </div>

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

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./QAOAtutorial/index.html" style="text-decoration: none; color: inherit;">Solving combinatorial optimization problems with QAOA</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./QAOAtutorial/index.html">
                    <img src="../../_static/maxcut_tutorial.png" alt="QAOA Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will break down the **theory** behind this promising algorithm before implementing it for the **MaxCut** problem, the **graph coloring** problem, as well as providing a new **constrained mixer type** letting you reduce the search space! We also include tutorials on how to solve **QUBO problems** and **portfolio rebalancing** with QAOA. Disclaimer: the tutorial might or might not involve **crayons**.

        .. raw:: html

            </div>

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./QIROtutorial.html" style="text-decoration: none; color: inherit;">Quantum-Informed Recursive Optimization</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./QIROtutorial.html">
                    <img src="../../_static/maxIndepSet.png" alt="QIRO Tutorial">
                </a>
            </div>

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

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./CD.html" style="text-decoration: none; color: inherit;">Counterdiabatic Driving Protocols</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./CD.html">
                    <img src="../../_static/cold.png" alt="Counteradiabatic Driving Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial explains the concept of counterdiabatic driving and solves a QUBO problem with LCD (local counterdiabatic driving) and COLD (counterdiabatic optimized local driving).

        .. raw:: html

            </div>

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./Sudoku.html" style="text-decoration: none; color: inherit;">Solving Sudoku using Quantum Backtracking</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./Sudoku.html">
                    <img src="../../_static/backtracking.svg" alt="Quantum Backtracking Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            You will learn how to efficiently implement the Sudoku problem specific quantum oracles, and how to use the general **Quantum Backtracking** implementation within Qrisp.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./QMCItutorial.html" style="text-decoration: none; color: inherit;">Quantum Monte Carlo Integration</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./QMCItutorial.html">
                    <img src="../../_static/qmci.svg" alt="Quantum Monte Carlo Integration Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will show you how to numerically calculate integrals with Quantum Monte Carlo Integration using Iterative Quantum Amplitude Estimation.

        .. raw:: html

            </div>


High-Performance Compilation and Execution with Jasp
----------------------------------------------------
Scale your quantum algorithms to practically relevant problem sizes by bypassing Python’s performance bottlenecks. 
By targeting the MLIR toolchain and QIR specification, Jasp enables high-speed compilation and seamless real-time control—allowing classical logic to execute within the quantum coherence window. 
This is essential for hardware-efficient protocols like Gidney’s adder, error-correcting syndrome decoding or repeat-until-success protocols like HHL.

You will explore the Jasp pipeline for hybrid real-time control, learn how to build highly performant custom optimization algorithms, 
and delve into Fault-Tolerant compilation to ensure your code is ready for next-generation hardware.
You'll also discover how to leverage Jasp and the BigInteger class to compile and perform resource estimations for Shor's algorithm at 2048 bit.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./Jasp.html" style="text-decoration: none; color: inherit;">Hybrid real-time algorithm control with Jasp</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./Jasp.html">
                    <img src="../../_static/hybrid_realtime.png" width="180" alt="Hybrid Realtime Tutorial">
                </a>
            </div>

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

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./JaspQAOAtutorial.html" style="text-decoration: none; color: inherit;">Building a QAOA implementation in Jasp</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./JaspQAOAtutorial.html">
                    <img src="../../_static/maxcut_jasp.png" alt="MaxCut QAOA Tutorial">
                </a>
            </div>

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

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./FT_compilation.html" style="text-decoration: none; color: inherit;">Fault-Tolerant compilation</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./FT_compilation.html">
                    <img src="../../_static/torus.png" alt="Fault-Tolerant Compilation Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial delves into the realm of **compiling for fault-tolerant quantum devices**, exploring the specialized techniques and considerations that set this stage apart from the compilation challenges encountered in NISQ environments. At the end you will also optimize the implementation of Shor's from the tutorial above.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./BigInteger.html" style="text-decoration: none; color: inherit;">Using the BigInteger to compile Shor's algorithm at 2048 bit</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./BigInteger.html">
                <img src="../../_static/order_finding_qre.svg" alt="Shor Quantum Resource Estimation Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            Learn how to use the BigInteger class to avoid integer overflow in Jasp and perform resource estimation of large-scale quantum algorithms.

        .. raw:: html

            </div>


Scientific Computing and Quantum Numerical Linear Algebra
---------------------------------------------------------
Unlock the potential of quantum computers for scientific discovery and advanced mathematics. 
Here, you will use Qrisp's high-level abstractions to tackle problems in quantum chemistry, while having block encodings as a programming abstraction for quantum numerical linear algebra. 
From simulating molecular dynamics (like the $H_2$ molecule) to solving systems of linear equations with the HHL algorithm and preparing ground states via quantum signal processing,
this track demonstrates how to bridge the gap between complex scientific theory and executable quantum code.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./H2.html" style="text-decoration: none; color: inherit;">Simulate the dynamics of the $H_2$ molecule</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./H2.html">
                    <img src="../../_static/hydrogen.png" alt="Hydrogen Molecule Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            This tutorial will show you how to leverage Qrisp’s advanced capabilities to perform **molecular simulations** on quantum computers.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./BE_tutorial/index.html" style="text-decoration: none; color: inherit;">Quantum Linear Algebra</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./BE_tutorial/index.html">
                    <img src="../../_static/BE_thumbnail.png" alt="Block Encoding Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            Use block encodings as programming abstractions to perform Quantum Linear Algebra using the numpy-like interface of our BlockEncoding class.

        .. raw:: html

            </div>

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./GQSP_filtering.html" style="text-decoration: none; color: inherit;">Eigenstate filtering using quantum signal processing</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./GQSP_filtering.html">
                    <img src="../../_static/filtering_thumb_placeholder.png" alt="QSP Filtering Tutorial">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            Explore how to prepare the ground state by applying a Gaussian filter to enhance the overlap of the prepared state with the ground state using GQSP.

        .. raw:: html

            </div>

    .. grid-item-card::
        :shadow: none

        .. raw:: html

            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">

        .. raw:: html

            <h4 style="font-weight: bold; margin-top: 0;">
                <a href="./HHL.html" style="text-decoration: none; color: inherit;">Solving systems of linear equations via HHL</a>
            </h4>

        .. raw:: html

            <div style="text-align: center;">
                <a href="./HHL.html">
                    <img src="../../_static/HHL.png" alt="HHL Algorithm diagram">
                </a>
            </div>

        .. raw:: html
            
            <p style="margin-top: 5px;"></p>

        .. dropdown:: :fa:`eye me-1` Description
            :color: primary

            The Harrow-Hassidim-Lloyd (HHL) quantum algorithm offers an exponential speed-up over classical methods for solving linear system problems $Ax=b$ for certain sparse matrices $A$.
            The tutorial demonstrates hybrid quantum-classical workflows using the Catalyst framework and highlights how Qrisp and Catalyst work together for implementing and compiling advanced quantum algorithms.

        .. raw:: html

            </div>


You’ve now explored the breadth of what’s possible when you trade gate-level surgery for high-level logic. From your first QuantumVariable to fault-tolerant resource estimation, you have the roadmap to develop utility-scale applications that once seemed out of reach.

The quantum landscape is evolving rapidly. By mastering these utorials, you’re no longer just a spectator, you’re an architect of the next generation of algorithms. So, take these concepts, experiment, and start building the future of quantum computing with intuitive, clean, and qrisy code.

.. toctree::
   :maxdepth: 2
   :hidden:
   
   tutorial.ipynb
   Shor.ipynb
   TSP.ipynb
   QAOAtutorial/index
   QIROtutorial.ipynb
   CD.ipynb
   Sudoku.ipynb
   QMCItutorial.ipynb
   Jasp.ipynb
   JaspQAOAtutorial.ipynb
   FT_compilation.ipynb
   BigInteger.ipynb
   H2.ipynb
   BE_tutorial/index
   GQSP_filtering.ipynb
   HHL.ipynb 