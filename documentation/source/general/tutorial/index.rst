.. _tutorial:

Tutorials
=========
 
Welcome to the Qrisp tutorials! This gallery is designed to take you from a curious beginner to a proficient developer of utility-scale quantum algorithms. By shifting the focus from low-level gate gymnastics to high-level programming with :ref:`QuantumVariables <QuantumVariable>`, Qrisp allows you to express complex logic with ease.

The tutorials are organized into four topics, each containing multiple hands-on examples that build on each other:

- **Foundations and first Steps**: Start here to master the core syntax of Qrisp, where you will learn to replace manual circuit building with high-level variables and implement textbook algorithms like Shor's and Grover's.

- **Optimization and Numerics**: Explore how to solve combinatorial optimization problems in logistics and scheduling using hybrid approaches like QAOA, QIRO, and COLD, as well as learning about our qrispy Quantum Backtracking implementation.

- **High-Performance Compilation and Execution with Jasp**: Leverage the JAX-based Jasp pipeline to enable hybrid real-time control flow, while scaling your code and estimating quantum resources for real-world-sized applications.

- **Scientific Computing and Quantum Numerical Linear Algebra**: Dive into high-level abstractions for quantum chemistry, physics and linear systems using our new :ref:`BlockEncoding` class.

If you’re the type who learns best by breaking things (and then fixing them), you can download any of these tutorials as a Jupyter notebook. Just look for the download box on the right side of the page within each specific tutorial to grab the code and run it in your own local environment.

By the end of these tutorials, you'll have a solid foundation of our high-level framework and be ready to tackle more complex projects. So let's get **started**!


Foundations and First Steps
---------------------------
Kickstart your quantum programming journey here! This section is designed to familiarize you with Qrisp's core philosophy:
shifting away from low-level circuit manipulation and toward intuitive, high-level programming using QuantumVariables. 

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./tutorial.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Solve a quadratic equation using Grover's algorithm and get acquainted with Quantum Phase Estimation.</p>
            </div>
            </a>

        .. image:: ../../_static/hello_world.png
            :alt: Qrisp Tutorial

        +++
        **Getting familiar with Qrisp**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./Shor.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Factor numbers and experiment with encrypting and decrypting hidden messages using our state-of-the-art implementation.</p>
            </div>
            </a>

        .. image:: ../../_static/period_finding.svg
            :alt: Shor Tutorial

        +++
        **Factoring integers using Shor's algorithm**


Optimization and Numerics
-------------------------
Discover how quantum algorithms can tackle complex search and optimization problems. In this section, you will dive into famous computational challenges using a diverse range of quantum algorithms.
Learn how to optimize solutions across a variety of real-world domains using hybrid variational models, backtracking, and counterdiabatic driving.

.. grid:: 1 1 2 2
            
    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./TSP.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Set up a superposition of all routes, evaluate route distance, and create a Grover's oracle to evaluate it.</p>
            </div>
            </a>

        .. image:: ../../_static/tsp.svg
            :alt: Traveling Salesperson Problem Tutorial

        +++
        **Solving the Traveling Salesman Problem**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./QAOAtutorial/index.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Implement QAOA for the MaxCut problem, graph coloring, and explore a new constrained mixer type to reduce search space.</p>
            </div>
            </a>

        .. image:: ../../_static/maxcut_tutorial.png
            :alt: QAOA Tutorial

        +++
        **Solving combinatorial optimization problems with QAOA**

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./QIROtutorial.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Discover a variational algorithm that dynamically updates the problem structure after each optimization round to solve the Maximal Independent Set problem.</p>
            </div>
            </a>

        .. image:: ../../_static/maxIndepSet.png
            :alt: QIRO Tutorial

        +++
        **Quantum-Informed Recursive Optimization**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./CD.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Solve a QUBO problem with local counterdiabatic driving and counterdiabatic optimized local driving.</p>
            </div>
            </a>

        .. image:: ../../_static/cold.png
            :alt: Counteradiabatic Driving Tutorial

        +++
        **Counterdiabatic Driving Protocols**

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./Sudoku.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Learn how to efficiently implement quantum oracles for Sudoku and apply Qrisp's general Quantum Backtracking algorithm.</p>
            </div>
            </a>

        .. image:: ../../_static/backtracking.svg
            :alt: Quantum Backtracking Tutorial

        +++
        **Solving Sudoku using Quantum Backtracking**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./QMCItutorial.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Numerically calculate integrals with Quantum Monte Carlo Integration using Iterative Quantum Amplitude Estimation.</p>
            </div>
            </a>

        .. image:: ../../_static/qmci.svg
            :alt: Quantum Monte Carlo Integration Tutorial

        +++
        **Quantum Monte Carlo Integration**


High-Performance Compilation and Execution with Jasp
----------------------------------------------------
Scale your quantum algorithms to practically relevant problem sizes by bypassing Python’s performance bottlenecks. 
By targeting the MLIR toolchain and QIR specification, Jasp enables high-speed compilation and seamless real-time control, allowing classical logic to execute within the quantum coherence window.
Explore the Jasp pipeline to build highly performant algorithms, and ensure your code is ready for next-generation, fault-tolerant hardware.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./Jasp.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Learn how this new compilation pipeline accelerates quantum algorithm compilation, and allows for hybrid real-time computations.</p>
            </div>
            </a>

        .. image:: ../../_static/hybrid_realtime.png
            :width: 180
            :alt: Hybrid Realtime Tutorial

        +++
        **Hybrid real-time algorithm control with Jasp**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./JaspQAOAtutorial.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Build an efficient custom QAOA implementation in Jasp using a JAX-traceable objective function for the MaxCut problem.</p>
            </div>
            </a>

        .. image:: ../../_static/maxcut_jasp.png
            :alt: MaxCut QAOA Tutorial

        +++
        **Building a QAOA implementation in Jasp**

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./FT_compilation.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Compile for fault-tolerant quantum devices, exploring specialized techniques that set this stage apart from NISQ environments.</p>
            </div>
            </a>

        .. image:: ../../_static/torus.png
            :alt: Fault-Tolerant Compilation Tutorial

        +++
        **Fault-Tolerant compilation**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./BigInteger.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Use the BigInteger class to avoid integer overflow in JAX and perform resource estimation of large-scale quantum algorithms.</p>
            </div>
            </a>

        .. image:: ../../_static/order_finding_qre.svg
            :alt: Shor Quantum Resource Estimation Tutorial

        +++
        **Using BigInteger to compile Shor's at 2048 bit**


Scientific Computing and Quantum Numerical Linear Algebra
---------------------------------------------------------
Unlock the potential of quantum computers for scientific discovery and advanced mathematics.
Here, you will use Qrisp's high-level abstractions to tackle complex problems in quantum chemistry and physics.
This track demonstrates how to bridge the gap between complex scientific theory and executable quantum code using features like block encodings as programming abstractions and quantum signal processing.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./H2.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Simulate the quantum dynamics of molecules using chemistry data.</p>
            </div>
            </a>

        .. image:: ../../_static/hydrogen.png
            :alt: Hydrogen Molecule Tutorial

        +++
        **Simulate the dynamics of the hydrogen molecule**

    .. grid-item-card::
        :shadow: sm
        :text-align: center
        :class-body: d-flex justify-content-center align-items-center

        .. raw:: html

            <a href="./BE_tutorial/index.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Use block encodings as programming abstractions to perform Quantum Linear Algebra using a NumPy-like interface.</p>
            </div>
            </a>

        .. image:: ../../_static/BE_thumbnail.png
            :alt: Block Encoding Tutorial
            :align: center

        +++
        **Quantum Linear Algebra**

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./GQSP_filtering.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Prepare ground states of lattice models by applying a Gaussian filter to enhance the overlap of the initial state with the ground state using quantum signal processing.</p>
            </div>
            </a>

        .. image:: ../../_static/filtering_thumb_placeholder.png
            :alt: QSP Filtering Tutorial

        +++
        **Eigenstate filtering using quantum signal processing**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./HHL.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Explore hybrid quantum-classical workflows using Catalyst and Qrisp to solve linear system problems.</p>
            </div>
            </a>

        .. image:: ../../_static/HHL.png
            :alt: HHL Algorithm diagram

        +++
        **Solving systems of linear equations via HHL**


You’ve now explored the breadth of what’s possible when you trade gate-level surgery for high-level logic.
From your first QuantumVariable to fault-tolerant resource estimation, you have the roadmap to develop utility-scale applications that once seemed out of reach.

The quantum landscape is evolving rapidly. By mastering these tutorials, you’re no longer just a spectator, you’re an architect of the next generation of algorithms. 
So, take these concepts, experiment, and start building the future of quantum computing with intuitive, clean, and qrispy code.

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
   CUDAQ.ipynb

