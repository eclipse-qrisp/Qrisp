.. _tutorial:

Tutorials
=========

.. raw:: html

    <style>
        /* Isolate the hover overlay effect so it doesn't break Sphinx layouts */
        .tutorial-hover-overlay {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background: rgb(0 14 72);
            color: #ffffff;
            opacity: 0;
            transition: opacity 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            z-index: 10;
            text-decoration: none !important;
        }
        
        .sd-card:hover .tutorial-hover-overlay {
            opacity: 0.95;
        }
        
        .tutorial-hover-overlay p.sd-card-text {
            color: #ffffff !important;
            margin: 0;
            text-align: center;
            font-size: 0.95rem;
        }
    </style>
 
Welcome to the Qrisp tutorials! This gallery is designed to take you from a curious beginner to a proficient developer of utility-scale quantum algorithms. By shifting the focus from low-level gate gymnastics to high-level programming with :ref:`QuantumVariables <QuantumVariable>`, Qrisp allows you to express complex logic with ease.

The tutorials are organized into four topics, each containing multiple hands-on examples that build on each other:

- **Foundations and first Steps**: Start here to master the core syntax of Qrisp, where you will learn to replace manual circuit building with high-level variables and implement textbook algorithms like Shor's and Grover's.

- **Optimization and Numerics**: Explore how to solve combinatorial optimization problems in logistics and scheduling using hybrid approaches like QAOA, QIRO, and COLD, as well as learning about our qrispy Quantum Backtracking implementation.

- **High-Performance Compilation and Execution with Jasp**: Learn how to scale your code, while estimating your quantum resources, for the real world using the Jasp pipeline for hybrid real-time control. You will also learn about optimizing for Fault-Tolerant compilation for next-generation hardware.

- **Scientific Computing and Quantum Numerical Linear Algebra**: Dive into high-level abstractions for quantum chemistry, physics and linear systems (HHL, CKS, QSP) using our new :ref:`BlockEncoding` class.

If you’re the type who learns best by breaking things (and then fixing them), you can download any of these tutorials as a Jupyter notebook. Just look for the download box on the right side of the page within each specific tutorial to grab the code and run it in your own local environment.

By the end of these tutorials, you'll have a solid foundation of our high-level framework and be ready to tackle more complex projects. So let's get **started**!


Foundations and first steps
---------------------------
Kickstart your quantum programming journey here! This section is designed to familiarize you with Qrisp's core philosophy: shifting away from low-level circuit manipulation and toward intuitive, high-level programming using QuantumVariables. 

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
Discover how quantum algorithms can tackle complex search and optimization problems. In this section, you will dive into solving famous computational challenges (such as the Traveling Salesman Problem and Sudoku) using a versatile suite of quantum tools. 

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
            <p class="sd-card-text">Apply the theory to implement the algorithm and solve a Maximal Independent Set problem with it!</p>
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
            <p class="sd-card-text">Solve a QUBO problem with LCD (local counterdiabatic driving) and COLD (counterdiabatic optimized local driving).</p>
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
            <p class="sd-card-text">Learn how to efficiently implement the Sudoku problem specific quantum oracles using Quantum Backtracking.</p>
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
Scale your quantum algorithms to practically relevant problem sizes by bypassing Python’s performance bottlenecks. You will explore the Jasp pipeline for hybrid real-time control, learn how to build highly performant custom optimization algorithms, and delve into Fault-Tolerant compilation to ensure your code is ready for next-generation hardware.

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
            <p class="sd-card-text">Build an efficient custom QAOA implementation in Jasp using a Jasp-traceable objective function for the MaxCut problem.</p>
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
            <p class="sd-card-text">Use the BigInteger class to avoid integer overflow in Jasp and perform resource estimation of large-scale quantum algorithms.</p>
            </div>
            </a>

        .. image:: ../../_static/order_finding_qre.svg
            :alt: Shor Quantum Resource Estimation Tutorial

        +++
        **Using BigInteger to compile Shor's at 2048 bit**


Scientific Computing and Quantum Numerical Linear Algebra
---------------------------------------------------------
Unlock the potential of quantum computers for scientific discovery and advanced mathematics. Here, you will use Qrisp's high-level abstractions to tackle problems in quantum chemistry, while having block encodings as a programming abstraction for quantum numerical linear algebra. 

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./H2.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Leverage Qrisp’s advanced capabilities to perform molecular simulations on quantum computers.</p>
            </div>
            </a>

        .. image:: ../../_static/hydrogen.png
            :alt: Hydrogen Molecule Tutorial

        +++
        **Simulate the dynamics of the $H_2$ molecule**

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./BE_tutorial/index.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Use block encodings as programming abstractions to perform Quantum Linear Algebra using a NumPy-like interface.</p>
            </div>
            </a>

        .. image:: ../../_static/BE_thumbnail.png
            :alt: Block Encoding Tutorial

        +++
        **Quantum Linear Algebra**

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: sm
        :text-align: center

        .. raw:: html

            <a href="./GQSP_filtering.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Prepare the ground state by applying a Gaussian filter to enhance the overlap of the prepared state using GQSP.</p>
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
            <p class="sd-card-text">Explore hybrid quantum-classical workflows using Catalyst and Qrisp to solve linear system problems Ax=b.</p>
            </div>
            </a>

        .. image:: ../../_static/HHL.png
            :alt: HHL Algorithm diagram

        +++
        **Solving systems of linear equations via HHL**


You’ve now explored the breadth of what’s possible when you trade gate-level surgery for high-level logic. From your first QuantumVariable to fault-tolerant resource estimation, you have the roadmap to develop utility-scale applications that once seemed out of reach.

The quantum landscape is evolving rapidly. By mastering these tutorials, you’re no longer just a spectator, you’re an architect of the next generation of algorithms. So, take these concepts, experiment, and start building the future of quantum computing with intuitive, clean, and Qrisp code.

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
