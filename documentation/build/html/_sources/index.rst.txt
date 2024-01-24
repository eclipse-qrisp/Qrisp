.. Qrisp documentation master file, created by
   sphinx-quickstart on Tue May 17 13:01:30 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. |br| raw:: html

   <br />
   
.. raw:: html

    <style>
    .bd-sidebar-secondary {
        display: none;
    }

    .wy-nav-content-wrap {
        margin-left: 0;
    }

    .frontpage-equation {
        margin-top: 1.9em;
        margin-bottom: 1.6em
    }

    .bd-main .bd-content .bd-article-container {
        max-width: 75rem; /* Make homepage a little wider instead of 60em */
    }

    .code-example-text {
        max-width: 60rem;
        margin: 0 auto;
    }

    /* Extra top/bottom padding to the sections */
    article.bd-article section {
        padding: 3rem 0 7rem;
    }

    /* Override all h1 headers except for the hidden ones */
    h1:not(.sd-d-none) {
        font-weight: bold;
        font-size: 48px;
        text-align: center;
        margin-bottom: 4rem;
    }

    /* Override all h3 headers that are not in hero */
    h3:not(#hero h3) {
        font-weight: bold;
        text-align: center;
    }

    .homepage-button.secondary-button:hover {
        color: white;
    }

    /* New CSS for mobile layout */
    @media (max-width: 768px) {
        #hero {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        #hero-left,
        #hero-right {
            width: 100%;
        }

        .homepage-button-container {
            margin-top: 1rem;
        }
        
        img{
        
            max-width: 76%;
        }
    }
    </style>

    <div id="hero">
        <div id="hero-left">
            <img alt="./_static/ecplipse_font.png" class="align-bottom-left scaled-image" src="./_static/ecplipse_font.png" width="109" height="25">
            <h2 style="font-size: 60px; font-weight: bold; margin: -0.4rem auto 0;">Qrisp</h2>
            <h3 style="font-weight: bold; margin-top: 0; color: #000000;">The next generation of quantum algorithm development</h3>
            <p>Qrisp is a high-level programming language for creating and compiling quantum algorithms. Its structured programming model enables scalable development and maintenance.</p>

            <div class="homepage-button-container">
                <div class="homepage-button-container-row">
                    <a href="./general/tutorial/index.html" class="homepage-button primary-button"; style="color: #FFFFFF;">Get Started</a>
                    <a href="./reference/Examples/index.html" class="homepage-button secondary-button"; style=" .a:hover{color: #FFFFFF}">See Examples</a>
                </div>
                <div class="homepage-button-container-row">
                    <a href="./reference/index.html" class="homepage-button-link">See API Reference →</a>
                </div>
            </div>
        </div>
        
    <div id="hero-right">

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none
        
        .. raw:: html

            <a href="./reference/Algorithms/QuantumBacktrackingTree.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Solve backtracking problems by leveraging quantum walks</p>
            </div>
            </a>

    
        .. image:: ./_static/backtracking.svg
            


    .. grid-item-card::
        :shadow: none
    
        .. raw:: html

            <a href="./reference/Algorithms/Shor.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Factor integers using abitrary quantum adders with Shor's algorithm</p>
            </div>
            </a>

    
        .. image:: ./_static/period_finding.svg

.. grid:: 1 1 2 2

    .. grid-item-card::
        :shadow: none
            
    
        .. raw:: html
            
            <a href="./reference/Core/generated/qrisp.QuantumSession.statevector.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Explore parametrized algorithms with symbolic statevector simulation</p>
            </div>
            </a>
            <div class="frontpage-equation">
            
            
        .. math::
            
            \definecolor{qrispblue}{RGB}{32, 48, 111}
            \definecolor{textgray}{RGB}{68, 68, 68}
            
            \Large
            \textcolor{textgray}{
            \begin{align}
            \frac{\ket{\texttt{hello}} + e^{i \textcolor{red}{\phi}} \ket{\texttt{world}}}{\sqrt{2}}
            \end{align}
            }
            
        .. raw:: html
            
            </div>
            

    .. grid-item-card::
        :shadow: none
    
        .. raw:: html

            <a href="./general/tutorial/TSP.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Solve the traveling salesman problem using quantum search</code></p>
            </div>
            </a>

    
        .. image:: ./_static/tsp.svg


.. raw:: html


    </div>  <!-- End Hero Right -->
    </div>
    
|
|
|

Key Features
============

.. grid:: 1 1 2 2
    :gutter: 5


    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
        
        .. image:: ./_static/typed_variables_pic.png
            :target: ./reference/Quantum%20Types/index.html
            :align: center
    
    
        .. raw:: html
        
            <div class="key-features-text">
        
        
        **Typed quantum variables** |br|
        Qrisp algorithms are constituted of variables and functions instead of qubits and circuits, which helps you to structure your code and avoid technical debt.
       
        .. raw:: html
        
            </div>

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
        
        .. image:: ./_static/recycling.png
            :target: ./reference/Core/Uncomputation.html
            :align: center

        .. raw:: html
        
            <div class="key-features-text">
        

    
        **Automatic uncomputation** |br|
        QuantumVariables can be uncomputed automatically once they are not needed anymore. The uncomputation module is tightly integrated with an advanced qubit resource management system.
        
        .. raw:: html
        
            </div>

        
    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
        
        .. image:: ./_static/puzzle.png
            :align: center

        .. raw:: html
        
            <div class="key-features-text">

    
        **Modularity** |br|
        Automated qubit allocation allows separate modules to recycle qubit resources for each other without intertwining the code. This feature facilitates interoperability of code written by respective domain experts.

        .. raw:: html
        
            </div>
        
    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
        
        .. image:: ./_static/abakus.png
            :target: ./reference/Quantum%20Types/QuantumFloat.html
            :align: center


        .. raw:: html
        
            <div class="key-features-text">
            
    
        **Arithmetic** |br|
        A smoothly integrated system of floating point arithmetic streamlines the development of non-trivial applications.

        .. raw:: html
        
            </div>

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
        
        .. image:: ./_static/plug.png
            :align: center

        .. raw:: html
        
            <div class="key-features-text">
    
        **Compatibility** |br|
        Compilation results are circuit objects, implying they can be run on a variety of hardware providers such as IBM Quantum, Quantinuum, Rigetti etc. Further circuit processing is possible using circuit optimizers like PyZX. 
        
        .. raw:: html
        
            </div>
        
    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
        
        .. image:: ./_static/quantumcomputer.png
            :align: center

        .. raw:: html
        
            <div class="key-features-text">

    
        **Simulator** |br|
        Qrisp ships with a high-performance simulator, that utilizes sparse matrices to store and process quantum states. This allows for the simulation of :ref:`(some) quantum circuits <SimulationExample>` involving 100+ qubits.
        
        .. raw:: html
        
            </div>
        





Dive into Qrisp code
====================

.. raw:: html

    <div class="code-example-text">
    

Qrisp code can be significantly shorter and also more readable than the equivalent program in a gate based framework. Regardless, the compiled quantum circuits are in many cases more efficient since the compiler can leverage the code structure into assumptions about algorithmic structure, implying strong compilation optimizations. Below you find two code snippets that perform the same basic task: Multiplication of two n-bit integers. One is written using Qiskit and the other one utilizes Qrisp.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - .. image:: ./qiskit_logo.png
         :width: 100
         :alt: Qrisp Logo
         :align: center
     - .. image:: ./qrisp_logo.png
         :width: 100
         :alt: Qiskit Logo
         :align: center
   * - ::
         
         from qiskit import (QuantumCircuit, QuantumRegister, 
         ClassicalRegister, Aer, execute)
         from qiskit.circuit.library import RGQFTMultiplier
         n = 6
         a = QuantumRegister(n)
         b = QuantumRegister(n)
         res = QuantumRegister(2*n)
         cl_res = ClassicalRegister(2*n)
         qc = QuantumCircuit(a, b, res, cl_res)
         for i in range(len(a)):
             if 3 & 1<<i: qc.x(a[i]) 
         for i in range(len(b)):
             if 4 & 1<<i: qc.x(b[i]) 
         qc.append(RGQFTMultiplier(n, 2*n), 
         list(a) + list(b) + list(res))
         qc.measure(res, cl_res)
         backend = Aer.get_backend('qasm_simulator')
         counts_dic = execute(qc, backend).result().get_counts()
         print({int(k, 2) : v for k, v in counts_dic.items()})
         #Yields: {12: 1024}
         
         
     - ::
   
         from qrisp import QuantumFloat
         n = 6
         a = QuantumFloat(n)
         b = QuantumFloat(n)
         a[:] = 3
         b[:] = 4
         res = a*b
         print(res)
         #Yields: {12: 1.0}

Apart from simple scripts like the above, our :doc:`tutorial <general/tutorial/TSP>` showcases the utilization of Qrisp in solving the traveling salesman problem. This solution involves over 10 distinct :ref:`QuantumVariables <QuantumVariable>`, with their respective qubits being repeatedly disentangled and repurposed for other variables. The presented approach scales better in the qubit count than the previously known QUBO based solution:  $\mathcal{O}(n \text{log}(n))$ vs. $\mathcal{O}(n^2)$. 

This example illustrates how Qrisp, as a high-level language, permits novel and scalable solutions to intricate problems and furthermore that high-level quantum programming languages will be an integral part of the future of quantum information science.

.. raw:: html

    </div>

Who is behind Qrisp
===================

.. raw:: html

    <div class="code-example-text">
    

Qrisp is an open-source project developed at `Fraunhofer FOKUS <https://www.fokus.fraunhofer.de/en/>`_, an industrial research facility based in Berlin. It is publicly funded by the German ministry of econmic affairs with the aim to enable commercial use of quantum computation. To achieve this, we aim to open this field of research to a broad audience of developers. Furthermore we are proud to announce that Qrisp will become a part of the `Eclipse foundation <https://www.eclipse.org/>`_!

.. raw:: html

    </div>

.. grid:: 1 1 5 5

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/dlr_logo.svg
            :align: center
            :width: 150
            :class: no-scaled-link

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/fraunhofer_fokus_logo.png
            :align: center
            :width: 150
            :class: no-scaled-link
            
    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/incubating.png
            :align: center
            :width: 150
            :class: no-scaled-link
            
    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/eleqtron_logo.png
            :align: center
            :width: 150
            :class: no-scaled-link

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/bmwk_logo.png
            :align: center
            :width: 150
            :class: no-scaled-link
            

.. toctree::
   :hidden:
   
   general/tutorial/index
   reference/index
   general/setup
   general/changelog/index
   general/imprint
 
