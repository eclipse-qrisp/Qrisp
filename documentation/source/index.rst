.. Qrisp documentation master file, created by
   sphinx-quickstart on Tue May 17 13:01:30 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. |br| raw:: html

   <br />
   
.. raw:: html

    <style>
    /* ===== BASE LAYOUT ===== */
    .bd-sidebar-secondary {
        display: none;
    }

    .wy-nav-content-wrap {
        margin-left: 0;
    }

    .frontpage-equation {
        margin-top: 1.9em;
        margin-bottom: 1.6em;
    }

    .bd-main .bd-content .bd-article-container {
        max-width: 80rem;
    }

    .code-example-text {
        max-width: 60rem;
        margin: 0 auto;
    }

    article.bd-article section {
        padding: 3rem 0 7rem;
    }

    h1:not(.sd-d-none) {
        font-weight: 800;
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #0d1b3e;
        letter-spacing: -0.02em;
    }

    h3:not(#hero h3) {
        font-weight: bold;
        text-align: center;
    }

    .sd-row {
        justify-content: center;
    }

    /* ===== HERO — DARK GRADIENT ===== */
    #hero {
        background: linear-gradient(135deg, #070d1f 0%, #0d1b3e 35%, #122a5e 65%, #0f2347 100%);
        border-radius: 20px;
        padding: 3.5rem 3rem;
        position: relative;
        overflow: hidden;
        color: #ffffff;
        box-shadow: 0 20px 60px rgba(10, 22, 50, 0.3);
    }

    /* Animated grid overlay */
    #hero::before {
        content: '';
        position: absolute;
        inset: -50%;
        width: 200%;
        height: 200%;
        background-image:
            linear-gradient(rgba(100, 180, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(100, 180, 255, 0.04) 1px, transparent 1px);
        background-size: 48px 48px;
        animation: heroGrid 20s linear infinite;
        pointer-events: none;
    }

    @keyframes heroGrid {
        0% { transform: translate(0, 0); }
        100% { transform: translate(48px, 48px); }
    }

    /* Accent glow — top right */
    #hero::after {
        content: '';
        position: absolute;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0, 140, 255, 0.12) 0%, rgba(0, 60, 180, 0.04) 50%, transparent 70%);
        top: -150px;
        right: -80px;
        border-radius: 50%;
        pointer-events: none;
        animation: orbPulse 6s ease-in-out infinite;
    }

    @keyframes orbPulse {
        0%, 100% { opacity: 0.6; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.12); }
    }

    /* Secondary glow — bottom left */
    .hero-orb-secondary {
        position: absolute;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        bottom: -140px;
        left: -100px;
        border-radius: 50%;
        pointer-events: none;
        animation: orbPulse 8s ease-in-out infinite reverse;
        z-index: 0;
    }

    /* ===== HERO LEFT ===== */
    #hero-left {
        position: relative;
        z-index: 2;
    }

    #hero-left .scaled-image {
        filter: brightness(0) invert(1);
        opacity: 0.8;
    }

    #hero-left h2 {
        background: linear-gradient(135deg, #ffffff 0%, #a8d4ff 50%, #ffffff 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 68px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        animation: shimmer 6s ease-in-out infinite;
    }

    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    #hero-left h3 {
        color: rgba(168, 212, 255, 0.9) !important;
        font-weight: 600 !important;
        font-size: 1.15rem;
        letter-spacing: 0.01em;
    }

    #hero-left p {
        color: rgba(200, 220, 255, 0.7);
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* ===== HERO RIGHT — FLOATING CARDS ===== */
    #hero-right {
        position: relative;
        z-index: 2;
    }

    #hero-right .sd-card {
        background: rgba(255, 255, 255, 0.97) !important;
        border: 1px solid rgba(100, 180, 255, 0.15) !important;
        border-radius: 14px !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }

    #hero-right .sd-card:hover {
        border-color: rgba(0, 120, 255, 0.4) !important;
        transform: translateY(-5px);
        box-shadow: 0 14px 44px rgba(0, 80, 200, 0.25);
    }

    #hero-right .example-img-plot-overlay,
    #hero-right .example-img-plot-overlay p.sd-card-text {
        border-radius: 14px;
    }

    /* ===== BUTTONS ===== */
    .homepage-button {
        min-width: 150px;
        padding: 0.65em 1.4em;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none !important;
    }

    .primary-button {
        background: linear-gradient(135deg, #0088ff, #0055cc) !important;
        border: 1px solid transparent !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0, 100, 255, 0.35);
    }

    .primary-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 100, 255, 0.5) !important;
        background: linear-gradient(135deg, #0099ff, #0066dd) !important;
        color: #ffffff !important;
    }

    .secondary-button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
    }

    .secondary-button:hover {
        background: rgba(255, 255, 255, 0.18) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        color: #ffffff !important;
        transform: translateY(-2px);
    }

    .homepage-button-link {
        color: rgba(168, 212, 255, 0.8) !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .homepage-button-link:hover {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    /* ===== STATS STRIP ===== */
    .stats-strip {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        padding: 3.5rem 2rem;
        margin: 0 auto;
        max-width: 80rem;
    }

    .stat-item {
        text-align: center;
        padding: 1.5rem 2.5rem;
        background: linear-gradient(135deg, #f0f6ff 0%, #e8eef8 100%);
        border-radius: 14px;
        border: 1px solid rgba(1, 89, 153, 0.06);
        transition: all 0.3s ease;
        flex: 1;
        min-width: 180px;
        max-width: 250px;
    }

    .stat-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(1, 89, 153, 0.08);
        border-color: rgba(1, 89, 153, 0.12);
    }

    .stat-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #015999;
        display: block;
        letter-spacing: -0.02em;
    }

    .stat-label {
        font-size: 0.82rem;
        color: #5a6a85;
        margin-top: 0.3rem;
        display: block;
        font-weight: 500;
    }

    /* ===== SECTION DECORATIONS ===== */
    .section-divider {
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, #015999, #0099ff);
        border-radius: 2px;
        margin: 0 auto 1rem;
        border: none;
    }

    .section-subtitle {
        text-align: center;
        color: #5a6a85;
        font-size: 1.05rem;
        max-width: 600px;
        margin: 0 auto 3rem;
        line-height: 1.6;
    }

    /* ===== FEATURE CARDS ===== */
    #key-features .sd-card {
        border-radius: 16px !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(1, 89, 153, 0.05) !important;
        background: #ffffff !important;
    }

    #key-features .sd-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(1, 89, 153, 0.08);
        border-color: rgba(1, 89, 153, 0.1) !important;
    }

    #key-features .sd-card img {
        transition: transform 0.3s ease;
    }

    #key-features .sd-card:hover img {
        transform: scale(1.05);
    }

    /* ===== CODE COMPARISON SECTION ===== */
    #dive-into-qrisp-code {
        background: linear-gradient(180deg, #f8fafc 0%, #f0f4f8 100%);
        border-radius: 20px;
        padding: 3rem 2rem !important;
    }

    /* ===== PARTNER LOGOS ===== */
    #who-is-behind-qrisp .sd-card {
        transition: all 0.3s ease;
        border-radius: 12px !important;
    }

    #who-is-behind-qrisp .sd-card:hover {
        transform: translateY(-3px);
    }

    #who-is-behind-qrisp .sd-card img {
        filter: grayscale(20%);
        opacity: 0.85;
        transition: all 0.3s ease;
    }

    #who-is-behind-qrisp .sd-card:hover img {
        filter: grayscale(0%);
        opacity: 1;
    }

    /* ===== ENTRANCE ANIMATIONS ===== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }

    #hero-left {
        animation: fadeInUp 0.8s ease-out;
    }

    #hero-right {
        animation: fadeInUp 0.8s ease-out 0.15s both;
    }

    .stats-strip {
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }

    /* ===== MOBILE ===== */
    @media (max-width: 768px) {
        #hero {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 2rem 1.5rem;
            border-radius: 12px;
        }

        #hero-left,
        #hero-right {
            width: 100%;
        }

        #hero-left h2 {
            font-size: 48px !important;
        }

        #hero-right {
            min-width: unset !important;
        }

        .homepage-button-container {
            margin-top: 1rem;
        }

        img {
            max-width: 76%;
        }

        .stats-strip {
            padding: 2rem 1rem;
        }

        .stat-item {
            padding: 1rem 1.5rem;
            min-width: 140px;
        }

        #dive-into-qrisp-code {
            border-radius: 12px;
            padding: 2rem 1rem !important;
        }
    }

    </style>

    <div id="hero">
        <div class="hero-orb-secondary"></div>
        <div id="hero-left">
            <img alt="./_static/ecplipse_font.png" class="align-bottom-left scaled-image" src="./_static/ecplipse_font.png" width="109" height="25">
            <h2 style="font-size: 68px; font-weight: 800; margin: -0.4rem auto 0;">Qrisp</h2>
            <h3 style="font-weight: 600; margin-top: 0;">The next generation of quantum algorithm development</h3>
            <p>Qrisp is a high-level programming language for creating and compiling quantum algorithms. Its structured programming model enables scalable development and maintenance — with JAX integration for hybrid quantum-classical workflows.</p>

            <div class="homepage-button-container">
                <div class="homepage-button-container-row">
                    <a href="./general/tutorial/index.html" class="homepage-button primary-button">Get Started</a>
                    <a href="./reference/Examples/index.html" class="homepage-button secondary-button">See Examples</a>
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
            
            <a href="./general/tutorial/Jasp.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Exert hybrid real-time algorithm control with Catalyst and Jasp</p>
            </div>
            </a>
            
        .. image:: ./_static/hybrid_realtime.png
            :width: 180
            :align: center


.. grid:: 1 1 2 2

            
    .. grid-item-card::
        :shadow: none
    
        .. raw:: html

            <a href="./general/tutorial/H2.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Simulate the quantum dynamics of molecules using chemistry data</code></p>
            </div>
            </a>

    
        .. image:: ./_static/hydrogen.png

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

            <a href="./reference/Algorithms/QuantumBacktrackingTree.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Solve backtracking problems by leveraging quantum walks</p>
            </div>
            </a>

    
        .. image:: ./_static/backtracking.svg

    .. grid-item-card::
        :shadow: none
    
        .. raw:: html

            <a href="./reference/Primitives/LCU.html">
            <div class="example-img-plot-overlay">
            <p class="sd-card-text">Realize LCU and LCHS using block encodings</p>
            </div>
            </a>

    
        .. image:: ./_static/LCU.png


.. raw:: html


    </div>  <!-- End Hero Right -->
    </div>
    
.. raw:: html

    <div class="stats-strip">
        <div class="stat-item">
            <span class="stat-number">Open Source</span>
            <span class="stat-label">Eclipse Foundation Project</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">100+</span>
            <span class="stat-label">Qubit Simulation</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">JAX</span>
            <span class="stat-label">Hybrid Integration</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">Multi-Backend</span>
            <span class="stat-label">Hardware Agnostic</span>
        </div>
    </div>

Key Features
============

.. raw:: html

    <hr class="section-divider">
    <p class="section-subtitle">Powerful features for scalable quantum algorithm engineering</p>

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
        Compilation results are circuit objects, implying they can be run on a variety of hardware providers such as IQM, IBM Quantum, Quantinuum, Rigetti etc. Further circuit processing is possible using circuit optimizers like PyZX. 
        
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

    <hr class="section-divider">
    <p class="section-subtitle">Write quantum algorithms in fewer lines of more readable code</p>
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
		ClassicalRegister, transpile)
		from qiskit_aer import Aer
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
		qc = transpile(qc, backend)
		counts_dic = backend.run(qc).result().get_counts()
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

Apart from simple scripts like the above, our :ref:`tutorial` showcases the utilization of Qrisp implementing Shor's algorithm leveraging `Montgomery reduction <https://en.wikipedia.org/wiki/Montgomery_modular_multiplication>`_, fully agnostic to the particular quantum adder. This solution involves several conceptually distinct :ref:`QuantumVariables <QuantumVariable>`, with their respective qubits being repeatedly disentangled and repurposed for other variables. The presented approach improves the resource requirement of known open source implementations :ref:`significantly <shor_benchmark_plot>`, while retaining an `accesible form <https://www.qrisp.eu/general/tutorial/Shor.html>`_.

This example illustrates how Qrisp, as a high-level language, permits novel and scalable solutions to intricate problems and furthermore that high-level quantum programming languages will be an integral part of the future of quantum information science.

.. raw:: html

    </div>

Who is behind Qrisp
===================

.. raw:: html

    <hr class="section-divider">
    <div class="code-example-text">
    

Qrisp is an Eclipse open-source project developed mainly at `Fraunhofer FOKUS <https://www.fokus.fraunhofer.de/en/>`_, an industrial research facility based in Berlin. It is publicly funded by the `German ministry of economic affairs <https://www.digitale-technologien.de/DT/Navigation/DE/ProgrammeProjekte/AktuelleTechnologieprogramme/Quanten_Computing/Projekte/Qompiler/qompiler.html>`_ and the European Union with the aim to enable commercial use of quantum computation. To achieve this, we aim to open this field of research to a broad audience of developers. Furthermore we are proud to announce that Qrisp will become a part of the `Eclipse foundation <https://www.eclipse.org/>`_!

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


.. grid:: 1 1 5 5

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/bmwk_logo.png
            :align: center
            :width: 150
            :class: no-scaled-link

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/iqm_logo.jpg
            :align: center
            :width: 150
            :class: no-scaled-link

    .. grid-item-card::
        :class-card: sd-border-0
        :shadow: none
    
        .. image:: ./_static/eu.png
            :align: center
            :width: 150
            :class: no-scaled-link


.. toctree::
   :hidden:
   
   general/tutorial/index
   reference/index
   general/setup
   general/papers/index
   general/changelog/index
   general/imprint
