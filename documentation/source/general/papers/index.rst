.. _research:

Research
--------

Below, we present a collection of scientific papers and articles that have either contributed to the development of Qrisp or highlight its applications and advantages in the field of quantum computing. These citations provide a foundation for understanding the theoretical underpinnings and practical implications of Qrisp. We encourage researchers, developers, and enthusiasts to explore these resources to gain a deeper insight into the capabilities and potential of Qrisp.

Research from within the Qrisp community
========================================

.. grid::

    .. grid-item-card:: Qrisp: A Framework for Compilable High-Level Programming of Gate-Based Quantum Computers

        | Raphael Seidel, Sebastian Bock, René Zander, Matic Petrič, Niklas Steinmann, Nikolay Tcholtchev, Manfred Hauswirth 
        | `ArXiv, 2024 <https://arxiv.org/abs/2406.14792>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            While significant progress has been made on the hardware side of quantum computing, 
            support for high-level quantum programming abstractions remains underdeveloped compared to classical programming languages. 
            In this article, we introduce Qrisp, a framework designed to bridge several gaps between high-level programming paradigms in 
            state-of-the-art software engineering and the physical reality of today's quantum hardware. The framework aims to provide a systematic 
            approach to quantum algorithm development such that they can be effortlessly implemented, maintained and improved. We propose a number of 
            programming abstractions that are inspired by classical paradigms, yet consistently focus on the particular needs of a quantum developer. 
            Unlike many other high-level language approaches, Qrisp's standout feature is its ability to compile programs to the circuit level, making 
            them executable on most existing physical backends. The introduced abstractions enable the Qrisp compiler to leverage algorithm structure 
            for increased compilation efficiency. Finally, we present a set of code examples, including an implementation of Shor's factoring 
            algorithm. For the latter, the resulting circuit shows significantly reduced quantum resource requirements, strongly supporting the claim that 
            systematic quantum algorithm development can give quantitative benefits.

.. grid::
  
    .. grid-item-card:: Double-Bracket Quantum Algorithms for Quantum Imaginary-Time Evolution

        | Marek Gluza, Jeongrak Son, Bi Hong Tiang, René Zander, Raphael Seidel, Yudai Suzuki, Zoë Holmes, Nelly H. Y. Ng
        | `Physical Review Letters, 2026 <https://link.springer.com/chapter/10.1007/978-3-032-03924-8_11>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Efficiently preparing approximate ground states of large, strongly correlated systems on quantum hardware is challenging, and yet, nature is innately adept at this. 
            This has motivated the study of thermodynamically inspired approaches to ground-state preparation that aim to replicate cooling processes via imaginary-time evolution. 
            However, synthesizing quantum circuits that efficiently implement imaginary-time evolution is itself difficult, 
            with prior proposals generally adopting heuristic variational approaches or using deep block encodings. 
            Here, we use the insight that quantum imaginary-time evolution is a solution of Brockett’s double-bracket flow 
            and synthesize circuits that implement double-bracket flows coherently on the quantum computer. 
            We prove that our double-bracket quantum imaginary-time evolution (DB-QITE) algorithm inherits the cooling guarantees of imaginary-time evolution. 
            Concretely, each step is guaranteed to (i) decrease the energy of an initial approximate ground state by an amount proportional to the energy fluctuations of the initial state 
            and (ii) increase the fidelity with the ground state. We provide gate counts for DB-QITE through numerical simulations in qrisp that demonstrate scenarios where DB-QITE outperforms quantum phase estimation. 
            Thus, DB-QITE provides a means to systematically improve the approximation of a ground state using shallow circuits.

.. grid::
  
    .. grid-item-card:: Methods and Tools for Secure Quantum Clouds with a specific Case Study on Homomorphic Encryption

        | Aurelia Kusumastuti, Nikolay Tcholtchev, Philipp Lämmel, Sebastian Bock, Manfred Hauswirth
        | `ArXIv, 2025 <https://arxiv.org/abs/2512.17748>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            The rise of quantum computing/technology potentially introduces significant security challenges to cloud computing, 
            necessitating quantum-resistant encryption strategies as well as protection schemes and methods for cloud infrastructures offering quantum computing time and services (i.e. quantum clouds). 
            This research explores various options for securing quantum clouds and ensuring privacy, especially focussing on the integration of homomorphic encryption (HE) into Eclipse Qrisp, 
            a high-level quantum computing framework, to enhance the security of quantum cloud platforms. The study addresses the technical feasibility of integrating HE with Qrisp, 
            evaluates performance trade-offs, and assesses the potential impact on future quantum cloud architectures. 
            The successful implementation and Qrisp integration of three post-quantum cryptographic (PQC) algorithms demonstrates the feasibility of integrating HE with quantum computing frameworks. 
            The findings indicate that while the Quantum One-Time Pad (QOTP) offers simplicity and low overhead, 
            other algorithms like Chen and Gentry-Sahai-Waters (GSW) present performance trade-offs in terms of runtime and memory consumption. 
            The study results in an overall set of recommendations for securing quantum clouds, e.g. implementing HE at data storage and processing levels, 
            developing Quantum Key Distribution (QKD), and enforcing stringent access control and authentication mechanisms as well as participating in PQC standardization efforts.

.. grid::
  
    .. grid-item-card:: Role of Riemannian geometry in double-bracket quantum imaginary-time evolution

        | René Zander, Raphael Seidel, Li Xiaoyue, Marek Gluza
        | `International Conference on Geometric Science of Information, 2025 <https://link.springer.com/chapter/10.1007/978-3-032-03924-8_11>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Double-bracket quantum imaginary-time evolution (DB-QITE) is a quantum algorithm which coherently implements steps in the Riemannian steepest-descent direction for the energy cost function. 
            DB-QITE is derived from Brockett's double-bracket flow which exhibits saddle points where gradients vanish. 
            In this work, we perform numerical simulations of DB-QITE and describe signatures of transitioning through the vicinity of such saddle points. 
            We provide an explicit gate count analysis using quantum compilation programmed in Qrisp.

.. grid::
  
    .. grid-item-card:: End-to-end compilable implementation of quantum elliptic curve logarithm in Qrisp

        | Diego Polimeni, Raphael Seidel
        | `ArXiv, 2025 <https://arxiv.org/abs/2501.10228>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Elliptic curve cryptography (ECC) is a widely established cryptographic technique, recognized for its effectiveness and 
            reliability across a broad range of applications such as securing telecommunications or safeguarding cryptocurrency wallets. 
            Although being more robust than RSA, ECC is, nevertheless, also threatened by attacks based on Shor's algorithm, which made it 
            a popular field of study in quantum information science. A variety of techniques have been proposed to perform EC arithmetic in quantum devices; 
            however, software support for compiling these algorithms into executables is extremely limited. Within this work, we leverage 
            the Qrisp programming language to realize one of the first fully compilable implementations of EC arithmetic and verify its correctness 
            using Qrisp's built-in sparse matrix simulator. 

.. grid::
  
    .. grid-item-card:: Designing a Meta-Model for the Eclipse Qrisp eDSL for High-Level Quantum Programming

        | Sebastian Bock, Raphael Seidel, Matic Petrič , Nikolay Tcholtchev, Andreas Hoffmann and Niklas Porges
        | `MODELSWARD 2025, 13th International Conference on Model-Based Software and Systems Engineering. <https://www.scitepress.org/Papers/2025/131210/131210.pdf>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Eclipse Qrisp is a high-level programming language designed to simplify quantum programming and make it accessible to a wider range of developers and end users. 
            Initially developed at Fraunhofer FOKUS and now part of the Eclipse Foundation, Eclipse Qrisp abstracts complex quantum operations into user-friendly constructs, 
            enhancing code readability structure. Currently, Eclipse Qrisp is realized as an extension of the Python programming language, in the form of an embedded Domain Specific Language (eDSL), 
            allowing to develop hybrid quantum algorithms, while at the same time utilizing the potential of the overall Python ecosystem in terms of libraries and available developer resources. 
            We firmly believe that the eDSL approach to high-level quantum programming will prevail over the idea of defining specific languages-with their own grammar and ecosystem-due to its ease of integration within available ICT products and services. 
            However, in order to reach higher levels of scalability and market penetration, the Eclipse Qrisp eDSL should be available for various platforms and programming languages beyond Python,
            e.g. C/C++, Java or Rust. In order to provide the means for implementing Eclipse Qrisp in other programming languages, this paper specifies a meta-model, thereby outlining the pursued design philosophy, 
            architecture, and key features, including compatibility with existing frameworks. The purpose of such a Qrisp meta-model is two-fold: On one hand it formalizes and standardizes the Eclipse Qrisp programming model. 
            On the other hand, such a meta-model can be used to formally extend other programming languages and platforms by the capabilities and concepts specified and implemented within Eclipse Qrisp.

.. grid::

    .. grid-item-card:: Automatic quantum function parallelization and memory management in Qrisp

        | Raphael Seidel
        | `6th International Workshop on Quantum Compilation, 2024 <https://quantum-compilers.github.io/iwqc2024/papers/IWQC2024_paper_16.pdf>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Automated optimization of quantum programs has gathered significant attention amidst
            the recent advances of hardware manufacturers. In this work we introduce a novel data-
            structure for representing quantum programs called permeability DAG, which captures several useful properties of quantum programs
            across multiple levels of abstraction. Operating on this representation facilitates a variety of powerful transformations such as 
            automatic parallelization, memory management and synthesis of uncomputation. More potential use-cases are listed in the outlook section.
            At the core, our representation abstracts away a class of non-trivial commutation relations,
            which stem from a feature called permeability. Both memory management and parallelization
            can be made sensitive to execution speed details of each particular quantum gate, implying our
            compilation methods are not only retargetable between NISQ/FT but even for individual device instances.

.. grid::

    .. grid-item-card:: Solving the Product Breakdown Structure Problem with constrained QAOA

        | René Zander, Raphael Seidel, Matteo Inajetovic, Niklas Steinmann, Matic Petrič 
        | `ArXiv, 2024 <https://arxiv.org/abs/2406.15228>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Constrained optimization problems, where
            not all possible variable assignments are feasible solutions, comprise numerous practically
            relevant optimization problems such as the
            Traveling Salesman Problem (TSP), or portfolio optimization. Established methods such
            as quantum annealing or vanilla QAOA usually transform the problem statement into a
            QUBO (Quadratic Unconstrained Binary Optimization) form, where the constraints are
            enforced by auxiliary terms in the QUBO objective. Consequently, such approaches fail to
            utilize the additional structure provided by the
            constraints.
            In this paper, we present a method for solving the industry relevant Product Breakdown
            Structure problem. Our solution is based
            on constrained QAOA, which by construction
            never explores the part of the Hilbert space
            that represents solutions forbidden by the problem constraints. The size of the search space is
            thereby reduced significantly. We experimentally show that this approach has not only a
            very favorable scaling behavior, but also appears to suppress the negative effects of Barren
            Plateaus.

.. grid::

    .. grid-item-card:: **Quantum Backtracking in Qrisp Applied to Sudoku Problems** 

        | Raphael Seidel, René Zander, Matic Petrič, Niklas Steinmann, David Q.\ Liu, Nikolay Tcholtchev, Manfred Hauswirth
        | `ArXiv, 2024 <https://arxiv.org/abs/2402.10060>`__ 

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            The quantum backtracking algorithm proposed by Ashley Montanaro raised considerable interest, as it provides a 
            quantum speed-up for a large class of classical optimization algorithms. It does not suffer from Barren-Plateaus 
            and transfers well into the fault-tolerant era, as it requires only a limited number of arbitrary angle gates. 
            Despite its potential, the algorithm has seen limited implementation efforts, presumably due to its abstract 
            formulation. In this work, we provide a detailed instruction on implementing the quantum step operator for 
            arbitrary backtracking instances. For a single controlled diffuser of a binary backtracking tree with depth n, 
            our implementation requires only 6n+14 CX gates. We detail the process of constructing accept and reject 
            oracles for Sudoku problems using our interface to quantum backtracking. The presented code is written using 
            Qrisp, a high-level quantum programming language, making it executable on most current physical backends and 
            simulators. Subsequently, we perform several simulator based experiments and demonstrate solving 4x4 Sudoku 
            instances with up to 9 empty fields. This is, to the best of our knowledge, the first instance of a compilable 
            implementation of this generality, marking a significant and exciting step forward in quantum software engineering.

.. grid::

    .. grid-item-card:: Uncomputation in the Qrisp high-level Quantum Programming Framework

        | Raphael Seidel, Nikolay Tcholtchev, Sebastian Bock, Manfred Hauswirth
        | `International Conference on Reversible Computation, 2023 <https://link.springer.com/chapter/10.1007/978-3-031-38100-3_11>`__

        .. dropdown:: :fa:`eye me-1` Abstract
            :color: primary

            Uncomputation is an essential part of reversible computing and plays a vital role in quantum computing. 
            Using this technique, memory resources can be safely deallocated without performing a nonreversible deletion process. 
            For the case of quantum computing, several algorithms depend on this as they require disentangled states in the course of 
            their execution. Thus, uncomputation is not only about resource management, but is also required from an algorithmic point 
            of view. However, synthesizing uncomputation circuits is tedious and can be automated. In this paper, we describe the 
            interface for automated generation of uncomputation circuits in our Qrisp framework. Our algorithm for synthesizing uncomputation 
            circuits in Qrisp is based on an improved version of "Unqomp", a solution presented by Paradis et. al. Our paper also presents some 
            improvements to the original algorithm, in order to make it suitable for the needs of a high-level programming framework. Qrisp 
            itself is a fully compilable, high-level programming language/framework for gate-based quantum computers, which abstracts from 
            many of the underlying hardware details. Qrisp's goal is to support a high-level programming paradigm as known from classical software development.       


External research utilizing or citing Qrisp
===========================================

.. list-table::
    :widths: 50 30 10
    :header-rows: 1
    
    * - Title
      - Authors
      - Year
    * - `Integrating Quantum Software Tools with (in) MLIR <https://dl.acm.org/doi/full/10.1145/3773656.3773658>`_
      - P\. Hopf, E\. Ochoa, Y\. Stade, D\. Rovara, N\. Quetschlich, I\. A\. Florea, J\. Izaac, R\. Wille, L\. Burgholzer
      - 2026
    * - `Leveraging quantum computing for heat conduction analysis: A case study in thermal engineering <https://www.sciencedirect.com/science/article/pii/S2214157X26001759>`_
      - P\. Asinari, M\. M\. Piredda, G\. Barletta, P\. De Angelis, N\. Alghamdi, G\. Trezza, M\. Provenzano, M\. Fasano, E\. Chiavazzo
      - 2026
    * - `Investigating Retargetability Claims for Quantum Compilers <https://arxiv.org/abs/2601.16779>`_
      - L\. Southall, J\. Ammermann, R\. Kelmendi, D\. Eichhorn, I\. Schaefer
      - 2026
    * - `The grand challenge of quantum applications <https://arxiv.org/abs/2511.09124>`_
      - R\. Babbush, R\. King, S\. Boixo, W\. Huggins, T\. Khattar, G\. H\. Low, J\. R\. McClean, T\. O'Brien, N\. C\. Rubin
      - 2025
    * - `Analysis of Surface Code Algorithms on Quantum Hardware Using the Qrisp Framework <https://www.mdpi.com/2079-9292/14/23/4707>`_
      - J\. Krzyszkowski, M\. Niemiec
      - 2025
    * - `Qrisp Implementation and Resource Analysis of a T-Count-Optimized Non-Restoring Quantum Square-Root Circuit <https://arxiv.org/pdf/2507.12603>`_
      - H\. Kupryianau, M\. Niemiec
      - 2025
    * - `"A Framework for Debugging Quantum Programs <https://ieeexplore.ieee.org/abstract/document/11134329>`_
      - D\. Rovara, L\. Burgholzer, R\. Wille
      - 2025
    * - `"Pilot-Quantum: A Middleware for Quantum-HPC Resource, Workload and Task Management <https://ieeexplore.ieee.org/abstract/document/11044846>`_
      - P\. Mantha, F\. J\. Kiwit, N\. Saurabh. S\. Jha. A\. Luckow
      - 2025
    * - `Quantum Computing Today: A Status Overview <https://ieeexplore.ieee.org/document/11285931>`_
      - Y\. Liu
      - 2025
    * - `Provideq: A quantum optimization toolbox <https://www.computer.org/csdl/proceedings-article/qsw/2025/672000a206/29zHhWKtMUU>`_
      - D\. Eichhorn, N\. Poser, M\. Schweikart, I\. Schaefer
      - 2025
    * - `Qwerty: A Basis-Oriented Quantum Programming Language <https://ieeexplore.ieee.org/document/11285931>`_
      - A\. J\. Adams, S\. Khan, A\. S\. Bhamra, R\. R\. Abusaada, T\. S\. Humble, J\. S\. Young, T\. M\. Conte
      - 2025  
    * - `QML-Essentials–A Framework for Working with Quantum Fourier Models <https://ieeexplore.ieee.org/abstract/document/11134334>`_
      - M\. Strobl, M\. Franz, E\. Kuehn, W\. Mauerer, A\. Streit
      - 2025  
    * - `SLURM Heterogeneous Jobs for Hybrid Classical-Quantum Workflows <https://arxiv.org/abs/2506.03846>`_
      - Y\. Liu
      - 2025  
    * - `First Practical Experiences Integrating Quantum Computers with HPC Resources: A Case Study With a 20-qubit Superconducting Quantum Computer <https://dl.acm.org/doi/full/10.1145/3731599.3767551>`_
      - E\. Mansfield, S\. Seegerer, P\. Vesanen, J\. Echavarria, M\. N\. Farooqi, B\. Mete, L\. Schulz
      - 2025
    * - `QML-ESSENTIALS-A Framework for Working with Quantum Fourier Models <https://www.lfdr.de/Publications/2025/StFrKu+25.pdf>`_
      - M\. Strobl, M\. Franz, E\. Kuehn, W\. Mauerer, A\. Streit
      - 2025
    * - `The Internet of Quantum Things (IoQT)-A New Frontier in Quantum Emulation and Simulation <https://ceur-ws.org/Vol-3962/paper43.pdf>`_
      - I\. Kefaloukos, N\. Tcholtchev, M\.A\. Kourtis, G\. Oikonomakis
      - 2025
    * - `Verifiable End-to-End Delegated Variational Quantum Algorithms <https://arxiv.org/abs/2504.15410>`_
      - M\. Inajetovic, P\. Wallten, A\. Pappa
      - 2025
    * - `Is Productivity in Quantum Programming Equivalent to Expressiveness? <https://arxiv.org/abs/2504.08876v2>`_
      - F\. Corrales-Garro, D\. Valerio-Ramírez, et al.
      - 2025
    * - `Exploration of Design Alternatives for Reducing Idle Time in Shor's Algorithm: A Study on Monolithic and Distributed Quantum Systems <https://arxiv.org/abs/2503.22564>`_
      - M\. Schmidt, A\. Kole, L\. Wichette, R\. Drechsler, F\. Kirchner, E\. Mounzer
      - 2025
    * - `Scalable Memory Recycling for Large Quantum Programs <https://arxiv.org/abs/2503.00822>`_
      - I\. Reichental, R\. Alon, L\. Preminger, M\. Vax
      - 2025
    * - `A parameter study for LLL and BKZ with application to shortest vector problems <https://arxiv.org/abs/2502.05160>`_
      - T\. Köppl, R\. Zander, L\. Henkel, N\. Tcholtchev
      - 2025
    * - `Solving Drone Routing Problems with Quantum Computing: A Hybrid Approach Combining Quantum Annealing and Gate-Based Paradigms <https://arxiv.org/abs/2501.18432>`_
      - E\. Osaba, P\. Miranda-Rodriguez, A\. Oikonomakis
      - 2025
    * - `A Static Analysis of Entanglement <https://link.springer.com/chapter/10.1007/978-3-031-82703-7_3>`_
      - N\. Assolini, A\. Di Pierro, I\. Mastroeni
      - 2025
    * - `CQ: A high-level imperative classical-quantum programming language <https://www.thi.uni-hannover.de/fileadmin/thi/abschlussarbeiten/2025/Bachelorarbeit_Lennart_Binkowski_Website.pdf>`_
      - L\. Binkowski, H\. Vollmer
      - 2025
    * - `Static Analysis of Quantum Programs <https://link.springer.com/chapter/10.1007/978-3-031-74776-2_1>`_
      - N\. Assolini, A\. Di Pierro, I\. Mastroeni
      - 2025
    * - `Qurts: Automatic Quantum Uncomputation by Affine Types with Lifetime <https://dl.acm.org/doi/abs/10.1145/3704842>`_
      - K\. Hirata, C\. Heunen
      - 2025
    * - `D-Wave's Nonlinear-Program Hybrid Solver: Description and Performance Analysis <https://ieeexplore.ieee.org/abstract/document/10820320>`_
      - E\. Osaba, P\. Miranda-Rodriguez
      - 2025
    * - `Improving Quantum Developer Experience with Kubernetes and Jupyter Notebooks <https://ieeexplore.ieee.org/document/10821037>`_
      - O\. Kinaen, A\. D\. Muñoz-Moller, V\. Stirbu et al.
      - 2024
    * - `A Framework for Debugging Quantum Programs <https://arxiv.org/abs/2412.12269>`_
      - D\. Rovara, L\. Burgholzer, R\. Wille
      - 2024
    * - `Resilience of lattice-based Cryptosystems to Quantum Attacks <https://ieeexplore.ieee.org/document/10733570>`_
      - T\. Köppl, R\. Zander, N\. Tcholtchev
      - 2024
    * - `Quff: A Dynamically Typed Hybrid Quantum-Classical Programming Language <https://dl.acm.org/doi/abs/10.1145/3679007.3685063>`_
      - C\. J\. Wright, M\. Luján, P\. Petoumenos et al.
      - 2024
    * - `Quantum types: going beyond qubits and quantum gates <https://dl.acm.org/doi/abs/10.1145/3643667.3648225?casa_token=b2839-ZSiG8AAAAA:IW2Fb22kgZlzyuvK8gFIs7ZprzZwsRZGbwWMdTbho1Keh3u2Ul39GuFgU2h9b4mXdQmaD5Cd1Dg1Fw>`_
      - T\. Varga, Y\. Aragonés-Soria, M\. Oriol
      - 2024
    * - `Towards Continuous Development for Quantum Programming in Decentralized IoT environments <https://www.sciencedirect.com/science/article/pii/S1877050924012286>`_
      - M\. Kourtis, N\. Tcholtchev, I.D\. Gheorghe-Pop et al.
      - 2024
    * - `Testing multi-subroutine quantum programs: From unit testing to integration testing <https://dl.acm.org/doi/full/10.1145/3656339>`_
      - P\. Long, J\. Zhao
      - 2024
    * - `The T-Complexity Costs of Error Correction for Control Flow in Quantum Computation <https://dl.acm.org/doi/abs/10.1145/3656397>`_
      - C\. Yuan, M\. Carbin
      - 2024
    * - `Eclipse Qrisp QAOA: description and preliminary comparison with Qiskit counterparts <https://arxiv.org/abs/2405.20173>`_
      - E\. Osaba, M\. Petrič, I.\ Oregi et al.
      - 2024
    * - `An Abstraction Hierarchy Toward Productive Quantum Programming <https://arxiv.org/abs/2405.13918>`_
      - O\. Di Matteo, S\. Núñez-Corrales, M\. Stęchły et al.
      - 2024
    * - `Quantum Software Ecosystem Design <https://arxiv.org/abs/2405.13244>`_
      - A\. Basermann, M\. Epping et al.
      - 2024
    * - `Hybrid Meta-Solving for Practical Quantum Computing <https://arxiv.org/abs/2405.09115>`_
      - D\. Eichhorn, M\. Schweikart, N\. Poser et al.
      - 2024
    * - `Quantum computing with Qiskit <https://arxiv.org/abs/2405.08810>`_
      - A\. Javadi-Abhari, M\. Treinish, K\. Krsulich et al.
      - 2024
    * - `UAV Swarm Management Platform for Autonomous Area and Infrastructure Inspection <https://ieeexplore.ieee.org/abstract/document/10497082>`_
      - M\. Batistatos; A\. Mazilu et al.
      - 2024
    * - `Automated Software Engineering (2024) 31:36 <https://link.springer.com/article/10.1007/s10515-024-00436-x>`_
      - A\. Sarkar
      - 2024
    * - `Towards Higher Abstraction Levels in Quantum Computing <https://link.springer.com/chapter/10.1007/978-981-97-0989-2_13>`_
      - H\. Fürntratt, P\. Schnabel et al.
      - 2024
    * - `Quantum Software Ecosystem: Stakeholders, Interactions and Challenges <https://www.researchgate.net/publication/378066784_Quantum_Software_Ecosystem_Stakeholders_Interactions_and_Challenges>`_
      - V\. Stirbu, T\. Mikkonen
      - 2024
    * - `High-Level Quantum Programming <https://www.research-collection.ethz.ch/handle/20.500.11850/634879>`_
      - B\. Bichsel
      - 2023
    * - `A Testing Pipeline for Quantum Computing Applications <https://publica.fraunhofer.de/entities/publication/ff4f1dc4-ab7d-41a6-8157-0b663aee83eb/details>`_
      - C\. Becker, I.D\. Gheorghe-Pop, N\. Tcholtchev
      - 2023
