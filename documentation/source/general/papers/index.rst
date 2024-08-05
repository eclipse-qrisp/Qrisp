.. _research:

Research
--------

Below, we present a collection of scientific papers and articles that have either contributed to the development of Qrisp or highlight its applications and advantages in the field of quantum computing. These citations provide a foundation for understanding the theoretical underpinnings and practical implications of Qrisp. We encourage researchers, developers, and enthusiasts to explore these resources to gain a deeper insight into the capabilities and potential of Qrisp.

Research from within the Qrisp community
========================================

.. grid::

    .. grid-item-card:: Qrisp: A Framework for Compilable High-Level Programming of Gate-Based Quantum Computers

        | Raphael Seidel, Sebastian Bock, René Zander, Matic Petrič, Niklas Steinmann, Nikolay Tcholtchev, Manfred Hauswirth 
        | `ArXiv, 2024 <https://arxiv.org/abs/2406.14792>`_

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

    .. grid-item-card:: Solving the Product Breakdown Structure Problem with constrained QAOA

        | René Zander, Raphael Seidel, Matteo Inajetovic, Niklas Steinmann, Matic Petrič 
        | `ArXiv, 2024 <https://arxiv.org/pdf/2406.15228>`_

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
        | `ArXiv, 2024 <https://arxiv.org/abs/2402.10060>`_ 

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
        | `ArXiv, 2023 <https://arxiv.org/abs/2307.11417>`_ 

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


Research conducted externally that utilizes or cites Qrisp
==========================================================

.. list-table::
    :widths: 50 30 10
    :header-rows: 1
    
    * - Title
      - Authors
      - Year
    * - `Towards Higher Abstraction Levels in Quantum Computing <https://link.springer.com/chapter/10.1007/978-981-97-0989-2_13>`_
      - H. Fürntratt, P. Schnabel et al.
      - 2024
    * - `Eclipse Qrisp QAOA: description and preliminary comparison with Qiskit counterparts <https://arxiv.org/abs/2405.20173>`_
      - E. Osaba, Matic Petrič, Izaskun Oregi et al. 
      - 2023
    * - `A Testing Pipeline for Quantum Computing Applications <https://publica.fraunhofer.de/entities/publication/ff4f1dc4-ab7d-41a6-8157-0b663aee83eb/details>`_
      - C. Becker, I.D. Gheorghe-Pop, N. Tscholtchev
      - 2023
    * - `Quantum computing with Qiskit <https://arxiv.org/pdf/2405.08810>`_
      - A. Javadi-Abhari, M. Treinish, K. Krsulich et al.
      - 2024
    * - `Testing multi-subroutine quantum programs: From unit testing to integration testing <https://dl.acm.org/doi/full/10.1145/3656339>`_
      - P. Long, J. Zhao
      - 2024
    * - `Quantum Software Ecosystem: Stakeholders, Interactions and Challenges <https://www.researchgate.net/publication/378066784_Quantum_Software_Ecosystem_Stakeholders_Interactions_and_Challenges>`_
      - V. Stirbu, T. Mikkonen 
      - 2024
    * - `The T-Complexity Costs of Error Correction for Control Flow in Quantum Computation <https://dl.acm.org/doi/pdf/10.1145/3656397>`_
      - C. Yuan, M. Carbin
      - 2024
    * - `UAV Swarm Management Platform for Autonomous Area and Infrastructure Inspection <https://ieeexplore.ieee.org/abstract/document/10497082>`_,
      - M. Batistatos; A. Mazilu et al. 
      - 2024
    * - `Hybrid Meta-Solving for Practical Quantum Computing <https://arxiv.org/pdf/2405.09115>`_
      - D. Eichhorn, M. Schweikart, N. Poser et al. 
      - 2024
    * - `Quantum Software Ecosystem Design <https://arxiv.org/abs/2405.13244>`_
      - A. Basermann, M. Epping et al. 
      - 2024
    * - `Towards Continuous Development for Quantum Programming in Decentralized IoT environments <https://www.sciencedirect.com/science/article/pii/S1877050924012286>`_
      - M. Kourtis, N Tcholtchev, I.D. Gheorghe-Pop et al. 
      - 2024
    * - `An Abstraction Hierarchy Toward Productive Quantum Programming <https://arxiv.org/abs/2405.13918>`_
      - O. Di Matteo, S. Núñez-Corrales, M. Stęchły et al. 
      - 2024
    * - `High-Level Quantum Programming <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/634879/1/thesis_electronic.pdf>`_
      - B. Bichsel  
      - 2023
    * - `Automated Software Engineering (2024) 31:36 <https://link.springer.com/article/10.1007/s10515-024-00436-x>`_
      - A. Sarkar 
      - 2024
