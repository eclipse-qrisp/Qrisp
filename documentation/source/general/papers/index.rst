.. _research:

Research
--------

On this page you can find the research related to Qrisp. Have a look at the papers written by the developers, and all the citations of Qrisp!

Our Research
============

.. grid::

    .. grid-item-card:: Qrisp: A Framework for Compilable High-Level Programming of Gate-Based Quantum Computers

        | Raphael Seidel, Sebastian Bock, René Zander, Matic Petrič, Niklas Steinmann, Nikolay Tcholtchev, Manfred Hauswirth 
        | `Arxiv, 2024 <https://arxiv.org/abs/2406.14792>`_

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
        | `Arxiv, 2024 <https://arxiv.org/pdf/2406.15228>`_

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
        | `Arxiv, 2024 <https://arxiv.org/abs/2402.10060>`_ 

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
        | `Arxiv, 2023 <https://arxiv.org/abs/2307.11417>`_ 

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


Citations of Qrisp
==================

- `Eclipse Qrisp QAOA: description and preliminary comparison with Qiskit counterparts <https://arxiv.org/abs/2405.20173>`_, Eneko Osaba, Matic Petrič, Izaskun Oregi et al. (2023)

- `A Testing Pipeline for Quantum Computing Applications <https://dl.acm.org/doi/full/10.1145/3656339>`_, C.K.U. Becker, I.D. Gheorghe-Pop (2023)

- `Quantum computing with Qiskit <https://arxiv.org/pdf/2405.08810>`_, A. Javadi-Abhari, M. Treinish, K. Krsulich et al. (2024)

- `Testing multi-subroutine quantum programs: From unit testing to integration testing <https://dl.acm.org/doi/full/10.1145/3656339>`_, P. Long, J. Zhao (2024)

- `Quantum Software Ecosystem: Stakeholders, Interactions and Challenges <https://arxiv.org/pdf/2405.08810>`_, V. Stirbu, T. Mikkonen (2024)

- `The T-Complexity Costs of Error Correction for Control Flow in Quantum Computation <https://dl.acm.org/doi/pdf/10.1145/3656397>`_, C. Yuan, M. Carbin (2024)

- `Hybrid Meta-Solving for Practical Quantum Computing <https://arxiv.org/pdf/2405.09115>`_, D Eichhorn, M Schweikart, N Poser et al. (2024)

- `Towards Continuous Development for Quantum Programming in Decentralized IoT environments <https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050924X00083/1-s2.0-S1877050924012286/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECUaCXVzLWVhc3QtMSJHMEUCIBYZtxNGl0GH9T8jCl4KlBhk8R9F7t%2F0sCasgx63%2FxoQAiEAzf0E0MrQumX8ELtzazWFjK%2FEFtryZ%2FMI%2FK31NN%2B6XTQqvAUI7v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDINKDOZ1bTsM%2FqMj8yqQBYbO%2FEJAEI4tE%2FkKVT6qDe32i1RmOO7d5QtQtzJBFZHGFe8gh%2FUq5lduz%2BFPK77MBXWVH9RiwPjdehQL6qdLzaxEoCW4Abd0t%2FBbvxPzudLNnE%2BDMWCnHIMSphawgZYbiRKAVUtZnGkL9mHntHE5mnGpzmaLO0iLturWhjn7b5ZcKiJs1riNjohGpB4HIkol8Ztdzd82kUGPpT%2FlvzO21gaBZTWmrmNmLfQHSXVNq92DS%2FW4SOz%2FYKQml33Fig%2F4wcoQPtNXl3Zjz8RUwrfG%2Fe4A10fhGPdlA45PKhddoZfK3OmAyMQLGqJf%2FL71yDOTFeJkG82xgKSC%2BUXZWGyzCD66Fkc5DVZWX9Rw2Pz3ak6QuqI1XxmCUi%2Fh%2FnaFxGrE3H58jz89%2BRpHZQemhPw5iyZGUw1XkTPmcOCr2YCA5J7jB9nGN0FKwCtxxaJYrOtXZHyqmyzcvIGWXVO3vKG%2FSC65MnGcYOK9bWChKl9TMLKNqtYm17vk1143DBKf9CAiBBbBrZZlJxne43sDed4PsTCL%2B18RwPGta7uqzPnrB6wRJSzz1UepnLhvuv0voMs%2BWJEwEQFfvnYQQ7G7MdSBphZRzJwD3M8JJdKiud3mrp1JEswPRQY6I3r8yYnVMGtqzhA1lFrNMSmuxoIXjVEDMLWT9aqHJDx1cR%2Bg%2BPADz4mGoaRNTh4gtvBdK%2FrECjVZgrWgGVL4S6BMj7OsLTL%2BxSCznoWD%2FhkeZcl7lLuLjhO48PSjsZwYBbCoQAHM6iMH%2B8WvWqOgJcQDf0SRsBV4PMJ137Sf9NHiu6vVtfATXLAn7og3zPv8OxWCzvbhkmH2yiuXHTWAD%2FIIER3PkFzuLxPUxIe5lRSFtOuoi87WlDy8MJjw3rQGOrEBWDxs6euLlSd%2FuvmmpvCpWmVlI2lQ891VEmc%2FhCN3lxOiKDCqd72eKsz%2Fa%2BVAPJ8CF877RpluJiM3ugpEsIEv9iyri2Q2yEcwoDRsa7LvAMiIQK1Kg5fYZjgbuiEOI5gZOgUog0pL3q19aApafpOPRsmy1CUZeD6UG%2BhGGdg2BFfO6%2ByZGyaJrtoz9A07ehmxpmvFJ%2BGrUwzIkDn255z5ngDwXGD68vHtYi7LNTHalPkM&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240717T133132Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXRRE3IOU%2F20240717%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ce77a74a81ed76fd1bfff19fb5c255b5c8994d9fc385ccce2fb742209a88a187&hash=fd8e28e8211804bbb2c3398b157fdec5290aee3f8b2a13bd3cf27e4e4159840b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050924012286&tid=spdf-89598cc2-435d-449d-b051-8232bfc3d7c7&sid=3d735e5d8766364ed778c0c3a9b03e10ebd1gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=02015f0e520457530553&rr=8a4a97c41b84bf28&cc=de>`_, M.A. Kourtis, N Tcholtchev, I.D. Gheorghe-Pop et al. (2024)

- `An Abstraction Hierarchy Toward Productive Quantum Programming <https://arxiv.org/abs/2405.13918>`_, O. Di Matteo, S. Núñez-Corrales, M. Stęchły et al. (2024)

- `High-Level Quantum Programming <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/634879/1/thesis_electronic.pdf>`_, B. Bichsel  (2023)

- `Automated Software Engineering (2024) 31:36 <https://link.springer.com/article/10.1007/s10515-024-00436-x>`_, Aritra Sarkar (2024)
