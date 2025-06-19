Learn Quantum Programming with Qrisp
====================================

Welcome to the Qrisp learning module! If you are reading this, you
already made a great decision today by starting to learn about quantum
computing and choosing Qrisp as your companion. Together, we want to
explore the quantum world and how it can be of use for computational
tasks.

With qrisp, we focus on gate based quantum computing, following an
agnostic approach, meaning it runs independent of the hardware
architecture. The advantage of gate-based computing is its universality
and feasibility. Other models, such as annealing or measurement-based
computing, also exist and we will have a look at them in a later
section.

Let’s start our journey with this very first chapter and find out why
you should spend your time learning about quantum computers.

Why Quantum Computing?
----------------------

| To not get your hopes too high, I want to start by saying that quantum
  computers will not take all problems that ever existed and give your
  the best solution. There are many areas where the use of quantum
  computers doesn’t make sense, and the general consensus is that you
  won’t be having a quantum computer in your living room to do your
  online shopping.
| After these (maybe discouraging) words, let’s have a look at some
  fields that could heavily profit from the use of quantum computing: 

- optimization problems: In a use case study by Volkswagen and D-Wave
  Systems, buses for a convention had to be routed through the narrow
  streets of Lisbon. Taking the convention times, hotel locations,
  street conditions and sightseeing into account, the goal was to route
  the buses in the most efficient way to avoid jamming up. 1,275
  real-time optimization tasks had to be done to fulfill the actual
  constraints, creating new routes every time. [1]
| Another optimization problem poses portfolio optimization for finance.
  Being able to handle vast amounts of data and swiftly identify risks
  and profits would create a huge advantage. Currently, estimated 10 to
  40 billion dollars are lost every year to fraud and faulty data
  management. As present fraud detection is highly inaccurate with a
  high number of false alarms, quantum technology can revolutionize
  customer targeting and prediction modeling by efficiently analyzing
  large volumes of behavioral data to offer personalized financial
  products and services in real-time.

-  | computational biology and physics: Problems that grow exponentially
     on classical computers would only grow linearly on quantum
     hardware. One example is the simulation of molecules: The chemical
     reaction of three hydrogen atoms is already challenging, even for a
     supercomputer. With every added atom, the necessary calculations
     double. Now imagine the really interesting stuff like proteins or
     cancer cells, absolutely impossible to reliably model today. With
     quantum computing, bioinformatics would be on another level.
   | It is so obvious it is rarely mentioned: Through quantum computing,
     we hope to accurately simulate quantum mechanical systems and gain
     more insights into the physics, which is still partly a mystery.
     Chemical properties such as ionic bonds (essential in molecule
     building) are inherently quantum, and could receive an advantage
     being modeled by a quantum device.

-  material simulation: The field of material simulation is concerned
   with predicting properties to achieve desired outcomes, like
   increased absorption or decreased reflectivity. In order to give
   precise predictions, the Hamiltonians (energy functions) need to be
   solved and therefore the energy of the system is calculated. Taking
   the same prequisites in quantum chemistry as computational biology,
   quantum computers can be used to model underlying quantum mechanics.
   Note that these calculations are already done on classical hardware,
   but we could improve the accuracy and model more complex systems in
   the future.

As you can see, it is not just some theoretical toy that is cool and
interesting to some researchers, but has real-life use cases. In the
next chapters, we will back these examples up with theoretical
considerations, so stay tuned!

Time Complexity
---------------

We already gave some motivation for why quantum computing is interesting
and now want to delve a bit deeper in the theory behind that. If you
have dabbled in computer science before, you probably came across this
picture: |Alt text| Source: Theprogrammersfirst
(https://theprogrammersfirst.wordpress.com/2020/07/22/what-is-a-plain-english-explanation-of-big-o-notation/)

| which shows the different time complexity classes of algorithms. The
  algorithms are divided, based on their estimated runtime for a
  specific input size :math:`n`. Many algorithms that we are interested
  in have at least quadratic complexity :math:`O(n^2)` and are therefore
  restricted when it comes to solving bigger problem instances.
| Why should you care about quantum complexity? A popular example is the
  public-key cryptosystem RSA: This encryption scheme relies on the
  well-studied assumption that factoring a given number, i.e. finding
  its two prime factors, scales exponentially with the length of the
  given number :math:`O(\sim e^n)`, making the task impossible for large
  enough numbers. But since 1994 this assumption only holds for
  classical computers, because Peter Shor then presented an algorithm
  for quantum computers, which only has a polynomial time complexity
  :math:`O(\sim (\log n)^2).` This means that with a big enough quantum
  computer it would be possible to break current public-key
  cryptosystems, by exploiting quantum phenomena such as superposition,
  entanglement and interference. In the section :ref:`physics`, we will learn about these phenoma
  and how they make quantum advantage possible.

It’s important to emphasize here, that quantum computers aren’t just
“faster” than classical computers, but due to their entirely different
nature, there are certain problems for which quantum computers need
considerably less operations to calculate the result, thus making it
possible to solve problems that are intractable with classical
computers, even though the operations themselves might be slower. This
is why we develop entirely new algorithms for quantum computers instead
of merely running the already existing programs, recognizing they aren’t
these supermachines and operate fundamentally different.

Complexity Theory
-----------------

| If the examples we’ve shown you so far didn’t convince you enough to
  start learning about quantum computing, the next part surely will: We
  will now take a look at the even bigger picture of complexity classes.
  Two very well-known classes are P and NP with the major unsolved “P
  versus NP” problem. Both classes simulate a Turing machine in
  polynomial time, deterministic (P) or nondeterministic (NP) (you can
  imagine a Turing machine like a mathematical model of a very basic
  computer). Furthermore, the class BPP contains all problems that can
  be solved on a Turing Machine in polynomial time in a probabilistic
  manner. In this context, we cannot guarantee the right solution every
  time, but we demand at least :math:`\frac{2}{3}` of all answers to be
  correct (BPP= bounded-error probabilistic polynomial time). Similarly,
  the complexity class BQP (bounded-error quantum polynomial time)
  solves a problem on a quantum computer in polynomial time, up to an
  error of :math:`\frac{1}{3}` (we will talk more about the error rate
  of quantum computers in the chapter “error
  handling”). Both classes contain the same error rates, but simulate
  the problem on different machines.
| Shor’s algorithm for factoring is suspected to be in BQP, but not BPP,
  meaning we can simulate the algorithm in polynomial time on a quantum
  computer, but (likely) not on a classical computer like a Turing
  machine. It is assumed that multiple problems lay in BQP, but not BPP,
  making quantum algorithms attractive for us.

|image1| Source: Nicepng
(https://www.nicepng.com/png/detail/420-4201382_complexity-png.png)

As pictured, the suspected relation of the mentioned complexity classes
is :math:`P \subseteq BPP \subseteq BQP`.

| It is believed that :math:`BQP \neq P`, but a proof is still missing.
  However we do know that :math:`P \subseteq PQP` (meaning that any
  polynomial problem on a classical computer can also be solved on a
  quantum computer in polynomial time. It is unclear if there are
  problems that can only be efficiently solved on a quantum computer).
  Additionally, in the `original paper <https://arxiv.org/pdf/quant-ph/9701001.pdf>`__ introducing :math:`BQP`, the
  authors Bernstein and Vazirani provided evidence that
  :math:`BPQ \neq BPP`, by introducing the recursive Fourier sampling
  that can only be achieved using :math:`n^{\Omega(\log n)}` queries on
  a classical computer, but only requires :math:`n` queries on a
  quantum computer.
| Nevertheless, many complexity relations are still not entirely proven,
  as it is an ongoing research field. Even though Grover’s algorithm,
  that offers an exponential speedup to a NP-problem, could not prove
  that all problems in :math:`NP` can be solved by quantum computers
  (:math:`NP \subseteq BQP`).

After concerning ourselves with some theoretical aspect of why you
should invest your time into quantum computing, let’s have a look at
what the experts are already doing with it.

.. _qubits: 

Today’s Quantum Computers
-------------------------

| Classical computers work with bits, the smallest unit of information,
  that can either be 0 or 1. If we combine many of them, the amount
  information the system can store and process grows linearly. Similar
  to classical computers, quantum computers work with qubits, whose
  states we write as :math:`\ket{0}` and :math:`\ket{1}`.
| Given their quantum nature, they can also be in a :ref:` superposition <physics>` of 0
  and 1. What does that mean? You can imagine that a qubit can be in two
  states at once, 0 and 1 simultaneously. When
  combining qubits together, we get exponential growth :math:`2^n`
  instead of linear growth. Because of superposition, the qubits can be
  in all possible states :math:`2^n` at the same time. This means that
  :math:`n` qubits are equivalent to :math:`2^{n}` bits.

If we go back in time and have a look at the development stages of
quantum computers, we can see an exponential trend:

|image2| Source: Statista
(https://www.statista.com/chart/17896/quantum-computing-developments/)

| As of December 2023, the record for most qubits in one quantum
  computer holds Atom Computing, boasting an impressive 1,180 qubits and
  dethroning IBM’s Osprey with 433 qubits.
| Some might be reminded of Moore’s law by this tendency, which states
  that classical computer chips double in computational power every 1.5
  - 2 years, leading to exponential growth. The unique characteristic
  about quantum computers: The combination of an exponentially
  increasing qubit count and the exponential information gain with each
  added qubit results in a double-exponential increase in computational
  power (Rose’s law). While classical computers are limited by practical
  dimensions like the size of atoms or the number of available atoms in
  the universe, quantum computers could have the same (if not more)
  computational power with less resources. If this trend continues,
  quantum computer will soon outpower classical computers by far.
| Keep in mind that this computational power is not useful for every
  problem. Especially for problems that involve a large variety on
  possible solutions, quantum computing can offer advantages by
  exploring multiple solutions at once, for example in database searches
  (by using Grover’s algorithm). Quantum computers leverage the
  principles of superposition and entanglement to process information in
  ways classical computers cannot. Problems with inherent quantum
  properties, where solutions exist in multiple states simultaneously,
  align well with the strengths of quantum algorithms. However, for many
  everyday computing needs that utilize sequential paradigms, classical
  computers remain highly effective and, in some cases, more practical.

| Right now, experimental quantum computers are limited by the available
  qubit count and the error rates of the operations. Multiple labs are
  invested in building bigger and more accurate quantum computers to
  actually prove the quantum advantage described earlier. In addition,
  qubits are very delicate, so far quantum computers only exist in labs.
  Even there, external fields disturb the quantum properties and errors
  are no rarity. Due to this instability, faultiness and decoherence
  (quantum behavior is lost to the environment), the amount of required
  physical qubits build into a system is much higher than the number of
  theoretical logical qubits thought of in a circuit (this relationship
  lays anywhere between 10:1 to 10000:1, depending on the circuit). This
  also taints our relationship :math:`|bits| = 2^{|qubits|}`, since this
  only counts for logical qubits.
| Because this is a physical system, all results we will get after
  measuring are probabilistic, meaning that we only have a certain
  probability to get this result. If we repeat the same procedure, even
  on the same hardware, there will be no guarantee to achieve the same
  result. Therefore, experiments need to be revised many times and our
  algorithms need to deliver a high probability.

Even though there are many unsolved problems, don’t be intimidated by
quantum computing! Reliable hardware is currently being developed and as
it is a newly emerging field, you too could contribute to its growth. In
the next sections, we will walk through necessary math and physics to
understand the basics of quantum computing. After that, we will look at
simple quantum algorithms and work our way up to more complex ones.

Summary
-------

-  quantum computing allows for more time efficient algorithms through
   the special mechanics like superposition and entanglement,
   benefitting various industries
-  qubits can take more values that just 0 and 1 like classical bits
-  qubits hold exponential information gain, and combined with their
   current exponential architecture growth can exceed Moore’s law
-  building quantum computers is a delicate and time-consuming task,
   still prone to error and external influences

And in case I haven’t said in enough already, here you can hear it from
`Scott Aaronson <https://scottaaronson.blog/>`__: “If you take nothing
else from this blog: quantum computers won’t solve hard problems
instantly by just trying all solutions in parallel.”

[1] for more details, check out `the original
paper <https://www.dwavesys.com/media/2pojgtcx/dwave_vw_case_story_v2f.pdf>`__

.. |Alt text| image:: https://i.stack.imgur.com/6zHEt.png
.. |image1| image:: https://www.nicepng.com/png/detail/420-4201382_complexity-png.png
.. |image2| image:: https://cdn.statcdn.com/Infographic/images/normal/17896.jpeg
