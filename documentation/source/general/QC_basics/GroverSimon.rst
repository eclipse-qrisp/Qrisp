EVEN MORE Qrisp programming
===========================

Continuing where we left off, we still have some famous oracle
algorithms left on the menu. First up, we will serve Grover’s oracle, a
search algorithm with a side of linear algebra and geometric sauce. Our
second course consists of something familiar, in a whole new dressing.
Happy coding, and eat up!

Grover’s oracle
---------------

One of the most famous and first quantum algorithms developed is
Grover’s algorithm, designed for searching an unsorted database (even
though that may not be a feasible use case in practice).
When you have to find a certain value but no indication of where to
look, you would have to start in the beginning and go through the list
until you find it. This would take on average :math:`\frac{n}{2}`
tries for one key and :math:`n` possible values. If you are unlucky,
you would have to look through the whole list and find the element in
question at the last position, having a worst case time complexity of
:math:`O(n)`.
Grover’s algorithm is a simple example where quantum computers
outperform classical approaches with a complexity of
:math:`O(\sqrt{n})`, meaning that on average, you’d have to call the
oracle :math:`\sqrt{n}` times. As you can see, the secret lies in an
improved query complexity.
The general idea is to construct a **quantum oracle** that returns
:math:`f(x) \rightarrow 0` for every value except for the sought out
key :math:`w`, instead we get :math:`f(w) \rightarrow 1`. We can
express this through the eigenvalues of the oracle matrix (this
specific step makes it a phase oracle):
:math:`U_f \ket{x} = (-1)^{f(x)} \ket{x}`, which means that all
eigenvalues are 1 except for :math:`w`, whose eigenvalue is :math:`-1`
(this flips the amplitude of :math:`w`). In simple words: All inputs
stay the same, except for our wanted value, whose sign gets flipped.
This is the work of the actual oracle in the algorithm.

We want to have a look at a specific example to demystify the process:
Let’s say we have the entries 0 - 3: 00, 01, 10, 11. We now want to
filter out the entry 3 (binary: 11). The corresponding oracle matrix,
that flips the amplitude of 3, would be:

.. math::

   U =\left (\begin{array}{rrr}
       1 & 0 &0 &0 \\
       0& 1 &0 &0 \\
       0 & 0 & 1&0\\
       0 & 0& 0 &- 1 
   \end{array} \right)

When we multiply the key vector
:math:`\ket 3 =\left (\begin{array}{rrr}  0 \\ 0 \\ 0 \\1 \end{array} \right)`
with this matrix, you get -1 (the **amplitude flip**). However, this
flip would not affect measurements just yet, since the probability of a
state is its square.

To solve this problem, the next step is **amplitude amplification** done
by the **diffusion operator** :math:`\ket{s} \bra{s} - I` with
:math:`\ket{s}` being the initial state, which is repeated approximately
:math:`\sqrt{n}` times, the leading factor for this algorithms
complexity. This operator reflects all probabilty amplitudes about the
average of the amplitudes. For that, we first have to calculate the
average of all amplitudes. Notice that after the oracle, one amplitude
has a negative sign. Now, the inversion works by subtracting
:math:`amplitude- 2* average` for every state. This leads to all
:math:`x \neq w` having a probability of (almost) 0 and the desired
solution :math:`w` a probability of (almost) 1. The diffusion operator
turns all :math:`\ket s` into :math:`\ket{00...000}`. Since we created
:math:`\ket s` by applying a Hadamard gate, we can just apply it again
to undo it.

The circuit looks like this:

::

           ┌───┐┌────────────┐┌───────────┐┌────────────┐┌───────────┐
   qf_0.0: ┤ H ├┤0           ├┤0          ├┤0           ├┤0          ├
           ├───┤│            ││           ││            ││           │
   qf_1.0: ┤ H ├┤1 tag_state ├┤1 diffuser ├┤1 tag_state ├┤1 diffuser ├
           ├───┤│            ││           ││            ││           │
   qf_1.1: ┤ H ├┤2           ├┤2          ├┤2           ├┤2          ├
           └───┘└────────────┘└───────────┘└────────────┘└───────────┘

In this example, 8 database entries where considered. The oracle
construction starts with a Hadamard gate to put all possible solutions
into superposition:
:math:`\ket{\psi_0} = \frac{1}{\sqrt n}\sum_{x=0}^{n-1} \ket x`.
Geometrically, this state lays between the perpendicular :math:`\ket{k}`
and :math:`\ket{v}`, with :math:`\ket{k}` as the state of the
thought-out key and :math:`\ket{v}` as the state of all the other values
(these are orthogonal to each other since they have no shared basis).
The superposition basically means that we don’t have any clue where our
key might be in the beginning, so all guesses are equally plausible. The
tag state is unique to your specific problem and flips the amplitude of
the key. The diffuser :math:`\ket{\psi_0} \bra{\psi_0}` reflects the
joint state at :math:`\ket v` multiple times, so that it effectively is
rotated closer to :math:`\ket k`.

The most important question when using Grover’s algorithm is: How to
construct the oracle to find the wanted value? This of course depends
on the key you are looking for and needs to be constructed
individually for every problem.
Fortunately for you, many steps of this algorithm are already
implemented in qrisp and ready to go, including the oracle build. You
merely have to pick out your desired keys. Let’s see that for our
previous example:

.. code:: python

   from qrisp import QuantumFloat

   #Create list of QuantumFloats
   qf_list = [QuantumFloat(2, signed = False)]

   from qrisp.grover import tag_state, grovers_alg

   def test_oracle(qf_list):

       tag_dic = {qf_list[0]: 3} 
       tag_state(tag_dic)   # tag_state is specified to take a dictionary as argument

   grovers_alg(qf_list, test_oracle, iterations=1)

   print(qf_list)

   from qrisp.misc import multi_measurement
   print(multi_measurement(qf_list))

::

   >>> [<QuantumFloat 'qf_0'>]
   >>> {(3,): 1.0}

Grover’s algorithm can also look for a key consisting of more than one
entry:

.. code:: python

   from qrisp import QuantumFloat

   #Create list of QuantumFloats
   qf_list = [QuantumFloat(2, signed = True), QuantumFloat(2, signed = True)]

   from qrisp.grover import tag_state, grovers_alg

   def test_oracle(qf_list):               # define oracle: tag the states -3 and 2 (the key that we are looking for)
       tag_dic = {qf_list[0] : -3, qf_list[1] : 2}
       tag_state(tag_dic)


   grovers_alg(qf_list, test_oracle)       # perform grovers algorithm on the list with our oracle

   from qrisp.misc import multi_measurement
   print(multi_measurement(qf_list))

::

   {(-3, 2): 0.99659, (0, 0): 5e-05, (0, 1): 5e-05, (0, 2): 5e-05, (0, 3): 5e-05, (0, -4): 5e-05, (0, -3): 5e-05, (0, -2): 5e-05, (0, -1): 5e-05, (1, 0): 5e-05, (1, 1): 5e-05, (1, 2): 5e-05, (1, 3): 5e-05, (1, -4): 5e-05, (1, -3): 5e-05, (1, -2): 5e-05, (1, -1): 5e-05, (2, 0): 5e-05, (2, 1): 5e-05, (2, 2): 5e-05, (2, 3): 5e-05, (2, -4): 5e-05, (2, -3): 5e-05, (2, -2): 5e-05, (2, -1): 5e-05, (3, 0): 5e-05, (3, 1): 5e-05, (3, 2): 5e-05, (3, 3): 5e-05, (3, -4): 5e-05, (3, -3): 5e-05, (3, -2): 5e-05, (3, -1): 5e-05, (-4, 0): 5e-05, (-4, 1): 5e-05, (-4, 2): 5e-05, (-4, 3): 5e-05, (-4, -4): 5e-05, (-4, -3): 5e-05, (-4, -2): 5e-05, (-4, -1): 5e-05, (-3, 0): 5e-05, (-3, 1): 5e-05, (-3, 3): 5e-05, (-3, -4): 5e-05, (-3, -3): 5e-05, (-3, -2): 5e-05, (-3, -1): 5e-05, (-2, 0): 5e-05, (-2, 1): 5e-05, (-2, 2): 5e-05, (-2, 3): 5e-05, (-2, -4): 5e-05, (-2, -3): 5e-05, (-2, -2): 5e-05, (-2, -1): 5e-05, (-1, 0): 5e-05, (-1, 1): 5e-05, (-1, 2): 5e-05, (-1, 3): 5e-05, (-1, -4): 5e-05, (-1, -3): 5e-05, (-1, -2): 5e-05, (-1, -1): 5e-05}

In this code, we construct a list of QuantumFloats, starting at (-3, 2)
and going to (-1, -1) in integer steps, resulting in 64 elements. We
then use the predefinied function ``tag_state``, that creates an oracle
based on the sought after values that we turn over in the dictionary:
(-3, 2). We then apply Grover’s algorithms on the list with our oracle
engineered with ``tag_state``. And that’s it! With just a few lines and
minimal effort you just beat a classical computer.

As another variation, you can also look for multiple keys. Keep in mind
that with more keys to look for, the probability for each one will
decrease, since it all needs to add up to one. It can also be modified
to detect multiple matching entries or into a partial search, only
looking for certain categories instead of a precise value.

This algorithm can be used for one-way functions: hard to compute, easy
to verify. Finding a particular entry in a database is the primary
example, but you could also solve a Sudoku or plan a route. The hard
part is to find a way to express these problems as an oracle, but with a
little brainpower, you can adapt this algorithm for your own problems.

Now, there’s only one plothole left to fill: I mentioned earlier that
this search is probably not the ideal use case in the future, but
which one is? First, search problems might not be ideal, since you
need to read out the actual information from every entry (here, the
value the entry contains), which can be more complex and
time-consuming than the actual algorithm. This could be fixed by using
the index instead of value, but then we would look at a sorted
database, where efficient classical sorting algorithms already exist.
But lucky for us, there are still other use cases! It can be used in
any NP-complete problem that utilizes exhaustive search, as well as
function inversion in cryptography. As always, these algorithms are an
ongoing research field for many scholars and we don’t claim
completeness.

Simon’s algorithm
-----------------

Simon says: Learn his algorithm! Circling back on the oracle algorithms
from last chapter, Simon’s algorithm is the next in a chain of routines
to prove quantum supremacy. From Deutsch, who could show that there is
some advantage to classical computing, to Bernstein and Vazirani, who
could guarantee only one necessary query, to now Simon, who was the
first to offer an exponential speedup (and a show-off).

Simon’s alorithm is devoted to solve Simon’s problem: In
:math:`f(x) = f( x \oplus s)`, what is the bitstring s? You may also
find the formulation that we need to find :math:`x` and :math:`y` in
:math:`f(x) = f(y)`, which describes a collision, and is only true if
:math:`x \oplus y \in \{0^n, s\}`. For that, we need a two-to-one
function, which means that exactly two different inputs are mapped to
one output. This problem is of the form
:math:`f:\{0, 1\}^n \rightarrow \{0, 1\}^n`, with a black box oracle as
:math:`f`.

How can we tackle this problem classically? First, we would need to
send enough queries to our black box oracle to get the same output
twice, so :math:`f(x) = f(y)`. This is the leading factor for
complexity, as this can take :math:`2^{\frac{n}{2}} +1` quieries for a
n-bit string in the worst case, having to go through half the input
domain to get every possible answer and the first repitition in the
next query. Now, you would apply XOR to the two inputs to obtain s.
(This is especially easy, if one of your queries in all zeros. That
way, the bitstring s is simply the other query.) Unfortunately, this
is pretty time consuming, leading to exponential complexity
:math:`O(2^n)`.
Our quantum solution however only has a linear complexity, therefore
presenting an exponential speedup, the first of its kind. For the
quantum version, we initalize two registers with each :math:`n` qubits
all in the state :math:`\ket 0`. We apply Hadamard gates on the first
register and the oracle for :math:`f(x)` on both. Remember that an
oracle maps :math:`\ket x \ket y \rightarrow \ket x \ket{f(x)}`, so
our bitstring is stored in the ancilla register later. Lastly, we
apply Hadamard gates to the first register again for uncomputing.

Sounds simple, right? Let’s see for ourselves and program it with Qrisp:

We want to look at the two-qubit system, meaning that our main register
and ancilla register both have 2 qubits. For the sake of the example, we
also implement the oracle with the hidden bitstring ``b=11`` using CX
gates. In the table below, you can see what that would mean for the
output:

===== ======
input output
===== ======
00    00
01    11
10    11
11    00
===== ======

As you can also see in the oracle, the output stores
:math:`f(x)= (x_0 \oplus x_1, x_0 \oplus x_1)`, since both register
qubits are connected to both ancilla qubits using CX gates, analogous to
XOR. Since every output is seen twice, we did in fact implement a 2-to-1
function.

.. code:: python

   from qrisp import QuantumSession, QuantumArray, QuantumVariable, h, cx

   register = QuantumVariable(2)
   ancilla = QuantumVariable(2)

   def oracle(register, ancilla):
       # this black box oracle is not known in practice. For our test case, we need to implement it ourselves.
       cx(register[0], ancilla[0])
       cx(register[0], ancilla[1])
       cx(register[1], ancilla[0])
       cx(register[1], ancilla[1])
       return register, ancilla            # bitstring: 11


   h(register)
   oracle(register, ancilla)
   print("ancilla ")
   print(ancilla)
   print("register")
   h(register)

::

   ancilla 
   {'00': 0.5, '11': 0.5}
   register
   {'00': 0.5, '11': 0.5}

It might surpise you to see that we are given two different bitstrings
with equal probability. The second, ``11``, is the one we already
predicted. The other, ``00``, is also mathematically true:
:math:`f(x) = f(x \oplus s)` holds true for :math:`s=00`. This trivial
solution is not what we look for, but will be part of our output for
every bitstring. If you only receive this solution, it might be a sign
that you accidentally implemented a 1-to-1 function.

With different combination of CX gates in the oracle, you can create
your own with another bitstring. You should make sure beforehand that
you are actually implementing a 2-to-1 function.

In this simple algorithm, we can see some theoretical implications: This
oracle seperates the complexity classes BPP (bounded-error classical
query complexity) and BQP (bounded-error quantum query complexity),
similar to Bernstein-Vazirani, in an exponential manner.

Like Bernstein-Vazirani, the applicability of Simon’s algorithm in
crypotgraphy is currently researched and is even shown to break some
classical encryption codes. Also, Simon’s algorithm was the stepping
stone to Shor’s algorithm, which is probably the most famous quantum
algorithm, as well as Quantum Fourier Transformation, which will be
examinated in the next chapter.


Summary 
-------

- Grover's algorithm works with an oracle that finds a sought after value. We start in superposition, apply the oracle (which leads to an amplitude flip in the key), repeat amplitude amplification with the diffusion operator :math:`\sqrt n` times 

- Grover's time complexity is :math:`O(\sqrt n)`

- Grover in Qrisp: ``grovers_alg(qf_list, test_oracle, iterations=1) # define test_oracle yourself before, easiest with tag_state`` 

- Simon's problem: Find bitstring :math:`s` :math:`f(x) = f( x \oplus s)`: Hadamard, oracle, Hadamard for uncomputation

- both algorithms work with quantum oracles and are researched for crypotgraphy as potential use-case 
