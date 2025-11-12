.. _DQItutorial:

.. currentmodule:: qrisp

Decoding by Quantum Interferometry
==================================

In this tutorial, we will show you Qrisp implementation of the `Optimization by Decoded Quantum Interferometry <https://arxiv.org/abs/2408.08292>`_ and its relevant subroutines. This algorithm is rigourusly derived in the oringal paper, 
including a motiviation with physics intuition, so we highly recommend giving it a read! Additionally the 2025 paper  `Quantum Circuit Design for Decoded Quantum Interferometry <https://arxiv.org/abs/2504.18334>`_ goes into circuit design for this algorithm (as the title would suggest) and worth investigating aswell!
We will cross-reference these two papers throughout the text if applicable, such that you know where to look for a specific thing.
Let us quickly summarize what the original paper is about: 

The algorithm is called Decoded Quantum Interferometry (DQI), which uses the quantum
Fourier transform to arrange that amplitudes interfere constructively on symbol strings for which the objective value is large, thereby enhancing the
probability of obtaining good solutions upon measurement. The procedure reduces optimization problems to decoding problems. 
Specifically, sparse unstructured optimization problems such as max-k-XORSAT are
reduced to decoding of low-density parity check codes.

Let us dive into how this works with the example of the max-XORSAT problem:

Given an :math:`m \, \text{x} \, n` matrix :math:`B` with :math:`m \geq n`, the max-XORSAT problem is to find an $n$-bit string $\textbf{x}$
satisfying as many as possible among the $m$ linear mod-2 equations $B \textbf{x} = \textbf{v}$, where also $\textbf{v}  \in \{ 0,1 \}^{\otimes n}$

The max-XORSAT problem can be rephrased as maximizing the objective function

.. math::

    f( \textbf{x} ) = \sum^{m}_{i=1} (-1)^{v_i + \textbf{b}_i \times \textbf{x} },

where $\textbf{b}_i$ is the $i$-th row of $B$. So $f(\textbf{x})$ is the number of linear equations that are
satisfied minus the number of unsatisfied ones.

Now, the Hadamard transform of this function will be sparse. Only for the strings $\textbf{b}_1$,...,$\textbf{b}_m$ the amplitudes will be nonzero, 
so we only have $m$ nonzero amplitudes in total. 
So if we can prepare the superposition :math:`\sum_{i=1}^{m} (-1)^{v_i} \ket{\textbf{b}_i}` and apply the Hadamard-transform,
then we will arrive at $\sum_{\textbf{x} in \in \{ 0,1 \}^{\otimes n}} f(\textbf{x} ) \ket{ \textbf{x} }$. 
Measuring this state in the computational basis creates a sample with a slight bias to high objective value solutions. A string $\textbf{x}$ is obtained with probability $\propto f(\textbf{x})^2$, 
so higher objective values $f(\textbf{x})$ will be more likely received as outcomes.

A slight bias is obviously not enough, but how can we improve this? The answer is by introducing a polynom $P$ to encode our function $f$, and turning it into a decoding problem.
We therefore want to prepare the state

.. math::
    
    \ket{P(f)} =  \sum_{\textbf{x} in \in \{ 0,1 \}^{\otimes n}} P(f(\textbf{x} )) \ket{ \textbf{x} },

where $P$ is a polynom of degree $l$, appropriatly normalized.
For the max-XORSAT case, the derivation of $\ket{P(f)}$ can be found on `p. 19 in the first paper <https://arxiv.org/abs/2504.18334#page=19>`_ 

Steps in the algorithm
----------------------

To arrive at the state $\ket{P(f)}$ five steps are conducted. The first one is to prepare a Dicke state superposition
$\sum^{l}_{k=0} w_k \ket{D_{m,k}}$, where

.. math::

    \ket{D_{m,k}} = \frac{1}{\sqrt{\binom{m}{k}}} \sum_{ | y | = k} \ket{\textbf{y}}, 

i.e. the Dicke state of weight $k$. We will make use of our established `Dicke state tutorial [REF, the dike state tut] <>` for this. 
The coefficents $w_k$ emerge as the principal eigenvector of a matrix built in relation to the expected number of satisfied constraints. 
Please refer to Lemma 9.2 in the `DQI paper <https://arxiv.org/pdf/2408.08292>`_ to see how we can define this matrix. The weights will then be fanned out 
via unary amplitude encoding. We will touch on this again in the implementation section.

The next step is to consider the problem specific phase $(-1)^{\textbf{v} \cdot \textbf{y}}$ via conditional $Z$-Gates. Recall that $\textbf{v}$ emerges from the inital equation $B \textbf{x} = \textbf{v}$ that we are trying to solve.
Then, the problem matrix will be encoded into the second register, such that we receive the state $\ket{B^T \textbf{y}}$ as part of our superposition. 
After this step we will have the state

.. math::

    \sum^{l}_{k=0} w_k \frac{w_k }{\sqrt{\binom{m}{k}}} \sum_{ | y | = k} (-1)^{\textbf{v} \cdot \textbf{y}} \ket{\textbf{y}} \ket{B^T \textbf{y}}

Now at this point we want to refer again to the biased polynomial $\ket{P(f)}$, which encodes problem solutions with high likelyhood.
Its Hadamard-transform will always be of the form:

.. math::

    H^{\otimes n} \ket{P(f)}  = \sum^{l}_{k=0} w_k \frac{w_k }{\sqrt{\binom{m}{k}}} \sum_{ | y | = k} (-1)^{\textbf{v} \cdot \textbf{y}} \ket{B^T \textbf{y}}

So after the previous step of encoding the problem matrix B  we are just one uncomputation of $\ket{\textbf{y}}$ (and a straight-forward Hadamard-transform) away from our desired polynomial $\ket{P(f)}$.



Uncomputation and syndrome decoding
-----------------------------------

Unfortunately, said uncomputation is the part hard of the algorithm. With the structure of $B$, this the task of a decoding problem, a problem studied in the context of p.e. low-density parity check matrices for the problem of error correction.
Without going into much detail, we want to mention three approaches to tackle this problem for our purposes here, which are also considered in the `original DQI paper <https://arxiv.org/abs/2408.08292>`_, namely:
`Belief Propagation <https://arxiv.org/pdf/2102.01984>`_, `Look-Up tables <https://en.wikipedia.org/wiki/Decoding_methods>`_ and `the quantum version of a Gauss-Jordan-Elimination <https://arxiv.org/pdf/quant-ph/0511062>`_.

Belief Propagation is the most reliable and efficient approach to syndrome decoding, it is also the most extensive one in terms of implementation effort, which is why we omit it in our implementation. In the DQI paper it is shown that this apporach is 
the best choice for experimental outperformance of classical algorithms, given specific figures of merit. So if you feel like trying your hand at an implementation or integration of existing solutions, go for it and feel free to let us know! 

Look-up tables are a direct circuit encoding of all possible syndrome decodings. This makes it the easiest choice for implementation, but also the most quantum resource demanding one (in terms of gate count in this case). We have not included it in our solution, 
but example implementations can be found online, see `the github repository to DQI circuits paper <https://github.com/BankNatchapol/DQI-Circuit/blob/main/src/dqi/cirq/decoding/algorithms.py#L3`_.

Our solution relies on the quantum version of a Gauss-Jordan-Elimination, which lies in the somewhere middle of belief propagation and look-up tables in terms of efficiency and resource demand.
You might ask yourself: What is the Gauss-Jordan-Elimination? And how do we make it quantum? That is a great question, so lets answer it!

A `Gauss-Jordan-Elimination <https://en.wikipedia.org/wiki/Gaussian_elimination>`_ is used to bring a matrix in reduced row echelon form, i.e. applying row operations until you have reached the reduced upper triangle form. In this case of an invertible matrix, 
one calculate its inverse in this way. For our purpose, this inversion property effectively results in the isolation of a syndrome in the corresponding augmented matrix, i.e. the problem matrix extended by the syndrome vector. We will see what this looks like in the implementation part
You can find more information on the Quantum Gauss-Jordan-Elimination and its use in syndrome decoding in the `thesis on Independent Set Decoding <https://www.politesi.polimi.it/retrieve/ef311c33-5427-4869-865f-94dc5ee733ed/Thesis.pdf#page=92>`_ by S. Perillo et. al.

We need to note here, that our DQI implementation does not use the complete quantum version of Gauss-Jordan-Elimination, 
but instead translates the classical steps derived for the matrix transformation into quantum gates, which can be applied to the syndrome quantum state.
Since the problem matrix $B$ is given as a classical object, as opposed to being encoded in a quantum state, we can just figure out classically what operations would be necessary to bring this matrix
into reduced row-echelon form  and translate the operations affecting the syndrome quantum state into gates, s.t.  where the swaps are replaced by :func:`swap` gates and the additions by :func:`cx` gates.
This greatly reduces the resource overhead, and eliminates the chance of failure with this procedure in the case of a non-invertible matrix, see :sec:`our code example <Decoding implementation in Qrisp>`.


It is important to note here, that our chosen routine may still produce erroneous uncomputations, an we need to post-select on the correct ones. The latter are characterized by the respective register being measured as all $0$.


DQI Implementation in Qrisp
---------------------------

Now that we have covered the theoretical aspect of the algorithm, let us jump into our implementation! We want to remind you again, 
that the theoretical aspects of the DQI procedure are very complex, and we can only cover so much here. So, if there are any questions left unanswered, 
we strongly recommend taking the time to read the `original paper by S. Jordan et. al. <https://arxiv.org/abs/2408.08292>`_ 

Lets reiterate - The algorithm consists of five different steps, executed on two different registers, the error and syndrome register. 

* Prepare the correctly weighted Dicke state
* Encode the problem specific phases
* Encode the constraints given by the problem matrix
* Uncompute the syndrome quantum state
* Apply a Hadamard-transform  

Before we tackle these steps, let us explain how we define the quantum registers:
We operate on two different quantum registers, which in the context of Qrisp may be represented by any sort of multi-qubit QuantumVariable (or other type). Multiple registers, i.e. variables, may then be grouped within a QuantumArray.
The first register, which is in the literature refered to as the error register, will be represented by a QuantumVariable ``qv_error`` of size $n$. On the second register we will apply the matrix operations resulting from the Quantum Gauss-Jordan-Elimination, see the :sec:`uncomputation section <Uncomputation and syndrome decoding>`. 
As mentioned previously, this routine operates on an augmented matrix, namely the transposed problem matrix augmented by the syndrome quantum state. And as you probably guessed, there is some functionality for that in Qrisp!
It requires using QuantumArray though, so the syndrome register, which we name ``qv_syndrome`` will be represented by a QuantumArray of length $m$ consisting on single-qubit QuantumVariables.


We start our implementation by looking at the unary amplitude encoding of the error register. As mentioned in the theoretical overview, the amplitudes to encode emerge as the pricipal eigenvector of a matrix related to 
the expected number of satisfied constraints, which is defined in Lemma 9.2 of the `DQI paper <https://arxiv.org/pdf/2408.08292>`_.
It is depended on a number of parameters: We have $m$ as the number of constraints, i.e. the number of rows in the problem matrix. The prime $p$ defines our finite field $\mathbb{F}_p$, which is set to $p=2$ in the case of max-XORSAT. Additionally, $r=1$ represents number of inputs yielding $+1$.

So let us build this matrix and solve for the pricipal eigenvector

::

    def get_optimal_w(m: int, l: int, p: int, r: int):

        # define the parameter d
        d    = (p - 2*r) / np.sqrt(r * (p - r))
        # create the main diagonal and the two offdiagonals as numpy arrays.
        diag = np.arange(l + 1) * d
        off  = np.sqrt(np.arange(1, l + 1) * (m - np.arange(1, l + 1) + 1))

        A = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")

        # solve for principal eigenvector
        _, vecs = eigsh(A, k=1, which="LA")
        w = vecs.flatten()
        
        # pad the weights list that is returned, s.t. it is a power of 2
        orig = len(w)
        pad_len = 2**int(np.ceil(np.log2(l+1)))
        weights = list(w)
        weights += [0 for i in range((pad_len -orig)) ]
        return weights


Next, we need to encode these amplitudes into the state to create $\sum_k w_k \ket{k}$. A simple fan-out strategy based on :func:`ry`-gates will do the trick.

::
            
    def uae_encoding(qa_error, num_constraints, weights):
        
        # Normalize weights
        weights = weights/sum(weights)
        w2 = weights  
        
        # Calculate the cumulative
        cum = np.concatenate(([0.0], np.cumsum(w2[:-1])))  
        denom = 1.0 - cum

        # Avoid division by zero, clip ratios
        ratio = np.where(denom > 0, w2 / denom, 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        betas = 2.0 * np.arccos(np.sqrt(ratio))

        # Actual encoding 
        # Apply first RY if nonzero
        if betas[0] != 0.0 and not np.isnan(betas[0]):
        ry(betas[0], qa_error[0][0])
        # Controlled rotations
        for i in range(1, num_constraints):
            if i < betas.size:
                b = betas[i]
                if b != 0.0 and not np.isnan(b):
                    with control( qa_error[i-1][0]):
                        ry(b,  qa_error[i][0])

Then we prepare a Dicke state superposition, i.e. further transform the state into $\sum^{l}_{k=0} w_k \frac{w_k }{\sqrt{\binom{m}{k}}} \sum_{ | y | = k} \ket{\textbf{y}}$.
Now this would require quite a bit of effort in any other framework. But with Qrisp, 
we already have according functionality in place! Which renders this as easy as calling the :func:`dicke_state` function (or any of its variations)! 
We will see this again later in the full implementation example, see the :sec:`respective section <DQI example for max-XORSAT>`.

What follows after is are fairly simple, and does not require much explanation from an implementation standpoint. The specific phase encoding
applies a phase on the error register in dependence on the solution constraints $\textbf{v}$. We will arrive at $\sum^{l}_{k=0} w_k \frac{w_k }{\sqrt{\binom{m}{k}}} \sum_{ | y | = k} (-1)^{\textbf{v} \cdot \textbf{y}} \ket{\textbf{y}}$.

::

    def specific_phase_encoding(qa_error, v):

        for index in range(len(v)):
            if v[index] == 1:
                z(qa_error[index])

The last transformation before we tackle the complex decoding steps is to encode the problem matrix. 
By applying :func:`cx`-gates conditioned on the binary $B^T$ matrix entries, we will receive the state $\sum^{l}_{k=0} w_k \frac{w_k }{\sqrt{\binom{m}{k}}} \sum_{ | y | = k} (-1)^{\textbf{v} \cdot \textbf{y}} \ket{\textbf{y}} \ket{B^T \textbf{y}}$.

::
        
    def constraint_encoding(qa_error, qv_syndrome, B):

        # reverse indices due to circuit definition
        qa_error, qv_syndrome = qa_error[::-1], qv_syndrome[::-1]

        # create B^T
        B_t = np.transpose(B)

        # encode constraints
        i_ind, j_ind = B_t.shape
        for i in range(i_ind):
            for j in range(j_ind):
                if  B_t[i][j] == 1: 
                    cx(qa_error[j], qv_syndrome[i])

        return qa_error, qv_syndrome


Decoding implementation in Qrisp
--------------------------------

We have now arrived at the decoding step, i.e. the uncomputation of $\ket{y}$. For this, we need a varition of the Gauss-Jordan elimination.
Our implementation is closing related to the one by N. Patamawisut et. al., found in their 2025 paper `Quantum Circuit Design for Decoded Quantum Interferometry <https://arxiv.org/abs/2504.18334>`_. A code example can be found on `their github repository <https://github.com/BankNatchapol/DQI-Circuit/tree/main>`_.
Considering the resources given by their research and the simplicity of the code we will not go into large detail on what happens here. If anything remains unclear, do not hesitate to reach out to us! (Or the original authors)

What this code does in essence is to calculate the necessary operations to bring the classical input matrix to reduced row echelon form. These may be a series of row swaps or row additions. 
A series of quantum operations is then returned, where the swaps are replaced by :func:`swap` gates and the additions by :func:`cx` gates, applied to the column which augments the matrix, i.e. the syndrome register ``qv_syndrome``, as shown in the next step. 

::

    def gauss_jordan_operations_general(matrix):

        m = len(matrix)
        num_cols = len(matrix[0])
        n = num_cols - 1  # last column is RHS
        # work on a copy
        mat = [row.copy() for row in matrix]
        operations = []
        pivot_row = 0
        pivot_col = 0
        # perform Gauss-Jordan procedure
        while pivot_row < m and pivot_col < n:
            # find pivot in this column at or below pivot_row
            pivot_idx = None
            for r in range(pivot_row, m):
                if mat[r][pivot_col] == 1:
                    pivot_idx = r
                    break
            if pivot_idx is None:
                pivot_col += 1
                continue

            # swap into pivot position if needed
            if pivot_idx != pivot_row:
                # append the correct operations
                operations.append(('swap', pivot_row, pivot_idx))
                mat[pivot_row], mat[pivot_idx] = mat[pivot_idx], mat[pivot_row]

            # eliminate other rows
            pivot_data = mat[pivot_row]
            for r in range(m):
                if r != pivot_row and mat[r][pivot_col] == 1:
                # append the correct operations
                    operations.append(('xor', pivot_row, r))
                    # XOR rows from pivot_col onward
                    row_r = mat[r]
                    row_r[pivot_col:] = [a ^ b for a, b in zip(row_r[pivot_col:], pivot_data[pivot_col:])]

            pivot_row += 1
            pivot_col += 1

        return operations, mat

The following ``syndrome_decoding`` function is then pretty much just the application of the gates extracted in the previous step, applied to the correct qubits from the syndrome register.

::

    def syndrome_decoding(matrix, qv_syndrome):

        if isinstance(matrix, np.ndarray):
            mat_list = matrix.tolist()
        # extract the gates resulting from the Gauss-Jordan elimination 
        ops, mat = gauss_jordan_operations_general(mat_list)
        num_qubits = len(mat_list)

        # apply the gates on the syndrome register
        for op_type, src, tgt in ops:
            if op_type == "swap":
                swap(qv_syndrome[src][0], qv_syndrome[tgt][0])
            elif op_type == "xor":
                cx(qv_syndrome[src][0], qv_syndrome[tgt][0])
        
        return mat , qv_syndrome


**ATTENTION!** This decoding procedure may result in erroneous decodings. We only want to consider the states, where the decoding was succesful. Since for a correctly decoded syndrome the error register will be in the all $\ket{0}$-state this is easily achieved by post-selecting 
our final results on the measurement of the error register being n $0$s. This step can be seen in the :sec:`results evaluation section <Checking the results>`

And now the only thing we are missing is a Hadamard-transformation and we are done!
Let us now look at how to use these building bricks to solve an actual optimization problem.

DQI example for max-XORSAT
--------------------------

The subroutines shown below are encapsuled in the :func:`dqi_procedure` function and will be exectuded upon calling it. It that sense, the following shows you the inner workings of this our :func:`dqi_procedure` function.

As a final example let us go through the wole workflow again and apply it to solve a max-Cut optimization problem. 
We start by defining the problem matrix $B$ and the solution constraints $\textbf{v}$, which are all one, since any solution is valid.

::

    B  = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1]
        ])

    v = np.array(
        [1, 1, 1, 1, 1, 1],
        ) 

This will result in the following graph:

::

    G = nx.from_numpy_array(B)
    nx.draw(G, with_labels=True)
    import matplotlib.pyplot as plt
    plt.show()


So let us go through the routine.
We start by defining our two :class:`QuantumArray`s ``qa_error`` and ``qa_syndrome``, representing the error and syndrome register respectively.
::

    m = B.shape[0]
    q_type = QuantumFloat(1, 0)
    qa_error = QuantumArray(qtype= q_type , shape=(m,))

    # create syndrome quantumArray
    qa_syndrome = QuantumArray(qtype= q_type , shape=(m,))

Next, we prepare the correctly weighted Dicke state. We use the functions ``get_optimal_w`` to calculate the weights and ``uae_encoding`` to fan them out across the error register ``qa_error``.
A final call of the :func:`dicke_state` preparation routine results in the desired superposition.

::

    # l can be chosen to be equal to m, or as rank(B) as an optimized version (or even more optimized)
    # --> it defines the uae_eigenvector consideration and size qv_inner,
    princ_eigenvec = get_optimal_w(m,l,p,r)
    
    # UAE encoding encoding
    uae_encoding(qa_error, m, princ_eigenvec)

    # need to reverse due to dicke_state function definition
    dicke_state(qa_error[::-1], len(qa_error)) 

We continue with the problem specific encodings, namely the specific phase encoding and constraint encoding, via the respective ``specific_phase_encoding`` and ``constraint_encoding`` functions. 

::

    # phase encoding
    specific_phase_encoding(qa_error, v)

    # constraint encoding
    qa_error, qa_syndrome = constraint_encoding(qa_error, qa_syndrome, B)

After which, we have reached the heart of the algorithm, the decoding. As explained earlier, we employ our classical Gauss-Jordan elimination procedure ``syndrome_decoding`` and extract the gates to apply to the ``qa_syndrome``.

::

    syndrome_decoding(B.T, qa_syndrome)


We then further encode these actions on the error register ``qa_error``

::

    for i in range(m):
        cx(qa_syndrome[i], qa_error[i]) 

And further invert the ``syndrome_decoding`` step for proper uncomputation.

::
    
    with invert():
        syndrome_decoding(B.T, qa_syndrome)

The last is then the aforementioned Hadamard-transformation.

::

    h(qa_syndrome)   

And finally, we measure!

::

    res_dict = multi_measurement([qvin,  resqf])

Et voilá! We have executed all steps to find an optimal solution to the maxCut problem via the DQI algorithm! 


Checking the results
--------------------

Let us now investigate the results:
First we extract the correct and also the best solutions, post-selected on the ``qa_error`` being measured in the all $\ket{0}$ state

::

    res_dict = multi_measurement([qvin,  resqf])

    # get the best and all correct solutions
    best_sols_dqi = []
    corr_sols_dqi = {}
    max_val = max(res_dict.values())
    for key,val in res_dict.items():
        cor_val = True
        for i in range(len(key[0])):
            if key[0][i]!= 0:
                cor_val = False
                break
        if cor_val:
            #print(key, val)  
            corr_sols_dqi.setdefault(key[1],val)
            if val == max_val:
                best_sols_dqi.append(key[1])


We want to compare them to a brute force solution. The following can also be found in `Patamawisut's github repository <https://github.com/BankNatchapol/DQI-Circuit/blob/main/src/dqi/utils/solver.py>`_, where he provides a similar plot for his solution.
What we can see here is a comparison between the likelyhood of a result appearing as a result of our DQI algorithm, plotted against to objective value of a solutions aquired via brute force. The results speaks for itself. 
Remarkably, all high objective value results appear with the highest likelyhood among our post-selected states!

Also, note that is alot of code for a singular plot! This is additional material, we will omit a clear explanation here, since this is just visulization.

::

    from itertools import product
    from typing import Callable, List, Tuple, Dict
    import numpy as np
    import networkx as nx

    # retype solutions as ints to order them, and assign objective value
    def _enumerate_solutions(num_vars: int, score_fn: Callable[[Tuple[int, ...]], int]): 

        results = []
        for bits in product([0,1], repeat=num_vars):
            results.append((bits, score_fn(bits)))
        # sort so best scores come first
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # get all solutions
    def max_xorsat_all_solutions(G: nx.Graph) -> Tuple[List[Dict[int,int]], int]:
        def score(bits: Tuple[int,...]) -> int:
            return sum((bits[i] ^ bits[j]) == 1 for i, j in G.edges())
        raw = _enumerate_solutions(G.number_of_nodes(), score)
        best_score = raw[0][1]
        # collect all assignments tying the best_score
        best = [bits for bits, s in raw if s == best_score]
        return ([{i: bits[i] for i in range(len(bits))} for bits in best], best_score)

    # solve for all solutions
    def brute_force_max(B: np.ndarray, v: np.ndarray) -> List[Tuple[str,int]]:
        m, n = B.shape
        def score(bits: Tuple[int,...]) -> int:
            x = np.array(bits, dtype=int)
            return int(np.sum((-1)**(B.dot(x)+v)))
        raw = _enumerate_solutions(n, score)
        # return sorted ascending by bitstring integer (you can re-sort here if you like)
        # or leave descending by score—choose whichever API you prefer
        return [("".join(map(str,bits)), sc) for bits, sc in raw]

    # solve for all solutions - function call
    brute_force_results = brute_force_max(B, v)

    # create the cool plot
    def plot_results_union_matplotlib(
    brute_force_results: List[Tuple[str, int]],
    dqi_results: Dict[str, int],
    plot_name: str = "Comparison of DQI and True Objective Values",
    spline_smoothing: float = 1.0,
    left_axis_margin: float = 0.05  # fraction of max to leave above the curve
    ) -> None:
        """
        Dual-axis Matplotlib chart: brute-force objective vs. DQI probability.
        Secondary axis centered on 0 to align with primary axis.
        Primary axis has a tighter cutoff near the max value.
        """
        if not brute_force_results:
            raise ValueError("brute_force_results must be non-empty")

        # determine full bit-length from brute-force labels
        full_len = len(brute_force_results[0][0])

        # normalize DQI keys
        norm_dqi: Dict[str, int] = {}
        for key, count in dqi_results.items():
            parts = key.split()
            if len(parts) == 2:
                prefix, suffix = parts
            else:
                prefix, suffix = "", parts[0]
            bits = suffix
            if len(bits) < full_len:
                bits = "0" * (full_len - len(bits)) + bits
            full_bits = (prefix + " " + bits) if prefix else bits
            norm_dqi[full_bits] = norm_dqi.get(full_bits, 0) + count
        
        prefix = ""
        if dqi_results:
            key_split = list(dqi_results.keys())[0].split(" ")
            if len(key_split) == 2:
                prefix = "0"*len(key_split[0]) 
        bf_dict = {f"{prefix} {label}".strip(): val for label, val in brute_force_results}
        for bits in norm_dqi:
            if bits not in bf_dict:
                bf_dict[bits] = 0

        # union of all labels
        all_keys = set(bf_dict) | set(norm_dqi)
        sorted_keys = sorted(all_keys, key=lambda k: int(k.replace(" ", ""), 2))

        # prepare series
        bf_values  = [bf_dict.get(k, 0) for k in sorted_keys]
        ext_counts = [norm_dqi.get(k, 0) for k in sorted_keys]
        total_ext  = sum(ext_counts)
        ext_probs  = [(c / total_ext) if total_ext else 0 for c in ext_counts]

        x = np.arange(len(sorted_keys))

        # start plot
        fig, ax1 = plt.subplots(figsize=(14, 5))

        # spline smoothing for bf_values if more than 3 points
        if len(x) > 3:
            spline = UnivariateSpline(x, bf_values, s=spline_smoothing*len(x))
            x_smooth = np.linspace(x.min(), x.max(), 10000)
            bf_smooth = spline(x_smooth)
            
            # clip spline to original min/max
            bf_smooth = np.clip(bf_smooth, min(bf_values), max(bf_values))
            
            ax1.plot(x_smooth, bf_smooth, color="blue", label="Objective Value")
        else:
            ax1.plot(x, bf_values, color="blue", label="Objective Value")

        # set tighter y-axis using clipped spline
        ax1.set_ylim(min(bf_values) - 0.05*abs(min(bf_values)),
                    max(bf_smooth) + left_axis_margin*max(bf_values))

        ax1.fill_between(x_smooth, bf_smooth, color="blue", alpha=0.2)
        ax1.set_ylabel("Objective Value", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # tighter y-axis range for left axis
        bf_max = max(bf_values) if bf_values else 1
        bf_min = min(bf_values) if bf_values else 0
        ax1.set_ylim(-(bf_max + bf_max*left_axis_margin), bf_max + bf_max*left_axis_margin)

        # secondary axis for DQI probabilities
        ax2 = ax1.twinx()
        
        # make secondary y-axis symmetric around 0
        dqi_max = max(ext_probs) if ext_probs else 0.1
        #ax2.set_ylim(-dqi_max- dqi_max*0.2, dqi_max+ dqi_max*0.2)
        ax2.set_ylim(-dqi_max - dqi_max*0.2 , dqi_max+ dqi_max*0.2)
        # plot bars centered on 0
        ax2.bar(x, ext_probs, alpha=0.6, color="red", label="DQI (Probability)", bottom=0)

        ax2.set_ylabel("Probability (DQI)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.xticks(x, sorted_keys, rotation=45, ha="right")
        plt.title(plot_name)

        # combine legends
        lines, labels = ax1.get_legend_handles_labels()
        bars, blabels = ax2.get_legend_handles_labels()
        ax1.legend(lines + bars, labels + blabels, loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

    # create the cool plot - function call
    plot_results_union_matplotlib(
        brute_force_results,
        corr_sols_dqi,
        plot_name="Comparison of DQI Results with True Objective Values (Post-selected on |0⟩)",
        spline_smoothing=1.3
        )


Congratulations, you have implemented the full version of the DQI algorithm and are now able to tackle optimization problems all on your own and can investigate the results with adequate vizualisation! Check out the next section for a heads up on where to go from here.

Further reading
---------------

The interested reader has many avenues to explore, starting with other use cases from the `DQI paper <https://arxiv.org/abs/2408.08292>`_, namely general LINSAT and OPI. We have not implemented them in Qrisp, and if you are interested and need help feel free to reach out! 
Another topic of interest is the belief propapagation version of syndrome decoding, which according according to the original work seems to be the best option for achieving true quantum advantage with the DQI algorithm.  

Should you be interested in resource analysis, we again refer to `Patamawisut's paper <https://arxiv.org/abs/2504.18334>`_, where he gives a detailed description on all aspects of the algorithm.

Now that you have reached the end of the tutorial, we hope you had alot of fun! If you like eploring Qrisp more, go have a look at our other tutorials! 
