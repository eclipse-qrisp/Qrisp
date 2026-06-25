.. _ExactGrover:

Exact Grover's Algorithm
========================

In this example we will showcase how the ``exact`` keyword for :meth:`grovers_alg <qrisp.grover.grovers_alg>` can be applied. This keyword allows to produce states, which are exact solutions to the given oracle (ie. they have 0% theoretical failure rate). This feature is based on `this paper <https://arxiv.org/abs/quant-ph/0106071>`__. For this to work the amount of solutions to the oracle has to be known beforehand. Furthermore the oracle has to support the ``phase`` keyword which indicates how much of a phaseshift the winner states receive. For standard Grover oracles, this phaseshift is always $\pi$.

We demonstrate this functionality by preparing a state which is a superposition of all states that contain a fixed number of "ones":

To count the amount of ones we use quantum phase estimation on the operator

.. math::
    
    U = \text{exp}\left(\frac{i 2 \pi}{2^k} \sum_{i = 0}^{n-1} ( 1 - \sigma_{z}^i )\right)


::

    from qrisp import QPE, p, QuantumVariable, lifted
    from qrisp.grover import grovers_alg, tag_state
    import numpy as np

    def U(qv, prec = None, iter = 1):
        for i in range(qv.size):
            p(iter*2*np.pi/2**prec, qv[i])
    
    @lifted
    def count_ones(qv):
        prec = int(np.ceil(np.log2(qv.size+1)))
        res = QPE(qv, U, precision = prec, iter_spec = True, kwargs = {"prec" : prec})
        return res<<prec


Quick test:
    
>>> qv = QuantumVariable(5)
>>> qv[:] = {"11000" : 1, "11010" : 1, "11110" : 1}
>>> count_qf = count_ones(qv)
>>> count_qf.qs.statevector()
sqrt(3)*(|11000>*|2> + |11010>*|3> + |11110>*|4>)/3

We now define the oracle ::

    def counting_oracle(qv, phase = np.pi, k = 1):
        
        count_qf = count_ones(qv)
        
        tag_state({count_qf : k}, phase = phase)
        
        count_qf.uncompute()

And evaluate Grover's algorithm ::

    n = 5
    k = 3
    qv = QuantumVariable(n)
    
    import math
    
    grovers_alg(qv, counting_oracle, exact = True, winner_state_amount = math.comb(n,k), kwargs = {"k" : k})


>>> print(qv)
{'11100': 0.1, 
 '11010': 0.1, 
 '10110': 0.1, 
 '01110': 0.1, 
 '11001': 0.1, 
 '10101': 0.1, 
 '01101': 0.1, 
 '10011': 0.1, 
 '01011': 0.1, 
 '00111': 0.1}

We see that contrary to regular Grover's algorithm, the states which have not been tagged by the oracle have 0 percent measurement probability.