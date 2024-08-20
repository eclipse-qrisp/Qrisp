.. _PortfolioRebalancing:

Portfolio rebalancing with constrained mixers
=============================================
In this tutorial we will explain how solve discrete portfolio rebalancing problems within the scope of the :ref:`Alternating Operator Ansatz <AOA>` .

Portofolio rebalancing is well studied and widely applicable problem in the world of financial mathematics. Possible quantum algorithms for improvements have been investigated in the last years. Within Qrisp we have created the infrastructure to employ the findings in a straight forward manner!

The most relevant research for this tutorial can be found in `Portfolio rebalancing experiments using the
Quantum Alternating Operator Ansatz <https://arxiv.org/pdf/1911.05296.pdf>`_ by M. Hudson et. al. This paper sets the baseline for translating the portofolio rebalancing optimization problem into the QAOA structure. We will give an outline of the mathematical intrecacies in the following section.

Additionally, the paper `Grover Mixers for QAOA: Shifting Complexity
from Mixer Design to State Preparation <https://arxiv.org/pdf/2006.00354.pdf>`_ by A. Baertschi et. al. introduces the notion of using constrained mixers, limiting the state subspace to only feasible states, i.e. allowed portfolios. This is achieved using the algorithm from `Deterministic Preparation of Dicke States <https://arxiv.org/pdf/1904.07358.pdf>`_ by A. Baertschi et. al. We will also cover this novel approach in our tutorial, as it is already embedded in our infrastructure.

For additional info on problem emergence, real world applications and details on the construction of the optimization problem, we refer to the `paper by M. Hudson et. al. <https://arxiv.org/pdf/1911.05296.pdf>`_ . Including it here is out scope for the implementation and application on quantum frameworks, which we want to focus on in this tutorial. We will therefore only guide you through the steps of translating the mathematical description to code, which can be solved with QAOA.


The portfolio rebalancing problem
---------------------------------


We define the problem of portfolio rebalancing to be a discrete portfolio optimization problem, as

.. math::

    \textbf{z} = \text{argmin}_{\textbf{z}} C_{RR}(\textbf{z}) + C_{TC}(\textbf{z}),

where

*  :math:`\textbf{z} = \{ -1, 0, +1 \}^N` defines the solution vector of discrete portfolio asset positions to be held, representing long (+1), short (−1) or no-hold (0) states, with :math:`N` being the number of available portfolio assets,
*  :math:`C_{RR}(\textbf{z})` describes the normalized risk-return function, and
*  :math:`C_{TC}(\textbf{z})` describes the normalized trading cost function

This is subject to an investment constraint, i.e. 

.. math:: 
    
    \sum_{i=1}^N z_i = D 

Here, :math:`D` describes the net total of discrete :math:`\textit{lots}` to be invested in. The classical risk-return function is given as 

.. math::

    C_{RR}(\textbf{z}) = \lambda \sum_{i=1}^N \sum_{i=j}^N \sigma_{ij} - (1 - \lambda) \sum_{i=1}^N \mu_i z_i , 

where

* :math:`\lambda \in \mathbb{R}, 0 \leq \lambda \leq 1`, is an asset manager control parameter to favor lower risk the higher the factor,  
* :math:`\sigma` is the normalized asset returns covariance matrix and
* :math:`\mu` is the normalized average asset returns vector

The trading cost function is given as 

.. math::

    C_{TC}(\textbf{z}) = \lambda \sum_{i=1}^N (1 - \delta (z_i -y_i))T, 

where

* :math:`\textbf{z} = \{ -1, 0, +1 \}^N` is the previous portfolio position,  
* :math:`\delta` is the dirac delta function and
* :math:`T` is the normalized asset trading cost.


The Quantum Portfolio rebalancing formulation
---------------------------------------------

:math:`\textbf{Attention!}` Do not confuse things here! Our original binary variables are described by :math:`\textbf{z}`, and the spin operators are referred to by :math:`s^{-}_i`/ :math:`s^{+}_i` (+ and - exponents are explained below) going forward. In the literature spin operators are usually referred to as :math:`Z`, given that they are :math:`Z` gates on the hardware. This is not the case in this piece of literature, and we decided to go with the literature formulation.

We have a given number of assets on which can we invest in short and/or long positions. We store this info in a ``QuantumArray`` of two ``QuantumVariables`` of length :math:`N`, i.e. the number of assets. The short postions are stored in the ``QuantumArray`` -index ``0`` and the long postions are stored in the ``QuantumArray`` -index ``1``. A ``1`` on index ``3`` on ``QuantumArray`` -index ``1`` then means we have long position on asset 4 (arrays are ``0``-indexed). This ``QuantumArray`` is therefore our quantum argument on which we run the the QAOA optimization on to find it in the the state of the optimial solution afterwards. You should be familiar with this concept from other QAOA problem instances in Qrisp. 

So how do we go from the classical formulation above to the quantum formulation? In the well know Ising model fashion we first convert the binary variables :math:`\textbf{z}` to spin variables:

We use two different spin operators: :math:`s^{+}_i`, which only operates on the :math:`\textbf{long}` postion :math:`i`, and :math:`s^{-}_i`, which only operates on the :math:`\textbf{short}` postion :math:`i`. They therefore operate on the two different indices of the ``QuantumArray`` argument: 

:math:`s^{-}_i` operates on the short postions in the ``QuantumArray`` -index ``0`` and :math:`s^{+}_i` operates on the long postions in the ``QuantumArray`` -index ``1``

The binary variables then turn into 

.. math::

    z_i  = \frac{s^{+}_i - s^{-}_i}{2},

Now we can plug this into the equations above to receive

.. math::

    C_{RR}(\textbf{s}) = \lambda \sum_{i=1}^N \sum_{i=j}^N \frac{\sigma_{ij}}{4} (s^{+}_i s^{+}_j  ) - (1 - \lambda) \sum_{i=1}^N \mu_i z_i , 

and 

.. math::

    C_{TC}(\textbf{s}) = \frac{1}{4} T (3 + (1-y_i^2 -y_i)s^{+}_i + (1-y_i^2 +y_i)s^{-}_i + (2y_i^2 -1)s^{+}_i s^{-}_i ).

With these two equations we can represent the full cost Hamitonian as :math:`H_C = C_{RR}(\textbf{s}) + C_{TC}(\textbf{s})`. The code that is "under the hood" for our implementation is pretty much just this equation typed out into gates, so will omit the code example here. Feel free to have a look at the source code if you're interested though!


Constrained Mixer
-----------------

As mentioned in the introduction, `the paper by A. Baertschi et. al <https://arxiv.org/pdf/2006.00354.pdf>`_ introduces the notion of preparing an allowed state as an initial starting value and then using a constrained mixer to reduce the search space to only feasible states. This fits perfectly into our established infrastructure!

We therefore start by creating a properly weighted intitial state. A clear description on the conditions here are given in the paper above in page 9 onwards. So, instead of copying the text you can find there word for word, we will refer to this paper here. The two important terms here are *lots* and *bands*. The number of *lots* :math:`d`, is the net total of long minus short positions in the portfolio, and determines the feasible subspace in this context.
The solution state for a portfolio with 4 assets, that is defined via the number of lots being 1, can have 

* 4 ones in the long, 3 ones in the short position ``QuantumVariable`` s of the QuantumArray, i.e. a state such as $\ket{0111}\ket{1111}$
* 3 ones in the long, 2 ones in the short position, i.e. a state such as $\ket{0011}\ket{0111}$
* 2 ones in the long, 1 ones in the short position, i.e. a state such as $\ket{0001}\ket{0011}$
* 1 one in the long, 0 ones in the short position, i.e. a state such as $\ket{0000}\ket{0001}$

The *band* refers to the number of short positions held.
Now, any permutation within the short and long postions, that still contain the correct number of ones (i.e. for the first example state $\ket{1110}\ket{1111}$) is also a valid solution, and part of *band* 3. This resembles the subspace of Dicke states, which we aim to prepare with our mixer.

We therfore start by a correctly weighted superposition with one state from each band, and turn this superposition into a superposition of **all** allowed states with a Dicke mixer. 

The code example below creates said initial superposition with correct weights for a portfolio with 2 lots

:: 
        
    def state_prep(q_array):
        # In the actual implementation the lots are defined outside of this function in a wrapper function
        # this function works for an arbitrary number of lots
        lots = 2

        l = q_array[1]
        s = q_array[0]
        
        n = len(l)
        band_prefix = dict()
        max_pref = 0
        for index in range(n- lots +1): 
            max_pref += math.comb(n, index)*math.comb(n, lots+ index)
            this_pref = math.comb(n, index)*math.comb(n, lots+ index)
            band_prefix.setdefault(str(index), this_pref)

        x(l[-lots:])
        param = 2 * np.arccos(np.sqrt((band_prefix["0"])/(max_pref)))
        ry(param,  s[-1])
        qc_s = s[-1].qs()

        for index1 in range(1,lots):
            param = 2 * np.arccos(np.sqrt((band_prefix[str(index1)])/(max_pref)))
            cry_gate = RYGate(param).control(1)
            qc_s.append(cry_gate, [s[-index1], s[-index1-1]])

        for index2 in range(1,lots+1):
            cx(s[-index2],l[-lots -index2])


With this initial superposition we now turn to the creation of the multichannel constrained Dicke mixer to receive the superposition of all allowed states. It is based on the ``dicke_state`` function, with creates the Dicke state $\ket{D^n_k}$ with Hamming weight :math:`k` from an input state with Hamming weight k, i.e. $\ket{0}^{\otimes n-k} \ket{1}^{\otimes k}$. This function is in turn based on the `deterministic algorithm by A. Baertschi et. al. <https://arxiv.org/pdf/1904.07358.pdf>`_ 

The code example defines our mixer. It calls the inverse conjugated formulation of the ``dicke_state`` function on the short and long position ``QuantumVariables`` separately. With additional multi-controlled phase gates we then create our multi-channeled mixer.

:: 
        
    def portfolio_mixer():

        from qrisp.misc.dicke_state import dicke_state

        def inv_prepare_dicke(qv, k):
            with invert():
                dicke_state(qv, k)

        def apply_mixer(q_array, beta):
            half = int(len(q_array[0])/2)
            qv1 = q_array[0]
            qv2 = q_array[1]

            # apply mcp gates for all dicke weights
            with conjugate(inv_prepare_dicke)(qv1, half):
                for i in range(half):
                    ctrl_state = "0" * (half-i-1) + ("1"*(i+1))
                    mcp(beta, qv1, ctrl_state = ctrl_state)
            
            with conjugate(inv_prepare_dicke)(qv2, half):
                for i in range(half):
                    ctrl_state = "0" * (half-i-1) + ("1"*(i+1))                   
                    mcp(beta, qv2, ctrl_state = ctrl_state)
            
        return apply_mixer

Et voilà! We now have what we need to run and optimize a portfolio rebalancing instance with QAOA! Well, except for the classical cost function, which is just the original equations for the portfolio costs translated into code. We will omit closer investigation, but feel free to look into the source code if you're interested!


Example implementation
----------------------

Let us now look at what these building blocks look like in working code example. We start off by defining the relevant parameters for a portfolio rebalancing instance, namely the number of assets and lots, the asset covariance matrix, the previous portfolio positions, the risk return factor, the normalized asset returns and the trading cost.   

::

    # assign problem definitions
    
    # number of assets
    n_assets = 4

    # lots
    lots = 2

    # old positions
    old_pos = [1, 1, 0, 1, 0, 0]
    # risk return factor
    risk_return = 0.9
    # trading costs
    T = 0 

    # covariance between assets -- create covar_matrix 
    covar_string = "99.8 42.5 37.2 40.3 38.0 30.0 46.8 14.9 42.5 100.5 41.1 15.2 71.1 27.8 47.5 12.7 37.2 41.1 181.3 17.9 38.4 27.9 39.0 8.3 40.3 15.2 17.9 253.1 12.4 48.7 33.3 3.8 38.0 71.1 38.4 12.4 84.7 28.5 42.0 13.1 30.0 27.8 27.9 48.7 28.5 173.1 28.9 -12.7 46.8 47.5 39.0 33.3 42.0 28.9 125.8 14.6 14.9 12.7 8.3 3.8 13.1 -12.7 14.6 179.0"
    li = list(covar_string.split(" ")) 
    fin_list = [float(item) for item in li]
    norm_sigma_array = preprocessing.normalize([fin_list])
    norm_sigma_full = np.reshape(norm_sigma_array, (8, 8))


    # normalized asset returns
    import numpy as np
    # first entry is return, second is covariance
    data1 = [0.000401, 0.009988 ,-0.000316 ,0.014433, 0.000061, 0.010024  ,0.001230 ,0.014854 ,0.000916 ,0.013465 , -0.000176, 0.010974 ,-0.000619 ,0.015910 , 0.000396 ,0.010007 ,0.000212, 0.009201 , -0.000881, 0.013377 ,0.001477, 0.013156 , 0.000184, 0.009907 ,0.001047, 0.011216 , 0.000492, 0.008399 ,0.000794, 0.010052 , 0.000291, 0.013247 ,0.000204, 0.009193 , 0.000674 ,0.008477,0.001500, 0.014958 , 0.000491, 0.010873]
    mu = [data1[i] for i in range(len(data1)) if i%2 ==0]
    from sklearn import preprocessing
    mu_array = np.array(mu)
    #this is the normalized asset return array
    norm_mu_full = preprocessing.normalize([mu_array])


    # full problem instance
    asset_return = list(norm_mu_full[0][:n_assets])
    covar_matrix = norm_sigma_full[0:n_assets,0:n_assets]
    problem = [old_pos,risk_return,covar_matrix,asset_return, tc]


What you see above is the example from the original portfolio rebalancing paper as referenced in the introduction. We hope to be able to provide more real world application data soon!

With established example, all that remains to do is assign the operators for the QAOA and we can run the optimization!

::

    from qrisp.qaoa.mixers import portfolio_mixer 
    from qrisp.qaoa.problems.portfolio_rebalancing import * 
    from qrisp.qaoa import QAOAProblem

    # assign operators
    cost_op = portfolio_cost_operator(problem=problem)
    cl_cost = portfolio_cl_cost_function(problem=problem)
    init_fun = portfolio_init(lots=lots)
    mixer_op = portfolio_mixer()

To run the problem, we initialize the ``QAOAProblem`` and ``QuantumArray`` instance:

::

    # assign QuantumArray to operate on
    qv = QuantumVariable(n_assets)
    q_array = QuantumArray(qtype=qv, shape=(2))

    # run the problem!
    theproblem = QAOAProblem(cost_operator=cost_op, mixer=mixer_op, cl_cost_function=cl_cost)
    theproblem.set_init_function(init_fun)
    theNiceQAOA = theproblem.run(q_array,depth = 3)

At the end, we can also have a look at the final results:

::

    #create a classical cost function to evaluate the results
    def a_cost_fct(key):
        half = len(key[0])
        new_key = [int(key[0][i])-int(key[1][i]) for i in range(half)]
        rr1 = sum([risk_return*covar_matrix[i][j] *new_key[i]*new_key[j] for i in range(half) for j in range(half)])
        rr2 = sum([(1-risk_return)*asset_return[j] *new_key[j] for j in range(half)])
        c_tc= sum([tc  for i in range(half) if new_key[i] != old_pos[i]])
        energy = -(rr1+ rr2+ c_tc)
        return energy

    maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
    for key, val in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if key in maxfive:

            print((key, val))
            print(a_cost_fct(key))


You've made it to the end of the tutorial! This is one of many financial applications to come in Qrisp, so stay tuned!