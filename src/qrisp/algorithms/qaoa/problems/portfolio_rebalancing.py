import math 
from qrisp import cx, RYGate, ry, x, rzz, rz
import numpy as np 



def portfolio_cost_operator(problem):
    """
    | Quantum cost operator  for the discrete portfolio rebalancing problem, as described in https://arxiv.org/pdf/1911.05296.pdf.
    | It is depended on the problem instance, including the old portfolio positions, the normalized covariance matrix, the normalized asset returns and trading costs. See example implementation for formatting. 

    Parameters
    ----------
    problem : List
        A list containing a the relevant data for the problem instance.

    Returns
    -------
    portfolio_cost_op: function
        A callable function to be applied to a QuantumArray for solving the problem instance.

    """


    old_pos = problem[0]
    risk_return = problem[1] 
    covar_matrix = problem[2]
    asset_return = problem[3] 
    #tradiing_cost
    tc = problem[4]/4
    def portfolio_cost_op(q_array, gamma):
        #does this work???? -- seems like it would
        #print("cost_op")
        #print(len(qv))
        l = q_array[1]
        s = q_array[0]

        #risk-return function
        for i in range(len(s)):
            prefac = 0
            for j in range(len(s)):
                
                prefac = gamma * covar_matrix[i][j]/4 * risk_return
                rzz(-prefac, l[i], s[j])
                rzz(-prefac, s[i], l[j])

                if i == j:
                    # case of rzz on same spin vars
                    rz(2*prefac, l[i])
                    rz(2*prefac, s[i])
                    continue

                rzz(prefac, l[i], l[j])
                rzz(prefac, s[i], s[j])
            
            prefac2 = gamma *asset_return[i] * (1-risk_return)/2
            rz(prefac2, l[i])
            rz(- prefac2, s[i])

        #trading cost function
        # first added term can be dropped, as it just results in a phase
        for i in range(len(s)):
            old_val = old_pos[i]
            rz(gamma* tc*(1 - old_val^2 - old_val), l[i])
            rz(gamma* tc*(1 - old_val^2 - old_val), s[i])
            rzz(gamma* tc* (2*old_val^2 -1), l[i], s[i] )

    return portfolio_cost_op


def portfolio_cl_cost_function(problem):
    """
    | Classical cost function for the discrete portfolio rebalancing problem, as described in https://arxiv.org/pdf/1911.05296.pdf.
    |It is depended on the problem instance, including the old portfolio positions, the normalized covariance matrix, the normalized asset returns and trading costs. See example implementation for formatting. 

    Parameters
    ----------
    problem : List
        A list containing a the relevant data for the problem instance.

    Returns
    -------
    cl_cost_function: function
        A callable function to calculate the cost value of the problem solution.

    """

    old_pos = problem[0]
    risk_return = problem[1] 
    covar_matrix = problem[2]
    asset_return = problem[3] 
    #tradiing_cost
    tc = problem[4]
    def cl_cost_function(res):
        #print(res)
        energy = 0
        counts = 0
        half = int(len(list(res.keys())[0][0]))
        
        key_list = []
        for key, val in res.items():
            
            #half = len(key)/2
            #new_key = [int(key[i])-int(key[i+half]) for i in range(half)]
            new_key = [int(key[0][i])-int(key[1][i]) for i in range(half)] # ??????
            key_list.append(new_key)
            rr1 = sum([risk_return*covar_matrix[i][j] *new_key[i]*new_key[j] for i in range(half) for j in range(half)])
            rr2 = sum([(1-risk_return)*asset_return[j] *new_key[j] for j in range(half)])
            c_tc= sum([tc  for i in range(half) if new_key[i] != old_pos[i]])
            energy -= (rr1+ rr2+ c_tc)*val
            counts += val
        final = energy/counts 
        #print(final)  
        return final
    
    return cl_cost_function





def portfolio_init(lots):
    """
    | Initial state for the discrete portfolio rebalancing problem, as described in https://arxiv.org/abs/1904.07358.
    | Depending on the number of lots a QuantumArray is prepared, where the first index describes the short positions held and the second index describes the long postions held.

    Parameters
    ----------
    lots : Int
        The number of lots in the initial portfolio position

    Returns
    -------
    state_prep: function
        A callable function to be applied to a QuantumArray to receive the initial state for the problem.

    """

    def state_prep(q_array):

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

        # how does one do the superpos? is everything controlled with everything that came before? or just the first? or only one before?
        for index1 in range(1,lots):
            param = 2 * np.arccos(np.sqrt((band_prefix[str(index1)])/(max_pref)))
            cry_gate = RYGate(param).control(1)
            qc_s.append(cry_gate, [s[-index1], s[-index1-1]])

        # the lots+ index thing below may cause problems in the future...
        for index2 in range(1,lots+1):
            cx(s[-index2],l[-lots -index2])
    return state_prep
