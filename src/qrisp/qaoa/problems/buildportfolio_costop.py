"""
* risk return
* the trading costs of rebalancing the outdated portfolio into (l, s)
* a penalty term for the unwanted case of holding a long and a short position of the same asset



"""

#risk return
# c_rr = lamba sum^N_i sum^N_j sig_ij z_i z_j    -  (1-lamba) sum^N_i mu_i z_i

# where: lamba in 0,1 ;  sig is normalized covariance matrix ; mu normalized asset return vector
# z_i = s^+_i - s^-_i :: plus acts on long-qv, minus acts on short-qv, see direct substitution below

#second guy also has a closed form, but wants the previous iteration results?? but only once for the initial cost_op

from qrisp import * 

def portfolio_cost_operator(problem):
    """
    need to change this into a function which only considers one qv, instead of 2
    """

    old_pos = problem[0]
    risk_return = problem[1] 
    covar_matrix = problem[2]
    asset_return = problem[3] 
    #tradiing_cost
    tc = problem[4]/4
    def portfolio_cost(q_array, gamma):
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

    return portfolio_cost


def portfolio_cl_cost_function_for_qv(problem):


    old_pos = problem[0]
    risk_return = problem[1] 
    covar_matrix = problem[2]
    asset_return = problem[3] 
    #tradiing_cost
    tc = problem[4]
    def cl_cost_function_old(res):
        #print(res)
        energy = 0
        counts = 0
        half = int(len(list(res.keys())[0])/2)
        key_list = []
        for key, val in res.items():
            
            #half = len(key)/2
            new_key = [int(key[i])-int(key[i+half]) for i in range(half)]
            key_list.append(new_key)
            rr1 = sum([risk_return*covar_matrix[i][j] *new_key[i]*new_key[j] for i in range(half) for j in range(half)])
            rr2 = sum([(1-risk_return)*asset_return[j] *new_key[j] for j in range(half)])
            c_tc= sum([tc  for i in range(half) if new_key[i] != old_pos[i]])
            energy -= (rr1+ rr2+ c_tc)*val
            counts += val
        final = energy/counts 
        #print(final)  
        return final
    
    return cl_cost_function_old

            
def portfolio_cl_cost_function(problem):


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


""" qv = QuantumVariable(4)
q_array = QuantumArray(qtype=qv, shape=(2))

from qrisp.qaoa.buildportfolio_init import * 
lots = 2
init_fun = portfolio_init(lots=lots)
init_fun(q_array)
 """