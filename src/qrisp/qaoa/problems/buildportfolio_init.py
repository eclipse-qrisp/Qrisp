#notes:

"""
encode portfolio: portfolio of n assets 
- two length n bit string
- str s: if 1 short on asset i
- str l: if 1 long on asset i

lots: num l - num s = d lots

cost function C(l,s):
riskreturn
trading costs
penalty for s and l on same i 

states: band k -- Hamming-weight k in short positions
--> dicke states as init

state prep: this is the essential part: based on dicke states --
see the attached picture
"""

from qrisp import * 
import math

def portfolio_init(lots):

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

""" lots = 2

init_fun = portfolio_init(lots=lots)

qv = QuantumVariable(8)

init_fun(qv)
print(qv.qs)
print(qv)
 """