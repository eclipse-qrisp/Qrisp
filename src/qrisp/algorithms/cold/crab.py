import numpy as np
from sympy import Symbol

''' Class to incorporate CRAB (chopped random basis) optimization for the COLD method. '''

class CRABObjective:
    def __init__(self, H_p, qarg, qc, N_opt):
        self.H_p = H_p      # Problem Hamiltonian
        self.qarg = qarg
        self.qc = qc
        self.N_opt = N_opt  # Number of optimizaion parameters
        self.last_x = None  # Keep track of last result
        self.random_pulses = np.random.uniform(-0.5, 0.5, N_opt)
        self.iteration = 0

    def __call__(self, params):
        
        # Detect new iteration by comparing parameter vectors
        if self.last_x is None or np.allclose(params, self.last_x) == False:
            # When optimizer moves away from last point, refresh random distribution
            self.random_pulses = np.random.uniform(-0.5, 0.5, self.N_opt)
            self.iteration += 1
        self.last_x = params.copy()

        # Parameters to give to the quantum circuit for compilation
        # Optimization params
        subs_dic = {Symbol(f"par_{i}"): params[i] for i in range(len(params))}
        # Random pulse params
        subs_dic.update({Symbol(f"r_{k}"): self.random_pulses[k] for k in range(self.N_opt)})

        # Evluate cost
        cost = self.H_p.expectation_value(self.qarg, compile=False,
                                          subs_dic=subs_dic, precompiled_qc=self.qc)()
        return cost