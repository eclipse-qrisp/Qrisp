from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import itertools
from qrisp import QuantumVariable
from qrisp.algorithms.cold.trotter_COLD import LCD_routine, COLD_routine
from qrisp.algorithms.cold.qubo_problems import *

### Some Functions for evaluating algorithms ###

def qubo_cost(Q, P):
    expected_cost = 0.0
    for bitstring, prob in P.items():
        # Convert bitstring (e.g., "10110") to numpy array of ints
        x = np.array([int(b) for b in bitstring], dtype=float)
        # Compute quadratic form x^T Q x
        cost = x @ Q @ x
        # Weight by probability
        expected_cost += prob * cost
    return expected_cost

def success_prob(meas, solution):
    sp = 0
    for s in solution.keys():
        try:
            sp += meas[s]
        except KeyError:
            continue
    return sp

def approx_ratio(Q, meas, solution):
    cost = qubo_cost(Q, meas)
    opt_cost = list(solution.values())[0]
    ar = cost/opt_cost
    return ar


# Prepare pandas dataframe 
header = ['N', 'tau', 'dt', 'alg', 'avgcost', 'sp', 'ar', 'runtime']


# Function to evaluate each algorithm for the problem given by Q with optimal 
# solution "solution",  evolution time T and timestep dt
def evaluate(Q, solution, T, dt, method):

    # Number of qubits
    N = Q.shape[0]
    # Number of timesteps
    N_steps = int(T/dt)
    # Number of control pulse parameters
    N_opt = 1

    print("\n--- Simulation parameters ---")
    print(f'T = {T}')
    print(f'Method: {method}')

    if method == 'LCD':

        qarg = QuantumVariable(N)

        t0 = time.time()
        meas_lcd = LCD_routine(Q, qarg, N_steps, T)
        t1 = time.time()

        data = [N, T, T/N_steps, 'lcd',
                qubo_cost(Q, meas_lcd), 
                success_prob(meas_lcd, solution), 
                approx_ratio(Q, meas_lcd, solution), 
                t1-t0]

    elif method == 'COLD':

        qarg = QuantumVariable(N)
    
        t0 = time.time()
        meas_cold = COLD_routine(Q, qarg, N_steps, T, N_opt, CRAB=False)
        t1 = time.time()

        data = [N, T, T/N_steps, 'cold', 
                qubo_cost(Q, meas_cold), 
                success_prob(meas_cold, solution), 
                approx_ratio(Q, meas_cold, solution), 
                t1-t0]
        
    elif method == 'COLD-CRAB':

        qarg = QuantumVariable(N)

        t0 = time.time()
        meas_cold_crab = COLD_routine(Q, qarg, N_steps, T, N_opt, CRAB=True)
        t1 = time.time()

        data = [N, T, T/N_steps, 'cold-crab', 
                qubo_cost(Q, meas_cold_crab), 
                success_prob(meas_cold_crab, solution), 
                approx_ratio(Q, meas_cold_crab, solution), 
                t1-t0]

    # Save as csv
    df = pd.DataFrame(data=data, columns=header)
    filename = f'eval_N{N}_T{T}_{method}.csv'
    df.to_csv(filename, index=False)


# Paralell ececution of each evloution time and method
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# methods = ["LCD", "COLD", "COLD-CRAB"]
# times = [1, 2, 3, 4, 5]

# tasks = list(itertools.product(methods, times))  # all (method, T) combinations
# n_tasks = len(tasks)

# # Run every combination of T and method on different GPU node
# if rank < n_tasks:
#     method, T = tasks[rank]
#     print(f"[Rank {rank}] Running {method} with T={T}")
    
#     evaluate(Q, solution, T, dt=0.01, method=method)
    
# else:
#     print(f"[Rank {rank}] idle.")