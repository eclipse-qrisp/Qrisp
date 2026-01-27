import pandas as pd
import numpy as np
from qrisp.algorithms.cold.problems.QUBO import solve_QUBO
from qrisp.algorithms.cold.cold_benchmark import most_likely_cost_and_prob
import time

if __name__ == '__main__':

    # Example QUBO
    Q = np.array([
    [ 2.000e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 4.000e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 2.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 4.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+02,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  7.080e+02, -1.416e+03, 1.416e+03, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 1.416e+03, -1.416e+03, 0.000e+00, 0.000e+00, 0.000e+00,  1.416e+03, -1.416e+03, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 1.416e+03, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, -1.416e+03, 1.416e+03],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 7.080e+02, 1.416e+03, -1.416e+03,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 7.080e+02, -1.416e+03,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 7.080e+02,  0.000e+00, 0.000e+00, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  7.080e+02, -1.416e+03, 0.000e+00],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 1.416e+03, -1.416e+03],
    [ 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00, 0.000e+00, 7.080e+02]])

    N = Q.shape[0]

    # Initialize dataframe
    header = ["method", "N", "T", "N_step", "uniform params", "N_opt", "CRAB", 
            "runtime", "most_likely_res", "most_likely_prob", "full_res"]
    data = []


    # Set algorithm parameters
    uniform = False                     # uniform AGP coeffs
    N_step = 15                         # number of timesteps (15 for depth = 103)
    T_range = np.arange(1, 15, 1)       # evolution times
    N_opt = 1                           # number of optimizable parameters in H_control
    objective = 'agp_coeff_magnitude'   # objective for optimization
    bounds = ()                         # bounds for optimization


    print(f'N = {N} \nObjective = {objective} \nUniform = {uniform} \nN_opt = {N_opt}\n')

    # Iterate over evolution times
    for T in T_range:
        print(f'T = {T}')

        ####################
        ## Solve with LCD ##
        ####################

        problem_args = {"method": "LCD", "uniform": uniform, "agp_type": "order1"}
        run_args = {"N_steps": N_step, "T": T}
        
        t0 = time.time()
        LCD_result = solve_QUBO(Q, problem_args, run_args)
        runtime = time.time() - t0

        # Get most likely cost and its probability
        ml_cost, ml_prob = most_likely_cost_and_prob(LCD_result, 1)

        # Save to list
        data.append(["LCD", N, T, N_step, uniform, 0, None, 
                    runtime, ml_cost, ml_prob, LCD_result])


        #####################
        ## Solve with COLD ##
        #####################

        CRAB = False

        problem_args = {"method": "COLD", "uniform": uniform}
        run_args = {"N_steps": N_step, "T": T, "N_opt": N_opt, "CRAB": CRAB, "objective": objective, "bounds": bounds}

        t0 = time.time()
        COLD_result = solve_QUBO(Q, problem_args, run_args)
        runtime = time.time() - t0

        # Get most likely cost and its probability
        ml_cost, ml_prob = most_likely_cost_and_prob(LCD_result, 1)

        # Save to list
        data.append(["COLD", N, T, N_step, uniform, N_opt, CRAB, 
                    runtime, ml_cost, ml_prob, COLD_result])
        

        ##########################
        ## Solve with COLD-CRAB ##
        ##########################

        CRAB = True

        problem_args = {"method": "COLD", "uniform": uniform}
        run_args = {"N_steps": N_step, "T": T, "N_opt": N_opt, "CRAB": CRAB, "objective": objective, "bounds": bounds}
        
        t0 = time.time()
        COLD_result = solve_QUBO(Q, problem_args, run_args)
        runtime = time.time() - t0

        # Get most likely cost and its probability
        ml_cost, ml_prob = most_likely_cost_and_prob(LCD_result, 1)

        # Save to list
        data.append(["COLD-CRAB", N, T, N_step, uniform, N_opt, CRAB, 
                    runtime, ml_cost, ml_prob, COLD_result])

        df = pd.DataFrame(data, columns=header)
        df.to_csv("Q_proj.csv") 
