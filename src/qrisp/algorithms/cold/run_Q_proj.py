import pandas as pd
import numpy as np
from qrisp import QuantumVariable
from qrisp.algorithms.cold.problems.QUBO_COLD import create_COLD_instance
from qrisp.algorithms.cold.problems.QUBO_LCD import create_LCD_instance
from qrisp.algorithms.cold.dcqo_problem import DCQOProblem
from qrisp.algorithms.cold.cold_benchmark import most_likely_res
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

        qarg = QuantumVariable(size=N)
        lam, alpha, H_init, H_prob, A_lam, J, h = create_LCD_instance(Q, "order1", uniform)
        LCD_prob = DCQOProblem(lam, alpha, H_init, H_prob, A_lam, J, h)
        t0 = time.time()
        LCD_result = LCD_prob.run(qarg, N_steps=N_step, T=T, method='LCD')
        runtime = time.time() - t0

        # Get most likely result and its probability
        ml = most_likely_res(Q, LCD_result, 1)
        ml_prob = [LCD_result[k] for k in ml.keys()]

        # Save to list
        data.append(["LCD", N, T, N_step, uniform, 0, None, 
                    runtime, ml, ml_prob, LCD_result])


        #####################
        ## Solve with COLD ##
        #####################

        CRAB = False
        qarg = QuantumVariable(size=N)
        lam, g, alpha, H_init, H_prob, A_lam, J, h, H_control = create_COLD_instance(Q, uniform)
        COLD_prob = DCQOProblem(lam, alpha, H_init, H_prob, A_lam, J, h, g, H_control)
        t0 = time.time()
        COLD_result = COLD_prob.run(
            qarg, N_step, T, 'COLD', N_opt, CRAB, 
            optimizer="COBYQA", objective=objective, bounds=bounds
            )
        runtime = time.time() - t0

        # Get most likely result and its probability
        ml = most_likely_res(Q, COLD_result, 1)
        ml_prob = [COLD_result[k] for k in ml.keys()]

        # Save to list
        data.append(["COLD", N, T, N_step, uniform, N_opt, False, 
                    runtime, ml, ml_prob, COLD_result])
        

        ##########################
        ## Solve with COLD-CRAB ##
        ##########################

        CRAB = True
        qarg = QuantumVariable(size=N)
        lam, g, alpha, H_init, H_prob, A_lam, J, h, H_control = create_COLD_instance(Q, uniform)
        COLD_prob = DCQOProblem(lam, alpha, H_init, H_prob, A_lam, J, h, g, H_control)
        t0 = time.time()
        COLD_result = COLD_prob.run(
            qarg, N_step, T, 'COLD', N_opt, CRAB, 
            optimizer="COBYQA", objective=objective, bounds=bounds
            )
        runtime = time.time() - t0

        # Get most likely result and its probability
        ml = most_likely_res(Q, COLD_result, 1)
        ml_prob = [COLD_result[k] for k in ml.keys()]

        # Save to list
        data.append(["COLD-CRAB", N, T, N_step, uniform, N_opt, True, 
                    runtime, ml, ml_prob, COLD_result])

        df = pd.DataFrame(data, columns=header)
        df.to_csv("Q_eval.csv") 
