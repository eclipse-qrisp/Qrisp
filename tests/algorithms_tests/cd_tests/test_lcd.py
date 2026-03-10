import numpy as np
from qrisp.algorithms.cold import solve_QUBO

def test_lcd_order1_uniform():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "LCD", "agp_type": "order1", "uniform": True}
    run_args = {"N_steps": 50, "T": 10}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_lcd_order1_nonuniform():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "LCD", "agp_type": "order1", "uniform": False}
    run_args = {"N_steps": 50, "T": 10}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_lcd_nc_uniform():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "LCD", "agp_type": "nc", "uniform": True}
    run_args = {"N_steps": 50, "T": 10}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_lcd_nc_nonuniform():
    
    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "LCD", "agp_type": "nc", "uniform": False}
    run_args = {"N_steps": 50, "T": 10}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]