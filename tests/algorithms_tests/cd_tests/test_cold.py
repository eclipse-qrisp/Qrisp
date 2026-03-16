import numpy as np
from qrisp.algorithms.cold import solve_QUBO

def test_cold_uniform_magnitude():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, 
                "objective": "agp_coeff_magnitude", "CRAB": False}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_nonuniform_magnitude():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, 
                "objective": "agp_coeff_magnitude", "CRAB": False}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_uniform_cost():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, 
                "objective": "exp_value", "CRAB": False}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_nonuniform_cost():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, 
                "objective": "exp_value", "CRAB": False}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_coldcrab_uniform_cost():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, 
                "objective": "exp_value", "CRAB": True}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_coldcrab_uniform_magnitude():

    Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
    
    solution = "1011"
    
    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, 
                "objective": "agp_coeff_magnitude", "CRAB": True}
    
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_expvalue_method_statevector_measurement():
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])
    problem_args = {"method": "COLD", "uniform": True} #, "agp_type": "order1"}

    for exp_method in ["statevector", "measurement"]:
        run_args = {
            "N_steps": 4,
            "T": 1.0,
            "N_opt": 1,
            "objective": "exp_value",
            "precision": 0.1,
            "exp_value_method": exp_method,
        }
        res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)
        assert isinstance(res, dict)
        assert len(res) > 0


def test_cold_expvalue_method_backend():
    from qrisp.interface.provider_backends.qiskit_backend import QiskitBackend

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])
    problem_args = {"method": "COLD", "uniform": True} #, "agp_type": "order1"}
    backend = QiskitBackend()
    run_args = {
        "N_steps": 4,
        "T": 1.0,
        "objective": "exp_value",
        "N_opt": 1,
        "precision": 0.1,
        "exp_value_method": "backend",
        "exp_value_backend": backend,
    }
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)
    assert isinstance(res, dict)
    assert len(res) > 0