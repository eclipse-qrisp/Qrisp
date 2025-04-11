from qrisp import QuantumFloat, measure, jaspify, y
from qrisp.alg_primitives.switch_case_tree_jasp import tree_switch_tree_jasp
import numpy as np

def test_switch_case_jasp_tree():
    @jaspify
    def fun(i: int):
        def case_fun(i, args):
            y(args[i])
        operand = QuantumFloat(8)
        case = QuantumFloat(3)
        case[:] = i
        case, operand = tree_switch_tree_jasp(operand, case, case_fun)
        return measure(case), measure(operand)

    c, o = fun(0)
    assert(np.abs(o - 2**c) < 0.001)
    assert(np.abs(c - 0) < 0.001)
    c, o = fun(3)
    assert(np.abs(o - 2**c) < 0.001)
    assert(np.abs(c - 3) < 0.001)
    c, o = fun(7)
    assert(np.abs(o - 2**c) < 0.001)
    assert(np.abs(c - 7) < 0.001)