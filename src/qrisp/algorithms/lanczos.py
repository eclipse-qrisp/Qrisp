from qrisp.operators import *
from qrisp import *
from qrisp.algorithms.grover import*
import numpy as np

def Lanczos(H, D):

    unitaries, coeffs = H.unitaries()

    num_unitaries = len(unitaries)

    def state_prep(case):
        prepare(case, np.sqrt(coeffs))

    n = np.int64(np.ceil(np.log2(num_unitaries)))



    def UR(case_indicator, operand, unitaries):
        qswitch(operand, case_indicator, unitaries)
        diffuser(case_indicator, state_function = state_prep)
    for k in jrange(0, 2*D):
        case_indicator = QuantumFloat(n)
        operand = QuantumVariable(2)
        if k % 2 == 0:
            with conjugate(state_prep)(case_indicator):
                for _ in jrange(k//2):
                    UR(case_indicator, operand, unitaries)
            print(case_indicator.qs)
            case_indicator.get_measurement()
    
    else:
        state_prep(case_indicator)
        for _ in jrange(k//2):
            UR(case_indicator, operand, unitaries)
        qv = QuantumVariable(1)
        h(qv)
        with control(qv[0]):
            qswitch(operand, case_indicator, unitaries)
        h(qv)
        print(case_indicator.qs)
        qv.get_measurement()
        qv.delete()
    
    case_indicator.delete()
    operand.delete()

