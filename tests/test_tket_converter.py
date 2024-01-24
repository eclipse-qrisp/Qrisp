
#from qrisp.circuit.standard_operations import op_list
from qrisp.circuit.standard_operations import     XGate,YGate, ZGate,    CXGate,  CYGate,    CZGate,MCXGate,PGate,  CPGate,u3Gate,  HGate,RXGate,   RYGate,   RZGate,   MCRXGate,SGate , TGate, RXXGate,RZZGate,  SXGate,   SXDGGate,  Barrier, Measurement,  Reset,  QubitAlloc, QubitDealloc,GPhaseGate,  SwapGate,U1Gate,  IDGate
import numpy as np
from qrisp import QuantumVariable

def pytket_rand_test():
    qvRand = QuantumVariable(10)
    qcRand = qvRand.qs
    rotation = np.pi/2
    single_gates = [    
        XGate(),
        YGate(),
        ZGate(),
        HGate(),
        SXGate(), 
        SGate()]

    rot_gates = [
        RXGate,
        RYGate,
        RZGate,
        PGate]

    c_gates = [
        CXGate(),
        CYGate(),
        CZGate()]

    mc_gates = [
            SwapGate(),     
            ]

    mc_rot_gates =[
        RXXGate,
        RZZGate
            ]

    special_gates = [
                    MCXGate(control_amount=3)
                    ]


    op_list = [*mc_gates, *c_gates, *single_gates, *rot_gates, 
                *special_gates,
                ]

    used_ops = []
    for index in range(30):
        randInteg = np.random.randint(0,len(op_list)+1)
        if randInteg == len(op_list): 
            used_ops.append("mcrxxx")
        else:
            used_ops.append(op_list[randInteg])
        

    for op in used_ops:
        qubit_1 = qvRand[np.random.randint(7,9)]
        qubit_2 = qvRand[np.random.randint(3,6)]

        if op == "mcrxxx": 
            qcRand.append(MCRXGate(rotation, control_amount=3),[qvRand[0],qvRand[2],qubit_2,qubit_1])
        elif op in single_gates:
            qcRand.append(op, qubit_1)
        elif op in rot_gates:
            qcRand.append(op(rotation), qubit_1)
        # this is being called first due to mcx is subclass of cx reasons
        elif op in special_gates:
            qcRand.append(op, [qvRand[0],qvRand[2],qubit_2,qubit_1])
        elif op in mc_rot_gates:
        #elif op in c_gates:
            qcRand.append(op(rotation), [qubit_1,qubit_2])
        elif op in c_gates or mc_gates:
        #elif op in c_gates:
            qcRand.append(op, [qubit_1,qubit_2])

    #from qrisp.interface.converter.PyTket.convert_from_tket import convert_to_tket



    tket_qcRand = qcRand.to_pytket()
    #tket_qcRand = convert_to_tket(qc=qcRand)
    from pytket.extensions.qiskit import AerBackend
 
    tket_qcRand.measure_all()


    backend = AerBackend()
    if not backend.valid_circuit(tket_qcRand):
        compiled_circ = backend.get_compiled_circuit(tket_qcRand)
        assert backend.valid_circuit(compiled_circ)
    else: 
        compiled_circ = backend.get_compiled_circuit(tket_qcRand)
    #compiled_circ = tket_qcRand
    handle = backend.process_circuit(compiled_circ, n_shots=1000)
    result = backend.get_result(handle)
    cnt = result.get_counts()
    d = {}
    for key, value in cnt.items():
        Key = (str(index) for index in key)
        converted2 = ''.join(Key)
        #reverse string because of pytket specific return??
        #converted = converted2[::-1]
        converted = converted2
        d.setdefault(converted,value/1000)

    print(d)

    #print(result.get_counts(basis=BasisOrder.dlo)) 
    theRes = qvRand.get_measurement()
    print(theRes)

    for index4 in list(d.keys()):
        if not index4 in list(theRes.keys()):
            print("NOT IN THERE")
            print(index4)
        assert index4 in list(theRes.keys())
        if not theRes[index4]*0.6 <= d[index4] <= theRes[index4]*1.4:
            print(index4)
            print(theRes[index4])
            print(d[index4])
        if not d[index4] < 0.05:
            assert theRes[index4]*0.6 <= d[index4] <= theRes[index4]*1.4
    

