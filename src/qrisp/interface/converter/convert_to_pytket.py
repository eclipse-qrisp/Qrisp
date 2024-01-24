
from qrisp import QuantumCircuit
from qrisp.core import QuantumVariable
from qrisp import ControlledOperation
from qrisp.interface import BackendClient
import numpy as np

def create_tket_instruction(op):
    from pytket import OpType
    
    from pytket.circuit import CircBox
    if op.name == "rxx":
        tket_ins = OpType.XXPhase
    elif op.name == "rzz":
        tket_ins = OpType.ZZPhase
    elif op.name == "ryy":
        tket_ins = OpType.YYPhase
    elif op.name == "measure":
        tket_ins = OpType.Measure
    elif op.name == "swap":
        tket_ins = OpType.SWAP
    elif op.name == "h":
        tket_ins = OpType.H
    elif op.name == "p":
        tket_ins =  OpType.Rz
    elif op.name == "x":
        tket_ins = OpType.X
    elif op.name == "y":
        tket_ins = OpType.Y
    elif op.name == "z":
        tket_ins = OpType.Z
    elif op.name == "rx":
        tket_ins = OpType.Rx
    elif op.name == "ry":
        tket_ins = OpType.Ry
    elif op.name == "rz":
        tket_ins = OpType.Rz
    elif op.name == "s":
        tket_ins = OpType.S
    elif op.name == "s_dg":
        tket_ins = OpType.Sdg
    elif op.name == "t":
        tket_ins = OpType.T
    elif op.name == "t_dg":
        tket_ins = OpType.Tdg
    elif op.name == "u3":
        tket_ins = OpType.U3
    
    elif op.definition:
        # if complex definition we create an abstract circBox for the section
        tket_definition = pytket_converter(op.definition, boxFlag= True)
        # might need adjustment
        #if len(op.definition.clbits):
        if tket_definition.n_qubits != op.num_qubits:
            raise Exception
        
        tket_ins = CircBox(tket_definition)

    else:
        raise Exception("Could not convert operation " + str(op.name) + " to PyTket") 

    return tket_ins



def pytket_converter(qc, boxFlag = False):
    from pytket import Circuit, Qubit, Bit, OpType
    
    from pytket.circuit import CircBox, QControlBox, Op

    # This dic gives the qiskit qubits/clbits when presented with their identifier
    qubit_dic = {}
    tket_qc = Circuit()
    #stringListQubs = []
    tketQubits = []
    for i in range(len(qc.qubits)):
        # add a named qubit
        tketQubits.append(Qubit(name = str(qc.qubits[i].identifier), index = i))
        qubit_dic[qc.qubits[i].identifier] = tketQubits[-1]
        tket_qc.add_qubit(tketQubits[-1])
    
    # Flag for alternative qubit assignment if we try to create an abstract CircBox
    if boxFlag:
        tket_qc = Circuit(len(qc.qubits))
        qubit_dic = dict()
        for i in range(len(qc.qubits)):
            qubit_dic[qc.qubits[i].identifier] = i

    clbit_dic = {}
    # Add Clbits
    tketClbits = []
    if len(qc.clbits):
        c_reg = tket_qc.add_c_register(name = "creg_std", size = len(qc.clbits))
    for i in range(len(qc.clbits)):
        clbit_dic[qc.clbits[i].identifier] = c_reg[i]
        #this will hopefully be used one day, when other simulators other than Aer are used with this backend, or... quantinuum decides to fix their backend integration
        # will throw an error on Aer backend and QASM converter, since they apparently only supports a single classical register, which is a lie
        """ tketClbits.append(Bit( name = str(qc.clbits[i].identifier)))
        clbit_dic[qc.clbits[i].identifier] = tketClbits[-1]
        tket_qc.add_bit(tketClbits[-1]) """

    for i in range(len(qc.data)):
        op = qc.data[i].op

        params = list(op.params)
        # Prepare qubits
        qubit_list = [qubit_dic[qubit.identifier] for qubit in qc.data[i].qubits]
        clbit_list = [clbit_dic[clbit.identifier] for clbit in qc.data[i].clbits]

        if op.name in ["cp", "p", "rx", "rz", "ry", "rxx", "rzz", "ryy", "u1", "u3"]: #and not boxFlag:
            #pytket expects angles in pi multiples
            params = [index/np.pi for index in params]  

        #add_gate
        if op.name in ["qb_alloc", "qb_dealloc"]:
            continue

        elif op.name == "cx":
            # maybe adjustment necessary here
            if hasattr(op, "ctrl_state"):
                tket_ins = OpType.CX
            else:
                tket_ins = OpType.CX

        elif op.name == "cy":
            if hasattr(op, "ctrl_state"):
                tket_ins = OpType.CY
            else:
                tket_ins = OpType.CY

        elif op.name == "cz":
            if hasattr(op, "ctrl_state"):
                tket_ins = OpType.CZ
            else:
                tket_ins = OpType.CZ
        
        elif op.name == "cp":
            if hasattr(op, "ctrl_state"):
                tket_ins = OpType.CRz
            else:
                tket_ins = OpType.Rz
        
        elif op.name == "sx":
            #bugged -> params empty
            params = []
            tket_ins = OpType.SX
        elif op.name == "sx_dg":
            #bugged -> params empty
            params = []
            tket_ins = OpType.SXdg

        elif op.name == "u1":
            params[0] = params[0]/np.pi
            #bugged
            tket_ins = OpType.Rz    
        elif op.name == "id":
            params = []
            #bugged
            tket_ins = OpType.noop


        elif issubclass(op.__class__, ControlledOperation):
            base_name = op.base_operation.name

            if len(base_name) == 1:
                base_name = base_name.upper()

            if 0==1:
            #if op.base_operation.definition:
                # old code relic -- buggy 
                # base_operation.definition doesnt convert correctly if mutiple abstract/costum gates are included
                tket_definition = pytket_converter(op.base_operation.definition)
                base_gate = tket_definition
                if isinstance( base_gate, Circuit):
                    tket_definition = pytket_converter(op.base_operation.definition, boxFlag=True)
                    tket_definition.name = base_name
                    tket_ins = CircBox(tket_definition)
                else: 
                    tket_ins = QControlBox(Op.create(base_gate) , len(op.controls))


            else:
                base_gate = pytket_converter(op.definition)
                
                if isinstance( base_gate, Circuit):
                    #circuit is returned as abstract CircBox --> pytket specific
                    tket_definition = pytket_converter(op.definition, boxFlag=True)
                    tket_definition.name = base_name
                    tket_ins = CircBox(tket_definition)

                else: 
                    #else a simpler multi controlled gate can be created
                    tket_ins = QControlBox(Op.create(base_gate) , len(op.controls))
                

        else:
            tket_ins = create_tket_instruction(op)

        if isinstance(tket_ins, CircBox):
            tket_qc.add_circbox(tket_ins, qubit_list)

        elif isinstance(tket_ins, QControlBox):
            tket_qc.add_qcontrolbox(tket_ins, qubit_list)
            
        elif len(clbit_list):
            # add other isinstance checks from above here aswell?
            tket_qc.add_gate(tket_ins, params, qubit_list + clbit_list)

        else:
            tket_qc.add_gate(tket_ins, params, qubit_list)

    return tket_qc
    










