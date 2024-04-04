

from qrisp import *
import numpy as np

"""which one is the lower index?? i tajke it, that l2 is the lower one??"""

def dicke_state(qv, k):

    n = len(qv)

    for index2 in reversed(range(k+1, n+1)):
        split_cycle_shift(qv, index2, k)


    for index in reversed(range(2,k+1)):
        split_cycle_shift(qv, index, index-1)
    
    #print(qv)
    
    #return qv
    

def split_cycle_shift(qv, highIndex, lowIndex):

    index_range = [highIndex - i for i in range(lowIndex)]
    
    for index in index_range:
        param = 2 * np.arccos(np.sqrt((highIndex - index + 1 ) /(highIndex)) )
        #mcry_gate = RYGate(param).control(2)
        #cry_gate = RYGate(param).control(1)
        # below: we do some weakref magic... this should NOT be the final code version

        """ #try with control envirnment
        if isinstance(qv,list):
            qc = qv[0].qs()
        elif isinstance(qv,QuantumVariable):
            qc = qv.qs
        else:
            raise Exception("we dont know what is going on here") """

        if index == highIndex:
            
            cx(qv[highIndex - 2], qv[highIndex-1]) 
            with control( qv[highIndex-1] ):
                ry(param, qv[highIndex - 2])
            cx(qv[highIndex - 2], qv[highIndex -1])
            """ cx(qv[highIndex - 2], qv[highIndex-1]) 
                qc.append(cry_gate, [ qv[highIndex-1], qv[highIndex - 2]])
                cx(qv[highIndex - 2], qv[highIndex -1]) """ 
        else: 
            
                #with control(qv[index -1], "1"):
            cx(qv[index -2], qv[highIndex-1]) 
            with control([qv[highIndex -1],qv[index -1]]):
                ry(param, qv[index - 2])
            cx(qv[index -2], qv[highIndex-1]) 
            """ cx(qv[index -2], qv[highIndex-1]) 
                qc.append(mcry_gate, [ qv[highIndex -1], qv[index -1], qv[index-2]])
                cx(qv[index -2], qv[highIndex-1])  """


def inv_prepare(qv, k):
    with invert():
        dicke_state(qv, k)


#formulate on q_array
def portfolio_mixer():

    def apply_mixer(q_array, beta):
        half = int(len(q_array[0])/2)
        qv1 = q_array[0]
        qv2 = q_array[1]
        #print("mixer")
        #print(len(qv))

        with conjugate(inv_prepare)(qv1, half):
            # mehrere mcp-gates, as hamiltonian
            mcp(beta, qv1, ctrl_state = "0001")
            mcp(beta, qv1, ctrl_state = "0011")
            mcp(beta, qv1, ctrl_state = "0111")
            mcp(beta, qv1, ctrl_state = "1111")

        with conjugate(inv_prepare)(qv2, half):
            mcp(beta, qv2, ctrl_state = "0001")
            mcp(beta, qv2, ctrl_state = "0011")
            mcp(beta, qv2, ctrl_state = "0111")
            mcp(beta, qv2, ctrl_state = "1111")
        
        #return qv

    return apply_mixer









"""#formulate on q_array
def portfolio_mixer():

    def apply_mixer(qv, beta):
        half = int(len(qv)/2)
        #print("mixer")
        #print(len(qv))

        with conjugate(inv_prepare)(qv[:half], half):
            # mehrere mcp-gates, as hamiltonian
            mcp(beta, qv[:half], ctrl_state = "0001")
            mcp(beta, qv[:half], ctrl_state = "0011")
            mcp(beta, qv[:half], ctrl_state = "0111")
            mcp(beta, qv[:half], ctrl_state = "1111")

        with conjugate(inv_prepare)(qv[half:], half):
            mcp(beta, qv[half:], ctrl_state = "0001")
            mcp(beta, qv[half:], ctrl_state = "0011")
            mcp(beta, qv[half:], ctrl_state = "0111")
            mcp(beta, qv[half:], ctrl_state = "1111")
        
        #return qv

    return apply_mixer


"""






