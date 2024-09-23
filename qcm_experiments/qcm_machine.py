"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

from qrisp import *


operations = ["nop", 
                #"h",
                #"x",
                "swap",
                #"get",
                "add",
                #"double",
                #"mul",
                #"square",
                "jmp",
                #"jz",
                #"jmp*"
                ]

class OperationQuantumVariable(QuantumVariable):
    
    def __init__(self):
        
        QuantumVariable.__init__(self, int(np.ceil(np.log2(len(operations)))))
    
    def decoder(self, i):
        
        return operations[i]


class QCMProgram:
    
    def __init__(self, code, init_state, default_wordsize = 3):
        """
        Initializes a QCMProgram instance

        Parameters
        ----------
        code : str
            The QCM code describing the script. All variables have to be named r{number} 
            (ie. r0, r1, r2..)
        init_state : list
            A list of integers describing the initial state of the r values.
        default_wordsize : int, optional
            The default word size of the machine. The default is 3.

        Returns
        -------
        None.

        """
        
        n = len(init_state)
        
        # Initiate the QuantumArray holding the r values
        self.r = QuantumArray(qtype = QuantumFloat(default_wordsize))
        self.r[:] = init_state
        
        # The stage is the sub-array, where values are moved to be processed
        # Here we use the first and second entry (any other choice would be possible)
        self.stage = self.r[:2]
        
        # Initiate the branch control QuantumFloat
        self.rb = QuantumFloat(default_wordsize)        
        self.rb[:] = 1
        
        # The idea is now to represent the program as three QuantumArrays:
        #    1. The array describing which operation to apply
        #    2. The array describing whether the operation is inverted
        #    3. The array of first arguments
        #    4. The array of second arguments
        
        self.operations_array = QuantumArray(qtype = OperationQuantumVariable())
        self.inversions_array = QuantumArray(qtype = QuantumBool())
        
        arg_index_qf = QuantumFloat(int(np.ceil(np.log2(n))))        
        self.qargs_0_array = QuantumArray(arg_index_qf)
        self.qargs_1_array = QuantumArray(arg_index_qf)
        
        # Together they form the "tape". The tape can be moved forwards and backwards
        # by self.rb steps using the move_tape method.
        
        # Thereby the current instruction always sits at the 0 index of the tape.
        
        # We now parse the code and initiate the tape.
        instructions = qcm_code.splitlines()
        
        # Remove the potentially empty first line
        if len(instructions[0]) == 0:
            instructions.pop(0)
        
        # For this we first extract the classical values that the tape in it's initial
        # state has and subsequently write these values
        program_operations = []
        inversions = []
        q_args_0 = []
        q_args_1 = []
        
        for i in range(len(instructions)):
            
            # Split the instruction into a list
            instructions[i] = instructions[i].split(" ")
            
            operation = instructions[i][0]
            
            # Write down the reversal state
            if operation[0] == "r":
                inversions.append(True)
                operation = operation[1:]
            else:
                inversions.append(False)
            
            # Write down which operation to apply
            program_operations.append(operation)
            
            # We now identify the indices of the arguments
            if operation in ["nop", "jmp"]:
                q_args_0.append(0)
                q_args_1.append(1)
                continue
            
            if operation == "jz":
                q_args_0.append(int(instructions[i][2][1:]))
                q_args_1.append(1)
                continue
            
            if operation in ["U", "jmp*"]:
                q_args_0.append(int(instructions[i][1][1:]))
                q_args_1.append(1)
                continue
            
            q_args_0.append(int(instructions[i][1][1:]))
            
            if len(instructions[i]) > 2:
                arg_1 = int(instructions[i][2][1:])
                if q_args_0[-1] - arg_1 < 0:
                    q_args_1.append((n + q_args_0[-1] - arg_1)%(n-1))
                else:
                    q_args_1.append((-q_args_0[-1] + arg_1)%(n-1))
            else:
                q_args_1.append(0)
            
            
        # Initiate the QuantumArrays
        self.operations_array[:] = program_operations        
        self.inversions_array[:] = inversions
        self.qargs_0_array[:] = q_args_0
        self.qargs_1_array[:] = q_args_1
        
        self.instr_reg = self.operations_array[0]
        
    def unload_arguments(self):
        """
        Moves the content of the stage back into their original position
        """
        
        cyclic_shift(self.r[1:], shift_amount = self.qargs_1_array[0])
        cyclic_shift(self.r, shift_amount = self.qargs_0_array[0])
    
        
    def load_arguments(self):
        """
        Moves the relevant arguments into the stage

        """
        
        with invert():
            self.unload_arguments()
        
    def move_tape(self):
        """
        Moves the tape forward by self.rb steps.
        """
        
        with invert():
            cyclic_shift(self.operations_array, shift_amount = self.rb)
            cyclic_shift(self.inversions_array, shift_amount = self.rb)
            cyclic_shift(self.qargs_0_array, shift_amount = self.rb)
            cyclic_shift(self.qargs_1_array, shift_amount = self.rb)
    
    def exec_loaded_instruction(self):
        """
        Executes the loaded instruction on the loaded arguments

        """
        
        for op in operations:
            tmp = (op == self.instr_reg)
            with control(tmp):
                execute_operation(op, self.stage, self.rb)
            tmp.uncompute()
                
    def proceed(self):
        """
        Performs the QCM instruction processing procedure
        """
        self.load_arguments()
        self.exec_loaded_instruction()
        self.unload_arguments()
        self.move_tape()
        
        

qcm_code = """
add r1 r2
swap r1 r0
nop
"""

# Initializes the above program with r0 = 0, r1 = 1, r2 = 2
qcm_prog = QCMProgram(qcm_code, init_state = [0,1,2])

# Test the loading mechanism
qcm_prog.load_arguments()
print(qcm_prog.stage)
# {OutcomeArray([1, 2]): 1.0}
# This is expected because argument 0 is r1 = 1 and argument 1 is r2 = 2

qcm_prog.unload_arguments()

print(qcm_prog.stage)
# {OutcomeArray([0, 1]): 1.0}
# Unloading worked.

# We now bring the movement speed rb into superposition
h(qcm_prog.rb[1])
print(qcm_prog.rb)
# {1: 0.5, 3: 0.5}
# The tape will therefore now be moved in superpostion:
# In one case by 1 step
# and in the other case by 3 steps (which brings it to the initial position)

# Move the tape and load the arguments
qcm_prog.move_tape()

qcm_prog.load_arguments()
print(multi_measurement([qcm_prog.instr_reg, qcm_prog.stage]))
# {('swap', OutcomeArray([1, 0])): 0.5, ('add', OutcomeArray([1, 2])): 0.5}
# This is a superposition of the instruction from the first and the second line!

#%%

# Instructions are implemented here

def U(op, ra):
    
    if op == "h":
        h(ra)
    elif op == "x":
        x(ra)
    else:
        raise Exception(f"Don't know operation {op}")
        
from qrisp import swap

def get_bit(ra, rb, rc):
    demux(rc, ra, rb)

def add(ra, rb):
    if ra is rb:
        cyclic_shift(ra)
    else:
        ra += rb

def multiply(ra, rb, verify_uncomputation = False):
    
    res_type = QuantumFloat(ra.size + rb.size)
    
    comp_q_dic = QuantumDictionary(return_type = res_type)
    
    if ra is rb:
        
        for a in [ra.decoder(i) for i in range(2**ra.size)]:
                comp_q_dic[a] = a**2
                
        rres = comp_q_dic[ra]
        
    else:
        
        for a in [ra.decoder(i) for i in range(2**ra.size)]:
            for b in [rb.decoder(i) for i in range(2**rb.size)]:
                comp_q_dic[a, b] = a*b
                
        rres = comp_q_dic[ra, rb]
        
        
        
        
    def uncomputation_wrapper(rres, rb):
        
        uncomp_q_dic = QuantumDictionary(return_type = ra)
        
        for b in [rb.decoder(i) for i in range(2**rb.size)]:
            for res in [rres.decoder(i) for i in range(2**rres.size)]:
                
                uncomp_q_dic[res, b] = 0
                
                if b == 0:
                    continue
                
                if res%b == 0:
                    if res//b < 2**ra.size:
                        
                        if ra is rb:
                            uncomp_q_dic[res] = res//b
                        else:
                            uncomp_q_dic[res, b] = res//b
        
        if ra is rb:
            return uncomp_q_dic[rres]
        else:
            return uncomp_q_dic[rres, rb]
            
    with invert():
        redirect_qfunction(uncomputation_wrapper)(rres, rb, target = ra)

    ra.extend(rb.size)
    
    swap(ra, rres)
    
    rres.delete(verify = verify_uncomputation)

"""    
# Test multiplication
ra = QuantumFloat(3)
rb = QuantumFloat(3)

ra[:] = 3
rb[:] = 2
multiply(ra, rb, verify_uncomputation = True)
print(multi_measurement([ra, rb]))

# Test squaring
ra = QuantumFloat(3)

ra[:] = 3
multiply(ra, ra, verify_uncomputation = True)
print(multi_measurement([ra, rb]))
"""
    
def jump(br, p):
    br += p

def conditional_jump(br, ra, p):
    tmp = ra == 0
    with control(tmp):
        jump(br, p)
    tmp.uncompute()    
        
def indirect_jump(br, ra):
    br += ra
    

def execute_operation(operation, args, br):
    
    if operation == "nop":
        return
    elif operation == "h":
        h(args[0])
    elif operation == "x":
        x(args[0])
    elif operation == "swap":
        swap(args[0], args[1])
    elif operation == "get":
        get_bit(args[0], args[1], args[2])
    elif operation == "add":
        add(args[0], args[1])
    elif operation == "double":
        add(args[0], args[0])
    elif operation == "mul":
        multiply(args[0], args[1])
    elif operation == "jmp":
        jump(br, args[0])
    elif operation == "jz":
        conditional_jump(br, args[0], args[1])
    elif operation == "jmp*":
        indirect_jump(br, args[0])
    else:
        raise Exception(f"Don't know operation {operation}")



#%%

# Test QCM execution

qcm_code = """
add r1 r2
swap r2 r1
add r1 r2
"""

# Initializes the above program with r0 = 0, r1 = 1, r2 = 2
qcm_prog = QCMProgram(qcm_code, init_state = [0,1,2])
print(qcm_prog.r)
# {OutcomeArray([0, 1, 2]): 1.0}

# Start the execution

qcm_prog.proceed()
print(qcm_prog.r)
# {OutcomeArray([0, 3, 2]): 1.0}

qcm_prog.proceed()
print(qcm_prog.r)
# {OutcomeArray([0, 2, 3]): 1.0}

qcm_prog.proceed()
print(qcm_prog.r)
# {OutcomeArray([0, 5, 3]): 1.0}


#%%
qcm_code = """
jmp r0
swap r2 r1
add r1 r2
"""

# Initializes the above program with r0 = |0>+|2>, r1 = 1, r2 = 2
qcm_prog = QCMProgram(qcm_code, init_state = [0,1,2])
r0 = qcm_prog.r[0]
h(r0[1])
print(qcm_prog.r)
# {OutcomeArray([0, 1, 2]): 0.5, OutcomeArray([2, 1, 2]): 0.5}

# Start the execution
qcm_prog.proceed()
print(multi_measurement([qcm_prog.rb, qcm_prog.instr_reg]))
# {(1, 'swap'): 0.5, (3, 'jmp'): 0.5}

qcm_prog.proceed()
print(multi_measurement([qcm_prog.rb, qcm_prog.instr_reg]))
# {(1, 'add'): 0.5, (5, 'add'): 0.5}

qcm_prog.proceed()
print(multi_measurement([qcm_prog.rb, qcm_prog.instr_reg, qcm_prog.r]))
# {(1, 'jmp', OutcomeArray([0, 3, 1])): 0.5, (5, 'swap', OutcomeArray([2, 3, 2])): 0.5}
# The jump doesn't increase rb because r0 = 0

qcm_prog.proceed()
print(multi_measurement([qcm_prog.rb, qcm_prog.instr_reg]))
# {(1, 'swap'): 0.5, (5, 'jmp'): 0.5}
