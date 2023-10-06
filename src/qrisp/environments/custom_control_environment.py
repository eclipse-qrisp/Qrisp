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

from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.environments.gate_wrap_environment import GateWrapEnvironment

def custom_control(func):
    """
    The ``custom_control`` decorator allows to specify the controlled version of 
    the decorated function. If this function is called within a :ref:`ControlEnvironment`
    or a :ref:`ConditionEnvironment` the controlled version is executed instead.
    
    Specific controlled versions of quantum functions are very common in many
    scientific publications. This is because the general control procedure can
    signifcantly increase resource demands.
    
    In order to use the ``custom_control`` decorator, you need to add the ``ctrl``
    keyword to your function signature. If called within a controlled context,
    this keyword will receive the corresponding control qubit.
    
    For more details consult the examples section.
    

    Parameters
    ----------
    func : function
        A function of QuantumVariables, which has the ``ctrl`` keyword.

    Returns
    -------
    adaptive_control_function : function
        A function which will execute it's controlled version, if called
        within a :ref:`ControlEnvironment` or a :ref:`ConditionEnvironment`.
        
    Examples
    --------
    
    We create a swap function with custom control.
    
    ::
        
        from qrisp import mcx, cx, custom_control
        
        @custom_control
        def swap(a, b, ctrl = None):
            
            if ctrl is None:
                
                cx(a, b)
                cx(b, a)
                cx(a, b)
                
            else:
                
                cx(a, b)
                mcx([ctrl, b], a)
                cx(a, b)
                
    
    Test the non-controlled version:
        
    ::
        
        from qrisp import QuantumBool
        
        a = QuantumBool()
        b = QuantumBool()
        
        swap(a, b)
        
        print(a.qs)
        
    
    ::
        
        QuantumCircuit:
        --------------
                  ┌───┐     
        a.0: ──■──┤ X ├──■──
             ┌─┴─┐└─┬─┘┌─┴─┐
        b.0: ┤ X ├──■──┤ X ├
             └───┘     └───┘
        Live QuantumVariables:
        ---------------------
        QuantumBool a
        QuantumBool b        
        
    
    Test the controlled version:
        
    ::
        
        from qrisp import control
        
        a = QuantumBool()
        b = QuantumBool()
        ctrl_qbl = QuantumBool()
        
        with control(ctrl_qbl):
            
            swap(a,b)
            
        print(a.qs.transpile(1))
        
    ::
        
                         ┌───┐     
               a.0: ──■──┤ X ├──■──
                    ┌─┴─┐└─┬─┘┌─┴─┐
               b.0: ┤ X ├──■──┤ X ├
                    └───┘  │  └───┘
        ctrl_qbl.0: ───────■───────
    

    """
    
    def adaptive_control_function(*args, **kwargs):
        
        from qrisp.core import recursive_qs_search
        from qrisp import merge, ControlEnvironment, ConditionEnvironment, QuantumEnvironment, InversionEnvironment, ConjugationEnvironment
        
        qs_list = recursive_qs_search([args, kwargs])
        
        merge(qs_list)
        
        if len(qs_list) == 0:
            return func(*args, **kwargs)
        
        qs = qs_list[0]
        
        control_qb = None
        for env in qs.env_stack[::-1]:
            if isinstance(env, (ControlEnvironment, ConditionEnvironment)):
                control_qb = env.condition_truth_value
                break
            if not isinstance(env, (QuantumEnvironment, InversionEnvironment, ConjugationEnvironment)):
                break

        if control_qb is None:
            return func(*args, **kwargs)
        
        
        with CustomControlEnvironment(control_qb, func.__name__):
            res = func(*args, ctrl = control_qb, **kwargs)
        
        return res
        
    return adaptive_control_function


class CustomControlEnvironment(GateWrapEnvironment):
    
    def __init__(self, control_qb, name):
        
        GateWrapEnvironment.__init__(self, name = name)
        
        self.control_qb = control_qb
    
    def compile(self):
        
        GateWrapEnvironment.compile(self)
        
        if hasattr(self, "instruction"):
            if self.control_qb in self.instruction.qubits:
                self.instruction.op = CustomControlOperation(self.instruction.op, self.instruction.qubits.index(self.control_qb))
        

from qrisp.circuit import Operation
class CustomControlOperation(Operation):
    
    def __init__(self, init_op, control_qubit_index):
        
        Operation.__init__(self, init_op = init_op)
        
        self.name = "c" + self.name
        self.init_op = init_op
        self.control_qubit_index = control_qubit_index
        self.permeability = init_op.permeability
        self.permeability[control_qubit_index] = True
        self.is_qfree = init_op.is_qfree

    def inverse(self):
        temp = self.init_op.inverse()
        return CustomControlOperation(temp, self.control_qubit_index)
    
    def copy(self):
        return CustomControlOperation(self.init_op.copy(), self.control_qubit_index)

