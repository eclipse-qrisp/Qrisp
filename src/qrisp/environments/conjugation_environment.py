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


from qrisp.environments import QuantumEnvironment, control
from qrisp.environments.custom_control_environment import custom_control
from qrisp.circuit import Operation, ControlledOperation
from qrisp.core.session_merging_tools import recursive_qs_search, merge
from qrisp.misc import get_depth_dic

class ConjugationEnvironment(QuantumEnvironment):
    r"""
    This :ref:`QuantumEnvironment<QuantumEnvironment>` can be used for perfoming conjugated operations.
    An arbitrary unitary :math:`U \in SU(2^n)` can be conjugated by another unitary
    :math:`V \in SU(2^n)`:
        
    .. math::
        
        \text{conj}(U,V) = V^\dagger U V


    This structure appears in many quantum algorithms such as `Grover <https://arxiv.org/abs/quant-ph/9605043>`_,
    `Quantum backtracking <https://arxiv.org/abs/1509.02374>`_ or
    `Fourier arithmetic <https://arxiv.org/abs/quant-ph/0008033>`_.
    
    Using the ``ConjugationEnvironment`` not only helps to structure the code, 
    but can also grant performance advantages.
    
    This is because the controlled circuit of such a conjugation is can be
    realized by just controlling :math:`U` instead of all three operations.
    
    .. math::
        
        C\text{conj}(U,V) = V^\dagger CU V

    
    The ``ConjugationEnvironment`` can be called using the alias ``conjugate``.
    Conjugate takes the conjugation function (in our example :math:`V`) and returns
    a function that takes the arguments for the conjugation function and returns
    the corresponding ``ConjugationEnvironment``.
    For more information consult the examples section.
    
    .. note::
        
        Note that every QuantumVariable that is created by the conjugation
        function :math:`V` must be deleted/uncomputed before function conclusion.
    
    Parameters
    ----------

    conjugation_function : function
        The function performing the operation :math:`V`.
    args : iterable
        The arguments for the conjugation function.
    kwargs : dict
        The keyword arguments for the conjugation function.
        
    Examples
    --------
    
    We perform Fourier addition on a :ref:`QuantumFloat`
    
    ::
        
        from qrisp import conjugate, QuantumFloat, p, QFT
        
        def fourier_adder(qf, n):
            
            with conjugate(QFT)(qf):
                
                for i in range(qf.size):
                    p(n*np.pi*2**(i-qf.size+1), qf[i])

    >>> qf = QuantumFloat(5)
    >>> fourier_adder(qf, 3)
    >>> print(qf)
    {3: 1.0}
    >>> fourier_adder(qf, 2)
    {5: 1.0}
    
    Investigate the effects of a controlled addition:
        
    ::
        
        from qrisp import control
        
        ctrl = QuantumFloat(1)
        qf = QuantumFloat(5)
        
        with control(ctrl):
            fourier_adder(qf, 3)
    

    To see that indeed only the conjugand has been controlled we take a look
    at the circuit:
        
    >>> print(qf.qs.transpile(1))
    
    ::
    
        ctrl.0: ─────────■──────────■─────────■─────────■─────────■─────────────────
                ┌──────┐ │P(3π/16)  │         │         │         │      ┌─────────┐
          qf.0: ┤0     ├─■──────────┼─────────┼─────────┼─────────┼──────┤0        ├
                │      │            │P(3π/8)  │         │         │      │         │
          qf.1: ┤1     ├────────────■─────────┼─────────┼─────────┼──────┤1        ├
                │      │                      │P(3π/4)  │         │      │         │
          qf.2: ┤2 QFT ├──────────────────────■─────────┼─────────┼──────┤2 QFT_dg ├
                │      │                                │P(3π/2)  │      │         │
          qf.3: ┤3     ├────────────────────────────────■─────────┼──────┤3        ├
                │      │                                          │P(3π) │         │
          qf.4: ┤4     ├──────────────────────────────────────────■──────┤4        ├
                └──────┘                                                 └─────────┘


    """
    
    
    def __init__(self, conjugation_function, args, kwargs):
        
        self.conjugation_function = conjugation_function
        
        self.args = args
        
        self.kwargs = kwargs
        
        self.manual_allocation_management = True
        
        QuantumEnvironment.__init__(self)
        
    def __enter__(self):
        
        # merge(recursive_qs_search(self.args) + [self.env_qs])
        QuantumEnvironment.__enter__(self)
        
        merge(recursive_qs_search(self.args) + [self.env_qs])
        
        
        qv_set_before = set(self.env_qs.qv_list)        
        res = self.conjugation_function(*self.args, **self.kwargs)
        
        temp_data = list(self.env_qs.data)
        self.env_qs.data = []
        i = 0

        while temp_data:
            instr = temp_data.pop(i)
            if isinstance(instr, QuantumEnvironment):
                instr.compile()
            else:
                self.env_qs.append(instr)
        
        if qv_set_before != set(self.env_qs.qv_list):
            raise Exception(f"Tried to create/destroy QuantumVariables {qv_set_before.symmetric_difference(set(self.env_qs.qv_list))} within a conjugation")
        
        self.conjugation_circ = self.env_qs.copy()
        
        self.env_qs.data = []
        
        return res
    
    def __exit__(self, exception_type, exception_value, traceback):
        
        self.perform_conjugation()
        
        QuantumEnvironment.__exit__(self, exception_type, exception_value, traceback)
        
    
    @custom_control
    def perform_conjugation(self, ctrl = None, ctrl_method = None):
        
        temp = list(self.env_qs.data)
        self.env_qs.data = []
        
        for instr in self.conjugation_circ.data:
            self.env_qs.append(instr)
            
        # self.env_qs.data.extend(self.conjugation_circ.data)
        
        if ctrl is not None:
            with control(ctrl, ctrl_method = ctrl_method):
                self.env_qs.data.extend(temp)
        else:
            
            self.env_qs.data.extend(temp)
        
        for instr in self.conjugation_circ.inverse().data:
            self.env_qs.append(instr)
        
    
    def compile_(self, ctrl = None):
        
        temp = list(self.env_qs.data)
        self.env_qs.data = []
        
        for instr in self.conjugation_circ.data:
            if isinstance(instr, QuantumEnvironment):
                instr.compile()
            else:
                self.env_qs.append(instr)
                
        self.conjugation_circ = self.env_qs.copy()
        self.env_qs.data = []
        
        QuantumEnvironment.compile(self)
        
        content_circ = self.env_qs.copy()
        self.conjugation_circ.qubits = list(content_circ.qubits)
        
        conjugation_depth_dic = get_depth_dic(self.conjugation_circ)
        content_depth_dic = get_depth_dic(content_circ)
        
        added_depth_dic = {qb : conjugation_depth_dic[qb] + content_depth_dic[qb] for qb in content_circ.qubits}
        
        instruction_qubits = []
        
        i = 0
        while i < len(content_circ.qubits):
            
            qb = content_circ.qubits[i]
            
            if added_depth_dic[qb]:
                instruction_qubits.append(qb)
                i += 1
            else:
                content_circ.qubits.pop(i)
                self.conjugation_circ.qubits.pop(i)
        
        self.env_qs.data = temp
        
        conj_op = ConjugatedOperation(self.conjugation_circ, content_circ)
        
        alloc_instr = [instr for instr in self.conjugation_circ.data + content_circ.data if instr.op.name == "qb_alloc"]
        
        for instr in alloc_instr:
            self.env_qs.append(instr)
        
        self.env_qs.append(conj_op, content_circ.qubits)
        
        dealloc_instr = [instr for instr in self.conjugation_circ.data + content_circ.data if instr.op.name == "qb_dealloc"]
        
        for instr in dealloc_instr:
            self.env_qs.append(instr)
        
        
class ConjugatedOperation(Operation):
    
    def __init__(self, conjugation_circ, content_circ):
        
        
        
        self.conjugation_gate = conjugation_circ.to_gate(name = "conjugator")
        self.content_gate = content_circ.to_gate(name = "conjugand")
        
        
        definition = conjugation_circ.clearcopy()
        
        definition.append(self.conjugation_gate, definition.qubits)
        definition.append(self.content_gate, definition.qubits)
        definition.append(self.conjugation_gate.inverse(), definition.qubits)
        
        Operation.__init__(self, 
                           name = "conjugation_env", 
                           definition = definition,
                           num_qubits = definition.num_qubits())
        
    def control(self, num_ctrl_qubits=1, ctrl_state=-1, method=None):
        
        controlled_conjugand = self.content_gate.control(num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state, method=None)
        
        res = type(controlled_conjugand)(self, num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state, method=method)
        
        res.definition.data = []
        
        res.definition.append(self.conjugation_gate, self.definition.qubits)
        res.definition.append(controlled_conjugand, res.definition.qubits[:num_ctrl_qubits] + self.definition.qubits)
        res.definition.append(self.conjugation_gate.inverse(), self.definition.qubits)
        
        return res
    
    def inverse(self):
        return ConjugatedOperation(self.conjugation_gate.definition, self.content_gate.inverse().definition)
        
        
        
def conjugate(conjugation_function):

     def conjugation_env_creator(*args, **kwargs):
         
         return ConjugationEnvironment(conjugation_function, args, kwargs)
     
     return conjugation_env_creator
        
        
            
    
        