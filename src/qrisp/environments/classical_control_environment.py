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

from jax.lax import cond

from qrisp.environments import QuantumEnvironment
from qrisp.jisp import extract_invalues, insert_outvalues



class ClControlEnvironment(QuantumEnvironment):
    r"""
    .. _ClControlEnvironment:
    
    The ``ClControlEnvironment`` enables execution of quantum code conditioned on
    classical values. The environment works with similar semantics as the 
    :ref:`ControlEnvironment`, implying this environment can also be entered
    using the ``control`` keyword.
    
    .. warning::
        
        Contrary to the :ref:`ControlEnvironment` the ``ClControlEnvironment`` must
        not have "carry values". This means that no value that is created inside this
        environment may be used outside of the environment.
        
    Examples
    ========
    
    We condition a quantum computation on the outcome of a previous measurement.
    
    ::
        
        from qrisp import *
        from qrisp.jisp import make_jispr
        def test_f(i):
            
            a = QuantumFloat(3)
            a[:] = i
            b = measure(a)
            
            with control(b == 4):
                x(a[0])
            
            return measure(a)

        jispr = make_jispr(test_f)(1)
    
    This jispr receives an integer and encodes that integer into the :ref:`QuantumFloat`
    `a`. Subsequently `a` is measured and an X gate is applied onto the 0-th
    qubit of `a` if the measurement value is 4.
    
    We can now evaluate the jispr on several inputs
    
    >>> jispr(1)
    1
    >>> jispr(2)
    2
    >>> jispr(3)
    3
    >>> jispr(4)
    5
    
    We see that in the case where 4 was encoded, the X gate was indeed executed.
    
    To elaborate the restriction of carry values, we give an example that would
    be illegal:
        
    ::
        
        def test_f(i):
            
            a = QuantumFloat(3)
            a[:] = i
            b = measure(a)
            
            with control(b == 4):
                c = QuantumFloat(2)
            
            return measure(c)

        jispr = make_jispr(test_f)(1)
    
    This script creates a ``QuantumFloat`` `c` within the classical control
    environment and subsequently uses `c` outside of the environment (in the
    return statement).
    
    It is however possible to create (quantum-)values within the environment
    and use them still within the environment:
        
    ::
        
        from qrisp import *
        from qrisp.jisp import make_jispr
        def test_f(i):
            
            a = QuantumFloat(3)
            a[:] = i
            b = measure(a)
            
            with control(b == 4):
                c = QuantumFloat(2)
                h(c[0])
                d = measure(c)
                
                # If c is measured to 1
                # flip a and uncompute c
                with control(d == 1):
                    x(a[0])
                    x(c[0])
                
                c.delete()
            
            return measure(a)
        
        jispr = make_jispr(test_f)(1)
    
    
    This script allocates another :ref:`QuantumFloat` `c` within the ClControlEnvironment
    and applies an Hadamard gate to the 0-th qubit. Subsequently the whole
    ``QuantumFloat`` is measured. If the measurement turns out to be one,
    the zeroth qubit of `a` is flipped (similar to the above examples) and
    furthermore `c` is brought back to the $\ket{0}$ state.
    
    >>> jispr(4)
    5
    >>> jispr(4)
    4
    
    
        
    
    """
    
    def __init__(self, ctrl_bls, ctrl_state=-1, invert = False):
        
        if not isinstance(ctrl_bls, list):
            ctrl_bls = [ctrl_bls]
        
        self.ctrl_bls = ctrl_bls
        
        QuantumEnvironment.__init__(self)
        
        self.env_args = ctrl_bls
    
    def jcompile(self, eqn, context_dic):
        
        args = extract_invalues(eqn, context_dic)
        
        ctrl_vars = args[1:len(self.ctrl_bls)+1]
        env_vars = [args[0]] + args[len(self.ctrl_bls)+1:]
        
        body_jispr = eqn.params["jispr"]
        
        if len(body_jispr.outvars) > 1:
            raise Exception("Found ClControlEnvironment with carry value")
        
        if len(ctrl_vars) > 1:
            cond_bl = ctrl_vars[0]
            for i in range(1, len(ctrl_vars)):
                cond_bl = cond_bl & ctrl_vars[i]
        else:
            cond_bl = ctrl_vars[0]
        
        def false_fun(*args):
            return args[0]
        
        flattened_body_jispr = body_jispr.flatten_environments()
        res_abs_qc = cond(cond_bl, flattened_body_jispr.eval, false_fun, *env_vars)
        
        insert_outvalues(eqn, context_dic, res_abs_qc)