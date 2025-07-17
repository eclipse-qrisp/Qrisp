"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
********************************************************************************
"""

from jax.lax import cond
from jax.extend.core import ClosedJaxpr
import jax


from qrisp.environments import QuantumEnvironment
from qrisp.jasp import extract_invalues, insert_outvalues, check_for_tracing_mode, get_last_equation


class ClControlEnvironment(QuantumEnvironment):
    r"""
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
        from qrisp.jasp import make_jaspr
        def test_f(i):

            a = QuantumFloat(3)
            a[:] = i
            b = measure(a)

            with control(b == 4):
                x(a[0])

            return measure(a)

        jaspr = make_jaspr(test_f)(1)

    This jaspr receives an integer and encodes that integer into the :ref:`QuantumFloat`
    `a`. Subsequently `a` is measured and an X gate is applied onto the 0-th
    qubit of `a` if the measurement value is 4.

    We can now evaluate the jaspr on several inputs

    >>> jaspr(1)
    1
    >>> jaspr(2)
    2
    >>> jaspr(3)
    3
    >>> jaspr(4)
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

        jaspr = make_jaspr(test_f)(1)

    This script creates a ``QuantumFloat`` `c` within the classical control
    environment and subsequently uses `c` outside of the environment (in the
    return statement).

    It is however possible to create (quantum-)values within the environment
    and use them still within the environment:

    ::

        from qrisp import *
        from qrisp.jasp import make_jaspr
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

        jaspr = make_jaspr(test_f)(1)


    This script allocates another :ref:`QuantumFloat` `c` within the ClControlEnvironment
    and applies an Hadamard gate to the 0-th qubit. Subsequently the whole
    ``QuantumFloat`` is measured. If the measurement turns out to be one,
    the zeroth qubit of `a` is flipped (similar to the above examples) and
    furthermore `c` is brought back to the $\ket{0}$ state.

    >>> jaspr(4)
    5
    >>> jaspr(4)
    4




    """

    def __init__(self, ctrl_bls, ctrl_state=-1, invert=False):

        if not isinstance(ctrl_bls, list):
            ctrl_bls = [ctrl_bls]

        self.ctrl_bls = ctrl_bls

        QuantumEnvironment.__init__(self, env_args=ctrl_bls)

        # Process the ctrl_state
        self.ctrl_state = ctrl_state

        # If the ctrl state is a string, convert into an integer
        if isinstance(self.ctrl_state, str):
            if ctrl_state == len(ctrl_bls) * "1":
                self.ctrl_state = -1
            else:
                self.ctrl_state = int(self.ctrl_state, 2)

        self.ctrl_state = self.ctrl_state % (2 ** len(self.ctrl_bls))

        self.invert = invert

    def compile(self):
        for i in range(len(self.ctrl_bls)):
            if self.ctrl_bls[i] != bool((self.ctrl_state >> i) & 1):
                break
        else:
            QuantumEnvironment.compile(self)

    def __exit__(self, exception_type, exception_value, traceback):

        # "with" blocks in Python are always executed - the ClControlEnvironment
        # is no exception. It can therefore happen that code is run, which is not
        # intended to run by the semantics, because the classical condition is not
        # True. While this can not really prevented, it can lead to the
        # (annoying) behavior, that Exceptions are raised (for instance
        # out of bounds errors) from code that is not intended to be executed anyways.

        # If there was an exception raised, we mitigate this problem
        # by checking if the control condition was not True in the first place.
        # If this is the case, we simply exit the QuantumEnvironment without
        # any further action.
        static_error_appeared = False
        if not check_for_tracing_mode():
            if exception_type is not None:
                for i in range(len(self.ctrl_bls)):
                    ctrl_bl = self.ctrl_bls[i]
                    if (ctrl_bl ^ (self.ctrl_state >> i)) & 1:
                        self.env_qs.data = []
                        static_error_appeared = True
                        exception_type = None
                        exception_value = None
                        traceback = None
                        break

        QuantumEnvironment.__exit__(self, exception_type, exception_value, traceback)

        if static_error_appeared:
            return True

    def jcompile(self, eqn, context_dic):

        args = extract_invalues(eqn, context_dic)

        # This list stores the the variables representing the control variables
        ctrl_vars = args[: len(self.ctrl_bls)]

        # This list stores the variables used in the environment body
        env_vars = args[len(self.ctrl_bls) :]

        # Flatten the environments in the body
        body_jaspr = eqn.params["jaspr"].flatten_environments()

        if len(body_jaspr.outvars) > 1:
            raise Exception("Found ClControlEnvironment with carry value")

        # Compute the control bool
        tmp = ctrl_vars[0]
        cond_bl = tmp

        # Process the control state requirement
        if self.ctrl_state != -1 and ((self.ctrl_state & 1) == 0):
            cond_bl = ~tmp

        # If there is more than one control variable, loop through
        if len(ctrl_vars) > 1:

            for i in range(1, len(ctrl_vars)):
                tmp = ctrl_vars[i]
                if self.ctrl_state != -1 and ((self.ctrl_state & 1 << i) == 0):
                    tmp = ~tmp

                cond_bl = cond_bl & tmp

        if self.invert:
            cond_bl = ~cond_bl

        @jax.jit       
        def identity_fun(*args):
            return args[-1]
        
        true_fun = identity_fun
        false_fun = identity_fun
        
        res_abs_qc = cond(cond_bl, true_fun, false_fun, *env_vars)
        
        insert_outvalues(eqn, context_dic, [res_abs_qc])
        
        traced_eqn = get_last_equation()
        
        branch_0 = traced_eqn.params["branches"][0]
        branch_0.jaxpr.eqns.pop(0)
        branch_0.jaxpr.outvars[-1] = branch_0.jaxpr.invars[-1]
        
        branch_1 = traced_eqn.params["branches"][1]
        
        from qrisp.jasp import Jaspr
        
        traced_eqn.params["branches"] = (jax.extend.core.ClosedJaxpr(Jaspr.from_cache(branch_0.jaxpr),
                                                                     branch_0.consts),
                                         jax.extend.core.ClosedJaxpr(body_jaspr,
                                                                     branch_1.consts))
        
        