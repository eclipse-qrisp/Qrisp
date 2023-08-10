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


from qrisp.core.quantum_variable import QuantumVariable
from qrisp.core.session_merging_tools import merge
from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.environments.quantum_inversion import invert
from qrisp.misc import lock, unlock


def temp_qv(function):
    def q_env_generator(*args, **kwargs):
        return ConjugationEnvironment(function, args, kwargs=kwargs)

    return q_env_generator


class CConjugationEnvironment(QuantumEnvironment):
    """
    This environment allows to temporarily compute a QuantumVariable which is available
    only inside this environment and automatically uncomputed once it is left.

    Similarly to the ConditionEnvironment, functions can be quickly transformed into
    this QuantumEnvironment using the temp_qv decorator.

    Parameters
    ----------
    qv_retrieval_function : function
        A function returning a QuantumVariable. It is not neccessary to uncompute
        intermediate quantum results of this function.
    args : list
        The list of arguments on which to call qv_retrieval_function.
    kwargs : dict, optional
        A dictionary of keyword arguments on which to call qv_retrieval_function.
        The default is {}.

    Examples
    --------

    We create a QuantumFloat, and temporary compute the tripled value. We then evaluate
    the condition of the tripled value being less than 6: ::

        from qrisp import temp_qv, QuantumFloat, QuantumBool, multi_measurement

        qf = QuantumFloat(5)
        q_bool = QuantumBool()

        qf[:] = {1 : 1, 2 : 1}

        @temp_qv
        def triple(qf):
            return 3*qf

        with triple(qf) as tripled_qf:

            with tripled_qf < 6:
                q_bool.flip()


    >>> print(multi_measurement([qf, q_bool]))
    {(1, True): 0.5, (2, False): 0.5}
    >>> print(tripled_qf)
    Exception: Tried to get measurement from deleted QuantumVariable


    """

    # Constructor of the class

    def __init__(self, qv_retrieval_function, args, kwargs={}):
        # Which
        self.qv_retrieval_function = qv_retrieval_function

        # Save the arguments on which the function should be evaluated
        self.args = args

        # Save the keyword arguments
        self.kwargs = kwargs

        # Note the QuantumSession of the arguments of the arguments
        self.arg_qs = merge(args)

        self.manual_allocation_management = True

    # Method to enter the environment
    def __enter__(self):
        self.user_qv = self.qv_retrieval_function(*self.args, **self.kwargs)

        super().__enter__()

        if not isinstance(self.user_qv, QuantumVariable):
            raise Exception("Retrieval function did not return a QuantumVariable")

        from qrisp import recursive_qv_search

        lock(recursive_qv_search(self.args))
        unlock(self.user_qv.reg)
        # perm_lock(self.user_qv.reg)

        merge(self.arg_qs, self.env_qs)

        return self.user_qv

    def __exit__(self, exception_type, exception_value, traceback):
        from qrisp import recursive_qv_search, redirect_qfunction

        unlock(recursive_qv_search(self.args))
        # perm_unlock(self.user_qv.reg)

        QuantumEnvironment.__exit__(self, exception_type, exception_value, traceback)

        redirected_function = redirect_qfunction(self.qv_retrieval_function)

        with invert():
            redirected_function(*self.args, target=self.user_qv, **self.kwargs)
