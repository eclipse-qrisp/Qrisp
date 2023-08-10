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


# TO-DO implement all the concepts discussed in
# https://stackoverflow.com/questions/30081275/why-is-1000000000000000-in-range1000000000000001-so-fast-in-python-3?rq=1

from qrisp import control, cx, perm_lock, perm_unlock, x
from qrisp.core import QuantumVariable
from qrisp.environments.quantum_conditionals import (
    ConditionEnvironment,
    quantum_condition,
)
from qrisp.logic_synthesis.truth_tables import TruthTable


class QuantumIterator:
    def __init__(self, quantum_condition, init_index=None):
        self.quantum_condition = quantum_condition

    def __iter__(self):
        self.quantum_condition.start_dumping()

        return self

    def __next__(self):
        raise Exception(
            "__next__ method for abstract quantum iterator class not defined"
        )


class qRange:
    def __init__(self, max_index_qf, create_index_qf=False):
        from qrisp.qtypes.quantum_float import QuantumFloat

        if not isinstance(max_index_qf, QuantumFloat):
            raise Exception("Can only create quantum iterators from quantum variables")

        if max_index_qf.exponent != 0:
            raise Exception(
                "qRange can only be intialized with integer quantum indices"
            )

        self.create_index_qf = create_index_qf
        self.max_index_qf = max_index_qf

    def __iter__(self):
        if self.create_index_qf:
            self.index_qf = self.max_index_qf.duplicate(
                qs=self.max_index_qf.qs, name="index_qf"
            )
            self.index_qf.init_from(self.max_index_qf)

        else:
            self.index_qf = self.max_index_qf

        if not self.index_qf.signed:
            self.index_qf.extend(1, position=-1)

        x(self.index_qf)
        self.index_qf += 1

        self.quantum_condition_env = control(self.index_qf[-1])

        self.c_index = -1

        perm_lock(self.index_qf)

        return self

    def __next__(self):
        perm_unlock(self.index_qf)
        self.c_index += 1

        if self.c_index != 0:
            self.quantum_condition_env.__exit__(None, None, None)

            if self.c_index >= 2 ** (self.index_qf.size - 1):
                if self.create_index_qf:
                    x(self.index_qf)
                    self.index_qf += self.c_index - 1

                    if not self.max_index_qf.signed:
                        cx(self.max_index_qf, self.index_qf[:-1])
                    else:
                        cx(self.max_index_qf, self.index_qf)
                    self.index_qf.delete()
                else:
                    x(self.index_qf)
                    self.index_qf += self.c_index - 1

                    self.index_qf.reduce(self.index_qf[-1])

                perm_unlock(self.index_qf)
                raise StopIteration

            self.index_qf += 1
        # @quantum_condition
        # def neg_condition(sign_qubit):
        #     return sign_qubit

        # self.quantum_condition_env = neg_condition(self.index_qf[-1])

        self.quantum_condition_env.__enter__()
        perm_lock(self.index_qf)
        return int(self.c_index)
