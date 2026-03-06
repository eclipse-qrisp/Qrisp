"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from qrisp.jasp import make_jaspr


def jasp_function_test(func):

    def testing_function(*args):

        qv = func(*args)
        from qrisp.core import QuantumVariable

        qv.__class__ = QuantumVariable

        old_counts_dic = qv.get_measurement()

        jaspr = make_jaspr(func)(*args)

        qv_qubits, qc = jaspr.to_qc(*args)

        clbit_list = []
        for qb in qv_qubits:
            clbit_list.append(qc.measure(qb))

        counts = qc.run(shots=None)

        # Remove other measurements outcomes from counts dic
        new_counts_dic = {}
        for key in counts.keys():
            # Remove possible whitespaces
            new_key = key.replace(" ", "")
            # Remove other measurements
            new_key = new_key[: len(clbit_list)][::-1]

            # new_key = int(new_key, base=2)
            try:
                new_counts_dic[new_key] += counts[key]
            except KeyError:
                new_counts_dic[new_key] = counts[key]

        for k in old_counts_dic.keys():
            if abs(old_counts_dic[k] - new_counts_dic[k]) > 1e-4:
                return False
        return True

    return testing_function
