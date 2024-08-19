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



from qrisp.qaoa.parameters.fourier_params import fourier_params_helper
def fourier_optimization_wrapper(array_uk_vk, qc, symbols, qarg_dupl , cl_cost_function, fourier_depth, index_p ,mes_kwargs):
        """
        Wrapper function for the optimization method used in QAOA, based on the Fourier heuristic.

        This function calculates the value of the classical cost function after post-processing if a post-processing function is set, otherwise it calculates the value of the classical cost function.

        Parameters
        ----------
        array_uk_vk : list
            The parameters to be optimized
        qc : QuantumCircuit
            The compiled quantum circuit.
        symbols : list
            The list of symbols used in the quantum circuit.
        qarg_dupl : QuantumVariable
            The duplicated quantum variable to which the quantum circuit is applied.
        mes_kwargs : dict
            The keyword arguments for the measurement function.

        Returns
        -------
        float
            The value of the classical cost function.
        """

        theta = fourier_params_helper(array_uk_vk, fourier_depth ,index_p)

        subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}
        res_dic = qarg_dupl.get_measurement(subs_dic = subs_dic, precompiled_qc = qc, **mes_kwargs)
        cl_cost = cl_cost_function(res_dic)
        

        return cl_cost