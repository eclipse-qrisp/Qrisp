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

import numpy as np

def fourier_params_helper(array_uk_vk, fourier_depth, index_p ):
    """
    Helper function to instanciate the Optimization params via the Fourier heuristic approach, see `Zhou et al. <https://arxiv.org/pdf/1812.01041.pdf>`_  p.5

    Parameters
    ----------
    array_uk_vk : list
        initial random inits, optimization parameter
    fourier_depth : int
        depth (q) of fourier transformation
    index_p : int
        depth of (iteration of) the circuit
    """
    theta = []
    #for gamma: go through first half
    for index in range(index_p):
            # index = index*depth?
            saverlist = []
            # then go through whatever the necessary uk_s for this gamma
            for sgl_res_index in range(fourier_depth):  
                saverlist.append(array_uk_vk[sgl_res_index] * np.sin( (sgl_res_index + 1 - 1/2)*( index + 1 - 1/2)* np.pi/index_p))                     

            theta.append(sum(saverlist))

    #for beta: go through second half 
    for index2 in range(index_p):
        saverlist2 = []
        # then go through whatever the necessary vk_s for this beta 
        for sgl_res_index2 in range(fourier_depth, 2*fourier_depth):                                                              # have to subtract the depth here again
            saverlist2.append(array_uk_vk[sgl_res_index2] * np.sin( (sgl_res_index2 - fourier_depth + 1 - 1/2)*( index2 + 1 - 1/2)* np.pi/index_p))

        theta.append(sum(saverlist2))

    return theta

