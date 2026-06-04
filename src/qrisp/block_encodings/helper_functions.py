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

from qrisp.core import QuantumArray, QuantumVariable

def _flatten_qargs(qargs: list[QuantumVariable | QuantumArray]) -> list[QuantumVariable]:
    """
    Flattens a list containing QuantumVariable and QuantumArray objects into a 1D list of QuantumVariables.

    This helper function iterates through the input arguments, preserving QuantumVariables 
    and extracting all individual elements from QuantumArrays.

    Parameters
    ----------
    qargs : list
        A list of QuantumVariable or QuantumArray objects.

    Returns
    -------
    list of QuantumVariable
        A flat list containing all underlying QuantumVariable instances.

    Raises
    ------
    TypeError
        If any element in the input list is not a QuantumVariable or a QuantumArray.
    """
    flattened_qargs: list[QuantumVariable] = []

    for arg in qargs:
        if isinstance(arg, QuantumVariable):
            flattened_qargs.append(arg)

        elif isinstance(arg, QuantumArray):
            # QuantumArray.flatten() returns a 1D QuantumArray
            flattened_qargs.extend([qv for qv in arg.flatten()])

        else:
            raise TypeError("Arguments must be of type QuantumVariable or QuantumArray")
        
    return flattened_qargs
