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

from functools import lru_cache

from jax import make_jaxpr
from jax.core import JaxprEqn, ClosedJaxpr
from jax.lax import add_p, sub_p, while_loop

from qrisp.jasp.primitives import AbstractQuantumCircuit, OperationPrimitive
from qrisp.jasp.jasp_expression.centerclass import Jaspr


def copy_jaxpr_eqn(eqn):
    return JaxprEqn(
        primitive=eqn.primitive,
        invars=list(eqn.invars),
        outvars=list(eqn.outvars),
        params=dict(eqn.params),
        source_info=eqn.source_info,
        effects=eqn.effects,
    )


@lru_cache(int(1e5))
def injection_transform(jaspr, qubit_array_outvar):
    """
    This function takes in a Jaspr that returns a QubitArray, which has been
    created in it's body. The function then transforms it to a Jaspr, which
    DOESN'T create this QubitArray but instead receives it as a parameter.
    This functionality is required to realize the redirect_qfunction decorator,
    which turns out-of-place functions into in-place functions.

    This function can't process Jaspr that modify the QubitArray via slicing.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to transform.
    qubit_array_outvar : jax.core.Var
        The QubitArray variable.

    Raises
    ------
    Exception
        Tried to redirect quantum function returning a sliced qubit array

    Returns
    -------
    injected_jaspr : Jaspr
        The transformed jaspr.

    """

    if not qubit_array_outvar in jaspr.outvars:
        raise Exception("Specified ")

    # We will now iterate through the function body to find the equation that
    # created qubit_array_outvar. Essentially there are three primitives, that can
    # produce a QubitArray

    # 1. create_qubits
    # 2. pjit
    # 3. slice

    # For the first case we delete the equation. Since the create qubits
    # primitive also returns a new QuantumCircuit object, we need to replace
    # the invar of the following equation to use the invar of the deleted
    # equation instead.

    # For the second case, we recursively apply this functions and update the
    # equation signature.

    # The third case is illegal and results in a raised Exception.

    # Create a new list to store the new instructions
    new_eqns = []

    # If a create_qubits equation is deleted, this variable will store the
    # invar of the equation, such that it can be replaced later.
    deleted_quantum_circuit_variable = None

    for i in range(len(jaspr.eqns)):

        eqn = jaspr.eqns[i]

        # Delete the equation by skipping the last line of the loop
        if eqn.primitive.name == "jasp.create_qubits":
            if eqn.outvars[0] is qubit_array_outvar:
                deleted_quantum_circuit_variable = eqn.invars[-1]
                continue

        # Recursively apply the injection transform
        elif eqn.primitive.name == "pjit":
            if qubit_array_outvar in eqn.outvars:

                # Retrieve the Jaspr to be transformed
                sub_jaspr = eqn.params["jaxpr"].jaxpr

                # Retrieve the QubitArray to be injected
                sub_qubit_array_outvar = sub_jaspr.outvars[
                    eqn.outvars.index(qubit_array_outvar)
                ]

                # Copy the equation to prevent in-place modification errors
                eqn = copy_jaxpr_eqn(eqn)

                # Modify the copied equation
                eqn.params["jaxpr"] = ClosedJaxpr(
                    injection_transform(sub_jaspr, sub_qubit_array_outvar), []
                )
                eqn.invars.insert(0, qubit_array_outvar)
                eqn.outvars.remove(qubit_array_outvar)

        # Raise exception for the illegal case
        elif eqn.primitive.name == "jasp.slice":
            if eqn.outvars[0] is qubit_array_outvar:
                raise Exception(
                    "Tried to redirect quantum function returning a sliced qubit array"
                )

        # Replace the QuantumCircuit invar
        if not deleted_quantum_circuit_variable is None:
            eqn = copy_jaxpr_eqn(eqn)
            for j in range(len(eqn.invars)):
                invar = eqn.invars[j]
                if isinstance(invar.aval, AbstractQuantumCircuit):
                    eqn.invars[j] = deleted_quantum_circuit_variable
                    deleted_quantum_circuit_variable = None
                    break

        new_eqns.append(eqn)

    # Create a copy to in-place modifications problems
    new_jaspr = jaspr.copy()

    # Update the signature of the new_jaspr
    new_jaspr.outvars.remove(qubit_array_outvar)
    new_jaspr.invars.insert(0, qubit_array_outvar)

    # Update the body
    new_jaspr.eqns.clear()
    new_jaspr.eqns.extend(new_eqns)

    # If the QuantumCircuit invar was never replaced, the QuantumCircuit is
    # is returned by the Jaspr
    if not deleted_quantum_circuit_variable is None:
        new_jaspr.outvars[-1] = deleted_quantum_circuit_variable

    return new_jaspr
