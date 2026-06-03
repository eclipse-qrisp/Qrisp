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

import numpy as np
from functools import partial
from qrisp.operators import QubitOperator
from qrisp.block_encodings.block_encoding_methods.foqcs_lcu.foqcs_preps import (
    foqcs_prep_heisenberg,
    foqcs_prep_spin_glass
    )

_FOQCS_SPIN_GLASS_TOL = 1e-12

def foqcs_analyze_operator(
        O: QubitOperator,
        L: int = -1,
        tol: float = _FOQCS_SPIN_GLASS_TOL,
        raise_errors: bool = True
) -> dict:
    r"""
    Parameters
    ----------
    O : QubitOperator
            Qubit operator of form: O = X(0) + X(1) + 0.5 * Y(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(1)

    L : int = -1
        Number of operand qubits.
        If not specified, will default to -1, and infer the number of operand qubits from the operator

    tol : float = _FOQCS_SPIN_GLASS_TOL (1e-12)
        Tolerance for considering the entry zero

    raise_errors : bool
        Flag determining whether to raise ValueError in the case of operator being incompatible with FOQCS-LCU.

    Returns
    ----------
    dict {
        "method": str ("heisenberg" || "spin_glass")
        "L": int (number of intracting qubits)
        "g": dict ("X": np.arr, "Z": np.arr, "Y": np.arr)
        "J": dict ("X": np.arr, "Z": np.arr, "Y": np.arr) (In case of spin_glass, numpy arrays are 2-dim)
    }

    Raises
    ----------
    ValueError
            When the operator is not compatible with FOQCS-LCU (fails the spin-glass check) and `raise_errors` is set to `True`.
    
    """
    terms = O._to_pauli_coeff_dict()

    # When analysis fails, it either throws a ValueError with the reason for failure,
    # or silently returns None if raise_errors flag is set to False.
    def fail(arg: str):
        if raise_errors:
            raise ValueError(arg)
        else:
            return None

    # Verifies that operator exists and checks its length
    max_ind = -1
    for pauli_str in terms:
        for ind, _ in pauli_str:
            max_ind = max(max_ind, ind)
    if max_ind < 0:
        return fail(f"Received empty or constant operator: {O}")

    # Infers the length or sanity checks the passed length.
    if L == -1:
        L = max_ind + 1
    elif L < max_ind + 1:
        return fail(f"Received L = {L}, while operator acts on {max_ind + 1} qubits.")

    g_dict = {"X": np.zeros(L, dtype="complex"), "Y": np.zeros(L, dtype="complex"), "Z": np.zeros(L, dtype="complex")}
    J_dict = {"X": np.zeros((L, L), dtype="complex"), "Y": np.zeros((L, L), dtype="complex"), "Z": np.zeros((L, L), dtype="complex")}

    # Verify operator against the spin-glass model.
        # Every non-zero term is Xi, Yi, Zi, XiXj, YiYj, ZiZj.
        # Fails if:
            # Has const cI
            # Mixed terms: XiZj, XiYj, YiZj
            # Three-or-more-body interaction: XiXjXk
    for pauli_str, coeff in terms.items():
        # All coeffs in arrays are zeroes, so just skip entry
        if np.isclose(coeff, 0, atol=tol):
            continue

        # Fail on a constant: cI
        if len(pauli_str) == 0:
            return fail(f"FOQCS-LCU does not support constant/identity terms, but received {(pauli_str, coeff)}")

        # Inspect one-body terms Xi, Yi, Zi:
        if len(pauli_str) == 1:
            (i, pauli), = pauli_str
            g_dict[pauli][i] += coeff
            continue

        # Inspect two-body terms XiXj, YiYj, ZiZj:
        if len(pauli_str) == 2:
            (i, pauli_i), (j, pauli_j) = sorted(pauli_str)

            if pauli_i != pauli_j:
                return fail(f"FOQCS-LCU supports only same-axis couplings, but received: {pauli_i}({i}) * {pauli_j}({j})")

            J_dict[pauli_i][i, j] += coeff
            J_dict[pauli_i][j, i] += coeff
            continue

        if len(pauli_str) > 2:
            return fail(f"FOQCS-LCU supports only one and two-body interactions.")

    # Spin-glass has passed by this point. YAY! :D

    # Verify ourselves against more specific Heisenberg-model.
    # Considering the Heisenberg model for quantum spins on a one-dimensional nearest-neighbour
    # chain (https://arxiv.org/abs/2507.20887, Page 8, Paragraph 1)
        # Spin-glass-compatible
        # Must have form: Xi, Yi, Zi, XiXi+1, YiYi+1, ZiZi+1; local fields must be uniform.
        # Fails if:
            # Local field is position dependent: g_0X != g_1X, 0.5*X(0) + 0.7*X(1)
            # Has long-range couplings: X_0*X_2
            # Has NN couplings with different sterngths: 0.5*X(0)*X(1) + 0.9*X(1)*X(2)
            # Zeroes matter, don't pass 0.5*X(0)*X(1) + 0*X(1)*X(2) , so depends on L.
    is_heisenberg = True # Assume it is Heisenberg before the check
    g_heis_dict = {}
    J_heis_dict = {}
 
    # Verify that all one-body (local) coeffs are the same (uniform).
    for pauli in {"X", "Z", "Y"}:
        values = g_dict[pauli][:]

        if not np.allclose(values, values[0], atol=tol):
            is_heisenberg = False
            # print("Heisenberg Rejected: non-uniform local interactions")
            break

        g_heis_dict[pauli] = values[0]
    
    # Verify couplings
    if is_heisenberg:
        for pauli in {"X", "Z", "Y"}:
            if L == 1:
                J_heis_dict[pauli] = 0
                continue
            
            nn_val = np.array(
                [J_dict[pauli][i, i + 1] for i in range(L - 1)],
                dtype = complex,
            )
            # All same-axis nearest-neighbours interactions must be uniform
            if not np.allclose(nn_val, nn_val[0], atol=tol):
                is_heisenberg = False
                # print("Heisenberg Rejected: non-uniform NN interactions")
                break

            J_heis_dict[pauli] = nn_val[0]

            # Reject non-nearest-neighbour couplings
            for i in range(L):
                for j in range(i + 1, L):
                    if j == i + 1:
                        continue
                    if not np.isclose(J_dict[pauli][i, j], 0, atol=tol):
                        is_heisenberg = False
                        # print("Heisenberg Rejected: not NN interactions present")
                        break
                if not is_heisenberg:
                    break
            if not is_heisenberg:
                break

    # Heisenberg check passed, build simpler structure
    if is_heisenberg:
        return {
            "method": "heisenberg",
            "L": L,
            "g": g_heis_dict,
            "J": J_heis_dict,
        }
    # General spin-glass / position-dependent XYZ spin model.
    else:
        return {
            "method": "spin_glass",
            "L": L,
            "g": g_dict,
            "J": J_dict,
        }

def is_operator_foqcs_compatible(
        O: QubitOperator,
        L: int = -1,
        tol: float = _FOQCS_SPIN_GLASS_TOL
) -> tuple: # tuple is of the form -> (bool, dict || None)
    r"""
    Parameters
    ----------
    O : QubitOperator
            Qubit operator of form: O = X(0) + X(1) + 0.5 * Y(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(1)

    L : int = -1
        Number of operand qubits.
        If not specified, will default to -1, and infer the number of operand qubits from the operator

    tol : float = _FOQCS_SPIN_GLASS_TOL (1e-12)
        Tolerance for considering the entry zero

    Returns
    ----------
    tuple : (bool, dict || None)
        bool : True if compatible with FOQCS-LCU
        dict : Output of `foqcs_analyze_operator`

    """
    res = foqcs_analyze_operator(O, L = L, tol = tol, raise_errors = False)
    return (res != None, res)

def build_foqcs_lcu_prep_from_analysis(aresult: dict) -> dict:
    r"""
    Parameters
    ----------
    aresult : dict
            Output of `foqcs_analyze_operator`

    Returns
    ----------
    prep : Callable[[QuantumVariable], None]
        Partial PREP_R function with all relevant parameters passed except QuantumVariable.
    
    num_q_ops : int
        Number of operand qubits (L argument for FOQCS-LCU PREP routines).
        The default is 1.

    unprep : Callable[[QuantumVariable], None] = None
        Complex conjugate transpose of PREP_R with conjugated parameters, 
        also a partial function variable with all relevant parameters passed except QuantumVariable.
        The default is None, in which case the unprep is calculated using the prep parameter.

    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian.
        The default is False.
    
    norm : "ArrayLike"
        Normalization factor.
        The default is 1 in case no normalization factor is passed.

    Raises
    ----------
    KeyError
        If function received an unsupported FOQCS-LCU PREP method
    """
    if aresult["method"] == "heisenberg":
        #print("Parsing as Heisenberg model")
        # Do Heisenberg model preprocessing here
        heis_L = aresult["L"]
        g = aresult["g"]
        J = aresult["J"]

        # Preprocess coefficients into sqrt-amplitude dictionaries.
        paulis = ("X", "Y", "Z")
        _g = {
            "X": np.sqrt(g["X"] * heis_L),
            "Y": np.sqrt(g["Y"] * heis_L * -1j),
            "Z": np.sqrt(g["Z"] * heis_L),
        }
        _J = {
            "X": np.sqrt(J["X"] * (heis_L - 1)),
            "Y": np.sqrt(J["Y"] * -(heis_L - 1)),
            "Z": np.sqrt(J["Z"] * (heis_L - 1)),
        }
        # Normalization for state preparation.
        coeff_vec = np.array(
            [_g[p] for p in paulis] + [_J[p] for p in paulis],
            dtype=complex,
        )
        norm = np.linalg.norm(coeff_vec)

        _g = {
            p: _g[p] / norm
            for p in paulis
        }
        _J = {
            p: _J[p] / norm
            for p in paulis
        }
        # Calculate the norm factor
        alpha = norm**2
        # Create partial PREP_R and PREP_L functions
        prep = partial(
            foqcs_prep_heisenberg,
            L=heis_L,
            g=_g,
            J=_J,
        )
        unprep = partial(
            foqcs_prep_heisenberg,
            L=heis_L,
            g=_g,
            J=_J,
            conjugate=True
        )

        return prep, heis_L, unprep, False, alpha

    elif aresult["method"] == "spin_glass":
        #print("Parsing as Spin-Glass model")

        sg_L = aresult["L"]
        g = aresult["g"]
        J = aresult["J"]

        paulis = ("X", "Y", "Z")

        # Convert matrix-form J from foqcs_analyze_operator into the
        # diagonal-list format expected by foqcs_prep_spin_glass:
        #
        #   _J[p][k - 1][i] couples sites i and i + k.
        #
        # Example for L = 4:
        #   _J[p][0] = [J_01, J_12, J_23]
        #   _J[p][1] = [J_02, J_13]
        #   _J[p][2] = [J_03]

        J_diag = {
            p: [
                np.array(
                    [J[p][i, i + k] for i in range(sg_L - k)],
                    dtype=complex,
                )
                for k in range(1, sg_L)
            ]
            for p in paulis
        }

        # Preprocess physical coefficients into square-root PREP amplitudes.
        #
        # SELECT convention used by from_foqcs_lcu_prep:
        #   X  branch -> X
        #   Z  branch -> Z
        #   Y  branch -> iY
        #   YY branch -> -YY
        _g = {
            "X": np.sqrt(np.asarray(g["X"], dtype=complex)),
            "Y": np.sqrt(-1j * np.asarray(g["Y"], dtype=complex)),
            "Z": np.sqrt(np.asarray(g["Z"], dtype=complex)),
        }

        _J = {
            "X": [np.sqrt(diag) for diag in J_diag["X"]],
            "Y": [np.sqrt(-diag) for diag in J_diag["Y"]],
            "Z": [np.sqrt(diag) for diag in J_diag["Z"]],
        }

        # Normalization factor alpha.
        #
        # foqcs_prep_spin_glass internally normalizes the PREP state group-wise,
        # but the amplitudes it prepares are proportional to the entries we pass.
        # Hence alpha is the squared 2-norm of the flattened PREP-amplitude vector.
        coeff_vec = []

        for p in paulis:

            coeff_vec.extend(_g[p])

            for diag in _J[p]:

                coeff_vec.extend(diag)

        coeff_vec = np.array(coeff_vec, dtype=complex)
        norm = np.linalg.norm(coeff_vec)
        alpha = norm**2

        prep = partial(
            foqcs_prep_spin_glass,
            L=sg_L,
            g=_g,
            J=_J,
        )

        unprep = partial(
            foqcs_prep_spin_glass,
            L=sg_L,
            g=_g,
            J=_J,
            conjugate=True,
        )

        return prep, sg_L, unprep, False, alpha

    raise KeyError(f"Failed to handle FOQCS-LCU method: \"{aresult['method']}\"")
