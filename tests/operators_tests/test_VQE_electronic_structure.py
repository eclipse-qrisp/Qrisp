# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests VQE ground-state energy for the H2 molecule via qrisp.vqe.problems.electronic_structure."""

import jax.numpy as jnp
import numpy as np
import pytest
from qrisp.vqe.problems.electronic_structure import *

from qrisp import QuantumFloat, QuantumVariable
from qrisp.jasp import jaspify


def _electronic_data(num_orb=4, num_elec=2):
    """Builds a minimal (all-zero) electronic data dictionary."""
    return {
        "one_int": np.zeros((num_orb, num_orb)),
        "two_int": np.zeros((num_orb, num_orb, num_orb, num_orb)),
        "num_orb": num_orb,
        "num_elec": num_elec,
    }


@pytest.mark.parametrize("i, j, expected", [(0, 0, 1), (0, 1, 0), (2, 2, 1), (3, 5, 0)])
def test_delta(i, j, expected):
    """Tests the Kronecker delta helper."""
    assert delta(i, j) == expected


@pytest.mark.parametrize("x, expected", [(0, 0), (1, 1), (4, 0), (5, 1)])
def test_omega(x, expected):
    """Tests the spin-component helper."""
    assert omega(x) == expected


def test_verify_symmetries():
    """Tests symmetry verification of the two-electron integral tensor."""
    assert verify_symmetries(np.zeros((2, 2, 2, 2)))

    broken = np.zeros((2, 2, 2, 2))
    broken[0, 1, 0, 1] = 1.0
    assert not verify_symmetries(broken)


def test_spacial_to_spin():
    """Tests the spacial-to-spin orbital integral transformation."""
    one_int = np.arange(4, dtype=float).reshape(2, 2)
    two_int = np.zeros((2, 2, 2, 2))
    one_int_spin, two_int_spin = spacial_to_spin(one_int, two_int)
    assert one_int_spin.shape == (4, 4)
    assert two_int_spin.shape == (4, 4, 4, 4)


def test_create_electronic_hamiltonian_invalid_type():
    """Tests that an unsupported argument type raises a ``TypeError``."""
    with pytest.raises(TypeError):
        create_electronic_hamiltonian("not a molecule")


def test_create_electronic_hamiltonian_invalid_active_space():
    """Tests that an invalid active space raises an exception."""
    with pytest.raises(Exception):
        create_electronic_hamiltonian(_electronic_data(num_orb=4, num_elec=2), active_orb=2, active_elec=3)


def test_create_electronic_hamiltonian_active_space():
    """Tests the active-space (inactive Fock operator) construction path."""
    H = create_electronic_hamiltonian(_electronic_data(num_orb=4, num_elec=4), active_orb=2, active_elec=2)
    assert H.find_minimal_qubit_amount() <= 2


@pytest.mark.parametrize("M, N", [(4, 2), (8, 4)])
def test_qccsd_ansatz_spin_combinations(M, N):
    """Tests the QCCSD ansatz including spin-up/spin-down combination terms."""
    ansatz, num_params = create_QCCSD_ansatz(M, N)
    qv = QuantumVariable(M)
    ansatz(qv, [0.1] * num_params)
    assert num_params > 0


def test_electronic_structure_problem_from_dict():
    """Tests ``electronic_structure_problem`` with an electronic data dictionary."""
    vqe = electronic_structure_problem(_electronic_data())
    assert vqe is not None


def test_electronic_structure_problem_active_space():
    """Tests ``electronic_structure_problem`` with an explicit active space."""
    vqe = electronic_structure_problem(_electronic_data(num_orb=4, num_elec=4), active_orb=2, active_elec=2)
    assert vqe is not None


def test_electronic_structure_problem_invalid_type():
    """Tests that an unsupported argument type raises a ``TypeError``."""
    with pytest.raises(TypeError):
        electronic_structure_problem("not a molecule")


#
# H2 molecule
#


def test_vqe_electronic_structure_H2():

    try:
        from pyscf import gto
    except:
        return

    mol = gto.M(atom="""H 0 0 0; H 0 0 0.74""", basis="sto-3g")

    H = create_electronic_hamiltonian(mol).to_qubit_operator()
    assert np.abs(H.ground_state_energy() - (-1.85238817356958)) < 1e-5

    vqe = electronic_structure_problem(mol)

    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(4), depth=1, max_iter=50)
        results.append(res)

    assert np.abs(min(results) - (-1.85238817356958)) < 3e-1


def test_jasp_vqe_electronic_structure_H2():

    try:
        from pyscf import gto
    except:
        return

    @jaspify(terminal_sampling=True)
    def main():

        mol = gto.M(atom="""H 0 0 0; H 0 0 0.74""", basis="sto-3g")

        vqe = electronic_structure_problem(mol)

        results = jnp.array([0.0] * 5)
        for i in range(5):
            res = vqe.run(QuantumFloat(4), depth=1, max_iter=100, optimizer="SPSA")
            results = results.at[i].set(res)

        return results

    results = main()

    assert np.abs(min(results) - (-1.85238817356958)) < 3e-1


#
# BeH2 molecule, active space
#
"""
def test_vqe_electronic_structure_BeH2():

    mol = gto.M(
        atom = f'''Be 0 0 0; H 0 0 3.0; H 0 0 -3.0''',
        basis = 'sto-3g')
    
    H = create_electronic_hamiltonian(mol,active_orb=6,active_elec=4).to_qubit_operator()
    assert np.abs(H.ground_state_energy()-(-16.73195995959339)) < 1e-9

    # runs for >1 minute
    vqe = electronic_structure_problem(mol,active_orb=6,active_elec=4)
    
    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(6),
                depth=1,
                max_iter=50)
        results.append(res)
    
    print(min(results))
    assert np.abs(min(results)-(-16.73195995959339)) < 1e-1
"""
