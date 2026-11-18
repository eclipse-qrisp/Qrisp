# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
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

"""Tests for the LCD algorithm on QUBO problems."""

import numpy as np
import pytest
from qubo_problems import Q_coupled, field_blind_coupled, solution_coupled

from qrisp.algorithms.cold import solve_QUBO


def test_lcd_local_uniform():
    """LCD with 1st-order AGP, uniform coefficients, finds the known solution."""
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "LCD", "agp_type": "local", "uniform": True}
    run_args = {"N_steps": 50, "T": 10}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_lcd_local_nonuniform():
    """LCD with 1st-order AGP, non-uniform coefficients, finds the known solution."""
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "LCD", "agp_type": "local", "uniform": False}
    run_args = {"N_steps": 50, "T": 10}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_lcd_nc_uniform():
    """LCD with nested-commutator AGP, uniform coefficients, finds the known solution."""
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "LCD", "agp_type": "nc", "uniform": True}
    run_args = {"N_steps": 50, "T": 10}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_lcd_nc_nonuniform():
    """LCD with nested-commutator AGP, non-uniform coefficients, finds the known solution."""
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "LCD", "agp_type": "nc", "uniform": False}
    run_args = {"N_steps": 50, "T": 10}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


@pytest.mark.parametrize("uniform", [True, False])
def test_lcd_solves_a_coupling_dominated_qubo(uniform):
    """LCD must beat the answer that reads the local fields and ignores the couplings.

    The other instances in this suite are field-dominated, so a run can land on their optimum
    without the ZZ terms contributing anything. Here "00110" is what the local fields alone say
    and "10110" is the actual optimum, so an encoding or a drive that does not carry the couplings
    through correctly ends up on the wrong state instead of merely on a less peaked distribution.
    """
    (solution,) = solution_coupled

    res = solve_QUBO(
        Q_coupled,
        problem_args={"method": "LCD", "agp_type": "nc", "uniform": uniform},
        run_args={"N_steps": 20, "T": 2},
    )

    assert list(res.keys())[0] == solution
    assert res[solution][0] > res[field_blind_coupled][0]
