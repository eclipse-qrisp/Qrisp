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

"""Tests for the behavior shared by the count_ops, depth and num_qubits resource estimators."""

import itertools

import pytest

from qrisp import QuantumBool, control, h, measure, x
from qrisp.jasp import jrange, qache
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import normalize_slice_bounds
from qrisp.jasp.interpreter_tools.interpreters.utilities import meas_rng


def test_normalize_slice_bounds_matches_python():
    """``normalize_slice_bounds`` follows Python slicing semantics for every combination of bounds."""
    for size, start, stop in itertools.product(range(5), range(-7, 8), range(-7, 8)):
        expected = range(size)[start:stop]
        norm_start, norm_stop = normalize_slice_bounds(size, start, stop)

        assert int(norm_stop) - int(norm_start) == len(expected), (size, start, stop)
        if expected:
            assert int(norm_start) == expected.start, (size, start, stop)


def _measure_and_flip(target):
    """Measure a fresh qubit in superposition, and if the outcome is 1, flip ``target``.

    The outcome 1 also allocates (and uses) an extra qubit, so every metric can
    tell how many outcomes were 1: count_ops counts the X gates, num_qubits the
    extra qubits, and depth the sequential X gates on ``target``.
    """
    qb = QuantumBool()
    h(qb)
    with control(measure(qb)):
        extra = QuantumBool()
        h(extra)
        x(target)


@qache
def _measure_and_flip_subroutine(target):
    """Call ``_measure_and_flip`` through a qache'd subroutine."""
    _measure_and_flip(target)


def _loop_program(num_iterations):
    """Measure inside a jrange loop."""
    target = QuantumBool()
    for _ in jrange(num_iterations):
        _measure_and_flip(target)


def _subroutine_program():
    """Measure inside 16 calls of the same qache'd subroutine."""
    target = QuantumBool()
    for _ in range(16):
        _measure_and_flip_subroutine(target)


def _branch_program(num_iterations):
    """Measure inside a classical branch, then again after it."""
    target = QuantumBool()
    with control(num_iterations > 0):
        for _ in jrange(num_iterations):
            _measure_and_flip(target)
    for _ in jrange(num_iterations):
        _measure_and_flip(target)


@pytest.mark.parametrize(
    "program,args,num_measurements",
    [
        (_loop_program, (16,), 16),
        (_subroutine_program, (), 16),
        (_branch_program, (8,), 16),
    ],
    ids=["jrange loop", "qache subroutine", "branch"],
)
def test_metrics_see_the_same_random_outcomes(estimate_resources, program, args, num_measurements):
    """With a random measurement behavior, all metrics see the same outcomes, and the outcomes vary.

    The k-th measured qubit is sampled with the key ``jax.random.key(k)``.

    Regression test: depth and num_qubits numbered the measurements with a counter
    that was not carried through loops, branches and subroutine calls. They reused
    the same random key, so every measurement of a loop had the same outcome, and
    the metrics took different branches than count_ops for the same program.
    """
    resources = estimate_resources(program, *args, meas_behavior=meas_rng)
    num_ones = resources["count_ops"].get("x", 0)

    assert resources["count_ops"]["measure"] == num_measurements
    # Neither all 0 nor all 1, so a repeated key would change the counts below
    assert 0 < num_ones < num_measurements

    # The target, one measured qubit per measurement, and one extra qubit per outcome 1
    assert resources["num_qubits"]["total_allocated"] == 1 + num_measurements + num_ones

    # All other gates act on fresh qubits in parallel, only the X gates on the target are sequential
    assert resources["depth"] == num_ones
