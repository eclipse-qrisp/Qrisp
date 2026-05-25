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

import math

from qrisp import *


def sampled_target_value(bitstring, ctrl_state, masks):
    mismatch_mask = 0

    for index, bit in enumerate(bitstring):
        if bit != ctrl_state[index]:
            mismatch_mask |= 1 << index
    # Target flips iff every sampled parity over the mismatch bits is zero.
    return all(((mask & mismatch_mask).bit_count() & 1) == 0 for mask in masks)


def test_approx_mcx_matches_sampled_truth_table():
    control_amount = 5
    ctrl_state = "10110"
    k = 3
    seed = 7

    masks = sample_approx_mcx_masks(control_amount, k, seed=seed)

    controls = QuantumVariable(control_amount)
    target = QuantumBool()

    h(controls)
    approx_mcx(controls, target, k=k, seed=seed, ctrl_state=ctrl_state)

    measurement = multi_measurement([controls, target])

    false_positives = []

    for value in range(2**control_amount):
        bitstring = bin_rep(value, control_amount)
        expected_target = sampled_target_value(bitstring, ctrl_state, masks)

        assert (bitstring, expected_target) in measurement

        if bitstring != ctrl_state and expected_target:
            false_positives.append(bitstring)

    assert (ctrl_state, True) in measurement
    assert false_positives