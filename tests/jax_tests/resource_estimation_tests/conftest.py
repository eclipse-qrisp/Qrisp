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

"""Shared fixtures for the tests of resource estimation with Jasp."""

# The fixtures evaluate a program with the three resource estimators at once
# (count_ops, depth and num_qubits), so that a test can compare all of them
# with the expected resources in a single assertion:
#
#     assert estimate_resources(main) == {
#         "count_ops": {"h": 1, "cx": 1},
#         "depth": 2,
#         "num_qubits": {
#             "total_allocated": 2,
#             "total_deallocated": 0,
#             "peak_allocations": 2,
#             "finally_allocated": 2,
#         },
#     }

import pytest

from qrisp.jasp import count_ops, depth, num_qubits


def _estimate_resources(program, *args, meas_behavior="0"):
    """Return the results of count_ops, depth and num_qubits for ``program(*args)``."""
    return {
        "count_ops": count_ops(meas_behavior=meas_behavior)(program)(*args),
        "depth": depth(meas_behavior=meas_behavior)(program)(*args),
        "num_qubits": num_qubits(meas_behavior=meas_behavior)(program)(*args),
    }


@pytest.fixture
def estimate_resources():
    """Provide a function returning the results of all three resource estimators for a program."""
    return _estimate_resources
