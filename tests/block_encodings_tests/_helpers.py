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

"""Shared post-selection and distribution-comparison helpers for block-encoding tests."""

import numpy as np

from qrisp import terminal_sampling


def postselect_results(res_dict, num_ops=1):
    """Filter a terminal_sampling dict to all-zero ancillas and renormalize."""
    if not res_dict:
        return {}
    sample_key = next(iter(res_dict))
    if not isinstance(sample_key, tuple):
        return res_dict

    filtered = {}
    for key, prob in res_dict.items():
        if all(anc == 0 for anc in key[num_ops:]):
            out_key = key[0] if num_ops == 1 else key[:num_ops]
            filtered[out_key] = filtered.get(out_key, 0.0) + prob

    success_prob = sum(filtered.values())
    assert success_prob > 0, "Post-selection onto all-zero ancillas had zero success probability."
    return {k: p / success_prob for k, p in filtered.items()}


def sample_postselected(block_encoding, operand_prep):
    """Apply ``block_encoding`` once via ``.apply()`` and post-select on all ancillas being 0."""

    @terminal_sampling
    def main(be):
        prepared = operand_prep()
        operands = prepared if isinstance(prepared, tuple) else (prepared,)
        ancillas = be.apply(*operands)
        if len(operands) == 1 and len(ancillas) == 0:
            return operands[0]
        return (*operands, *ancillas)

    return postselect_results(main(block_encoding), num_ops=block_encoding.num_ops)


def compare_distributions(res_dict_1, res_dict_2, n, atol=1e-8):
    """Assert two outcome probability dictionaries match on all n-qubit computational basis states."""
    for k in range(2**n):
        assert np.isclose(res_dict_1.get(k, 0), res_dict_2.get(k, 0), atol=atol)
