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

"""Provides the find_max and sorted_correlations subroutines for locating maximal qubit correlations in QIRO."""

import numpy as np


def find_max(single_cor, double_cor, res, solutions):
    """Subroutine for finding the values with maximal correlation in the QIRO algorithm.

    Parameters
    ----------
    single_cor : list
        Single qubit correlations to check.
    double_cor : list[tuple]
        Multi qubit correlations to check.
    res : dict
        Result dictionary of QAOA optimization procedure.
    solutions : List
        Qubits which were found to be positivel correlated, i.e., part of the problem solution.

    Returns
    -------
    max_item, sign
        The item with maximal correlation and the sign of the correlation.

    """
    max = 0
    max_item = None
    sign = None

    for item2 in double_cor:
        if abs(item2[0]) == abs(item2[1]):
            continue
        summe = 0

        # calc correlation expectation
        for key, val in res.items():
            summe += val * pow(-1, int(key[int(abs(item2[0]))])) * pow(-1, int(key[int(abs(item2[1]))]))

        # find max
        if abs(summe) > abs(max):
            max, max_item = summe, item2
            sign = np.sign(summe)

    for node in single_cor:
        if node in solutions:
            continue
        summe = 0

        for key, val in res.items():
            summe += val * pow(-1, int(key[int(node)]))

        if abs(summe) > abs(max):
            max, max_item = summe, node
            sign = np.sign(summe)

    return max_item, sign


def sorted_correlations(single_cor, double_cor, res, solutions):
    """Subroutine returning all non-zero correlations, ordered by decreasing absolute value.

    The order of equally large correlations matches ``find_max`` (two-qubit correlations
    before single-qubit ones, then input order), so the first entry is the item ``find_max``
    would return. Replacement routines can use the remaining entries to fall back to the
    next-largest correlation if a rule does not reduce the problem.

    Parameters
    ----------
    single_cor : list
        Single qubit correlations to check.
    double_cor : list[tuple]
        Multi qubit correlations to check.
    res : dict
        Result dictionary of QAOA optimization procedure.
    solutions : List
        Qubits which were found to be positivel correlated, i.e., part of the problem solution.

    Returns
    -------
    list[tuple]
        Pairs ``(item, correlation)``, sorted by decreasing ``abs(correlation)``.

    """
    correlations = []

    for item2 in double_cor:
        if abs(item2[0]) == abs(item2[1]):
            continue
        summe = 0
        for key, val in res.items():
            summe += val * pow(-1, int(key[int(abs(item2[0]))])) * pow(-1, int(key[int(abs(item2[1]))]))
        if summe != 0:
            correlations.append((item2, summe))

    for node in single_cor:
        if node in solutions:
            continue
        summe = 0
        for key, val in res.items():
            summe += val * pow(-1, int(key[int(node)]))
        if summe != 0:
            correlations.append((node, summe))

    # sorted() is stable, so ties keep the find_max order
    return sorted(correlations, key=lambda entry: -abs(entry[1]))
