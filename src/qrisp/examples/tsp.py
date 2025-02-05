"""
\********************************************************************************
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
********************************************************************************/
"""


import time
from math import factorial
from itertools import permutations
from numpy import binary_repr
from qrisp import *
from qrisp import auto_uncompute
from qrisp.grover import grovers_alg
from qrisp import QuantumFloat


city_amount = 4

distance_matrix = (
    np.array(
        [
            [0, 0.25, 0.125, 0.5],
            [0.25, 0, 0.625, 0.375],
            [0.125, 0.625, 0, 0.75],
            [0.5, 0.375, 0.75, 0],
        ]
    )
    / 4
)

# Create a function that generates a state of superposition of all permutations


# Receives a QuantumArray qa and a QuantumFloat index and
# then swaps the entry specified by index to the first position of the QuantummArray
def swap_to_front(qa, index):
    with invert():
        demux(qa[0], index, qa, permit_mismatching_size=True)


def eval_perm(perm_specifiers):
    # Specify the size of the QuantumFloats, which will represent the cities
    city_specifier_size = int(np.ceil(np.log2(city_amount)))

    # Create the QuantumArray, which will hold the permutations
    qa = QuantumArray(QuantumFloat(city_specifier_size), city_amount)

    # Initiate the QuantumArray with the identity permutation, ie. (0,1,2..)
    qa[:] = np.arange(city_amount)

    # Iteratively swap
    for i in range(city_amount - 1):
        swap_to_front(qa[i:], perm_specifiers[i])

    return qa


# Create function that returns QuantumFloats specifying the permutations (these will be
# in uniform superposition)
def create_perm_specifiers(city_amount, init_seq=None):
    perm_specifiers = []

    for i in range(city_amount - 1):
        qf_size = int(np.ceil(np.log2(city_amount - i)))

        perm_specifier = QuantumFloat(qf_size)

        if not init_seq is None:
            perm_specifier[:] = init_seq[i]

        perm_specifiers.append(perm_specifier)

    return perm_specifiers


# Create function that evaluates if a certain permutation is below a certain distance


# First implement distance function
@as_hamiltonian
def trip_distance(i, j, iter=1):
    return distance_matrix[i, j] * 2 * np.pi * iter


def phase_apply_summed_distance(itinerary, iter=1):
    n = len(itinerary)
    for i in range(n):
        trip_distance(itinerary[i], itinerary[(i + 1) % n], iter=iter)


@gate_wrap(permeability="args", is_qfree=True)
def qpe_calc_perm_travel_distance(itinerary, precision):
    return QPE(
        itinerary, phase_apply_summed_distance, precision=precision, iter_spec=True
    )


def qdict_calc_perm_travel_distance(itinerary, precision):
    # A QuantumFloat with n qubits and exponent -n
    # can represent values between 0 and 1
    res = QuantumFloat(precision, -precision)

    # Fill QuantumDictionary
    qd = QuantumDictionary(return_type=res)
    for i in range(city_amount):
        for j in range(city_amount):
            qd[(i, j)] = distance_matrix[i, j]

    # Evaluate result
    for i in range(city_amount):
        trip_distance = qd[itinerary[i], itinerary[(i + 1) % city_amount]]
        res += trip_distance
        trip_distance.uncompute(recompute=True)

    return res


@auto_uncompute
def eval_distance_threshold(perm_specifiers, precision, threshold, method="qpe"):
    itinerary = eval_perm(perm_specifiers)

    # distance = calc_perm_travel_distance(itinerary, precision)
    if method == "qdict":
        distance = qdict_calc_perm_travel_distance(itinerary, precision)
    elif method == "qpe":
        distance = qpe_calc_perm_travel_distance(itinerary, precision)
    else:
        raise Exception(f"Don't know method {method}")

    z(distance <= threshold)


# Create permutation specifiers
perm_specifiers = create_perm_specifiers(city_amount)

winner_state_amount = (
    2 ** sum([qv.size for qv in perm_specifiers])
    / factorial(city_amount)
    * city_amount
    * 2
)  # average number of state per permutation * (4 cyclic shifts)*(2 directions)

# Evaluate Grovers algorithm
grovers_alg(
    perm_specifiers,  # Permutation specifiers
    eval_distance_threshold,  # Oracle function
    kwargs={
        "threshold": 0.4,
        "precision": 5,
        "method": "qdict",
    },  # Specify the keyword arguments for the Oracle
    winner_state_amount=winner_state_amount,
)  # Specify the estimated amount of winners
# eval_distance_threshold(perm_specifiers , 5, 0.4)

qc = perm_specifiers[0].qs.compile()
print(qc.depth())
print(qc.cnot_count())


# %%
# Retrieve measurement


start_time = time.time()
res = multi_measurement(perm_specifiers)
print(time.time() - start_time)
# %%
# Check results


print("Perm specifier solution: ", list(res)[0])
perm_specifiers = create_perm_specifiers(city_amount, init_seq=list(res)[0])

perm_qa = eval_perm(perm_specifiers)
print("Permutation solution: ", perm_qa)


# %%

city_amount = 4

distance_matrix = (
    np.array(
        [
            [0, 0.25, 0.125, 0.5],
            [0.25, 0, 0.625, 0.375],
            [0.125, 0.625, 0, 0.75],
            [0.5, 0.375, 0.75, 0],
        ]
    )
    / 4
)


# Create a function that generates a state of superposition of all permutations
def swap_to_front(qa, index):
    with invert():
        # The keyword ctrl_method = "gray_pt" allows the controlled swaps to be
        # synthesized using Margolus gates. These gates perform the same operation as a
        # regular Toffoli, but add a different phase for each input. This phase will not
        # matter though, since it will be reverted once the ancilla values of the oracle
        # are uncomputed.
        demux(qa[0], index, qa, permit_mismatching_size=True, ctrl_method="gray_pt")


def eval_perm(perm_specifiers):
    N = len(perm_specifiers)

    # To filter out the cyclic permutations, we impose that the first city is always
    # city 0. We will have to consider this assumption later when calculating the route
    # distance by manually adding the trip distance of the first trip (from city 0) and
    # the last trip (to city 0).
    qa = QuantumArray(QuantumFloat(int(np.ceil(np.log2(city_amount)))), city_amount - 1)

    qa[:] = np.arange(1, city_amount)

    for i in range(N):
        swap_to_front(qa[i:], perm_specifiers[i])

    return qa


# Create function that returns QuantumFloats specifying the permutations (these will be
# in uniform superposition)
def create_perm_specifiers(city_amount, init_seq=None):
    perm_specifiers = []

    for i in range(city_amount - 1):
        qf_size = int(np.ceil(np.log2(city_amount - i)))

        if i == 0:
            continue

        temp_qf = QuantumFloat(qf_size)

        if not init_seq is None:
            temp_qf[:] = init_seq[i - 1]

        perm_specifiers.append(temp_qf)

    return perm_specifiers


# Create function that evaluates if a certain permutation is below a certain distance

# First implement distance function
@as_hamiltonian
def trip_distance(i, j, iter=1):
    return distance_matrix[i, j] * 2 * np.pi * iter


@as_hamiltonian
def distance_to_0(j, iter=1):
    return distance_matrix[0, j] * 2 * np.pi * iter


def phase_apply_summed_distance(itinerary, iter=1):
    # Add the distance of the first trip
    distance_to_0(itinerary[0], iter=iter)

    # Add the distance of the last trip
    distance_to_0(itinerary[-1], iter=iter)

    # Add the remaining trips
    for i in range(city_amount - 2):
        trip_distance(itinerary[i], itinerary[i + 1], iter=iter)


@lifted
def qpe_calc_perm_travel_distance(itinerary, precision):
    if precision is None:
        raise Exception("Tried to evaluate oracle without specifying a precision")

    return QPE(
        itinerary, phase_apply_summed_distance, precision=precision, iter_spec=True
    )


def qdict_calc_perm_travel_distance(itinerary, precision):
    # A QuantumFloat with n qubits and exponent -n
    # can represent values between 0 and 1
    res = QuantumFloat(precision, -precision)

    # Fill QuantumDictionary
    qd = QuantumDictionary(return_type=res)
    for i in range(city_amount):
        for j in range(city_amount):
            qd[(i, j)] = distance_matrix[i, j]

    # This dictionary contains the distances of each city to city 0
    qd_to_zero = QuantumDictionary(return_type=res)

    for i in range(city_amount):
        qd_to_zero[i] = distance_matrix[0, i]

    # The distance of the first trip is acquired by loading from qd_to_zero
    res = qd_to_zero[itinerary[0]]

    # Add the distance of the final trip
    final_trip_distance = qd_to_zero[itinerary[-1]]
    res += final_trip_distance
    final_trip_distance.uncompute(recompute=True)

    # Evaluate result
    for i in range(city_amount - 2):
        trip_distance = qd[itinerary[i], itinerary[(i + 1) % city_amount]]
        res += trip_distance
        trip_distance.uncompute(recompute=True)

    return res


@auto_uncompute
def eval_distance_threshold(perm_specifiers, precision, threshold, method="qpe"):
    itinerary = eval_perm(perm_specifiers)

    if method == "qdict":
        distance = qdict_calc_perm_travel_distance(itinerary, precision)
    elif method == "qpe":
        distance = qpe_calc_perm_travel_distance(itinerary, precision)
    else:
        raise Exception(f"Don't know method {method}")

    distance.add_sign()
    distance -= distance.truncate(threshold)
    z(distance.sign())


# Create permutation specifiers
perm_specifiers = create_perm_specifiers(city_amount)


# eval_distance_threshold(perm_specifiers, 5, 0.53125)


winner_state_amount = 2 ** sum([qv.size for qv in perm_specifiers]) / factorial(
    city_amount - 2
)  # average number of state per permutation * (4 cyclic shifts)*(2 directions)


# Evaluate Grovers algorithm
grovers_alg(
    perm_specifiers,  # Permutation specifiers
    eval_distance_threshold,  # Oracle function
    kwargs={
        "threshold": 0.4,
        "precision": 5,
        "method": "qdict",
    },  # Specify the keyword arguments for the Oracle
    winner_state_amount=winner_state_amount,
)  # Specify the estimated amount of winners


# Retrieve measurement

res = multi_measurement(perm_specifiers)

# Check results
# print(list(res)[0])

# %%
# Check which distance the result corresponds to

print("Perm specifier solution: ", list(res)[0])

perm_specifiers = create_perm_specifiers(city_amount, init_seq=list(res)[0])


perm_qa = eval_perm(perm_specifiers)

print("Permutation solution: ", perm_qa)

# distance = calc_perm_travel_distance(perm_specifiers, 5)
# trut_values = eval_distance_threshold(perm_specifiers, 5, 0.4)
# distances = multi_measurement([distance] + perm_specifiers)

# distances


# %%

hamming_locations = [0, 5, 2, 7, 1, 9, 4, 10, 13, 14]

city_amount = len(hamming_locations)

precision = 4


def trip_distance(qv_a, qv_b, iter=1):

    k = precision + 1

    phase = 2 * np.pi / 2**k * iter
    gray_gate = gray_synth_gate([0, phase, phase, 0])

    qv_a.qs.append(gray_gate, [qv_a, qv_b])


def route_distance(city_qa, iter=1):
    m = len(city_qa)

    for i in range(m):
        trip_distance(city_qa[i], city_qa[(i + 1) % m], iter=iter)


def eval_perm(perm_specifiers, locations=hamming_locations):
    N = len(perm_specifiers)

    qa = QuantumArray(QuantumFloat(int(np.ceil(np.log2(max(locations) + 1)))))

    qa[:] = locations

    for i in range(N):
        swap_to_front(qa[i + 1 :], perm_specifiers[i])

    return qa


# Check which distance the result corresponds to
# perm_specifiers = create_perm_specifiers(city_amount)
# h(perm_specifiers)
# distance = calc_perm_travel_distance(perm_specifiers, precision)
# distances = multi_measurement([distance] + perm_specifiers)
# truth_values = eval_distance_threshold(perm_specifiers, precision, 0.2)

# distances = multi_measurement([truth_values] + perm_specifiers)

perm_specifiers = create_perm_specifiers(city_amount)
grovers_alg(
    perm_specifiers,  # Permutation specifiers
    eval_distance_threshold,  # Oracle function
    oracle_type="bool",  # Specify that the oracle returns a QuantumBool
    kwargs={
        "threshold": 0.2,
        "precision": precision,
    },  # Specify the keyword arguments for the Oracle
    winner_state_amount=1,
)  # Specify the estimated amount of winners

# print(perm_specifiers[0].qs.cnot_count())
# print(len(perm_specifiers[0].qs.compile().qubits))

# res = multi_measurement(perm_specifiers)

# %%


def checksum(x):
    return sum([1 for c in binary_repr(x) if c == "1"])


def eval_hamming_distance(seq):
    s = 0

    for i in range(len(seq)):
        s += checksum(seq[i] ^ seq[(i + 1) % len(seq)])

    return s


perms = list(permutations(hamming_locations))

distances = [eval_hamming_distance(perm) for perm in perms]

print(perms[np.argmin(distances)])

# %%

compiled_qc = perm_specifiers[0].qs.compile()
print(len(compiled_qc.qubits))
print((compiled_qc.depth()))
print((compiled_qc.cnot_count()))

# %%

qf = QuantumFloat(3)

qf += 3
qf -= 3
