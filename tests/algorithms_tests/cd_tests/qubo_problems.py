"""Example QUBO matrices and their known optimal solutions, indexed by problem size N."""

"""QUBO instances with their known optimal solutions, used as test fixtures.

These used to live in ``qrisp.algorithms.cold.problems`` and were re-exported into the public
namespace by a star import, although nothing outside the test suite ever consumed them.
"""

import numpy as np

# N = 3
Q3 = np.array([[-0.9, 0.5, 0.3], [0.5, -0.7, -0.4], [0.3, -0.4, 0.2]])
solution3 = {"011": -1.3}

# N = 4
Q4 = np.array(
    [
        [-1.0, 0.5, 0.4, 0.0],
        [0.5, -0.9, 0.6, 0.0],
        [0.4, 0.6, -0.8, -0.5],
        [0.0, 0.0, -0.5, 0.2],
    ]
)
solution4 = {"1011": -1.8}

# N = 5
Q5 = np.array(
    [
        [-1.2, 0.6, 0.6, 0.0, 0.0],
        [0.6, -0.8, 0.6, 0.0, 0.0],
        [0.6, 0.6, -0.9, -0.7, 0.0],
        [0.0, 0.0, -0.7, -0.4, 0.5],
        [0.0, 0.0, 0.0, 0.5, 0.3],
    ]
)
solution5 = {"00110": -2.7, "10110": -2.7}


# N = 6
Q6 = np.array(
    [
        [-1.1, 0.6, 0.4, 0.0, 0.0, 0.0],
        [0.6, -0.9, 0.5, 0.0, 0.0, 0.0],
        [0.4, 0.5, -1.0, -0.6, 0.0, 0.0],
        [0.0, 0.0, -0.6, -0.5, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.6, -0.3, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.5, -0.4],
    ]
)
solution6 = {"101101": -3.4}

# N = 7
Q7 = np.array(
    [
        [-1.2, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.5, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.5, -0.9, -0.6, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.6, -0.7, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, -0.4, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, -0.3, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.2],
    ]
)
solution7 = {"1011010": -3.5}

# N = 8
Q8 = np.array(
    [
        [-1.3, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.6, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.5, -1.1, -0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.6, -0.6, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, -0.4, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, -0.3, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.25, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, -0.2],
    ]
)
solution8 = {"10110101": -3.9}

# N = 9
Q9 = np.array(
    [
        [-1.0, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        [0.4, -0.9, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.6, -1.2, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, -0.8, 0.70, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, -0.3, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.6, -0.4, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.25, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.35],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, -0.15],
    ]
)
solution9 = {"101101010": -4.0}

Q10 = np.array(
    [
        [-1.5, 0.5, 0.3, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.5, -1.4, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.3, 0.4, -1.3, -0.5, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0],
        [0.2, 0.0, -0.5, -1.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.5, -1.0, 0.4, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.4, -0.9, 0.3, 0.0, 0.0, 0.1],
        [0.1, 0.0, 0.0, 0.0, 0.1, 0.3, -0.8, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.7, 0.25, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.25, -0.6, 0.2],
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, -0.5],
    ]
)
solution10 = {"0111001101": -5.4, "1011010101": -5.4}

# A coupling-dominated instance.
#
# The 4x4 matrix the COLD and AGP tests hard-code, and the 6x6 one in the CD tutorial, are both
# field-dominated: the state that minimises sum_i h_i z_i on its own, with h_i = -0.5 sum_j Q_ij,
# already is the optimum, so their couplings never have to be taken into account at all. Q4 and Q5
# above share that property. An encoding that gets the local fields only roughly right still solves
# every one of them, which is how a QUBO-to-Ising conversion that double-counted diag(Q) survived
# in the source for as long as it did.
#
# Here the couplings decide. The field-blind answer "00110" is one bit away from the optimum and
# costs 0.6 more, -0.8 against -1.4, and the double-counting encoding puts its ground state at
# "00010" -- neither of them the optimum.
Q_coupled = np.array(
    [
        [0.6, 0.1, -0.6, 0.0, 0.4],
        [0.1, 0.6, -0.2, 0.6, 0.2],
        [-0.6, -0.2, 0.5, -0.4, 0.6],
        [0.0, 0.6, -0.4, -0.5, -0.1],
        [0.4, 0.2, 0.6, -0.1, 0.4],
    ]
)
solution_coupled = {"10110": -1.4}
field_blind_coupled = "00110"
