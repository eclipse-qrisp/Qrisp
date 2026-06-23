"""
********************************************************************************
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
********************************************************************************
"""

# This file implements the algorithm presented here:
# https://arxiv.org/abs/2304.02921

# We begin with some functions for carry propagation.
# For this we implement the Brent-Kung-Tree in a recursive manner. See:
# https://web.stanford.edu/class/archive/ee/ee371/ee371.1066/lectures/lect_04.pdf Page 13

from qrisp import *


# Returns the PROPAGATE status of a group of entries
def calc_P_group(P):
    new_p = QuantumBool(name="p_group*")
    if len(P) == 2:
        mcx(P, new_p, method="gidney")
    else:
        mcx(P, new_p)
    return new_p


# Returns the GENERATE status of a group of entries
# A group has GENERATE status True, if any of the members
# has GENERATE status True and all of the following members
# have PROPAGATE status True.


# The GENERATE status of the group is calculated into the last
# member of the group (G[-1])
def calc_G_group(P, G):
    for i in range(len(G) - 1):
        mcx([G[i]] + P[i + 1 :], G[-1], method="jones")
    return G[-1]


counter = 0


# Receives the values PROPAGATE and GENERATE values of a group.
# It is assumed that the 0-th entry of G already contains the carry status
# This function then calculates the carry status for every other entry of G
def propagate_carry(P, G):

    # This loop iteratively calculates the CARRY status into all entries
    # of G.

    # We calculate the CARRY status of position i by considering the CARRY status of
    # position 0 and the GENERATE status of any position above that. To successfully
    # generate a CARRY status of True, all the other positions in between need to have
    # PROPAGATE status True.

    # Note that in order to calculate the CARRY status of position i, we need the
    # GENERATE status (not the CARRY status) of j for all j < i.
    # This is why we reverse the loop, such that the position with the highest i
    # are calculated first (the computation can access all the GENERATE
    # information because none of the CARRY status has been calculated yet)
    for i in range(1, len(G))[::-1]:
        for j in range(i):
            mcx([G[j]] + P[j + 1 : i + 1], G[i], method="jones")


# This function recursively calculates the layers of the Brent-Kung tree.
# It receives the GENERATE and CARRY status of a set of positions and returns
# the evaluated CARRY status

# The function takes the parameters P and G, which represent the PROPAGATE and
# the GENERATE status of the corresponding digit.

# The parameter r describes the radix of the Brent-Kung tree.

# Finally the parameter k_out describes a the layer where the carry calculation
# should be ended prematurely. That is if k_out = 2 and r = 4, the output will
# be every 4**2 = 16-th bit of G, that contains a properly calculated carry value


def brent_kung_tree(P, G, r, k_out=np.inf):
    # if len(G):
    # print(G[0].qs().compile(40, compile_mcm = True, gate_speed = gate_speed).t_depth())
    if r <= 1 or not isinstance(r, int):
        raise Exception("Provided invalid radix base (needs to be integer > 1)")

    # We group the positions and calculate the GENERATE and PROPAGATE status
    # of the groups. Subsequently the function is called recursively to evaluate the
    # CARRY status of the respective groups.

    # Finally the CARRY status of the groups is propagated to the remaining positions
    # to yield the complete set of carries (not only the groups)

    # Recursion cancellation condition
    if len(G) == 0:
        return []

    # Set up list for grouped GENERATE and PROPAGATE states
    grouped_P = []
    grouped_G = []
    for i in range(len(G) // r):
        interval = slice(r * i, r * (i + 1))

        if i == 0:
            # We can skip the 0-th position
            # This is because the propagate_carry function and the calc_G_group
            # function are not taking the 0-th position into consideration
            grouped_P.append(None)
        else:
            grouped_P.append(calc_P_group(P[interval]))

        grouped_G.append(calc_G_group(P[interval], G[interval]))

    # Call the function recursively
    grouped_carry = brent_kung_tree(grouped_P, grouped_G, r, k_out=k_out - 1)

    if k_out > 0:
        return grouped_carry

    # If working with a radix > 2, the carry also needs to be propagated
    # to the position of the first group. In the case of r = 2, this is not neccessary
    # because the first group consists just of the 0-th position (which already contains
    # the carry in its GENERATE status) and the 1st position which contains the CARRY status
    # due to the GENERATE computation of this group.
    initial_interval = slice(0, r - 1)
    propagate_carry(P[initial_interval], G[initial_interval])

    # Go through all the groups and propagate the carry.
    # Note that the last interval has not neccessarily size r (depending on the size of G)
    for i in range(1, len(G) // r + 1):
        interval = slice(r * i - 1, r * (i + 1) - 1)
        propagate_carry(P[interval], G[interval])

    # Return G (now contains the carry positions)
    return G


# The following function calculates the carry status from two QuantumVariables of equal size

# The function takes two QuantumFloats a, b, the radix base and the radix exponent.
# The function returns a QuantumVariable c, that contains the CARRY status of every
# radix_base**radix_exponent digit of CARRY(a,b)

# Example:

# a = 3 = 0011
# b = 6 = 0110
# CARRY(a,b) = 0110

# Assume we choose radix_base = 2. If we choose radix_exponent = 0,
# this function returns

# c = 0110

# If we choose radix_exponent = 1

# c = 01

# ie. position 1 and 3 of CARRY(a,b)

# If we chose radix_exponent = 2

# c = 0

# ie. position 3 of CARRY(a,b)


# Furthermore this function uncomputes all garbage. What is the garbage here?
# The cancelation of the brent kung tree at an early layer produce many GENERATE
# entries, that are not holding CARRY values. Furthermore, also the PROPAGATE values
# of groups are uncomputed.


@auto_uncompute
def calc_carry(a, b, radix_base=2, radix_exponent=0):

    R = radix_base**radix_exponent
    # How can we achieve that the GENERATE entries, that don't contain a relevant
    # CARRY information are uncomputed while the CARRY entries stay?

    # We do this by creating two QuantumVariables, one which holds the carry and
    # one which holds the GENERATE garbage. We then arrange both of them in a list g,
    # that represents the GENERATE array.

    # This variable will hold the result
    # If b can be divided into k blocks of size R,
    # we only need k-1 ancillae qubit, because we have no need for
    # the carry of the last bock.
    c = QuantumVariable(int(np.ceil(len(b) / R)) - 1)

    # This variable will hold the intermediate GENERATE values, that are supposed
    # to be uncomputed. The uncomputation is performed using the auto_uncompute
    # decorator. This decorator uncomputes all local variables.
    brent_kung_ancilla = QuantumVariable(c.size * (R - 1))

    # Create the g list
    anc_list = list(brent_kung_ancilla)
    c_list = list(c)
    g = []
    for i in range(len(c)):
        for j in range(R - 1):
            g.append(anc_list.pop(0))
        g.append(c_list.pop(0))

    # Append the remaining ancillae
    g.extend(anc_list)

    # We now compute the GENERATE and the PROPAGATE status of each digit
    # g_i = a_i * b_i
    # p_i = a_i XOR b_i

    # This could be achieved by applying a sequence of mcx gates to compute g
    # followed by a sequence of cx gates that compute p.

    # This however breaks the automatic uncomputation algorithm since it induces
    # a cycle in the dag representation. We can still use automatic uncomputation
    # with a hack: We use both operation together and mark the argument of b as
    # permeable.

    # This gate is synthesized by the anc_gate function.

    for i in range(min(len(g), len(a), len(b))):
        a.qs.append(anc_gate(method="gidney"), [a[i], b[i], g[i]])

    # In some rare cases this hack seems to also disturb the gate ordering, since
    # the dag linearization assumes commutativity due to the permeability. We fix this problem
    # by enforcing the order with a non-permeable identity. Ie. two subseqeuent x gates
    x(g)
    x(g)
    # These x gates are cancelled by the compile method so there is no overhead.

    p = b

    # Calculate compute the carry using the Brent-Kung tree
    brent_kung_tree(p, g, radix_base, radix_exponent)

    # print("======")
    # print(c.qs.compile(40, compile_mcm = True, gate_speed = gate_speed).t_depth())

    # Undo the previous cx to keep b unchanged
    # For the entries that are uncomputed automatically, the inverted anc_gate
    # will restore b So we only need to restore.
    for i in range(min(len(g), len(a), len(b))):
        if g[i] in list(c):
            cx(a[i], b[i])
        else:
            pass

    # Return the result
    return c


# This function returns a gate, that performs the toffoli and cx gate.
# We mark the second qubit as permeable to "hack" the uncomputation algorithm.
def anc_gate(method="gray"):
    anc_qc = QuantumCircuit(3)
    anc_qc.mcx([0, 1], 2, method=method)
    anc_qc.cx(0, 1)
    res = anc_qc.to_gate(name="anc_gate")
    res.permeability = {0: True, 1: True, 2: False}
    res.is_qfree = True
    return res


# This function performs the in-place addition
# b += a
# based on the higher radix qcla
# The overall radix can be specified as an exponential of the form
# R = radix_base**radix_exponent
def qcla(a, b, radix_base=2, radix_exponent=0, reduce_t_depth=True):

    if len(a) > len(b):
        raise Exception(
            "Tried to add QuantumFloat of higher precision onto QuantumFloat of lower precision"
        )

    R = radix_base**radix_exponent

    # The case that a only has a single qubit is simple.
    if len(b) == 1:
        cx(a, b)
        return
    elif len(b) <= R:
        qcla_anc = QuantumVariable(len(b) - len(a))
        gidney_adder(list(a) + list(qcla_anc), b)
        qcla_anc.delete()
        return

    # Calculate the carry
    c = calc_carry(a, b, radix_base, radix_exponent)

    sum_path_gidney(a, b, c, R)


def sum_path(a, b, c, R):

    if R == 1:
        # If R = 1, we are in the case of Drapers QCLA:
        # https://arxiv.org/abs/quant-ph/0406142
        # We can use the formula
        # S = A (+) B (+) C
        cx(c[:-1], b[1:])
        cx(a, b[: len(a)])
    else:
        i = 0
        # Execute addition using the corresponding carry values
        for i in range(len(c) + 1):

            # Determine the radix qubits to perform the addition on
            a_block = a[R * i : R * (i + 1)]
            b_block = b[R * i : R * (i + 1)]

            # If a_block is empty we break the loop.
            # This doesn't necessarily imply that b_block is empty because
            # len(b) could be larger then len(a). In this case we execute an
            # increment block after this loop
            if not len(a_block):
                i = i - 1
                break

            # If a_block and b_block have different sizes, we create a padding
            # variable to fill up the remaining entries
            if not len(a_block) == len(b_block):
                padding_var = QuantumVariable(len(b_block) - len(a_block))
                a_block = a_block + list(padding_var)
            # Perform Cuccarro addition
            if i == 0:
                # cuccaro_procedure(a.qs, a_block, b_block)

                gidney_adder(a_block, b_block)
            else:

                # cuccaro_procedure(a.qs, a_block, b_block, carry_in = c[i-1])
                gidney_adder(a_block, b_block, c[i - 1])

            # Delete carry var
            try:
                padding_var.delete()
            except NameError:
                pass

        # This loop treats the case that len(a) != len(b)
        # In this case we need to perform the increment function on the remaining
        # qubits of b (incase the carry is True)
        # We start at the index the last loop finished at
        for j in range(i + 1, len(b) // R + 1)[::-1]:
            b_block = b[R * j : R * (j + 1)]

            if not len(b_block):
                continue

            # Perform incrementation function
            if R * j == 0:
                qbl = QuantumBool()
                lin_incr(b_block, qbl[0], c_out=c[j])
                qbl.delete()
            else:
                lin_incr(b_block, c[j - 1], c_out=c[j])

    # To uncompute the carry we use Drapers strategy
    # CARRY(A,B) = CARRY(A, NOT(A+B))
    # We therefore bitflip the sum
    x(b)

    # Contrary to Draper's adder we don't need to uncompute every carry digit.
    # Because of the above equivalence, the carries agree on every digit, so especially
    # on the digits representing the output of the calc_carry function. We can therefore
    # uncompute using calc_carry (even with higher radix) by inverting calc_carry.

    with invert():
        # We use the redirect_qfunction decorator to steer the function onto c
        redirect_qfunction(calc_carry)(a, b, radix_base, radix_exponent, target=c)

    # Delete c. If needed, verify that the uncomputation indeed suceeded
    c.delete(verify=False)

    # Flip the sum back
    x(b)


def sum_path_gidney(a, b, c, R):

    if R == 1:
        # If R = 1, we are in the case of Drapers QCLA:
        # https://arxiv.org/abs/quant-ph/0406142
        # We can use the formula
        # S = A (+) B (+) C
        cx(c, b[1:])

        # We now do the mother of all evil hacks:
        # We can get rid of c using automatic uncomputation.
        # However since we already hacked the uncomputation by claiming
        # the anc_gate is permeable when in reality it's not, the anc_gate
        # (which is used to uncompute c) introduces an additional cx gate.
        # This cx gate incidentally cancels out with the one from below.

        c.uncompute()
        # cx(a, b[:len(a)])
        return

    # print(c)
    # # This loop treats the case that len(a) != len(b)
    # # In this case we need to perform the increment function on the remaining
    # # qubits of b (incase the carry is True)
    # print(c)
    # #We start at the index the last loop finished at
    if len(a) != len(b):
        # print("len(a): ", len(a))
        # print("R: ", R)
        # print("len(a)//R: ", len(a)//R)
        for j in range(int(np.ceil(len(a) / R)), len(b) // R + 2)[::-1]:

            b_block = b[R * j : R * (j + 1)]
            # print(j)

            if not len(b_block):
                continue
            # print(b_block)

            # Perform incrementation function
            if R * j == 0:
                lin_incr(b_block, c_out=c[j])
            else:
                if j < len(c):
                    lin_incr(b_block, c[j - 1], c_out=c[j])
                else:
                    lin_incr(b_block, c[j - 1])

    # Execute addition using the corresponding carry values
    for i in range(len(a) // R + 1)[::-1]:
        # Determine the radix qubits to perform the addition on
        a_block = a[R * i : R * (i + 1)]
        b_block = b[R * i : R * (i + 1)]

        # If a_block is empty we break the loop.
        # This doesn't necessarily imply that b_block is empty because
        # len(b) could be larger then len(a). In this case we execute an
        # increment block after this loop
        if not len(a_block):
            i = i - 1
            continue

        # If a_block and b_block have different sizes, we create a padding
        # variable to fill up the remaining entries
        if not len(a_block) == len(b_block):
            padding_var = QuantumVariable(len(b_block) - len(a_block))
            a_block = a_block + list(padding_var)

        # Perform Cuccarro addition
        if i == 0:
            # cuccaro_procedure(a.qs, a_block, b_block)
            if len(c):
                gidney_adder(a_block, b_block, c_out=c[i])
            else:
                gidney_adder(a_block, b_block)
        elif i < len(c):
            # cuccaro_procedure(a.qs, a_block, b_block, carry_in = c[i-1])
            gidney_adder(a_block, b_block, c_in=c[i - 1], c_out=c[i])
        else:
            gidney_adder(a_block, b_block, c_in=c[i - 1])

        # Delete carry var
        try:
            padding_var.delete()
        except NameError:
            pass

    c.delete(verify=False)


# This function performs the linear depth incrementation described in
# https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html
# This function also needs a linear amount of qubits. These qubits pose no overhead
# because in the above QCLA function, the calc_carry function already allocated and
# deallocated some ancillas, so the compiler can reuse these
def lin_incr(a, c_in=None, c_out=None):

    if c_out is not None:
        a = list(a) + [c_out]

    if len(a) == 1:
        cx(c_in, a)
        return

    incr_anc = QuantumVariable(len(a) - 1)

    if c_in is not None:
        mcx([c_in, a[0]], incr_anc[0], method="gidney")

    for i in range(len(a) - 2):
        mcx([incr_anc[i], a[i + 1]], incr_anc[i + 1], method="gidney")

    cx(incr_anc[-1], a[-1])

    for i in range(len(a) - 2)[::-1]:
        mcx([incr_anc[i], a[i + 1]], incr_anc[i + 1], method="gidney_inv")
        cx(incr_anc[i], a[i + 1])

    if c_in is not None:
        mcx([c_in, a[0]], incr_anc[0], method="gidney_inv")
        cx(c_in, a[0])

    incr_anc.delete()


a = QuantumFloat(4)
b = QuantumFloat(4)
d = b.duplicate()
# a = QuantumVariable(3)
# b = QuantumVariable(3)

# c = QuantumBool()
a[:] = 1
b[:] = 0
# c[:] = 1
# sum_path_gidney(a, b, c, 2)

# h(a)
# h(b)
# d[:] = b

# print(multi_measurement([a,b,c]))

# import time


radix_base = 2
radix_exponent = 1
c = calc_carry(a, b, radix_base, radix_exponent)
# # brent_kung_tree(a, b, radix_b1ase, radix_exponent)
# # print(counter)
# qcla(a, b, radix_base, radix_exponent)

# print(a.qs)
# b += a
# gidney_adder(a, b)

print(c)
from qrisp.interface import QiskitBackend

# print(multi_measurement([a,d,b]))
# print(b.get_measurement(compilation_kwargs = {"compile_mcm" : True}))
# print(b)


# print(c)
# print(a.qs.depth())
# gate_speed = lambda x : t_depth_indicator(x, 2**-6)
# print(a.qs.compile(40, compile_mcm = True, gate_speed = gate_speed).t_depth())
# print(a.qs.compile(2).cnot_count())
# print(a.qs.compile(2).num_qubits())
# print(a.qs.count_ops())
# %%
# Test functions
# Function to classically calculate the carry value
def full_adder_carry(bitstring1, bitstring2):
    if len(bitstring1) != len(bitstring2):
        raise ValueError("Bitstrings must be of equal length")

    n = len(bitstring1)
    carry = 0
    carry_values = []

    for i in range(n):
        a = int(bitstring1[i])
        b = int(bitstring2[i])

        # Calculate the carry-out (Cout) for this bit
        sum_ab = a + b + carry
        carry_out = sum_ab // 2

        carry_values.append(carry_out)

        # Update the carry for the next iteration
        carry = carry_out

    return "".join(str(c) for c in carry_values)


n = 4
for n in range(3, 8):
    for r in range(2, n):
        a = QuantumVariable(n)
        b = QuantumVariable(n)

        h(a)
        h(b)

        c = calc_carry(a, b, r)

        mes_res = multi_measurement([a, b, c])

        for k in mes_res.keys():
            if k[2] != full_adder_carry(k[0], k[1])[:-1]:
                print(k[0])
                print(k[1])
                print(k[2])
                print(full_adder_carry(k[0], k[1])[:-1])

                assert False

# %%
# for R in [1,2,4,8]:

import time

t0 = time.time()
for radix_base in [2, 3]:
    for radix_exponent in [1, 0, 2]:
        for m in range(1, 7):
            for n in range(1, m):

                a = QuantumFloat(n)
                b = QuantumFloat(m)
                c = QuantumFloat(m)

                h(a)
                h(b)
                c[:] = b

                print(a.size, b.size, radix_base, radix_exponent)
                qcla(a, b, radix_base=radix_base, radix_exponent=radix_exponent)
                mes_res = multi_measurement([a, c, b])

                for k in mes_res.keys():
                    if k[2] != (k[0] + k[1]) % 2**m:
                        print(k[0])
                        print(k[1])
                        print(k[2])
                        print((k[0] + k[1]) % 2**m)

                        assert False

                statevector_arr = a.qs.compile().statevector_array()
                angles = np.angle(
                    statevector_arr[
                        np.abs(statevector_arr) > 1 / 2 ** ((a.size + b.size) / 2 + 1)
                    ]
                )

                # Test correct phase behavior
                assert np.sum(np.abs(angles)) < 0.1


print(time.time() - t0)
