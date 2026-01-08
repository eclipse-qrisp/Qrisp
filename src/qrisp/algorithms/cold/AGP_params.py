import numpy as np
from collections import defaultdict

# ----- Pauli algebra (bitwise) -----

def pauli_mul(p1, p2):
    """
    Multiply two Pauli strings encoded as (X_mask, Z_mask).
    Returns (result_pauli, phase).
    """
    X1, Z1 = p1
    X2, Z2 = p2

    X3 = X1 ^ X2
    Z3 = Z1 ^ Z2

    omega = (
        (X1 & Z2).bit_count()
        - (Z1 & X2).bit_count()
    )
    phase = (1j) ** omega

    return (X3, Z3), phase

def commutator_dict(A, B):
    """
    Compute commutator of A and B.
    A, B are dicts: { (X,Z) : coeff }
    where (X, Z) denote the pauli and coeff is a factor of the pauli.
    """
    out = defaultdict(complex)

    for p1, c1 in A.items():
        for p2, c2 in B.items():
            pauli_12, phase_12 = pauli_mul(p1, p2)
            pauli_21, phase_21 = pauli_mul(p2, p1)

            out[pauli_12] += c1 * c2 * phase_12
            out[pauli_21] -= c2 * c1 * phase_21

    # Prune zeros
    return {p: c for p, c in out.items() if c != 0}

def trace(O1, O2):
    """
    Compute trace of D1*D2 (both pauli operators).
    Tr( O1 O2 ) = 2^N * sum_k coeff1(k) * coeff2(k), Pauli strings orthogonal.
    """
    tr = 0+0j
    if len(O1) < len(O2):
        for pauli, coeff in O1.items():
            if pauli in O2:
                tr += coeff * O2[pauli]
    else:
        for pauli, coeff in O2.items():
            if pauli in O1:
                tr += coeff * O1[pauli]
    return tr

# ----- Helpers to build strings -----

def pauli_from_ops(ops):
    """
    Create pauli dict from operator string dict.
    ops: dict {index: pauli_int}
    to: (X, Z) where (X_i, Z_i) denote the pauli on qubit i by binaries.
    """
    X = 0
    Z = 0
    for i, p in ops.items():
        if p == 1:        # X
            X |= 1 << i
        elif p == 2:      # Y
            X |= 1 << i
            Z |= 1 << i
        elif p == 3:      # Z
            Z |= 1 << i
    return (X, Z)


# ----- General builders for linear system -----

def build_H_and_dH(h, J, lam, B_val=0.0, Bp_val=0.0):
    """
    Create Hamiltonian H and derivative dH/dlam from model
    values h, J, lam. B_val and Bp_val . The last two are
    only necessary for the quantum control pulse in COLD.
    """
    N = len(h)
    H = defaultdict(complex)
    dH = defaultdict(complex)

    # X-field
    for i in range(N):
        pX = pauli_from_ops({i: 1})
        H[pX] += -(1 - lam)
        dH[pX] += 1.0

    # ZZ couplings
    for i in range(N):
        for j in range(i + 1, N):
            Jij = J[i, j]
            if Jij != 0:
                pZZ = pauli_from_ops({i: 3, j: 3})
                H[pZZ] += lam * Jij
                dH[pZZ] += Jij

    # local Z fields
    for i in range(N):
        hi = h[i]
        if hi != 0:
            pZ = pauli_from_ops({i: 3})
            H[pZ] += lam * hi
            dH[pZ] += hi

    # global Z field
    if B_val != 0.0 or Bp_val != 0.0:
        for i in range(N):
            pZ = pauli_from_ops({i: 3})
            H[pZ] += B_val
            dH[pZ] += Bp_val

    return dict(H), dict(dH)

def build_AGP_templates_NC(h, J):
    """
    Build AGP ansatz from nested commutators 1st order.
    A_i = -2 [ h_i Y_i + sum_{j<i} J_ij ( Z_i Y_j + Y_i Z_j ) ]
    """
    N = len(h)
    A = []

    for i in range(N):
        A_i = {}

        # -2 h_i Y_i
        if h[i] != 0.0:
            pY = pauli_from_ops({i: 2})
            A_i[pY] = A_i.get(pY, 0.0) - 2.0 * h[i]

        # -2 J_ij (Z_i Y_j + Y_i Z_j)
        for j in range(i):
            Jij = J[i, j]
            if Jij == 0.0:
                continue

            pZiYj = pauli_from_ops({i: 3, j: 2})
            pYiZj = pauli_from_ops({i: 2, j: 3})

            A_i[pZiYj] = A_i.get(pZiYj, 0.0) - 2.0 * Jij
            A_i[pYiZj] = A_i.get(pYiZj, 0.0) - 2.0 * Jij

        A.append(A_i)

    return A

def build_AGP_templates(N, uniform=False):
    """
    Build AGP ansatz for 2nd order with uniform or non-uniform parameters
    """
    A = []

    # Non-uniform case
    if not uniform:
        # alpha: Y_i
        for i in range(N):
            A_i = {pauli_from_ops({i: 2}): 1.0}
            A.append(A_i)

        # gamma: X/Y
        for i in range(N):
            A_i = defaultdict(complex)
            for j in range(N):
                if j == i:
                    continue
                if j > i:
                    A_i[pauli_from_ops({i: 1, j: 2})] += 1.0
                else:
                    A_i[pauli_from_ops({j: 1, i: 2})] += 1.0
            A.append(dict(A_i))

        # chi: Z/Y
        for i in range(N):
            A_i = defaultdict(complex)
            for j in range(N):
                if j == i:
                    continue
                if j > i:
                    A_i[pauli_from_ops({i: 3, j: 2})] += 1.0
                else:
                    A_i[pauli_from_ops({j: 3, i: 2})] += 1.0
            A.append(dict(A_i))

    # Uniform case
    else:
        A_alpha = defaultdict(complex)
        A_gamma = defaultdict(complex)
        A_chi   = defaultdict(complex)

        for i in range(N):
            A_alpha[pauli_from_ops({i: 2})] += 1.0
            for j in range(N):
                if j == i:
                    continue
                A_gamma[pauli_from_ops({min(i,j): 1, max(i,j): 2})] += 1.0
                A_chi[pauli_from_ops({min(i,j): 3, max(i,j): 2})] += 1.0

        A = [dict(A_alpha), dict(A_gamma), dict(A_chi)]

    return A

def build_Hg_from_templates(h, J, lam, B_val, Bp_val, A_lam):
    """
    Given templates A_lam (dict operators), build:
      C = [A_lam, H(lam)]
      Hmat_ij = Re Tr(C_i C_j)
      gvec = Re Tr(i dH/dlam * C)
    """
    H, dH = build_H_and_dH(h, J, lam, B_val=B_val, Bp_val=Bp_val)
    C = [commutator_dict(A, H) for A in A_lam]

    # Build H and gvec for linear system
    P = len(A_lam)
    Hmat = np.zeros((P, P), dtype=float)
    gvec = np.zeros(P, dtype=float)

    for i in range(P):
        gvec[i] = float(np.real(1j * trace(dH, C[i])))

        for j in range(i, P):
            val = float(np.real(trace(C[i], C[j])))
            Hmat[i, j] = val
            Hmat[j, i] = val

    return Hmat, gvec

# ----- Solvers -----
def solve_params(Hmat, gvec):
    """
    Solve Hmat x = gvec.
    """
    try:
        x = np.linalg.solve(Hmat, gvec)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(Hmat, gvec, rcond=None)[0]
    return x
def solve_alpha(h, J, lam, B_val=0.0, Bp_val=0.0):
    """
    Solve minimal action for first order AGP.
    Returns alpha array of length N (non-uniform).
    """
    N = len(h)

    # Build AGP from NC ansatz
    A = build_AGP_templates_NC(h, J)

    # Build H and dH
    H, dH = build_H_and_dH(h, J, lam, B_val=B_val, Bp_val=Bp_val)

    # Compute commutators
    C_list = [commutator_dict(A_i, H) for A_i in A]

    # Create linear system Hx = g
    Hmat = np.zeros((N, N), dtype=float)
    gvec = np.zeros(N, dtype=float)
    for i in range(N):
        C = C_list[i]
        # Tr{dH * i[A_lam, H]}
        gvec[i] = np.real(1j * trace(dH, C))
        # Tr{[A_lam, H] * [A_lam, H]}
        for j in range(i, N):
            val = np.real(trace(C, C_list[j]))
            Hmat[i, j] = val
            Hmat[j, i] = val

    # Solve
    alpha = solve_params(Hmat, gvec)
    return alpha

def solve_alpha_gamma_chi(h, J, lam, B_val=0.0, Bp_val=0.0, uniform=False):
    """
    Solve minimal action for 2nd order AGP (leading two three parameters alpha, gamma, chi).

      - uniform=False: each is length N
      - uniform=True: each is length N with identical entries

    """
    N = len(h)
    # Build AGP 
    A  = build_AGP_templates(N, uniform=uniform)
    # Create arrays for linear system
    Hmat, gvec = build_Hg_from_templates(h, J, lam, B_val, Bp_val, A)
    # Solve
    x = solve_params(Hmat, gvec)
    if uniform:
        a, g, c = map(float, x)
        return np.full(N, a), np.full(N, g), np.full(N, c)
    else:
        return x[:N], x[N:2*N], x[2*N:3*N]