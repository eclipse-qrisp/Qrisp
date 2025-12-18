
# unified_agp.py

import numpy as np
from collections import defaultdict

# Pauli encoding: 0=I, 1=X, 2=Y, 3=Z

# ----- Pauli algebra (dict backend) -----

def mul_site(p, q):
    # returns (result_pauli, phase) where phase in {1, -1, 1j, -1j}
    if p == 0:  # I * q
        return q, 1
    if q == 0:  # p * I
        return p, 1
    if p == q:  # same Pauli -> I
        return 0, 1
    # Use XY=iZ, YZ=iX, ZX=iY and reverse signs for swapped order
    if p == 1 and q == 2:   # X*Y = i Z
        return 3, 1j
    if p == 2 and q == 1:   # Y*X = -i Z
        return 3, -1j
    if p == 2 and q == 3:   # Y*Z = i X
        return 1, 1j
    if p == 3 and q == 2:   # Z*Y = -i X
        return 1, -1j
    if p == 3 and q == 1:   # Z*X = i Y
        return 2, 1j
    if p == 1 and q == 3:   # X*Z = -i Y
        return 2, -1j
    raise ValueError("Invalid Pauli pair")

def multiply_strings(s1, s2):
    # s1, s2 are tuples of ints length N; returns (result_tuple, phase)
    phase = 1
    N = len(s1)
    out = [0] * N
    for k in range(N):
        r, ph = mul_site(s1[k], s2[k])
        out[k] = r
        phase *= ph
    return tuple(out), phase

def add_to_dict(D, key, coeff):
    if coeff == 0:
        return
    D[key] = D.get(key, 0) + coeff
    if D[key] == 0:
        D.pop(key)

def commutator_dict(A, B):
    # [A, B] = A B - B A, with A,B as dicts {string: coeff}
    out = defaultdict(complex)
    for s1, c1 in A.items():
        for s2, c2 in B.items():
            s12, ph12 = multiply_strings(s1, s2)
            s21, ph21 = multiply_strings(s2, s1)
            add_to_dict(out, s12, c1 * c2 * ph12)
            add_to_dict(out, s21, -c2 * c1 * ph21)
    return dict(out)

def scale_dict(D, scalar):
    return {k: scalar * v for k, v in D.items()}

def add_dicts(*dicts):
    out = defaultdict(complex)
    for D in dicts:
        for k, v in D.items():
            out[k] += v
    return dict(out)

def trace_inner(D1, D2, N):
    # Tr( O1 O2 ) = 2^N * sum_k coeff1(k) * coeff2(k), Pauli strings orthogonal
    s = 0+0j
    if len(D1) < len(D2):
        for k, v in D1.items():
            if k in D2:
                s += v * D2[k]
    else:
        for k, v in D2.items():
            if k in D1:
                s += v * D1[k]
    return (2 ** N) * s

# ----- Helpers to build strings -----

def basis_string(N, ops):
    # ops: dict {index: pauli_int}, others are I
    s = [0] * N
    for i, p in ops.items():
        s[i] = p
    return tuple(s)

# ----- Model: H and dH/dlam -----

def build_H_and_dH(h, J, lam, B_val=0.0, Bp_val=0.0):
    """
    Build H_beta(lam) and dH/dlam in dict form:
      H(lam) = -(1-lam) sum_i X_i

             + lam sum_{i<j} J_ij Z_i Z_j
             + lam sum_i h_i Z_i

             + B_val sum_i Z_i

      dH/dlam = + sum_i X_i
                + sum_{i<j} J_ij Z_i Z_j

                + sum_i h_i Z_i
                + Bp_val sum_i Z_i     # if B depends on lam

    """
    N = len(h)
    H = defaultdict(complex)
    dH = defaultdict(complex)

    # X-field
    for i in range(N):
        sX = basis_string(N, {i: 1})
        add_to_dict(H, sX, -(1 - lam))
        add_to_dict(dH, sX, 1.0)

    # ZZ couplings
    for i in range(N):
        for j in range(i + 1, N):
            Jij = J[i, j]
            if Jij == 0:
                continue
            sZZ = basis_string(N, {i: 3, j: 3})
            add_to_dict(H, sZZ, lam * Jij)
            add_to_dict(dH, sZZ, Jij)

    # Z local fields h_i
    for i in range(N):
        hi = h[i]
        if hi != 0:
            sZ = basis_string(N, {i: 3})
            add_to_dict(H, sZ, lam * hi)
            add_to_dict(dH, sZ, hi)

    # Z global field B(lam)
    if B_val != 0.0 or Bp_val != 0.0:
        for i in range(N):
            sZ = basis_string(N, {i: 3})
            if B_val != 0.0:
                add_to_dict(H, sZ, B_val)
            if Bp_val != 0.0:
                add_to_dict(dH, sZ, Bp_val)

    return dict(H), dict(dH)

# ----- Templates -----

def build_B_templates(h, J):
    """
    Build B_l templates (alpha-only ansatz): Q_l = B_l
    B_l = -2 [ h_l Y_l + sum_{j<l} J_{l j} ( Z_l Y_j + Y_l Z_j ) ]
    """
    N = len(h)
    Q_list, labels = [], []
    for l in range(N):
        Bl = defaultdict(complex)
        hl = h[l]
        if hl != 0:
            sYl = basis_string(N, {l: 2})
            add_to_dict(Bl, sYl, -2.0 * hl)
        for j in range(l):
            Jij = J[l, j]
            if Jij == 0:
                continue
            sZlYj = basis_string(N, {l: 3, j: 2})
            sYlZj = basis_string(N, {l: 2, j: 3})
            add_to_dict(Bl, sZlYj, -2.0 * Jij)
            add_to_dict(Bl, sYlZj, -2.0 * Jij)
        Q_list.append(dict(Bl))
        labels.append(('alpha', l))
    return Q_list, labels

def build_AGP_templates(N, uniform=False):
    """
    Three-parameter ansatz templates in dict form:
      alpha: Y_i
      gamma: sum_{j>i} X_i Y_j + sum_{j<i} X_j Y_i
      chi:   sum_{j>i} Z_i Y_j + sum_{j<i} Z_j Y_i
    If uniform=True, returns 3 templates (sums over sites).
    If uniform=False, returns 3N templates, one per site.
    """
    def add_two_site(acc, i, pi, j, pj, coef=1.0):
        s = basis_string(N, {i: pi, j: pj})
        add_to_dict(acc, s, coef)

    if not uniform:
        Q_list, labels = [], []
        # alpha: Y_i
        for i in range(N):
            Qi = {basis_string(N, {i: 2}): 1.0}
            Q_list.append(Qi)
            labels.append(('alpha', i))
        # gamma: X/Y mixing
        for i in range(N):
            Qi = defaultdict(complex)
            for j in range(i+1, N):
                add_two_site(Qi, i, 1, j, 2, 1.0)  # X_i Y_j
            for j in range(0, i):
                add_two_site(Qi, j, 1, i, 2, 1.0)  # X_j Y_i
            Q_list.append(dict(Qi))
            labels.append(('gamma', i))
        # chi: Z/Y mixing
        for i in range(N):
            Qi = defaultdict(complex)
            for j in range(i+1, N):
                add_two_site(Qi, i, 3, j, 2, 1.0)  # Z_i Y_j
            for j in range(0, i):
                add_two_site(Qi, j, 3, i, 2, 1.0)  # Z_j Y_i
            Q_list.append(dict(Qi))
            labels.append(('chi', i))
        return Q_list, labels

    # uniform=True: sum over sites for each type
    Q_alpha = defaultdict(complex)
    for i in range(N):
        add_to_dict(Q_alpha, basis_string(N, {i: 2}), 1.0)

    Q_gamma = defaultdict(complex)
    for i in range(N):
        for j in range(i+1, N):
            add_to_dict(Q_gamma, basis_string(N, {i: 1, j: 2}), 1.0)
        for j in range(0, i):
            add_to_dict(Q_gamma, basis_string(N, {j: 1, i: 2}), 1.0)

    Q_chi = defaultdict(complex)
    for i in range(N):
        for j in range(i+1, N):
            add_to_dict(Q_chi, basis_string(N, {i: 3, j: 2}), 1.0)
        for j in range(0, i):
            add_to_dict(Q_chi, basis_string(N, {j: 3, i: 2}), 1.0)

    Q_list = [dict(Q_alpha), dict(Q_gamma), dict(Q_chi)]
    labels = ['alpha', 'gamma', 'chi']
    return Q_list, labels

# ----- General builder: Hmat and gvec -----

def build_Hg_from_templates(h, J, lam, B_val, Bp_val, Q_list):
    """
    Given templates Q_p (dict operators), build:
      C_p = [Q_p, H(lam)]
      Hmat_{pq} = Re Tr(C_p C_q)
      gvec_p    = Re Tr(i (dH/dlam) C_p)
    """
    N = len(h)
    H, dH = build_H_and_dH(h, J, lam, B_val=B_val, Bp_val=Bp_val)
    C_list = [commutator_dict(Q, H) for Q in Q_list]
    P = len(Q_list)
    Hmat = np.zeros((P, P), dtype=float)
    gvec = np.zeros(P, dtype=float)
    for p in range(P):
        Cp = C_list[p]
        g = 1j * trace_inner(dH, Cp, N)
        gvec[p] = float(np.real(g))
        for q in range(P):
            Cq = C_list[q]
            Hmat[p, q] = float(np.real(trace_inner(Cp, Cq, N)))
    return Hmat, gvec

# ----- Solvers -----

def solve_params(Hmat, gvec, ridge=0.0):
    """
    Solve Hmat x = gvec, with optional ridge regularization.
    """
    if ridge > 0.0:
        Hmat = Hmat + ridge * np.eye(Hmat.shape[0])
    try:
        x = np.linalg.solve(Hmat, gvec)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(Hmat, gvec, rcond=None)[0]
    return x

def solve_alpha(h, J, lam, B_val=0.0, Bp_val=0.0, ridge=0.0):
    """
    Alpha-only ansatz using B_l templates. Returns alpha (length N).
    Matches the linear system convention used in your first script.
    """
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    N = len(h)
    Q_list, _ = build_B_templates(h, J)
    Hmat, gvec = build_Hg_from_templates(h, J, lam, B_val, Bp_val, Q_list)
    x = solve_params(Hmat, gvec, ridge=ridge)
    alpha = x[:N]
    return alpha

def solve_alpha_gamma_chi(h, J, lam, B_val=0.0, Bp_val=0.0, uniform=False, ridge=0.0):
    """
    Three-parameter ansatz. Returns alpha, gamma, chi:

      - uniform=False: each is length N
      - uniform=True: each is length N with identical entries

    """
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    N = len(h)
    Q_list, labels = build_AGP_templates(N, uniform=uniform)
    Hmat, gvec = build_Hg_from_templates(h, J, lam, B_val, Bp_val, Q_list)
    x = solve_params(Hmat, gvec, ridge=ridge)
    if uniform:
        a, g, c = map(float, x.tolist())
        alpha = [a] * N
        gamma = [g] * N
        chi   = [c] * N
        return np.array(alpha), np.array(gamma), np.array(chi)
    else:
        alpha = x[0:N]
        gamma = x[N:2*N]
        chi   = x[2*N:3*N]
        return alpha, gamma, chi