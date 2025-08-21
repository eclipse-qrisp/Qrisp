from qrisp.operators import *
from qrisp import *
from qrisp.algorithms.grover import*
import numpy as np
import scipy

def inner_lanczos(H, D, state_prep_func):
    """
    Perform the quantum subroutine of the exact and efficient Lanczos method to estimate Chebyshev polynomials of a Hamiltonian.

    This function implements the Krylov space construction via block-encodings 
    of Chebyshev polynomials T_k(H), as described in "Exact and efficient Lanczos method 
    on a quantum computer" (arXiv:2208.00567). 
    
    At each order k = 0, ..., 2D-1 it prepares and measures circuits corresponding 
    either to $\bra{\psi\lfloor k/2\rfloor}R\ket{\psi\lfloor k/2\rfloor}$ for even k, or
    $\bra{\psi\lfloor k/2\rfloor U\ket{\psi\lfoor k/2\rfloor}$ for odd k. 
    The measured statistics encode the expectation values $⟨T_k(H)⟩$. 

    Parameters
    ----------
    H : Hamiltonian (QubitOperator)
        Hamiltonian represented as a Pauli sum operator with block-encoding accessible via `unitaries()`. 
    D : int
        Krylov space dimension. Determines maximum Chebyshev order (2D-1).
    state_prep_func : callable 
        Function preparing the initial system state (operand) $\ket{\psi_0}$ on a QuantumVariable.

    Returns
    -------
    meas_res : dictionary
        Measurement results for each Chebyshev polynomial order, suitable for conversion into expectation values.
    """

    # Extract unitaries and size of the case_indicator QuantumFloat.
    unitaries, coeffs = H.unitaries()
    num_unitaries = len(unitaries)
    n = np.int64(np.ceil(np.log2(num_unitaries)))

    # Prepare $\ket{G} = \sum_i \sqrt{\alpha_i}\ket{i}$ on the auxiliary register (case_indicator) (paper Def. 1, Eq. (4))
    def state_prep(case):
        prepare(case, np.sqrt(coeffs))

    def UR(case_indicator, operand, unitaries):
        qswitch(operand, case_indicator, unitaries) #qswitch applies $U = \sum_i\ket{i}\bra{i}\otimes P_i$.
        diffuser(case_indicator, state_function=state_prep) # diffuser implements the reflection operator R about $\ket{G}$.

    meas_res = {}
 
    for k in jrange(0, 2*D):
        case_indicator = QuantumFloat(n)
        operand = QuantumVariable(H.find_minimal_qubit_amount())
        state_prep_func(operand) # prepare operand $\ket{\psi_0}$

        if k % 2 == 0:
            # EVEN k: Figure 1 top
            with conjugate(state_prep)(case_indicator):
                for _ in jrange(k//2):
                    UR(case_indicator, operand, unitaries)
            
            meas = case_indicator.get_measurement()
            meas_res[k] = meas 

        else:
            # ODD k: Figure 1 bottom
            state_prep(case_indicator)
            for _ in jrange(k//2):
                UR(case_indicator, operand, unitaries)
            qv = QuantumVariable(1)
            h(qv) # Hadamard test for <U>
            with control(qv[0]):
                qswitch(operand, case_indicator, unitaries) # control-U on the case_indicator QuantumFloat
            h(qv) # Hadamard test for <U>
            meas = qv.get_measurement()
            meas_res[k] = meas
    
        case_indicator.delete()
        operand.delete()

    return meas_res

def compute_expectation(counts):
    """
    Convert measurement counts into an expectation value.

    Assumes measurement outcomes correspond to ±1 eigenvalues of observables 
    (reflection R or block-encoding unitary U). 

    For even k we measure the auxilary case_indicator QuantumFloat in the computational basis. We then map:
    +1 if all-zeroes outcome (projector 2|0><0| - I), else -1 (paper step 2).

    For odd k we perform the hadamard test for the unitary on the controlled unitary and then measure the 
    auxilary case_indicator QuantumFloat in the Z basis

    Parameters
    ----------
    counts : dictionary
        Measurement counts or probabilities, keyed by outcomes.

    Returns
    -------
    expval : float
        Expectation value of the measured observable.
    """
    expval = 0.0
    for outcome, prob in counts.items():

        if int(outcome) == 0:
            expval += prob * 1
        else:
            expval += prob * (-1)
    return expval

def build_S_H_from_Tk(Tk_expectation, D):
    """
    Construct Lanczos overlap matrix S and Hamiltonian matrix H from Chebyshev polynomials.

    Uses Chebyshev recurrence identities to compute matrix elements
    (Eq. (17), (19) in the reference paper) from measured expectations.

    Parameters
    ----------
    Tk_expectation : dictionary
        Mapping k to ⟨T_k(H)⟩.
    
    D : integer
        Krylov space dimension.

    Returns
    -------
    S : ndarray
        overlap (Gram) matrix S or Krylov states.
 
    H_mat : ndarray
        Hamiltonian matrix in Krylov subspace.
    """
    def Tk(k):
        k = abs(k)
        return Tk_expectation.get(k, 0)

    S = np.zeros((D, D))
    H_mat = np.zeros((D, D))

    for i in range(D):
        for j in range(D):
            S[i, j] = 0.5 * (Tk(i + j) + Tk(abs(i - j)))
            H_mat[i, j] = 0.25 * (
                Tk(i + j + 1)
                + Tk(abs(i + j - 1))
                + Tk(abs(i - j + 1))
                + Tk(abs(i - j - 1))
            )
    return S, H_mat

def regularize_S_H(S, H_mat, cutoff=1e-3):
    """
    Regularize overlap matrix S by thresholding eigenvalues below cutoff * max_eigenvalue.
    Project both S and H onto subspace defined by large eigenvalues.

    Parameters
    ----------
    S : Overlap matrix
    H_mat : Hamiltonian matrix
    cutoff : float
        Eigenvalue threshold for regularizing S

    Returns
    -------
    S_reg : ndarray
        regularized overlap matrix S.
    H_reg : ndarray
        regularized Hamiltonian matrix in Krylov subspace.
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    max_eigval = eigvals.max()
    threshold = cutoff * max_eigval
    mask = eigvals > threshold
    U = eigvecs[:, mask]
    S_reg = U.T @ S @ U
    H_reg = U.T @ H @ U
    return S_reg, H_reg

def lanczos_alg(H, D, state_prep_func, cutoff=1e-2):
    """
    Exact and efficient Lanczos method on a quantum computer for ground state energy estimation.

    Implements the following steps:
      1. Run quantum Lanczos subroutine to obtain Chebyshev expectation values.
      2. Build overlap and Hamiltonian subspace matrices (S, H).
      3. Regularize overlap matrix S and H.
      4. Solve generalized eigenvalue problem Hv=ESv.
      5. Return lowest eigenvalue as ground state energy estimate.

    Parameters
    ----------
        H : Hamiltonian (QuantumOperator)
            Hamiltonian operator expressed as the Pauli sum.
        D : int
            Krylov space dimension.
        state_prep_func : callable 
            Function preparing the initial system state (operand) $\ket{\psi_0}$ on a QuantumVariable.
        cutoff : float
            Regularization cutoff threshold for overlap matrix S.

    Returns
    -------
    energy : float
        Estimated ground state energy of the Hamiltonian H.
    results : dictionary
        Full details including:
            - 'Tk_expvals': dict of Chebyshev expectation values
            - 'energy': ground state estimate
            - 'eigvals': eigenvalues of regularized problem
            - 'eigvecs': eigenvectors
            - 'S_reg': regularized overlap matrix
            - 'H_reg': regularized Hamiltonian matrix
    """
    unitaries, coeffs = H.unitaries()
    
    # Step 1: Quantum Lanczos: Get expectation values of Chebyshev polynomials
    meas_counts = inner_lanczos(H, D, state_prep_func)

    # Step 2: Convert counts to expectation values
    Tk_expvals = {k: compute_expectation(counts) for k, counts in meas_counts.items()}

    # Step 3: Build matrices S and H
    S, H_mat = build_S_H_from_Tk(Tk_expvals, D)

    # Step 4: Regularize matrices via thresholding
    S_reg, H_reg = regularize_S_H(S, H_mat, cutoff=cutoff)

    # Step 5: Solve generalized eigenvalue problem Hv = ESv
    evals, evecs = scipy.linalg.eigh(H_reg, S_reg)

    ground_state_energy = np.min(evals)*sum(coeffs)

    results = {
        'Tk_expvals': Tk_expvals,
        'energy': ground_state_energy,
        'eigvals': evals,
        'eigvecs': evecs,
        'S_reg': S_reg,
        'H_reg': H_reg,
    }
    return ground_state_energy, results