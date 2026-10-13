import random
from scipy.sparse import coo_array
import numpy as np
from qrisp import QuantumVariable, prepare_sparse, measure, jaspify
import pytest


def _int_to_binary(i, width):
    r"""Convert an integer to a binary tuple and string."""
    bits = []
    while i > 0:
        bits.append(int(i & 1))
        i >>= 1
    bits.extend([0] * (width - len(bits)))

    return tuple(reversed(bits)), "".join(str(b) for b in reversed(bits))


def _reverse_sparse_array(x, width):
    def _reverse_bits(i, width):
        res = 0
        for _ in range(width):
            res = (res << 1) | (i & 1)
            i >>= 1
        return res

    ind = x.coords[0]
    val = x.data

    ind_new = []
    for i in ind:
        ind_new.append(_reverse_bits(i, width))

    return coo_array((val, (ind_new,)), shape=(2**width,))


def _create_sparse_state(n, S):
    """create a random sparse vector of length 2**n and with S non-zero entries"""
    random.seed(42)
    np.random.seed(42)

    x = np.random.random(S) * np.exp(1j * 2 * np.pi * np.random.random(S))
    x /= np.sum(np.abs(x) ** 2) ** 0.5
    indices = random.sample(list(range(2**n)), S)

    target_state_array = coo_array((x, (indices,)), shape=(2**n,))

    target_state_dict1 = {}
    target_state_dict2 = {}
    for i, ind in enumerate(indices):
        t, s = _int_to_binary(ind, n)
        target_state_dict1[t] = x[i]
        target_state_dict2[s] = x[i]

    return target_state_array, target_state_dict1, target_state_dict2


def test_gleinig_hoefler_no_ancilla():
    target_state_array, target_state_dict1, target_state_dict2 = _create_sparse_state(8, 100)

    qv1 = QuantumVariable(8)
    prepare_sparse(qv1, target_state_array, method="gleinig_hoefler")

    qv2 = QuantumVariable(8)
    prepare_sparse(qv2, target_state_dict1, method="gleinig_hoefler")

    qv3 = QuantumVariable(8)
    prepare_sparse(qv3, target_state_dict2, method="gleinig_hoefler")

    qv4 = QuantumVariable(8)
    prepare_sparse(qv4, target_state_array, reverse=True, method="gleinig_hoefler")

    res1 = qv1.qs.statevector_array()
    res2 = qv2.qs.statevector_array()
    res3 = qv3.qs.statevector_array()
    res4 = qv4.qs.statevector_array()

    assert np.max(np.abs(res1 - target_state_array.toarray())) < 1e-6
    assert np.max(np.abs(res2 - target_state_array.toarray())) < 1e-6
    assert np.max(np.abs(res3 - target_state_array.toarray())) < 1e-6

    target_state_array_reversed = _reverse_sparse_array(target_state_array, 8)
    assert np.max(np.abs(res4 - target_state_array_reversed.toarray())) < 1e-6


def test_gleinig_hoefler_with_ancilla():
    target_state_array, target_state_dict1, target_state_dict2 = _create_sparse_state(8, 50)

    qv1 = QuantumVariable(8)
    prepare_sparse(qv1, target_state_array, use_ancillas=True, method="gleinig_hoefler")

    qv2 = QuantumVariable(8)
    prepare_sparse(qv2, target_state_dict1, use_ancillas=True, method="gleinig_hoefler")

    qv3 = QuantumVariable(8)
    prepare_sparse(qv3, target_state_dict2, use_ancillas=True, method="gleinig_hoefler")

    res1 = qv1.qs.compile().statevector_array()[::2]
    res2 = qv2.qs.compile().statevector_array()[::2]
    res3 = qv3.qs.compile().statevector_array()[::2]

    assert np.max(np.abs(res1 - target_state_array.toarray())) < 1e-6
    assert np.max(np.abs(res2 - target_state_array.toarray())) < 1e-6
    assert np.max(np.abs(res3 - target_state_array.toarray())) < 1e-6


def test_gleinig_hoefler_basis_state():
    target_state = {"01": 1.0}
    qv = QuantumVariable(2)

    prepare_sparse(qv, target_state, method="gleinig_hoefler")
    res = qv.qs.statevector_array()

    target_vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    assert np.max(np.abs(res - target_vector)) < 1e-6


def test_gleinig_hoefler_jasp_mode():
    @jaspify
    def main():
        target_state_array, target_state_dict1, target_state_dict2 = _create_sparse_state(5, 10)

        qv1 = QuantumVariable(5)
        prepare_sparse(qv1, target_state_array, method="gleinig_hoefler")
        qv2 = QuantumVariable(5)
        prepare_sparse(qv2, target_state_dict1, method="gleinig_hoefler")
        qv3 = QuantumVariable(5)
        prepare_sparse(qv3, target_state_dict2, method="gleinig_hoefler")

        qv4 = QuantumVariable(5)
        prepare_sparse(qv4, target_state_array, use_ancillas=True, method="gleinig_hoefler")
        qv5 = QuantumVariable(5)
        prepare_sparse(qv5, target_state_dict1, use_ancillas=True, method="gleinig_hoefler")
        qv6 = QuantumVariable(5)
        prepare_sparse(qv6, target_state_dict2, use_ancillas=True, method="gleinig_hoefler")

        return measure(qv1), measure(qv2), measure(qv3), measure(qv4), measure(qv5), measure(qv6)

    main()


def test_error_handling():
    with pytest.raises(ValueError, match="If target_state is array, it must have length"):
        target_state = coo_array(([1.0, 2.0], ([0, 1],)), shape=(10,))
        qv = QuantumVariable(3)
        prepare_sparse(qv, target_state)
    with pytest.raises(ValueError, match="If target_state is a dictionary, its keys must be either tuples or strings"):
        target_state = {1: 1.0, 2: 0.5}
        qv = QuantumVariable(3)
        prepare_sparse(qv, target_state)
    with pytest.raises(ValueError, match="If target_state is a dictionary with tuples or strings as keys"):
        target_state = {(0, 1): 0.5**0.5, (1, 1): 0.5**0.5}
        qv = QuantumVariable(3)
        prepare_sparse(qv, target_state)
    with pytest.raises(ValueError, match="target_state must be either of type scipy.sparse.coo_array or dict"):
        target_state = [1, 2, 3]
        qv = QuantumVariable(3)
        prepare_sparse(qv, target_state)
    with pytest.raises(ValueError, match="Method xyz not known."):
        target_state = coo_array(([1.0, 2.0], ([0, 1],)), shape=(8,))
        qv = QuantumVariable(3)
        prepare_sparse(qv, target_state, method="xyz")
