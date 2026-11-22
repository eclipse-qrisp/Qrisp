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

"""Tests for LinearCombinationBlockEncoding semantics, validation, and compilation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import compare_distributions, postselect_results, sample_postselected
from jax.tree_util import tree_flatten, tree_unflatten

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    h,
    invert,
    jaspify,
    make_jaspr,
    measure,
    multi_measurement,
    s_dg,
    terminal_sampling,
    x,
    z,
)
from qrisp.block_encodings import BlockEncoding, LinearCombinationBlockEncoding
from qrisp.jasp import count_ops
from qrisp.operators import X, Y, Z


def _trace_apply(encoding, operand_size=3):
    def circuit():
        operand = QuantumFloat(operand_size)
        encoding.apply(operand)
        return operand

    return make_jaspr(circuit)()


#
# Structure, flattening, immutability, ancillas, and dagger
#


def test_flattened_linear_combination_and_pytree():
    """Linear combinations flatten into one LCU and round-trip through PyTree flatten/unflatten."""
    H1 = X(0) * X(1)
    H2 = 0.4 * Y(0) + Z(1)
    H3 = 0.7 * X(2)
    coefficients = [2.0, -1.0, 0.5]

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    BE3 = BlockEncoding.from_operator(H3)

    BE_chain = coefficients[0] * BE1 + coefficients[1] * BE2 + coefficients[2] * BE3
    BE_direct = BlockEncoding.linear_combination([BE1, BE2, BE3], coefficients=coefficients)

    assert isinstance(BE_chain, LinearCombinationBlockEncoding)
    assert isinstance(BE_direct, LinearCombinationBlockEncoding)
    assert len(BE_chain.terms) == 3
    assert len(BE_direct.terms) == 3
    assert BE_chain._anc_templates[0].qv_size == 2
    assert BE_chain.num_ancs == 2

    leaves, treedef = tree_flatten(BE_chain)
    reconstructed = tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, LinearCombinationBlockEncoding)
    assert len(reconstructed.terms) == 3
    assert len((reconstructed + BE1).terms) == 4

    H_target = coefficients[0] * H1 + coefficients[1] * H2 + coefficients[2] * H3
    BE_target = BlockEncoding.from_operator(H_target)
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount(), H3.find_minimal_qubit_amount())
    prep = lambda: QuantumVariable(n)  # noqa: E731

    res_target = sample_postselected(BE_target, prep)
    compare_distributions(res_target, sample_postselected(BE_chain, prep), n)
    compare_distributions(res_target, sample_postselected(BE_direct, prep), n)


def test_terms_are_kept_as_written():
    """Duplicate, zero-weighted, and cancelling terms are preserved without algebraic simplification."""
    be = BlockEncoding(1, [], lambda operand: None)
    other = BlockEncoding(1, [], lambda operand: None)

    repeated = BlockEncoding.linear_combination([be, be], coefficients=[2, 3])
    assert repeated.terms == ((2, be), (3, be))
    assert repeated.alpha == 5

    with_zero = BlockEncoding.linear_combination([be, other], coefficients=[0, 2])
    assert with_zero.terms == ((0, be), (2, other))

    cancelling = BlockEncoding.linear_combination([be, other], coefficients=[1, -1])
    assert cancelling.alpha == 2


def test_immutable_representation():
    """LCU terms and derived attributes are exposed immutably."""
    first = BlockEncoding(2, [], lambda operand: None, is_hermitian=True)
    second = BlockEncoding(3, [], lambda operand: None, is_hermitian=True)
    combination = BlockEncoding.linear_combination([first, second], coefficients=[4, -1])

    assert isinstance(combination, LinearCombinationBlockEncoding)
    assert isinstance(combination.terms, tuple)
    assert combination.alpha == 11
    assert combination.num_ops == 1
    assert combination.num_ancs == 1
    assert combination.is_hermitian

    with pytest.raises(TypeError):
        combination.terms[0] = (1, first)
    with pytest.raises(AttributeError):
        combination.terms = ()
    with pytest.raises(AttributeError):
        combination._terms = ()


def test_no_workspace_and_heterogeneous_ancillas():
    """LCU omits the workspace when no child has ancillas and shares one workspace otherwise."""
    be_no_anc = BlockEncoding(1, [], lambda qv: qv.__iadd__(1))
    be_sum = be_no_anc + be_no_anc
    qv = QuantumFloat(3)
    ancs = be_sum.apply(qv)
    assert len(ancs) == 1
    assert multi_measurement([qv] + ancs) == {(1, 0): 1.0}

    def two_anc_unitary(float_anc, bool_anc, operand):
        x(float_anc)
        x(bool_anc)

    be_two = BlockEncoding(1, [QuantumFloat(2), QuantumBool()], two_anc_unitary)
    be_one = BlockEncoding(1, [QuantumFloat(1)], lambda anc, operand: x(anc))
    be_het = BlockEncoding.linear_combination([be_two, be_one])

    assert be_het.num_ancs == 2
    assert be_het._anc_templates[1].qv_size == 3
    be_het.apply(QuantumVariable(1))

    @jaspify
    def main(block_encoding):
        operand = QuantumVariable(1)
        block_encoding.apply(operand)
        return measure(operand)

    assert main(be_het) == 0


def test_dagger_conjugates_terms_and_flattens():
    """LinearCombinationBlockEncoding.dagger conjugates coefficients and child encodings."""
    A = BlockEncoding.from_operator(X(0))
    B = BlockEncoding.from_operator(Y(0))
    C = BlockEncoding.from_operator(Z(0))

    combo = (1 + 2j) * A + 3.0 * B
    dag = combo.dagger()

    assert isinstance(dag, LinearCombinationBlockEncoding)
    assert [coeff for coeff, _ in dag.terms] == [1 - 2j, 3.0]
    assert np.isclose(dag.alpha, combo.alpha)
    assert len((dag + C).terms) == 3

    U = BlockEncoding.from_eye(1)
    V = BlockEncoding.from_eye(-1)
    non_herm = (1 + 1j) * U + 2.0 * V
    expected = (1 - 1j) * U.dagger() + 2.0 * V.dagger()
    prep = lambda: QuantumFloat(2)  # noqa: E731

    compare_distributions(sample_postselected(non_herm.dagger(), prep), sample_postselected(expected, prep), 2)


#
# Validation and coefficient handling
#


def test_input_validation():
    """Invalid term counts, types, operand counts, and non-scalar coefficients are rejected."""
    be1 = BlockEncoding(1, [], lambda a: None, num_ops=1)
    be2 = BlockEncoding(1, [], lambda a, b: None, num_ops=2)

    for empty_call in (
        lambda: LinearCombinationBlockEncoding([]),
        lambda: BlockEncoding._from_lcu_terms(()),
        lambda: BlockEncoding.linear_combination([]),
    ):
        with pytest.raises(ValueError, match="At least one block-encoding"):
            empty_call()

    with pytest.raises(ValueError, match="number of coefficients"):
        BlockEncoding.linear_combination([be1], coefficients=[1, 2])
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        BlockEncoding.linear_combination([be1], coefficients=2.0)
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        BlockEncoding.linear_combination([be1], coefficients=np.array([[1.0]]))
    with pytest.raises(TypeError, match="BlockEncoding"):
        BlockEncoding.linear_combination([object()])
    with pytest.raises(TypeError, match="BlockEncoding"):
        LinearCombinationBlockEncoding([(1, "not a block encoding")])
    with pytest.raises(ValueError, match="same number of operands"):
        BlockEncoding.linear_combination([be1, be2], coefficients=[1, 1])
    with pytest.raises(ValueError, match="same number of operands"):
        BlockEncoding.linear_combination([be1, be2], coefficients=[1, 0])

    for bad_coeff in (
        lambda: BlockEncoding.linear_combination([be1], coefficients=[np.array([1.0, 2.0])]),
        lambda: be1 * np.array([1.0, 2.0]),
        lambda: BlockEncoding.linear_combination([be1], coefficients=[jnp.array([1.0, 2.0])]),
        lambda: make_jaspr(lambda c: c * be1)(jnp.array([1.0, 2.0])),
        lambda: BlockEncoding.linear_combination([be1, be1], coefficients=[[1.0], [2.0]]),
    ):
        with pytest.raises(ValueError, match="scalar"):
            bad_coeff()


def test_numpy_coefficient_is_detached_and_jax_array_preserved():
    """Mutable 0-d NumPy coefficients are detached to Python scalars while JAX arrays pass through."""
    first = BlockEncoding(1, [], lambda operand: x(operand[0]))
    second = BlockEncoding(1, [], lambda operand: x(operand[1]))

    mutable_coeff = np.array(2.0)
    combination = BlockEncoding.linear_combination([first, second], coefficients=[mutable_coeff, jnp.array(1.0)])
    alpha_before = combination.alpha
    mutable_coeff[...] = 3.0

    assert combination.terms[0][0] == 2.0
    assert type(combination.terms[0][0]) is float
    assert type(combination.terms[1][0]).__name__ == "ArrayImpl"
    assert combination.alpha == alpha_before


def test_dynamic_coefficient_normalization():
    """Traced coefficients compute normalization without Python-level comparisons."""
    be = BlockEncoding(2, [], lambda operand: None)
    assert jax.jit(lambda c: BlockEncoding.linear_combination([be], coefficients=[c]).alpha)(jnp.array(3.0)) == 6.0


#
# Phase and Hermiticity
#


def _phase_kickback_distribution(coefficient, rotate_to_y_basis):
    child = BlockEncoding(1, [], lambda operand: z(operand[0]))
    encoding = BlockEncoding.linear_combination([child], [coefficient])

    @terminal_sampling
    def main():
        control_qbl = QuantumBool()
        h(control_qbl)
        operand = QuantumFloat(3)
        with control(control_qbl[0]):
            encoding.apply(operand)
        if rotate_to_y_basis:
            s_dg(control_qbl[0])
        h(control_qbl)
        return control_qbl

    return main()


@pytest.mark.parametrize(
    "coefficient, argument",
    [
        (1.0, 0.0),
        (-1.0, np.pi),
        (1j, np.pi / 2),
        (-1j, -np.pi / 2),
        ((1 + 1j) / np.sqrt(2), np.pi / 4),
    ],
)
def test_single_term_applies_coefficient_phase(coefficient, argument):
    """A single-term linear combination applies arg(coefficient) as a global phase."""
    x_basis = _phase_kickback_distribution(coefficient, rotate_to_y_basis=False)
    y_basis = _phase_kickback_distribution(coefficient, rotate_to_y_basis=True)

    assert np.isclose(x_basis.get(False, 0), np.cos(argument / 2) ** 2, atol=1e-4)
    assert np.isclose(y_basis.get(False, 0), np.cos((argument - np.pi / 2) / 2) ** 2, atol=1e-4)


def _unitary_hermiticity_error(block_encoding):
    operand = QuantumVariable(1)
    ancillas = block_encoding.create_ancillas()
    block_encoding.unitary(*ancillas, operand)
    unitary = operand.qs.get_unitary()
    return float(np.max(np.abs(unitary - unitary.conj().T)))


@pytest.mark.parametrize(
    "name, build, expected",
    [
        ("non-negative terms", lambda Z_be, X_be: Z_be + X_be, True),
        ("scaled non-negative terms", lambda Z_be, X_be: Z_be + 2.0 * X_be, True),
        ("signed sum", lambda Z_be, X_be: Z_be - X_be, True),
        ("negated single term", lambda Z_be, X_be: -Z_be, True),
        ("complex sum", lambda Z_be, X_be: Z_be + 1j * X_be, False),
        ("complex single term", lambda Z_be, X_be: 1j * Z_be, False),
    ],
)
def test_is_hermitian_matches_unitary(name, build, expected):
    """is_hermitian agrees with the actual block-encoding unitary matrix."""
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))
    encoding = build(Z_be, X_be)

    assert encoding.is_hermitian is expected
    if expected:
        assert _unitary_hermiticity_error(encoding) < 1e-9
    else:
        assert _unitary_hermiticity_error(encoding) > 1e-9


def test_complex_coefficients_use_non_hermitian_qubitization():
    """Qubitization adds a control ancilla for complex-weighted sums and omits it for real ones."""
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))

    real_combo = Z_be + 2.0 * X_be
    complex_combo = Z_be + 1j * X_be

    assert real_combo.qubitization().num_ancs == real_combo.num_ancs
    assert complex_combo.qubitization().num_ancs == complex_combo.num_ancs + 1


def test_traced_coefficient_reality_survives_pytree_boundary():
    """Traced JAX coefficients preserve real vs complex Hermiticity via tracer dtype."""
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))
    observed = {}

    @jaspify
    def main(block_encoding, key):
        assert block_encoding._has_traced_coefficients
        observed[key] = block_encoding.is_hermitian
        qv = QuantumFloat(1)
        block_encoding.apply(qv)
        return measure(qv)

    main(Z_be + jnp.array(2.0) * X_be, "real")
    main(Z_be + jnp.array(1j) * X_be, "complex")

    assert observed["real"] is True
    assert observed["complex"] is False


def test_signed_sum_polynomial_matches_direct_encoding():
    """Polynomial evaluation of a signed sum matches the directly encoded polynomial."""
    operator = Z(0) - X(0) + Z(1)
    combination = (
        BlockEncoding.from_operator(Z(0)) - BlockEncoding.from_operator(X(0)) + BlockEncoding.from_operator(Z(1))
    )
    reference = BlockEncoding.from_operator(operator**2)
    prep = lambda: QuantumVariable(2)  # noqa: E731

    from_combination = sample_postselected(combination.poly(np.array([0.0, 0.0, 1.0])), prep)
    from_reference = sample_postselected(reference, prep)

    for state in (0, 1):
        assert np.isclose(from_combination.get(state, 0), from_reference.get(state, 0), atol=1e-3)


@pytest.mark.parametrize("coefficients", [[1.0, 2.0], [1.0, -2.0], [1.0, 1j]])
def test_prep_amplitudes_are_real_and_non_negative(coefficients):
    """PREP amplitudes are always real and non-negative magnitudes."""
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))
    combination = BlockEncoding.linear_combination([Z_be, X_be], coefficients=coefficients)

    amplitudes = np.asarray(combination._lcu_amplitudes)
    assert np.all(np.isreal(amplitudes))
    assert np.all(np.real(amplitudes) >= 0)


#
# Compilation, tracing, and caching
#


def test_unitary_identity_is_stable():
    """Accessing encoding.unitary repeatedly returns the same callable object."""
    encoding = BlockEncoding.from_eye(0) + BlockEncoding.from_eye(1)
    assert encoding.unitary is encoding.unitary


def test_ancilla_templates_do_not_allocate_qubits_while_tracing():
    """Inspecting _anc_templates inside a trace emits no create_qubits equations."""
    encoding = BlockEncoding.from_eye(0) + BlockEncoding.from_eye(1)
    jaspr = make_jaspr(lambda: encoding._anc_templates and None)()
    assert not any(eqn.primitive.name == "jasp.create_qubits" for eqn in jaspr.eqns)


def test_nested_linear_combinations_do_not_amplify_tracing():
    """Nesting linear combinations and products traces the innermost child exactly once."""
    for nesting_depth in (1, 2, 3):
        executions = []

        def innermost(operand):
            executions.append(None)
            x(operand[0])

        encoding = BlockEncoding(1, [], innermost)
        for _ in range(nesting_depth):
            encoding = BlockEncoding.from_eye() + (encoding @ BlockEncoding(1, [], lambda operand: x(operand[1])))

        _trace_apply(encoding)
        assert len(executions) == 1


def test_repeated_application_traces_child_once():
    """Applying the same linear combination repeatedly traces each child unitary once."""
    executions = []

    def counted(operand):
        executions.append(None)
        x(operand[0])

    encoding = BlockEncoding(1, [], counted) + BlockEncoding(1, [], lambda operand: x(operand[1]))

    def circuit():
        operand = QuantumFloat(3)
        for _ in range(3):
            encoding.apply(operand)
        return operand

    make_jaspr(circuit)()
    assert len(executions) == 1


def test_nested_traced_sum_applied_repeatedly():
    """Nested sums with traced coefficients trace the child once and re-trace cleanly."""
    executions = []

    def counted(operand):
        executions.append(None)
        x(operand[0])

    base = BlockEncoding(1, [], counted)
    other = BlockEncoding(1, [], lambda operand: x(operand[1]))

    def circuit(scale):
        outer = (scale * base + other) + (scale * other)
        operand = QuantumFloat(3)
        outer.apply(operand)
        outer.apply(operand)
        return operand

    make_jaspr(circuit)(0.5)
    assert len(executions) == 1
    make_jaspr(circuit)(0.25)


@pytest.mark.parametrize("via_pytree", [False, True])
def test_concrete_coefficients_use_prepare_qiskit(monkeypatch, via_pytree):
    """Concrete coefficients stay concrete and use prepare_qiskit both when captured and via PyTree."""
    from qrisp.alg_primitives.state_preparation import prepare_func

    methods = []

    def record(name, implementation):
        def wrapper(*args, **kwargs):
            methods.append(name)
            return implementation(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(prepare_func, "prepare_qiskit", record("qiskit", prepare_func.prepare_qiskit))
    monkeypatch.setattr(prepare_func, "prepare_qswitch", record("qswitch", prepare_func.prepare_qswitch))

    encoding = 2 * BlockEncoding.from_eye(0) - BlockEncoding.from_eye(1) - BlockEncoding.from_eye(-1)
    observed = {}

    if via_pytree:

        def main(be):
            observed["coefficients"] = be._lcu_coefficients
            observed["amplitudes"] = be._lcu_amplitudes
            observed["alpha"] = be.alpha
            operand = QuantumFloat(3)
            be.apply(operand)
            return operand

        make_jaspr(main)(encoding)
    else:

        def main():
            observed["coefficients"] = encoding._lcu_coefficients
            observed["amplitudes"] = encoding._lcu_amplitudes
            observed["alpha"] = encoding.alpha
            operand = QuantumFloat(3)
            encoding.apply(operand)
            return operand

        make_jaspr(main)()

    for name, value in observed.items():
        assert not isinstance(value, jax.core.Tracer), f"{name} became a tracer"
    assert np.isclose(observed["alpha"], 4.0)
    assert methods and "qswitch" not in methods


def test_linear_combination_compiles_to_few_equations():
    """A small concrete linear combination compiles to fewer than 100 top-level equations."""
    encoding = 2 * BlockEncoding.from_eye(0) - BlockEncoding.from_eye(1) - BlockEncoding.from_eye(-1)
    assert len(_trace_apply(encoding).eqns) < 100


def test_traced_coefficients_fall_back_to_symbolic_prepare():
    """Dynamic JAX-array coefficients trace through symbolic PREP and match the reference."""
    operator = 0.5 * X(0) + 0.7 * Y(0) + 0.3 * Z(1)
    encoding = BlockEncoding.from_operator(operator)
    combination = jnp.array(1.0) * encoding + jnp.array(1.0) * BlockEncoding.from_operator(Z(0))
    reference = BlockEncoding.from_operator(operator + Z(0))
    qubit_amount = (operator + Z(0)).find_minimal_qubit_amount()

    @terminal_sampling
    def main(block_encoding):
        assert block_encoding._has_traced_coefficients
        operand = QuantumVariable(qubit_amount)
        ancillas = block_encoding.apply(operand)
        return operand, *ancillas

    res_combo = postselect_results(main(combination), num_ops=1)
    res_ref = sample_postselected(reference, lambda: QuantumVariable(qubit_amount))
    compare_distributions(res_combo, res_ref, qubit_amount)


@pytest.mark.parametrize(
    "name, build",
    [
        ("difference", lambda: BlockEncoding(1, [], lambda o: x(o[0])) - BlockEncoding(1, [], lambda o: x(o[1]))),
        (
            "terms with ancillas",
            lambda: BlockEncoding(1, [QuantumFloat(2)], lambda a, o: x(o[0])) + BlockEncoding(1, [], lambda o: x(o[1])),
        ),
        (
            "sum of products",
            lambda: (
                (BlockEncoding(1, [], lambda o: x(o[0])) @ BlockEncoding(1, [], lambda o: x(o[1])))
                + (BlockEncoding(1, [], lambda o: x(o[1])) @ BlockEncoding(1, [], lambda o: x(o[0])))
            ),
        ),
    ],
)
def test_reuse_across_transformations(name, build):
    """Reusing one linear combination across multiple JAX transformations leaks no tracers."""
    encoding = build()

    def circuit():
        operand = QuantumFloat(4)
        encoding.apply(operand)
        return operand

    make_jaspr(circuit)()
    count_ops(meas_behavior="0")(circuit)()
    make_jaspr(circuit)()


@pytest.mark.parametrize("term_count", [2, 3])
@pytest.mark.parametrize("controlled", [False, True])
def test_traced_coefficients_survive_inversion_and_control(term_count, controlled):
    """Traced-coefficient linear combinations compile under invert() and control(invert())."""

    def circuit(scale):
        encoding = None
        for idx in range(term_count):
            scaled = jnp.sqrt(scale) * BlockEncoding(1, [], lambda o, i=idx: x(o[i % 3]))
            encoding = scaled if encoding is None else encoding + scaled
        operand = QuantumFloat(3)
        if controlled:
            cond = QuantumBool()
            with control(cond[0]):
                with invert():
                    encoding.apply(operand)
        else:
            with invert():
                encoding.apply(operand)
        return operand

    make_jaspr(circuit)(0.5)
