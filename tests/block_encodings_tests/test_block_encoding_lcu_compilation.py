"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

"""Compilation-cost regression tests for LinearCombinationBlockEncoding.

A linear combination lowers to PREP-SELECT-PREP on top of the general ``q_switch``
and ``prepare`` primitives. Both of those have trace-time behaviour that can make
the compilation cost of a correct block encoding explode without changing any
result, so the tests here assert on *how much* gets traced rather than on what is
computed. The numerical behaviour is covered by test_block_encoding_arithmetic.py.

Two regressions are guarded:

1. ``q_switch`` traces every branch several times per call (once for its loop body
   and once for each arm of its final conditional), and custom_control /
   custom_inversion speculatively trace the controlled and inverted variants on
   top of that. If the SELECT branches are rebuilt on every invocation of the LCU
   unitary, none of those repeats hit Jasp's identity-keyed caches, and the cost
   multiplies once per level of nested linear combinations.

2. ``prepare`` dispatches on whether its amplitude vector converts to NumPy. Since
   a ``jnp`` operation inside a trace returns a tracer even for compile-time
   constants, deriving the amplitudes with ``jnp`` silently downgrades PREP from a
   single opaque gate to a fully symbolic binary-tree construction.
"""

import jax
import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable, make_jaspr, terminal_sampling, x
from qrisp.jasp import count_ops, depth, num_qubits
from qrisp.block_encodings import BlockEncoding, LinearCombinationBlockEncoding
from qrisp.operators import X, Y, Z


def _traced_equations(encoding, operand_size=3):
    """Trace ``encoding.apply`` on a fresh operand and return the resulting Jaspr."""

    def circuit():
        operand = QuantumFloat(operand_size)
        encoding.apply(operand)
        return operand

    return make_jaspr(circuit)()


#
# Derived state is reused
#


def test_derived_state_is_cached():
    """The unitary, branches, layouts and templates must be built exactly once.

    Jasp's caches (``qache``, and therefore ``jax.jit``) are keyed on object
    identity. Handing out a fresh unitary on every access silently disables them.
    """
    encoding = BlockEncoding.from_eye(0) + BlockEncoding.from_eye(1)

    assert encoding.unitary is encoding.unitary
    assert encoding._lcu_layouts is encoding._lcu_layouts
    assert all(first is second for first, second in zip(encoding._anc_templates, encoding._anc_templates))


def test_ancilla_templates_are_not_shared_mutable_state():
    """Callers may concatenate the returned template list without corrupting the cache."""
    encoding = BlockEncoding.from_eye(0) + BlockEncoding.from_eye(1)

    templates = encoding._anc_templates
    templates.append(None)

    assert len(encoding._anc_templates) == len(templates) - 1


#
# Trace amplification does not grow with nesting depth
#


def _count_child_traces(nesting_depth):
    """Trace a tower of ``nesting_depth`` nested linear combinations.

    Each level wraps the previous one as ``eye + (child @ other)``, which is the
    shape produced by building a product of sums. Returns how often the innermost
    child's Python body was executed while tracing.
    """
    executions = []

    def innermost(operand):
        executions.append(None)
        x(operand[0])

    def other(operand):
        x(operand[1])

    encoding = BlockEncoding(1, [], innermost)
    for _ in range(nesting_depth):
        encoding = BlockEncoding.from_eye() + (encoding @ BlockEncoding(1, [], other))

    _traced_equations(encoding)
    return len(executions)


def test_nested_linear_combinations_do_not_amplify_tracing():
    """Nesting linear combinations must not multiply the tracing cost per level.

    Before the SELECT branches were hoisted and cached, every nesting level
    multiplied the number of traces of the whole subtree by the number of times
    ``q_switch`` traces a branch, giving 12, 144 and 1728 traces for depths 1, 2
    and 3. The exact constant is an implementation detail of ``q_switch``; what
    must hold is that it stays constant in the nesting depth.
    """
    trace_counts = [_count_child_traces(depth) for depth in (1, 2, 3)]

    assert len(set(trace_counts)) == 1, f"tracing cost grows with nesting depth: {trace_counts}"
    assert trace_counts[0] <= 2, f"child traced {trace_counts[0]} times for a single linear combination"


def test_repeated_use_of_a_linear_combination_traces_the_child_once():
    """Applying the same linear combination twice must not trace its children twice."""
    executions = []

    def counted(operand):
        executions.append(None)
        x(operand[0])

    encoding = BlockEncoding(1, [], counted) + BlockEncoding(1, [], lambda operand: x(operand[1]))

    def circuit():
        operand = QuantumFloat(3)
        encoding.apply(operand)
        encoding.apply(operand)
        return operand

    make_jaspr(circuit)()

    assert len(executions) <= 2


#
# PREP keeps its concrete fast path
#


def test_concrete_coefficients_stay_concrete_during_tracing():
    """Concrete coefficients must not be turned into tracers by the derivation.

    ``prepare`` falls back to its symbolic implementation as soon as the amplitude
    vector cannot be converted to NumPy, so the derived quantities have to stay out
    of ``jnp`` whenever the inputs are known at build time.
    """
    encoding = 2 * BlockEncoding.from_eye(0) - BlockEncoding.from_eye(1) - BlockEncoding.from_eye(-1)
    observed = {}

    def circuit():
        operand = QuantumFloat(3)
        # Read the derived quantities from inside a trace, which is where the
        # regression appeared: outside a trace jnp returns concrete arrays.
        observed["coefficients"] = encoding._lcu_coefficients
        observed["amplitudes"] = encoding._lcu_amplitudes
        observed["alpha"] = encoding.alpha
        encoding.apply(operand)
        return operand

    make_jaspr(circuit)()

    for name, value in observed.items():
        assert not isinstance(value, jax.core.Tracer), f"{name} became a tracer inside the trace"
    assert np.isclose(observed["alpha"], 4.0)


def test_prepare_uses_the_concrete_state_preparation(monkeypatch):
    """PREP and its inverse must both take the concrete ``prepare_qiskit`` path."""
    import qrisp.alg_primitives.state_preparation.prepare_func as prepare_func

    methods = []

    def record(name, implementation):
        def wrapper(*args, **kwargs):
            methods.append(name)
            return implementation(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(prepare_func, "prepare_qiskit", record("qiskit", prepare_func.prepare_qiskit))
    monkeypatch.setattr(prepare_func, "prepare_qswitch", record("qswitch", prepare_func.prepare_qswitch))

    encoding = 2 * BlockEncoding.from_eye(0) - BlockEncoding.from_eye(1) - BlockEncoding.from_eye(-1)
    _traced_equations(encoding)

    assert methods, "prepare was never called"
    assert "qswitch" not in methods, "PREP fell back to the symbolic state preparation"


def test_linear_combination_compiles_to_few_equations():
    """A small linear combination must not blow up the traced program.

    This is a coarse smoke guard on the two regressions above: the same encoding
    traced to 449 top-level equations when PREP fell back to its symbolic path.
    """
    encoding = 2 * BlockEncoding.from_eye(0) - BlockEncoding.from_eye(1) - BlockEncoding.from_eye(-1)

    assert len(_traced_equations(encoding).eqns) < 100


#
# The traced fallback stays available and correct
#


def test_traced_coefficients_fall_back_to_the_symbolic_path():
    """Coefficients that only exist at runtime must still produce a valid encoding.

    Passing the block encoding through a jit boundary as a pytree turns its
    coefficients into tracers, which is exactly when the ``jnp`` derivation and the
    symbolic state preparation are the correct choice.
    """
    operator = 0.5 * X(0) + 0.7 * Y(0) + 0.3 * Z(1)
    encoding = BlockEncoding.from_operator(operator)
    combination = encoding + BlockEncoding.from_operator(Z(0))
    reference = BlockEncoding.from_operator(operator + Z(0))

    qubit_amount = (operator + Z(0)).find_minimal_qubit_amount()

    @terminal_sampling
    def main(block_encoding):
        return block_encoding.apply_rus(lambda: QuantumVariable(qubit_amount))()

    combination_result = main(combination)
    reference_result = main(reference)

    for state in range(2**qubit_amount):
        assert np.isclose(combination_result.get(state, 0), reference_result.get(state, 0))


@pytest.mark.parametrize(
    "H1, H2, H3",
    [
        (X(0) * X(1), Z(0) * Z(1), 0.4 * Y(0)),
        (0.5 * X(1) + 0.7 * Y(1), Z(0) + Z(1), 0.25 * Z(0) * X(1)),
    ],
)
def test_nested_linear_combination_is_numerically_unchanged(H1, H2, H3):
    """Nested sums must encode the same operator as the flattened reference.

    The SELECT branches take the shared ancilla workspace as an operand rather than
    capturing a view of it, which is what makes them reusable across traces. This
    guards that restructuring against a silent change of the encoded operator.
    """
    nested = (BlockEncoding.from_operator(H1) + BlockEncoding.from_operator(H2)) + BlockEncoding.from_operator(H3)
    reference = BlockEncoding.from_operator(H1 + H2 + H3)

    assert isinstance(nested, LinearCombinationBlockEncoding)

    qubit_amount = (H1 + H2 + H3).find_minimal_qubit_amount()

    @terminal_sampling
    def main(block_encoding):
        return block_encoding.apply_rus(lambda: QuantumVariable(qubit_amount))()

    nested_result = main(nested)
    reference_result = main(reference)

    for state in range(2**qubit_amount):
        assert np.isclose(nested_result.get(state, 0), reference_result.get(state, 0))


#
# Cached state must not outlive the trace that built it
#


def _reuse_across_transformations(encoding, operand_size=4):
    """Apply one encoding under several JAX transformations, reusing the same object.

    Anything the encoding cached during the first trace is handed to the later ones.
    A cached value holding a tracer from the first trace raises UnexpectedTracerError
    here, which is what makes this the shape that catches the regression.
    """

    def circuit():
        operand = QuantumFloat(operand_size)
        encoding.apply(operand)
        return operand

    make_jaspr(circuit)()
    count_ops(meas_behavior="0")(circuit)()
    depth(meas_behavior="0")(circuit)()
    num_qubits(meas_behavior="0")(circuit)()
    make_jaspr(circuit)()


def _first():
    return BlockEncoding(1, [], lambda operand: x(operand[0]))


def _second():
    return BlockEncoding(1, [], lambda operand: x(operand[1]))


@pytest.mark.parametrize(
    "name, build",
    [
        ("two terms", lambda: _first() + _second()),
        ("three terms", lambda: _first() + _second() + _first()),
        ("scaled", lambda: 2.5 * (_first() + _second())),
        ("difference", lambda: _first() - _second()),
        ("terms with ancillas", lambda: BlockEncoding(1, [QuantumFloat(2)], lambda a, o: x(o[0])) + _second()),
        ("sum of products", lambda: (_first() @ _second()) + (_second() @ _first())),
    ],
)
def test_linear_combination_survives_reuse_across_transformations(name, build):
    """A cached template must never carry a tracer out of the trace that built it.

    Ancilla templates record their register size, and in Jasp that size is a tracer,
    so a template first built inside a trace cannot be cached. Caching it made the
    second transformation fail with UnexpectedTracerError.
    """
    _reuse_across_transformations(build())
