"""
********************************************************************************
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

import jax
import jax.numpy as jnp
from jax import make_jaxpr

from qrisp.jasp.interpreter_tools.call_graph_analysis import (
    analyze_call_graph,
    JaxprStats,
)


# ── Tests ─────────────────────────────────────────────────────────

def test_flat_jaxpr():
    """A jaxpr with no sub-calls: inlined_eqn_count == local_eqn_count."""

    def f(x):
        x = x + 1.0
        x = x * 2.0
        x = x - 3.0
        return x

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    assert root.local_eqn_count == 3
    assert root.inlined_eqn_count == 3
    assert root.call_count == 1
    assert len(all_stats) == 1


def test_single_jit_call():
    """One jitted subroutine called once inside a jitted function.

    make_jaxpr(@jit f) produces:
      root (1 eqn: jit[f])
        └─ f_body (2 eqns: jit[sub] + add)
             └─ sub_body (2 eqns: add + mul)

    Inlined root = 1 + (f_inlined - 1)
    f_inlined    = 2 + (2 - 1) = 3
    root_inlined = 1 + (3 - 1) = 3
    """

    @jax.jit
    def sub(x):
        x = x + 1.0
        x = x * 2.0
        return x

    @jax.jit
    def f(x):
        return sub(x) + 3.0

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # 3 jaxprs: root wrapper, f body, sub body
    assert len(all_stats) == 3
    assert root.inlined_eqn_count == 3

    # sub_body should have 2 eqns, called once
    sub_stats = [s for s in all_stats.values()
                 if s.local_eqn_count == 2 and s.inlined_eqn_count == 2]
    assert len(sub_stats) == 1
    assert sub_stats[0].call_count == 1


def test_shared_jit_reuse():
    """Same jitted function called 3 times → call_count == 3.

    make_jaxpr(@jit g) produces:
      root (1 eqn: jit[g])
        └─ g_body (3 eqns: 3× jit[sub])
             └─ sub_body (2 eqns: add + mul)  ← shared, called 3×

    sub_inlined  = 2
    g_inlined    = 3 + 3*(2-1) = 6
    root_inlined = 1 + (6-1)   = 6
    """

    @jax.jit
    def sub(x):
        x = x + 1.0
        x = x * 2.0
        return x

    @jax.jit
    def g(x):
        x = sub(x)
        x = sub(x)
        x = sub(x)
        return x

    jaxpr = make_jaxpr(g)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    assert root.inlined_eqn_count == 6

    # sub should be reused 3 times
    sub_stats = [s for s in all_stats.values()
                 if s.local_eqn_count == 2 and s.inlined_eqn_count == 2]
    assert len(sub_stats) == 1
    assert sub_stats[0].call_count == 3


def test_nested_jit():
    """Chain: f -> outer -> inner.

    root (1 eqn) -> f_body (1 eqn: jit[outer])
                      -> outer_body (2 eqns: jit[inner] + mul)
                           -> inner_body (1 eqn: add)

    inner_inlined = 1
    outer_inlined = 2 + (1-1) = 2
    f_inlined     = 1 + (2-1) = 2
    root_inlined  = 1 + (2-1) = 2
    """

    @jax.jit
    def inner(x):
        return x + 1.0

    @jax.jit
    def outer(x):
        x = inner(x)
        x = x * 2.0
        return x

    @jax.jit
    def f(x):
        return outer(x)

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    assert root.inlined_eqn_count == 2
    assert len(all_stats) == 4  # root, f_body, outer_body, inner_body


def test_diamond_reuse():
    """inner called from two different outer functions → call_count == 2."""

    @jax.jit
    def inner(x):
        return x + 1.0

    @jax.jit
    def outer_a(x):
        return inner(x) * 2.0

    @jax.jit
    def outer_b(x):
        return inner(x) - 3.0

    @jax.jit
    def f(x):
        return outer_a(x) + outer_b(x)

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # inner_body has 1 local eqn and should be called 2×
    inner_stats = [s for s in all_stats.values()
                   if s.local_eqn_count == 1 and s.inlined_eqn_count == 1]
    assert len(inner_stats) == 1
    assert inner_stats[0].call_count == 2


def test_cond_branches():
    """Cond creates sub-jaxprs for each branch."""

    def f(x):
        return jax.lax.cond(x > 0, lambda x: x + 1.0, lambda x: x - 1.0, x)

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # Root: gt + convert_element_type + cond = 3 eqns; each branch has 1 eqn
    # Inlined: 3 + (1-1) + (1-1) = 3
    assert root.local_eqn_count == 3
    assert root.inlined_eqn_count == 3
    # root + 2 branches
    assert len(all_stats) == 3


def test_while_loop():
    """While loop has cond_jaxpr and body_jaxpr sub-jaxprs."""

    def f(x):
        return jax.lax.while_loop(lambda x: x < 10.0, lambda x: x + 1.0, x)

    jaxpr = make_jaxpr(f)(0.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # Root: 1 eqn (the while); cond: 1 eqn; body: 1 eqn
    assert root.local_eqn_count == 1
    assert root.inlined_eqn_count == 1
    assert len(all_stats) == 3


def test_scan():
    """Scan body is captured as a sub-jaxpr."""

    def f(x):
        def body(carry, _):
            return carry + 1.0, carry
        final, ys = jax.lax.scan(body, x, jnp.arange(5.0))
        return final

    jaxpr = make_jaxpr(f)(0.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # root: 2 eqns (iota/arange + scan), scan body: 1 eqn
    assert root.local_eqn_count == 2
    assert root.inlined_eqn_count == 2
    assert len(all_stats) == 2


def test_inflation_calculation():
    """Verify inflation = (call_count - 1) × inlined_eqn_count."""

    @jax.jit
    def sub(x):
        x = x + 1.0
        x = x * 2.0
        x = x - 3.0
        x = x + 4.0
        x = x * 5.0
        return x

    @jax.jit
    def f(x):
        for _ in range(10):
            x = sub(x)
        return x

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # Find sub (5 local eqns, called 10×)
    sub_stats = [s for s in all_stats.values()
                 if s.local_eqn_count == 5 and s.inlined_eqn_count == 5]
    assert len(sub_stats) == 1
    assert sub_stats[0].call_count == 10

    inflation = (sub_stats[0].call_count - 1) * sub_stats[0].inlined_eqn_count
    assert inflation == 45  # 9 extra copies × 5 eqns

    # f_body: 10 local + 10*(5-1) = 50
    f_body = [s for s in all_stats.values() if s.local_eqn_count == 10]
    assert len(f_body) == 1
    assert f_body[0].inlined_eqn_count == 50


def test_no_reuse_no_inflation():
    """Single-use sub-jaxprs produce zero total inflation."""

    @jax.jit
    def sub_a(x):
        return x + 1.0

    @jax.jit
    def sub_b(x):
        return x * 2.0

    @jax.jit
    def f(x):
        return sub_a(x) + sub_b(x)

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    total_inflation = sum(
        (s.call_count - 1) * s.inlined_eqn_count
        for s in all_stats.values()
    )
    assert total_inflation == 0


def test_deeply_nested_inlined_count():
    """Chain: root -> a -> b -> c, each adding 1 operation.

    c_body:     1 eqn,  inlined=1
    b_body:     2 eqns, inlined=2+(1-1)=2
    a_body:     2 eqns, inlined=2+(2-1)=3
    f_body:     1 eqn,  inlined=1+(3-1)=3
    root:       1 eqn,  inlined=1+(3-1)=3
    """

    @jax.jit
    def c(x):
        return x + 1.0

    @jax.jit
    def b(x):
        return c(x) * 2.0

    @jax.jit
    def a(x):
        return b(x) - 3.0

    @jax.jit
    def f(x):
        return a(x)

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    assert root.inlined_eqn_count == 3
    # root, f_body, a_body, b_body, c_body
    assert len(all_stats) == 5


def test_mixed_reuse_and_unique():
    """Mix: shared (3×) + unique (1×)."""

    @jax.jit
    def shared(x):
        return x + 1.0

    @jax.jit
    def unique(x):
        return x * 3.0

    @jax.jit
    def f(x):
        x = shared(x)
        x = shared(x)
        x = shared(x)
        x = unique(x)
        return x

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    # shared_body: 1 eqn, called 3×
    shared_stats = [s for s in all_stats.values() if s.call_count == 3]
    assert len(shared_stats) == 1
    assert shared_stats[0].inlined_eqn_count == 1

    # unique_body: 1 eqn, called 1×, inlined == 1
    # (exclude root wrapper which also has local==1 but inlined==4)
    unique_stats = [s for s in all_stats.values()
                    if s.call_count == 1 and s.local_eqn_count == 1
                    and s.inlined_eqn_count == 1]
    assert len(unique_stats) == 1


def test_empty_jaxpr():
    """A jaxpr with no equations (identity function)."""

    def f(x):
        return x

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    assert root.local_eqn_count == 0
    assert root.inlined_eqn_count == 0
    assert len(all_stats) == 1


def test_large_reuse_count():
    """High reuse: same subroutine called 100 times."""

    @jax.jit
    def sub(x):
        x = x + 1.0
        x = x * 2.0
        return x

    @jax.jit
    def f(x):
        for _ in range(100):
            x = sub(x)
        return x

    jaxpr = make_jaxpr(f)(1.0)
    root, all_stats = analyze_call_graph(jaxpr)

    sub_stats = [s for s in all_stats.values()
                 if s.local_eqn_count == 2 and s.inlined_eqn_count == 2]
    assert len(sub_stats) == 1
    assert sub_stats[0].call_count == 100

    # f_body: 100 local + 100*(2-1) = 200
    f_body = [s for s in all_stats.values() if s.local_eqn_count == 100]
    assert len(f_body) == 1
    assert f_body[0].inlined_eqn_count == 200


# ── Runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_flat_jaxpr,
        test_single_jit_call,
        test_shared_jit_reuse,
        test_nested_jit,
        test_diamond_reuse,
        test_cond_branches,
        test_while_loop,
        test_scan,
        test_inflation_calculation,
        test_no_reuse_no_inflation,
        test_deeply_nested_inlined_count,
        test_mixed_reuse_and_unique,
        test_empty_jaxpr,
        test_large_reuse_count,
    ]

    for t in tests:
        t()
        print(f"  {t.__name__} PASSED")

    print(f"\nAll {len(tests)} tests passed!")
