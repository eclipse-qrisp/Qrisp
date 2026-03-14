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

"""
Call Graph Analysis Utilities
=============================

This module provides functions for analyzing the call graph structure of Jaxpr
expression trees, in particular computing the **inlined equation count** (the
total number of equations that would result from fully inlining all sub-jaxpr
calls) and the **reuse count** of each sub-jaxpr (how many times it is
referenced across the entire call graph).

These metrics are useful for making informed decisions about compilation
strategies — for example, deciding whether a reused sub-jaxpr should be
wrapped in a ``jax.pure_callback`` to avoid XLA's ``flatten-call-graph``
pass duplicating it at every call site, which causes HLO explosion and
superlinear compilation time.

The analysis is performed as a single recursive tree walk with per-jaxpr
caching, making it efficient even for deeply nested call graphs.

Example usage::

    from qrisp.jasp.interpreter_tools.call_graph_analysis import analyze_call_graph

    stats = analyze_call_graph(my_jaspr)
    print(stats.inlined_eqn_count)  # Total equations after full inlining
    print(stats.call_count)         # How many call sites reference this jaxpr
    for sub_id, sub_stats in stats.sub_jaxpr_stats.items():
        print(f"  sub {sub_id}: {sub_stats.call_count}x reused, "
              f"{sub_stats.inlined_eqn_count} eqns inlined")
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from jax.extend.core import ClosedJaxpr, Jaxpr


@dataclass
class JaxprStats:
    """
    Analysis results for a single Jaxpr node.

    Attributes
    ----------
    jaxpr : Jaxpr | ClosedJaxpr
        The analyzed Jaxpr object.
    local_eqn_count : int
        Number of equations directly in this jaxpr (excluding sub-jaxpr bodies).
    inlined_eqn_count : int
        Total equations after recursively inlining all sub-jaxpr calls.
        This represents the HLO size XLA would produce after ``flatten-call-graph``.
    call_count : int
        Number of times this jaxpr is referenced as a sub-jaxpr across the
        entire expression tree. A value > 1 indicates reuse.
    """

    jaxpr: object
    local_eqn_count: int = 0
    inlined_eqn_count: int = 0
    call_count: int = 0


def _get_jaxpr_core(jaxpr):
    """Extract the core Jaxpr from a Jaspr, ClosedJaxpr, or plain Jaxpr."""
    # Jaspr inherits from ClosedJaxpr, so this covers both
    if isinstance(jaxpr, ClosedJaxpr):
        return jaxpr.jaxpr
    return jaxpr


def _iter_sub_jaxprs(eqn):
    """
    Yield all sub-jaxprs referenced by an equation.

    Handles the known equation types that contain sub-jaxprs:
    - ``pjit`` / ``jit``: ``eqn.params["jaxpr"]``
    - ``cond``: ``eqn.params["branches"]`` (list)
    - ``while``: ``eqn.params["cond_jaxpr"]``, ``eqn.params["body_jaxpr"]``
    - ``scan``: ``eqn.params["jaxpr"]``

    Parameters
    ----------
    eqn : JaxprEqn
        A single equation from a Jaxpr.

    Yields
    ------
    Jaxpr | ClosedJaxpr
        Each sub-jaxpr referenced by this equation.
    """
    name = eqn.primitive.name

    if name in ("pjit", "jit"):
        jaxpr = eqn.params.get("jaxpr")
        if jaxpr is not None:
            yield jaxpr

    elif name == "cond":
        for branch in eqn.params.get("branches", []):
            yield branch

    elif name == "while":
        cond_jaxpr = eqn.params.get("cond_jaxpr")
        body_jaxpr = eqn.params.get("body_jaxpr")
        if cond_jaxpr is not None:
            yield cond_jaxpr
        if body_jaxpr is not None:
            yield body_jaxpr

    elif name == "scan":
        jaxpr = eqn.params.get("jaxpr")
        if jaxpr is not None:
            yield jaxpr


def analyze_call_graph(jaxpr):
    """
    Recursively analyze the call graph of a Jaxpr expression tree.

    Computes the **inlined equation count** and **call/reuse count** for
    every sub-jaxpr in the call graph. Results are cached by jaxpr identity
    (``id()``) so each unique jaxpr is analyzed exactly once.

    The inlined equation count represents the total number of HLO operations
    XLA would see after its ``flatten-call-graph`` pass inlines all shared
    sub-computations. This is the key metric for predicting compilation cost.

    Parameters
    ----------
    jaxpr : Jaxpr | ClosedJaxpr | Jaspr
        The root Jaxpr to analyze.

    Returns
    -------
    JaxprStats
        The analysis results for the root jaxpr. 
        
    dict[int, JaxprStats]
        A dictionary mapping ``id(sub_jaxpr)`` to ``JaxprStats`` for every unique
        sub-jaxpr encountered in the call graph (including the root).

    Example
    -------
    ::

        root_stats, all_stats = analyze_call_graph(my_jaspr)

        # Total equations after full inlining
        print(root_stats.inlined_eqn_count)

        # Find heavily-reused large sub-jaxprs
        for jid, stats in all_stats.items():
            inflation = (stats.call_count - 1) * stats.inlined_eqn_count
            if inflation > 100:
                print(f"  jaxpr id={jid}: {stats.call_count}x reused, "
                      f"{stats.inlined_eqn_count} inlined eqns, "
                      f"inflation={inflation}")
    """
    # Cache: id(jaxpr) -> JaxprStats
    stats_cache: Dict[int, JaxprStats] = {}

    def _analyze(jaxpr) -> JaxprStats:
        jid = id(jaxpr)

        if jid in stats_cache:
            # Already analyzed — just bump the call count
            stats_cache[jid].call_count += 1
            return stats_cache[jid]

        core = _get_jaxpr_core(jaxpr)
        local_count = len(core.eqns)

        # Create the stats entry before recursing (handles cycles gracefully)
        stats = JaxprStats(
            jaxpr=jaxpr,
            local_eqn_count=local_count,
            inlined_eqn_count=0,  # computed below
            call_count=1,
        )
        stats_cache[jid] = stats

        # Compute the inlined count: start with local equations,
        # then for each sub-jaxpr reference, replace the call equation
        # with the sub-jaxpr's full inlined body
        inlined = local_count
        for eqn in core.eqns:
            for sub_jaxpr in _iter_sub_jaxprs(eqn):
                sub_stats = _analyze(sub_jaxpr)
                # The call equation itself is already counted in local_count,
                # so we add the sub-jaxpr's inlined size minus 1 (for the call)
                inlined += sub_stats.inlined_eqn_count - 1

        stats.inlined_eqn_count = inlined
        return stats

    root_stats = _analyze(jaxpr)
    return root_stats, stats_cache
