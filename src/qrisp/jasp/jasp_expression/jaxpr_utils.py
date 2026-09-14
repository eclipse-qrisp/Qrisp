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

"""Helper functions for rebuilding a Jaxpr or ClosedJaxpr with new equations and/or outvars."""

from typing import TYPE_CHECKING

from jax.extend.core import ClosedJaxpr, Jaxpr

if TYPE_CHECKING:
    from qrisp.jasp.jasp_expression.centerclass import Jaspr


def rebuild_jaxpr(base: "Jaxpr | Jaspr", *, eqns=None, outvars=None) -> Jaxpr:
    """Return a new bare Jaxpr, copying base's constvars/invars/effects/debug_info,
    with an optionally-overridden eqns and/or outvars list.

    Centralizes the "rebuild a Jaxpr with unchanged invars/constvars but new eqns"
    skeleton shared by control_transform.py's (former) copy_jaxpr,
    environment_collection.py's collect_environments, and inv_transform.py's
    invert_jaspr.

    Parameters
    ----------
    base : Jaxpr | Jaspr
        The Jaxpr (or Jaspr, which exposes the same fields via property
        delegation to its own wrapped jaxpr) whose constvars/invars/effects/
        debug_info are carried over unchanged.

    eqns : list[JaxprEqn] | None, optional
        The new equation list. Defaults to ``base.eqns`` (a fresh copy) when
        omitted.

    outvars : list | None, optional
        The new outvars list. Defaults to ``base.outvars`` (a fresh copy) when
        omitted.

    Returns
    -------
    Jaxpr
        A new Jaxpr with the requested overrides applied.

    """
    return Jaxpr(
        constvars=list(base.constvars),
        invars=list(base.invars),
        outvars=list(outvars) if outvars is not None else list(base.outvars),
        eqns=list(eqns) if eqns is not None else list(base.eqns),
        effects=base.effects,
        debug_info=base.debug_info,
    )


def rebuild_closed_jaxpr(base: "ClosedJaxpr | Jaspr", *, eqns=None, outvars=None) -> ClosedJaxpr:
    """Like rebuild_jaxpr, but also re-wraps the result in a ClosedJaxpr using
    base's own consts.

    Centralizes the "rebuild_jaxpr(...), then wrap in ClosedJaxpr(result,
    base.consts)" pairing that every current rebuild_jaxpr call site performs
    immediately afterward.

    Parameters
    ----------
    base : ClosedJaxpr | Jaspr
        The ClosedJaxpr (or Jaspr) whose constvars/invars/effects/debug_info/
        consts are carried over unchanged.

    eqns : list[JaxprEqn] | None, optional
        The new equation list. Defaults to base's own eqns when omitted.

    outvars : list | None, optional
        The new outvars list. Defaults to base's own outvars when omitted.

    Returns
    -------
    ClosedJaxpr
        A new ClosedJaxpr with the requested overrides applied, wrapping
        base's original consts.

    """
    return ClosedJaxpr(rebuild_jaxpr(base.jaxpr, eqns=eqns, outvars=outvars), base.consts)


def fold_extra_constvars_into_invars(
    closed_jaxpr: ClosedJaxpr, n_expected_const: int, insert_at: int = 0
) -> ClosedJaxpr:
    """Fold the leading constvars of a ClosedJaxpr back into genuine invars.

    Qrisp builds the very same quantum function in two ways, and the two disagree
    on how closed-over values are passed.

    ``qache`` traces the forward version through ``jax.jit``, so Jax performs its
    own closure conversion: values that the traced function closes over (e.g. the
    angle arrays captured by ``prepare_qswitch``'s case functions) become
    *leading invars* of the callee, and the wrapping ``jit`` equation supplies
    them as extra leading operands.

    ``make_jaspr`` traces through ``jax.make_jaxpr``, which does not closure
    convert: those same values stay behind in ``constvars``/``consts``. A Jaspr
    traced that way holds live tracers in ``consts`` - which breaks as soon as it
    is reused in a different tracing context - and exposes fewer invars than the
    ``jit`` equation supplies.

    This function rewrites the second shape into the first, so that a Jaspr
    produced by ``make_jaspr`` can stand in as the callee of a ``jit`` equation
    that was built for a ``qache``-produced one.

    Note that the *values* of the folded constvars (``consts[:n_extra]``) are
    dropped: the folded vars become plain invars, and it is the caller's
    responsibility to supply equivalent values positionally. That holds precisely
    because Jax's own closure conversion put the same values in the same leading
    positions on the forward path - see ``closure_convert_jaspr``, which pairs
    this rewrite with a signature check against the forward Jaspr.

    Parameters
    ----------
    closed_jaxpr : jax.extend.core.ClosedJaxpr
        The ClosedJaxpr to normalize.
    n_expected_const : int
        The number of constvars that are genuine and must be left alone
        (usually 0).
    insert_at : int, optional
        The position (within the resulting invars list) at which to re-insert the
        folded constvars. Use this when the signature starts with invars that
        must stay in front of the folded ones (e.g. the control qubit that
        ``custom_control`` prepends). Default is 0 (folded invars go first).

    Returns
    -------
    jax.extend.core.ClosedJaxpr
        A ClosedJaxpr with exactly ``n_expected_const`` constvars. The input is
        returned unchanged if there was nothing to fold.

    """
    core = closed_jaxpr.jaxpr
    n_extra = len(core.constvars) - n_expected_const
    if n_extra <= 0:
        return closed_jaxpr

    folded_invars = list(core.constvars[:n_extra])
    new_constvars = list(core.constvars[n_extra:])
    new_consts = list(closed_jaxpr.consts[n_extra:])
    new_invars = list(core.invars)
    new_invars[insert_at:insert_at] = folded_invars
    new_core = Jaxpr(
        constvars=new_constvars,
        invars=new_invars,
        outvars=list(core.outvars),
        eqns=list(core.eqns),
        effects=core.effects,
        debug_info=core.debug_info,
    )
    return ClosedJaxpr(new_core, new_consts)


def closure_convert_jaspr(jaspr: "Jaspr", insert_at: int = 0) -> "Jaspr":
    """Return ``jaspr`` in the calling convention that pjit uses for its callees.

    Every constvar is folded into an invar (see
    ``fold_extra_constvars_into_invars``), so the result takes all of its inputs
    through its signature and carries no ``consts``. Apply this to any Jaspr that
    was traced with ``make_jaspr`` but is going to be used as the callee of a
    ``jit`` equation, i.e. the cached ``inv_jaspr``/``ctrl_jaspr`` of the
    ``custom_inversion``/``custom_control`` decorators.

    Doing this once, where the Jaspr is created, is what keeps the transformation
    passes (``invert_eqn``, ``control_eqn``) free of rewrapping: a rewrap there
    would have to reattach every piece of Jaspr metadata by hand, and could not
    preserve a ``ControlledJaspr`` at all.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to convert.
    insert_at : int, optional
        Where to insert the folded invars, see
        ``fold_extra_constvars_into_invars``. Default is 0.

    Returns
    -------
    Jaspr
        The converted Jaspr, or ``jaspr`` itself if there was nothing to fold.

    """
    from qrisp.jasp.jasp_expression.centerclass import Jaspr

    normalized = fold_extra_constvars_into_invars(jaspr, 0, insert_at=insert_at)
    if normalized is jaspr:
        return jaspr

    # Rewrapping produces a fresh Jaspr, so every attribute that is not part of
    # the Jaxpr itself has to be carried over explicitly. The variable objects
    # are shared with the input, so the permeability dict keyed by them still
    # applies.
    res = Jaspr(
        normalized,
        permeability=jaspr.permeability,
        isqfree=jaspr.isqfree,
        ctrl_jaspr=jaspr.ctrl_jaspr,
        inv_jaspr=jaspr.inv_jaspr,
    )
    res.envs_flattened = jaspr.envs_flattened

    return res
