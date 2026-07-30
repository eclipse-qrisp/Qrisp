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

from typing import TYPE_CHECKING

from jax.extend.core import Jaxpr

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
