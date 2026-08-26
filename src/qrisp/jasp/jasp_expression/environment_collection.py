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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from jax.api_util import debug_info
from jax.extend.core import JaxprEqn, Literal, Var
from numba import njit

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.jasp.interpreter_tools import copy_jaxpr_eqn
from qrisp.jasp.jasp_expression.jaxpr_utils import rebuild_closed_jaxpr

if TYPE_CHECKING:
    from jax.extend.core import ClosedJaxpr

    from qrisp.jasp.jasp_expression.centerclass import Jaspr

# In newer versions, Jax enforces providing a debug info object
# to the Jaxpr constructor. This object contains metadata information
# about the Python function corresponding to the Jaxpr object.
# In this file, Jaxprs are created from QuantumEnvironments,
# i.e. blocks of code that undergo a particular compilation
# routine. The assumption that there is a underlying function
# therefore doesn't apply. Because of this reason, we create a
# dummy debug info object, to still enable compatibility with the
# latest Jax versions.
dummy_debug_info = debug_info(
    traced_for="env_jaspr",
    fun=(lambda: None),
    args=(),
    kwargs={},
    static_argnums=[],
    static_argnames=[],
    result_paths_thunk=lambda: (),
)


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def collect_environments(closed_jaxpr: "ClosedJaxpr | Jaspr") -> "ClosedJaxpr | Jaspr":
    """Turn a Jaxpr with QuantumEnvironments in enter/exit form into collected form.

    Collected means that each QuantumEnvironment's content is represented by
    a Jaspr.

    Parameters
    ----------
    closed_jaxpr : jax.extend.core.ClosedJaxpr | Jaspr
        The Jaxpr with QuantumEnvironment in enter/exit form.

    Returns
    -------
    jax.extend.core.ClosedJaxpr | Jaspr
        A Jaxpr with QuantumEnvironments in collected form.

    """
    from qrisp.jasp import Jaspr

    if isinstance(closed_jaxpr, Jaspr) and closed_jaxpr.envs_flattened:
        return closed_jaxpr

    # We iterate through the list of equations, appending the equations to
    # the new list containing the processed equations.

    # Once we hit an exit primitive, we collect the Equations between the enter
    # and exit primitive.
    eqn_list = list(closed_jaxpr.jaxpr.eqns)
    new_eqn_list: list[JaxprEqn] = []

    # An important part of collecting the quantum environments is determining
    # the input output variables. Doing this analysis can be prohibitvely costly
    # if implemented naively. For this reason the VarTracker class implements,
    # which tracks the I/O variables in a specialized data structured that
    # enables an efficient solution to this problem.
    eqn_var_tracker = VarTracker(eqn_list)
    new_eqn_var_tracker = VarTracker(new_eqn_list)

    for j, orig_eqn in enumerate(eqn_list):
        eqn = _recurse_into_subjaxprs(orig_eqn)

        # If an exit primitive is found, start the collecting mechanism.
        if eqn.primitive.name == "jasp.q_env" and "exit" in eqn.params.values():
            eqn, new_eqn_list, new_eqn_var_tracker = _collect_environment_body(
                eqn,
                eqn_var_tracker.slice_start(j + 1),
                new_eqn_list,
                new_eqn_var_tracker,
                closed_jaxpr,
            )

        # Append the equation
        new_eqn_list.append(eqn)
        new_eqn_var_tracker.append(eqn)

    if isinstance(closed_jaxpr, Jaspr):
        res = closed_jaxpr.update_eqns(new_eqn_list)

        if closed_jaxpr.ctrl_jaspr is not None:
            res.ctrl_jaspr = closed_jaxpr.ctrl_jaspr
        if closed_jaxpr.inv_jaspr is not None:
            res.inv_jaspr = closed_jaxpr.inv_jaspr
        return res

    # Return the transformed equation
    return rebuild_closed_jaxpr(closed_jaxpr, eqns=new_eqn_list)


def _recurse_into_subjaxprs(eqn: JaxprEqn) -> JaxprEqn:
    """Recursively collect environments inside a jit/cond/while sub-jaxpr.

    Equations that don't carry a sub-jaxpr of interest are returned unchanged.

    Parameters
    ----------
    eqn : jax.extend.core.JaxprEqn
        The equation to recurse into.

    Returns
    -------
    jax.extend.core.JaxprEqn
        Either eqn unchanged, or a copy with its "jaxpr"/"branches"/
        "body_jaxpr" param replaced by the collected equivalent.

    """
    if eqn.primitive.name == "jit":
        new_eqn = copy_jaxpr_eqn(eqn)
        new_eqn.params["jaxpr"] = collect_environments(eqn.params["jaxpr"])
        return new_eqn

    if eqn.primitive.name == "cond":
        new_eqn = copy_jaxpr_eqn(eqn)
        new_eqn.params["branches"] = tuple(collect_environments(branch) for branch in eqn.params["branches"])
        return new_eqn

    if eqn.primitive.name == "while":
        new_eqn = copy_jaxpr_eqn(eqn)
        new_eqn.params["body_jaxpr"] = collect_environments(eqn.params["body_jaxpr"])
        return new_eqn

    return eqn


def _collect_environment_body(
    exit_eqn: JaxprEqn,
    remaining_script_var_tracker: "VarTracker",
    new_eqn_list: list[JaxprEqn],
    new_eqn_var_tracker: "VarTracker",
    closed_jaxpr: "ClosedJaxpr | Jaspr",
) -> tuple[JaxprEqn, list[JaxprEqn], "VarTracker"]:
    """Collapse a QuantumEnvironment's enter/exit equation pair into one equation.

    The matching "enter" equation is located by scanning new_eqn_list
    backwards from its end; everything after it becomes the environment body.

    Parameters
    ----------
    exit_eqn : jax.extend.core.JaxprEqn
        The "jasp.q_env" exit equation that triggered the collection.
    remaining_script_var_tracker : VarTracker
        Tracks the I/O variables of the equations following exit_eqn in the
        original (uncollected) equation list, i.e. the code that still needs
        to run after the environment body.
    new_eqn_list : list[jax.extend.core.JaxprEqn]
        The equations collected so far, ending with the matching "enter"
        equation somewhere in its tail.
    new_eqn_var_tracker : VarTracker
        Tracks the I/O variables of new_eqn_list.
    closed_jaxpr : jax.extend.core.ClosedJaxpr | Jaspr
        The enclosing Jaxpr, used to determine which of its outvars must
        remain live after the environment body.

    Returns
    -------
    tuple[jax.extend.core.JaxprEqn, list[jax.extend.core.JaxprEqn], VarTracker]
        The collected "jasp.q_env" equation to append, together with
        new_eqn_list and new_eqn_var_tracker truncated to just before the
        matching "enter" equation.

    """
    from qrisp.jasp import AbstractQuantumState, Jaspr

    # Find the position of the enter primitive.
    for i in range(len(new_eqn_list))[::-1]:
        enter_eq = new_eqn_list[i]
        if enter_eq.primitive.name == "jasp.q_env" and "enter" in enter_eq.params.values():
            break
    else:
        raise Exception("Found a QuantumEnvironment exit equation without a matching enter equation")

    # The environment body is the slice of new_eqn_list after the enter equation.
    invars = new_eqn_var_tracker.slice_start(i + 1).find_invars()

    # Remove the AbstractQuantumState variable and prepend it.
    try:
        invars.remove(enter_eq.outvars[0])
    except ValueError:
        pass

    # Same for the outvars
    outvars = find_outvars(
        new_eqn_list[i + 1 :],
        remaining_script_var_tracker,
        [var for var in closed_jaxpr.jaxpr.outvars if not isinstance(var, Literal)],
    )

    # Filter the AbstractQuantumState (we add it manually to make sure
    # it is the last argument)
    for k, outvar in enumerate(outvars):
        if isinstance(outvar.aval, AbstractQuantumState):
            outvars.pop(k)
            break

    # Create the Jaxpr
    environment_body_jaspr = Jaspr(
        constvars=[],
        invars=invars + enter_eq.outvars,
        outvars=outvars + exit_eqn.invars[-1:],
        eqns=new_eqn_list[i + 1 :],
        debug_info=dummy_debug_info,
    )

    # Create the Equation
    collected_eqn = JaxprEqn(
        params={"type": exit_eqn.params["type"], "jaspr": environment_body_jaspr},
        primitive=exit_eqn.primitive,
        invars=enter_eq.invars[:-1] + invars + enter_eq.invars[-1:],
        outvars=outvars + exit_eqn.outvars[-1:],
        effects=exit_eqn.effects,
        source_info=exit_eqn.source_info,
        ctx=exit_eqn.ctx,
    )

    # Remove the collected equations from new_eqn_list
    return collected_eqn, new_eqn_list[:i], new_eqn_var_tracker.slice_end(i)


def find_outvars(
    body_eqn_list: list[JaxprEqn],
    script_remainder_var_tracker: "VarTracker",
    return_vars: list[Var],
) -> list[Var]:
    """Infer which variables a function body must return.

    The function takes the equations of a function body and some "follow-up"
    instructions and infers which variables need to be returned by the
    function.

    Parameters
    ----------
    body_eqn_list : list[jax.extend.core.JaxprEqn]
        A list of equations describing a function.
    script_remainder_var_tracker : VarTracker
        Tracks the I/O variables of the follow-up equations, i.e. the code
        that runs after the function body and may still need some of its
        variables.
    return_vars : list[jax.extend.core.Var]
        Variables that the enclosing Jaxpr returns and must therefore be kept
        alive even if the follow-up equations don't reference them.

    Returns
    -------
    list[jax.extend.core.Var]
        A list of variables that would have to be returned by the function.

    """
    # This list will contain all variables produced by the function
    outvars = []

    # Fill the list
    for eqn in body_eqn_list:
        outvars.extend(eqn.outvars)

    # Remove the duplicates
    outvars = list(set(outvars))

    # Find which variables are required for executing the follow-up
    required_remainder_vars = script_remainder_var_tracker.find_invars()

    # The result is the intersection between both sets of variables
    return list(set(outvars).intersection(required_remainder_vars + return_vars))


def _register_vars(
    variables: Sequence[Var | Literal],
    var_to_int_dic: dict[Var, int],
    int_to_var_dic: dict[int, Var],
) -> list[int]:
    """Translate a sequence of Jaxpr variables into their tracked integer ids.

    Assign a fresh id to any variable seen for the first time. Literal values
    (e.g. constants embedded directly in an equation) are silently skipped,
    since they aren't SSA variables that need id-tracking.
    """
    integers = []
    for var in variables:
        if isinstance(var, Literal):
            continue
        try:
            integers.append(var_to_int_dic[var])
        # This represents the case that the variable has not been converted yet
        except KeyError:
            var_to_int_dic[var] = len(var_to_int_dic)
            int_to_var_dic[var_to_int_dic[var]] = var
            integers.append(var_to_int_dic[var])
    return integers


def _slice_field(
    values: list[int],
    index_tracker: list[int],
    point: int,
    from_start: bool,
) -> tuple[list[int], list[int]]:
    """Slice one of VarTracker's parallel (values, index_tracker) pairs.

    If from_start, keep values/index positions from `point` onward (used by
    slice_start), re-basing the index_tracker to start at 0. Otherwise, keep
    values/index positions up to (and including) `point` (used by slice_end).
    """
    boundary = index_tracker[point]
    if from_start:
        return values[boundary:], [i - boundary for i in index_tracker[point:]]
    return values[:boundary], index_tracker[: point + 1]


# Equation-list sizes below this (in either invars or outvars) are cheaper to
# process with plain Python than to dispatch into the numba-jitted kernel.
_JIT_COMPILE_THRESHOLD = 20


class VarTracker:
    """Track the input and output variables of a growing equation list efficiently.

    This class is motivated by the task of identifying the inputs and outputs
    of the collected environments. For large Jaxpr, this can be prohibitively
    expensive, which is why this class tracks a specialized data structure, which
    allows an efficient solution of this problem.

    To describe how this works in detail, we start by noting that it is beneficial
    to assign every variable (input and outputs) a simply integer.

    The translation between this assignment is achieved through the var_to_int_dic
    and the int_to_var_dic.

    Given a list of equations, what this class tracks is now a list of integers
    for both inputs and outputs that describe all the inputs and outputs.

    i.e. if we are given the equations

    %1 = prim_0 %2 %3 %4
    %5 %6 = prim_1 %1

    These lists would look like this:

    inputs = [%2, %3, %4, %1]
    outputs = [%1, %5, %6]

    We can now compute efficiently which variables are the invars of this list
    of equations by initializing an array with 6 entries, first setting all
    the invar positions to 1, i.e. [1, 1, 1, 1, 0, 0]
    and then setting all the outvar position to 0, i.e. [0, 1, 1, 1, 0, 0].

    This is implemented in the find_invars method.


    An important feature that is required by the environment collection function
    is to slice the list of equations - afterall the environment body will
    be a slice of the equation list in most cases.

    Because of this, the class tracks another list of integers, demarking
    which interval of integers belongs to which equation. This list always
    starts with 0.

    For the above example we would have

    invar_index_tracker = [0, 3, 4]
    outvar_index_tracker = [0, 1, 3]

    Each entry of these lists therefore denotes where the corresponding equation
    starts.

    This list can be used to efficiently implement the slicing features.
    """

    def __init__(self, eqn_list: list[JaxprEqn]) -> None:
        """Build a VarTracker over the given (possibly empty) equation list."""
        self.var_to_int_dic: dict[Var, int] = {}
        self.int_to_var_dic: dict[int, Var] = {}

        self.eqn_invar_list: list[int] = []
        self.eqn_outvar_list: list[int] = []

        self.invar_eqn_index_tracker: list[int] = [0]
        self.outvar_eqn_index_tracker: list[int] = [0]

        for eqn in eqn_list:
            self.append(eqn)

    def append(self, eqn: JaxprEqn) -> None:
        """Add the equation eqn to the list of tracked equations.

        Parameters
        ----------
        eqn : jax.extend.core.JaxprEqn
            The equation whose invars/outvars should be registered and tracked.

        """
        self.eqn_invar_list.extend(_register_vars(eqn.invars, self.var_to_int_dic, self.int_to_var_dic))
        self.invar_eqn_index_tracker.append(len(self.eqn_invar_list))

        self.eqn_outvar_list.extend(_register_vars(eqn.outvars, self.var_to_int_dic, self.int_to_var_dic))
        self.outvar_eqn_index_tracker.append(len(self.eqn_outvar_list))

    def slice_start(self, starting_point: int) -> "VarTracker":
        """Return the equivalent of VarTracker(eqn_list[starting_point:]).

        Parameters
        ----------
        starting_point : int
            The equation index to start the slice from.

        """
        res = VarTracker([])

        res.eqn_invar_list, res.invar_eqn_index_tracker = _slice_field(
            self.eqn_invar_list, self.invar_eqn_index_tracker, starting_point, from_start=True
        )
        res.eqn_outvar_list, res.outvar_eqn_index_tracker = _slice_field(
            self.eqn_outvar_list, self.outvar_eqn_index_tracker, starting_point, from_start=True
        )

        res.int_to_var_dic = self.int_to_var_dic
        res.var_to_int_dic = self.var_to_int_dic

        return res

    def slice_end(self, end_point: int) -> "VarTracker":
        """Return the equivalent of VarTracker(eqn_list[:end_point]).

        Parameters
        ----------
        end_point : int
            The equation index to end the slice at (exclusive).

        """
        res = VarTracker([])

        res.eqn_invar_list, res.invar_eqn_index_tracker = _slice_field(
            self.eqn_invar_list, self.invar_eqn_index_tracker, end_point, from_start=False
        )
        res.eqn_outvar_list, res.outvar_eqn_index_tracker = _slice_field(
            self.eqn_outvar_list, self.outvar_eqn_index_tracker, end_point, from_start=False
        )

        res.int_to_var_dic = self.int_to_var_dic
        res.var_to_int_dic = self.var_to_int_dic

        return res

    def find_invars(self) -> list[Var]:
        """Compute the undefined invars of the currently tracked equation list.

        I.e. all the variables that are used as invars but not defined by one
        of the equations.

        Returns
        -------
        list[jax.extend.core.Var]

        """
        # If viable, call the jitted version.
        if len(self.eqn_invar_list) < _JIT_COMPILE_THRESHOLD or len(self.eqn_outvar_list) < _JIT_COMPILE_THRESHOLD:
            invar_index_list = find_invar_kernel([-1] + self.eqn_invar_list, [-1] + self.eqn_outvar_list)
        else:
            invar_index_list: np.ndarray = jitted_find_invar_kernel(
                np.array([-1] + self.eqn_invar_list, dtype=np.int32),
                np.array([-1] + self.eqn_outvar_list, dtype=np.int32),  # type: ignore[reportCallIssue]
            )

        # Convert from integers to variables
        res = [self.int_to_var_dic[idx] for idx in invar_index_list]

        # For some jaxpr transformations, it is important that the order of invars
        # returned by this function is according to the order of appearance in the
        # body.

        # We therefore create a dictionary that indicates the index of the first
        # usage as an invar and sort according to this dictionary.
        sorting_dic = {self.eqn_invar_list[i]: i for i in range(len(self.eqn_invar_list))[::-1]}

        res.sort(key=lambda x: sorting_dic[self.var_to_int_dic[x]])

        return res


def find_invar_kernel(invar_indices: list[int], outvar_indices: list[int]) -> np.ndarray:
    """Execute the algorithm described in the docstring of VarTracker.

    Parameters
    ----------
    invar_indices : list[int]
        Every invar occurrence across the tracked equations, as integer ids,
        prefixed with a sentinel -1.
    outvar_indices : list[int]
        Every outvar occurrence across the tracked equations, as integer ids,
        prefixed with a sentinel -1.

    Returns
    -------
    numpy.ndarray
        The integer ids of the variables that are invars but not outvars.

    """
    max_invar = np.max(invar_indices)
    max_outvar = np.max(outvar_indices)
    max_var = max(max_invar, max_outvar)

    if max_var == -1:
        return np.zeros(0, dtype=np.int64)

    invar_array = np.zeros(max_var + 1, dtype=np.int8)
    invar_array[invar_indices] = 1
    invar_array[outvar_indices] = 0
    res = np.nonzero(invar_array)[0]

    return res


jitted_find_invar_kernel = njit(find_invar_kernel)
