"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

import weakref

from jaxlib.xla_extension import ArrayImpl

# This module contains the necessary tools to merge QuantumSessions

# Due to the interplay with QuantumEnvironments and the behavior
# that QuantumEnvironments automatically detect QuantumSessions that operated
# within them there is quite a bit of complexity involved here.

# The idea to merge two session qs_0 and qs_1 is to first add the appropriate
# amount of qubits/clbits to qs_0, append all the instructions and
# fix the other attributes Then we let all the attributes of qs_1 point
# to the corresponding attributes of qs_0 such that any change on one quantum session
# also appears in the other quantum session

# Therefore, two QuantumSessions are declared as identical (or merged) if their
# attributes (like .data, .qubits) point to the same lists as the other one.
# The == operator for QuantumSessions checks if the .data attribute points to the same
# list.


# If a merge is requested, the first step is to "engage" the QuantumSessions in the
# environment stack. By default, QuantumEnvironments create their own QuantumSession
# and "engaging" here means that if merge is called on a QuantumSession within an
# environment, this QuantumSession will be merged with all the QuantumSessions of the
# environment stack.
def merge_sessions(qs_0, qs_1):
    # We now iterate through every QuantumEnvironment and insert its data into the
    # correct original_data attribute
    for qs in [qs_0, qs_1]:
        # We need to find the environment where the env_qs quantum session is not merged
        # into qs. This implies that the instructions of qs have been appended in this
        # environment's parent. Therefore, all the data needs to go into the
        # original_data attribute of this environment.
        for env in qs.env_stack:
            if not env.env_qs == qs:
                env.original_data.extend(qs.data)
                qs.data = []
                merge_sessions_inner(qs, env.env_qs)

    merge_sessions_inner(qs_0, qs_1)


def merge_sessions_inner(qs_0, qs_1, merge_env_stack_=True):
    if qs_0 == qs_1:
        return

    # from qrisp.quantum_network import QuantumNetworkSession

    # if isinstance(qs_0, QuantumNetworkSession):
    #     if qs_0.backend is not qs_1.backend:
    #         if isinstance(qs_1, QuantumNetworkSession):
    #             raise Exception(
    #                 "Tried to merge two QuantumNetworkSessions containing "
    #                 "differing clients"
    #             )

    #         if qs_1.backend is not None:
    #             raise Exception(
    #                 "Tried to merge a QuantumNetworkSession with a "
    #                 "QuantumSession with a non-trivial backend"
    #             )

    #     qs_1.backend = qs_0.backend
    #     qs_1.inbox = qs_0.inbox
    #     qs_1.__class__ = qs_0.__class__

    # if isinstance(qs_1, QuantumNetworkSession):
    #     if qs_1.backend is not qs_0.backend:
    #         if isinstance(qs_0, QuantumNetworkSession):
    #             raise Exception(
    #                 "Tried to merge two QuantumNetworkSessions containing "
    #                 "differing clients"
    #             )

    #         if qs_0.backend is not None:
    #             raise Exception(
    #                 "Tried to merge a QuantumNetworkSession with a "
    #                 "QuantumSession with a non-trivial backend"
    #             )

    #     qs_0.backend = qs_1.backend
    #     qs_0.inbox = qs_1.inbox
    #     qs_0.__class__ = qs_1.__class__

    # The problem we face here is the following:
    # If two sessions a, b are merged and b is merged with another session c at a later
    # point, the attributes of b are updated by that merge, but the attributes of a are
    # not. To solve this problem, we declare one of the merge participants the "parent"
    # of the merge. The parent has an attribute called "shadow_sessions" that contain
    # the sessions that the parent has been merged with. The attribute updates are then
    # performed on all shadow sessions.

    # To make sure we only operate on parent sessions, we check the parent attribute
    while hasattr(qs_0, "parent"):
        qs_0 = qs_0.parent

    while hasattr(qs_1, "parent"):
        qs_1 = qs_1.parent

    resolve_naming_collisions(qs_0, qs_1)

    intersecting_qubits = set([qb.identifier for qb in qs_0.qubits]).intersection(
        [qb.identifier for qb in qs_1.qubits]
    )

    if len(intersecting_qubits):
        raise Exception(
            f"Tried to merge two QuantumSessions containing identically named "
            f"qubits {intersecting_qubits}"
        )

    if len(qs_0.env_stack) < len(qs_1.env_stack):
        qs_0, qs_1 = qs_1, qs_0

    if merge_env_stack_:
        merge_env_stack(qs_0, qs_1)

        if qs_0 == qs_1:
            return

    # The qs_tracker list should only contain one Session instance per equivalence
    # class, i.e. if two sessions are merged only one of them should appear in the
    # qs_tracker
    i = 0
    while i < len(qs_0.qs_tracker):
        if qs_0.qs_tracker[i]() is None:
            qs_0.qs_tracker.pop(i)
        elif qs_0.qs_tracker[i]() == qs_1:
            qs_0.qs_tracker.pop(i)
        else:
            i += 1

    # Each environment contains a list of QuantumSessions that when it was entered.
    # This list determines if a QuantumSession is engaged into an QuantumEnvironment, if
    # merge is called from inside the QuantumEnvironment. We only want to have one of
    # the two QuantumSessions that are merged in this list, which is resolved by the
    # identify_sessions_in_environment
    for i in range(len(qs_1.env_stack)):
        identify_sessions_in_environment(qs_1.env_stack[-i - 1], qs_0, qs_1)

    if qs_0.backend is not None and qs_1.backend is not None:
        if id(qs_0.backend) != id(qs_1.backend):
            raise Exception(
                "Tried to merge QuantumSessions with differing, "
                "non-trivial default backends."
            )

    if qs_0.backend is None:
        qs_0.backend = qs_1.backend

    if set(qs_0.clbits).intersection(qs_1.clbits):
        raise Exception("Tried to merge sessions with common classical bits")

    for qb in qs_1.qubits:
        qb.qs = weakref.ref(qs_0)
    for cb in qs_1.clbits:
        cb.qs = weakref.ref(qs_0)

    qs_0.qubits.extend(qs_1.qubits)
    qs_0.clbits.extend(qs_1.clbits)

    for i in range(len(qs_0.clbits)):
        qs_0.clbits[i].identifier = f"clbit_{i}"

    qs_0.data.extend(qs_1.data)

    object.__setattr__(qs_1, "data", qs_0.data)

    # Update qv_list
    while len(qs_1.qv_list):
        qv = qs_1.qv_list.pop()
        # Patch quantum session attribute
        qv.qs = qs_0

        qs_0.qv_list.append(qv)

    for qv in qs_1.deleted_qv_list:
        # Patch quantum session attribute
        qv.qs = qs_0

    qs_0.deleted_qv_list.extend(qs_1.deleted_qv_list)

    reorder_quantum_variables(qs_0)

    qs_0.will_be_uncomputed = bool(qs_0.will_be_uncomputed) or bool(
        qs_1.will_be_uncomputed
    )
    # Add variables to the uncomputation stack
    qs_0.uncomp_stack.extend(qs_1.uncomp_stack)

    qs_0.shadow_sessions.extend(qs_1.shadow_sessions + [weakref.ref(qs_1)])

    i = 0
    while i < len(qs_0.shadow_sessions):
        qs = qs_0.shadow_sessions[i]()
        if qs is None:
            qs_0.shadow_sessions.pop(i)
            continue

        qs.parent = qs_0

        # Update all the attributes
        for var in qs_0.__dict__.keys():
            qs.__dict__[var] = qs_0.__dict__[var]

        i += 1


def resolve_naming_collisions(qs_0, qs_1):
    qs_names_0 = [qv.name for qv in qs_0.qv_list + qs_0.deleted_qv_list]
    qs_names_1 = [qv.name for qv in qs_1.qv_list + qs_1.deleted_qv_list]

    intersecting_qv_names = set(qs_names_0).intersection(qs_names_1)

    if len(intersecting_qv_names):
        for qv_name in intersecting_qv_names:
            qv_0_index = qs_names_0.index(qv_name)

            qv_0 = (qs_0.qv_list + qs_0.deleted_qv_list)[qv_0_index]

            qv_1_index = qs_names_1.index(qv_name)
            qv_1 = (qs_1.qv_list + qs_1.deleted_qv_list)[qv_1_index]

            if qv_1.user_given_name:
                if qv_0.user_given_name:
                    raise Exception(
                        "Tried to merge QuantumSession containing "
                        f"identically named QuantumVariables {qv_1.name}"
                    )

                qv_0, qv_1 = qv_1, qv_0

            else:
                if qv_0.creation_time > qv_1.creation_time:
                    if not qv_1.user_given_name:
                        qv_0, qv_1 = qv_1, qv_0

            proposed_new_name = qv_1.name + "_1"
            k = 1

            while proposed_new_name in qs_names_0 + qs_names_1:
                k += 1
                proposed_new_name = qv_1.name + "_" + str(k)

            qv_1.name = proposed_new_name

            for k in range(len(qv_1.reg)):
                qv_1.reg[k].identifier = qv_1.name + "." + str(k)


def merge_env_stack(qs_0, qs_1):
    # We need to find the environment where the env_qs quantum session is not merged
    # into qs. This implies that the instructions of this session have been appended in
    # this environment's parent. Therefore, all the data needs to go into the
    # original_data attribute of this environment.
    for env in qs_0.env_stack:
        if env.env_qs == qs_0 and not env.env_qs == qs_1:
            if env in qs_1.env_stack:
                env.original_data.extend(qs_1.data)
                qs_1.data = []

        if env.env_qs == qs_1 and not env.env_qs == qs_0:
            for env_ in qs_0.env_stack:
                if not env_.env_qs == qs_0:
                    env_.original_data.extend(qs_0.data)
                    qs_0.data = []
                    merge_sessions_inner(qs_0, env_.env_qs, False)
                if env.env_qs == env_.env_qs:
                    break


def reorder_quantum_variables(qs):
    overall_qv_list = qs.qv_list + qs.deleted_qv_list

    qs.qv_list.sort(key=lambda x: x.creation_time)

    overall_qv_list.sort(key=lambda x: x.creation_time)

    sorted_qubit_list = []
    for qv in overall_qv_list:
        sorted_qubit_list += qv.reg

    sorted_qubit_list.extend(list(set(qs.qubits) - set(sorted_qubit_list)))

    qs.qubits = sorted_qubit_list


def identify_sessions_in_environment(env, qs_0, qs_1):
    i = 0
    while i < len(env.active_qs_list):
        if env.active_qs_list[i]() is None:
            env.active_qs_list.pop(i)
            continue
        elif env.active_qs_list[i]() == qs_1:
            env.active_qs_list.pop(i)
            continue
        i += 1


def multi_session_merge(input_list):
    session_list = input_list
    session_list.sort(key=lambda x: -len(x.data))

    if not len(session_list):
        from qrisp.core import QuantumSession

        return QuantumSession()

    for i in range(len(session_list)):
        merge_sessions(session_list[0], session_list[i])

    return session_list[0]


def recursive_qs_search(input):
    if isinstance(input, str):
        return []

    from qrisp.core import QuantumSession
    from qrisp.environments import QuantumEnvironment

    if hasattr(input, "__iter__"):
        iterable = True

    # This case would in principle allow for also searching objects
    # this however requires a considerable chunk of resources
    elif hasattr(input, "__dict__") and not isinstance(input, QuantumSession):
        iterable = False
        # input = input.__dict__
    else:
        iterable = False

    result = []
    if iterable:
        if isinstance(input, dict):
            for key in input.keys():
                result += recursive_qs_search(key)
                result += recursive_qs_search(input[key])
        else:
            input = list(input)
            for i in range(len(input)):
                if isinstance(input[i], ArrayImpl):
                    continue
                result += recursive_qs_search(input[i])
    else:
        if isinstance(input, QuantumSession):
            result = [input]
        elif isinstance(input, QuantumEnvironment):
            result = [input.env_qs]
        else:
            try:
                if isinstance(input(), QuantumSession):
                    result = [input()]
            except TypeError:
                pass

            try:
                if isinstance(input.qs(), QuantumSession):
                    result = [input.qs()]
            except AttributeError:
                pass

    return result


def recursive_qv_search(input):
    if isinstance(input, str):
        return []
    from qrisp.core import QuantumVariable

    if isinstance(input, QuantumVariable):
        return [input]

    if hasattr(input, "__iter__"):
        iterable = True
    elif hasattr(input, "__dict__") and not isinstance(input, QuantumVariable):
        iterable = False
        # input = input.__dict__
    else:
        iterable = False

    result = []
    if iterable:
        if isinstance(input, dict):
            for key in input.keys():
                result += recursive_qv_search(key)
                result += recursive_qv_search(input[key])
        elif isinstance(input, (tuple, list)):
            for i in range(len(input)):
                result += recursive_qv_search(input[i])

    return result


def recursive_qa_search(input):
    if isinstance(input, str):
        return []
    from qrisp.core import QuantumArray

    if isinstance(input, QuantumArray):
        return [input]

    if hasattr(input, "__iter__"):
        iterable = True
    elif hasattr(input, "__dict__") and not isinstance(input, QuantumArray):
        iterable = False
        # input = input.__dict__
    else:
        iterable = False

    result = []
    if iterable:
        if isinstance(input, dict):
            for key in input.keys():
                result += recursive_qa_search(key)
                result += recursive_qa_search(input[key])
        elif isinstance(input, (tuple, list)):
            for i in range(len(input)):
                result += recursive_qa_search(input[i])

    return result


def merge(*args):
    session_list = recursive_qs_search(args)
    return multi_session_merge(session_list)
