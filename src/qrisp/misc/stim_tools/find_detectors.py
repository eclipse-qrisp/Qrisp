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

"""
Automatic detector identification for quantum error correction circuits.

This module provides the :func:`find_detectors` decorator, which leverages
`tqecd <https://github.com/QCHackers/tqecd>`_ to automatically identify
stim ``DETECTOR`` instructions at Jasp tracing time and emit the
corresponding ``parity()`` calls.

Usage
-----
.. code-block:: python

    from qrisp import *
    from qrisp.jasp.evaluation_tools.stim_extraction import extract_stim
    from qrisp.misc.stim_tools.find_detectors import find_detectors

    @find_detectors
    def syndrome_round(qa):
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        cx(qa[2], qa[3]); cx(qa[4], qa[3])
        return measure(qa[1]), measure(qa[3])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m1, m3 = syndrome_round(qa)
        return detectors, m1, m3
"""

import stim
import numpy as np
from jax.tree_util import tree_flatten

from qrisp import QuantumArray, QuantumBool
from qrisp.jasp import make_jaspr
from qrisp.jasp.primitives.parity_primitive import parity
from qrisp.circuit import QuantumCircuit as QC, Clbit


# ───────────────────── stim instruction categories ─────────────────────
_RESET_OPS = frozenset({"R", "RX", "RY"})
_MEASUREMENT_OPS = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY"})
_TWO_QUBIT_OPS = frozenset({"CX", "CY", "CZ", "XCZ", "YCZ", "SWAP", "ISWAP"})


# ───────────────────── internal helpers ────────────────────────────────

def _count_qubits(*args):
    """Count total qubits across all QuantumArray args."""
    total = 0
    for a in args:
        if isinstance(a, QuantumArray):
            total += a.size * a.qtype.size
    return total


def _build_analysis_args(inner_jaspr, num_qubits, original_args):
    """Build concrete args for calling ``to_qc`` on *inner_jaspr*.

    Each QuantumArray argument receives its own disjoint slice of qubits
    from the analysis circuit, matching the qubit count of the original
    QuantumArray.
    """
    analysis_qc = QC(num_qubits)
    qubit_list = list(analysis_qc.qubits)

    # Partition qubits into slices corresponding to each QuantumArray arg
    qubit_offset = 0
    qubit_slices = []
    for a in original_args:
        if isinstance(a, QuantumArray):
            n = a.size * a.qtype.size
            qubit_slices.append(qubit_list[qubit_offset:qubit_offset + n])
            qubit_offset += n

    slice_iter = iter(qubit_slices)

    invars = inner_jaspr.invars[:-1]  # exclude trailing QC
    args = []
    for v in invars:
        aval = v.aval
        aval_str = str(aval)
        if aval_str == "QubitArray":
            args.append(next(slice_iter))
        elif "int" in aval_str and hasattr(aval, "shape") and aval.shape:
            args.append(np.arange(num_qubits))
        elif "int" in aval_str:
            args.append(1)
        else:
            raise ValueError(f"Unexpected invar type: {aval_str}")
    return args


def _prepare_for_tqecd(stim_circ):
    """
    Re-structure a stim circuit into the fragment format expected by tqecd.

    tqecd requires each *fragment* to follow this layout:

    1. Zero or more moments of reset operations
    2. Zero or more moments of computation gates (no resets or measurements)
    3. Exactly one moment of measurement operations

    Moments are separated by ``TICK`` instructions.  Multi-round circuits are
    split at measurement boundaries so each round becomes its own fragment.
    The first fragment receives implicit ``R`` (Z-basis reset) for every qubit
    that was not explicitly reset by the user's circuit.
    """
    # Collect all qubit indices
    all_qubits = set()
    for inst in stim_circ.flattened():
        for t in inst.targets_copy():
            if not t.is_measurement_record_target:
                all_qubits.add(t.value)

    # --- Split instructions into rounds ---
    # A round boundary occurs when we've accumulated measurements and the
    # next instruction is no longer a measurement.
    rounds = []
    current = {"resets": [], "gates": [], "measurements": []}

    for inst in stim_circ.flattened():
        name = inst.name
        if name == "TICK":
            continue

        if current["measurements"] and name not in _MEASUREMENT_OPS:
            rounds.append(current)
            current = {"resets": [], "gates": [], "measurements": []}

        if name in _RESET_OPS:
            current["resets"].append(inst)
        elif name in _MEASUREMENT_OPS:
            current["measurements"].append(inst)
        else:
            current["gates"].append(inst)

    if current["measurements"]:
        rounds.append(current)

    # --- Build output circuit ---
    new_circ = stim.Circuit()

    for q in sorted(all_qubits):
        new_circ.append("QUBIT_COORDS", [q], [float(q), 0.0])

    for round_idx, rnd in enumerate(rounds):
        # Resets
        explicitly_reset = set()
        for inst in rnd["resets"]:
            for t in inst.targets_copy():
                explicitly_reset.add(t.value)

        # First round: implicit R for un-reset qubits (tqecd requires it)
        if round_idx == 0:
            unreset = sorted(all_qubits - explicitly_reset)
            if unreset:
                new_circ.append("R", unreset)

        for inst in rnd["resets"]:
            targets = [t.value for t in inst.targets_copy()]
            new_circ.append(inst.name, targets, inst.gate_args_copy())

        new_circ.append("TICK")

        # Gate moments — two-qubit gates each get their own moment
        for inst in rnd["gates"]:
            name = inst.name
            targets = inst.targets_copy()
            gate_args = inst.gate_args_copy()

            if name in _TWO_QUBIT_OPS:
                for i in range(0, len(targets), 2):
                    new_circ.append(
                        name,
                        [targets[i].value, targets[i + 1].value],
                        gate_args,
                    )
                    new_circ.append("TICK")
            else:
                target_vals = [t.value for t in targets]
                new_circ.append(name, target_vals, gate_args)
                new_circ.append("TICK")

        # Measurements
        for inst in rnd["measurements"]:
            targets = [t.value for t in inst.targets_copy()]
            new_circ.append(inst.name, targets, inst.gate_args_copy())

        if round_idx < len(rounds) - 1:
            new_circ.append("TICK")

    return new_circ


def _extract_detectors(annotated_circ, total_meas, inv_meas_map):
    """
    Extract detector definitions from an annotated stim circuit.

    Returns a list of lists of :class:`~qrisp.circuit.Clbit`, one list per
    ``DETECTOR`` instruction.
    """
    detectors = []
    for inst in annotated_circ.flattened():
        if inst.name == "DETECTOR":
            clbits = []
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    abs_idx = total_meas + t.value  # rec[] is negative
                    clbits.append(inv_meas_map[abs_idx])
            detectors.append(clbits)
    return detectors


# ───────────────────── public API ──────────────────────────────────────

def find_detectors(func=None, *, return_circuits=False):
    r"""Decorator that automatically identifies stim detectors via tqecd.

    ``find_detectors`` traces the decorated function, converts the resulting
    circuit to a stim circuit, re-structures it into the fragment format that
    `tqecd <https://github.com/QCHackers/tqecd>`_ expects, runs automatic
    detector annotation, and finally emits ``parity()`` calls for every
    discovered detector during the real Jasp tracing pass.

    .. note::

        This feature requires the optional ``tqecd`` package.  Install it
        with ``pip install tqecd``.

    Parameters
    ----------
    func : callable, optional
        The function to decorate.  When using keyword arguments (e.g.
        ``@find_detectors(return_circuits=True)``), *func* is ``None`` and
        the decorator returns a wrapper that accepts *func*.
    return_circuits : bool, default ``False``
        When ``True``, three additional items are appended to the return
        value (after the detector list and the original returns):

        * **raw_stim** — the stim circuit obtained directly from ``to_qc``
          / ``to_stim``, before any restructuring.
        * **tqecd_input** — the circuit after ``_prepare_for_tqecd`` has
          re-arranged it into proper fragments (the input fed to tqecd).
        * **annotated** — the circuit that comes back from
          ``tqecd.annotate_detectors_automatically``, containing the
          ``DETECTOR`` instructions.

        This is useful for debugging or understanding why detectors were
        or were not discovered.

    Returns
    -------
    The decorated function returns::

        (detector_bools, *original_returns)

    or, when ``return_circuits=True``::

        (detector_bools, *original_returns, raw_stim, tqecd_input, annotated)

    where *detector_bools* is a ``list`` of Jasp-traced boolean values, one
    per discovered detector, each representing the ``parity()`` of the
    measurements that constitute that detector.

    Pitfalls & Limitations
    ----------------------
    **Arguments must be QuantumArrays of QuantumBool**
        Every positional argument to the decorated function must be a
        ``QuantumArray`` with ``qtype=QuantumBool()``.  Other quantum types
        or classical arguments are not supported and will raise a
        ``TypeError``.

    **Only Clifford gates are supported**
        tqecd analyses stabilizer flows, which are only well-defined for
        Clifford circuits.  If the decorated function contains non-Clifford
        gates (e.g. T, Toffoli, arbitrary-angle rotations), stim will raise
        an error during circuit construction or tqecd will silently produce
        incorrect results.

    **No real-time classical computation inside the decorated function**
        The circuit is traced symbolically.  Conditional logic that depends
        on measurement outcomes (real-time ``if``/``else``) cannot be
        expressed in the stim circuit and will break analysis.

    **Composite detectors from flow products are not discoverable**
        tqecd finds detectors by tracking individual stabilizer flows from
        resets through gates to measurements.  It **cannot** discover
        composite detectors that arise as products of multiple stabilizer
        flows.  For example, in a GHZ-state circuit the parity
        :math:`Z_3 Z_4` is a product of :math:`Z_0 Z_3` and
        :math:`Z_0 Z_4`, but tqecd will not identify it.

    **Multi-round circuits require explicit resets**
        For tqecd to detect round-to-round detectors (comparing syndrome
        measurements across rounds), ancilla qubits **must** be explicitly
        reset between rounds.  Without resets, tqecd cannot establish the
        start of a new stabilizer flow and will miss cross-round detectors.

    **Fragment structure is automatically inferred**
        The decorator splits the flat stim circuit into tqecd *fragments*
        at measurement boundaries (a new fragment starts whenever a
        non-measurement instruction follows a measurement).  This heuristic
        works for standard syndrome-extraction patterns but may fail for
        unusual gate orderings or interleaved measurement/gate patterns.

    **First-round detectors are always included**
        Because all qubits are implicitly reset at the beginning (tqecd
        requires it), measurements in the first round that are deterministic
        given those resets will appear as detectors.  This is standard QEC
        behaviour — the first round's ancilla outcomes are compared against
        the known reset state.

    **Detectors referencing non-returned measurements are discarded**
        If a detector involves a measurement whose result is *not* part of
        the decorated function's return value, the detector is silently
        dropped.  Make sure all relevant measurements are returned.

    **Two-qubit gates are split into individual tqecd moments**
        To avoid qubit collisions within a single moment (which tqecd
        forbids), every two-qubit gate pair is placed in its own moment.
        This is correct but may create more moments than strictly necessary
        for circuits where the two-qubit gates already act on disjoint
        qubits.

    **Multiple QuantumArray arguments receive disjoint qubit slices**
        When the decorated function takes more than one QuantumArray
        argument, it is assumed that each argument indeed represents
        a distinct set of qubits. This can not be guaranteed at tracing
        time and needs to be ensured from user side.
        Each array is mapped to its own contiguous slice of
        qubits during analysis.  The slices are allocated in positional
        order.

    Examples
    --------
    Single-round syndrome extraction (three data qubits, two ancilla
    checks):

    >>> from qrisp import *
    >>> from qrisp.jasp.evaluation_tools.stim_extraction import extract_stim
    >>> from qrisp.misc.stim_tools.find_detectors import find_detectors
    >>>
    >>> @find_detectors
    ... def syndrome(qa):
    ...     reset(qa[3]); reset(qa[4])
    ...     cx(qa[0], qa[3]); cx(qa[1], qa[3])
    ...     cx(qa[1], qa[4]); cx(qa[2], qa[4])
    ...     return measure(qa[3]), measure(qa[4])
    ...
    >>> @extract_stim
    ... def main():
    ...     qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
    ...     detectors, m3, m4 = syndrome(qa)
    ...     return detectors, m3, m4

    Two-round repetition code (detectors compare across rounds):

    >>> @find_detectors
    ... def two_rounds(qa):
    ...     for _ in range(2):
    ...         reset(qa[1]); reset(qa[3])
    ...         cx(qa[0], qa[1]); cx(qa[2], qa[1])
    ...         cx(qa[2], qa[3]); cx(qa[4], qa[3])
    ...         m1 = measure(qa[1]); m3 = measure(qa[3])
    ...     return m1, m3  # only last round's measurements returned

    Debug mode — inspect intermediate circuits:

    >>> @find_detectors(return_circuits=True)
    ... def debug_syndrome(qa):
    ...     reset(qa[1])
    ...     cx(qa[0], qa[1]); cx(qa[2], qa[1])
    ...     return measure(qa[1])
    """
    try:
        import tqecd
    except ImportError:
        raise ImportError(
            "find_detectors requires the 'tqecd' package. "
            "Install it with: pip install tqecd"
        )

    def _decorator(fn):
        def wrapper(*args, **kwargs):
            # --- Validate args ---
            for a in args:
                if not isinstance(a, QuantumArray):
                    raise TypeError(
                        f"find_detectors: all args must be QuantumArray, "
                        f"got {type(a)}"
                    )
                if not isinstance(a.qtype, QuantumBool):
                    raise TypeError(
                        f"find_detectors: all args must have qtype "
                        f"QuantumBool, got {a.qtype}"
                    )

            num_qubits = _count_qubits(*args)

            # --- 1. Trace inner function to get sub-jaspr ---
            inner_jaspr = make_jaspr(fn)(*args, **kwargs)

            # --- 2. Analysis: to_qc → stim → tqecd ---
            analysis_args = _build_analysis_args(inner_jaspr, num_qubits, args)
            result = inner_jaspr.to_qc(*analysis_args)

            analysis_qc = result[-1]
            analysis_returns = result[:-1]

            # Flatten analysis Clbits (consistent ordering via tree_flatten)
            analysis_clbits_flat, _ = tree_flatten(analysis_returns)

            # Convert to stim
            raw_stim_circ, meas_map = analysis_qc.to_stim(
                return_measurement_map=True
            )
            inv_meas_map = {v: k for k, v in meas_map.items()}

            # Prepare for tqecd and annotate
            tqecd_circ = _prepare_for_tqecd(raw_stim_circ)
            annotated = tqecd.annotate_detectors_automatically(tqecd_circ)
            total_meas = tqecd_circ.num_measurements

            # Extract detector Clbit sets
            detector_clbit_sets = _extract_detectors(
                annotated, total_meas, inv_meas_map
            )

            # Keep only detectors whose Clbits all appear in returned meas
            returned_clbit_set = set(analysis_clbits_flat)
            relevant_detectors = [
                det
                for det in detector_clbit_sets
                if all(cb in returned_clbit_set for cb in det)
            ]

            # --- 3. Call real function to get traced results ---
            real_returns = fn(*args, **kwargs)

            if not isinstance(real_returns, tuple):
                real_returns = (real_returns,)

            traced_bools_flat, _ = tree_flatten(real_returns)

            # Map analysis Clbits to traced booleans
            clbit_to_traced = dict(
                zip(analysis_clbits_flat, traced_bools_flat)
            )

            # --- 4. Emit parity calls for each detector ---
            detector_results = []
            for det_clbits in relevant_detectors:
                meas_args = [clbit_to_traced[cb] for cb in det_clbits]
                det_result = parity(*meas_args, expectation=True)
                detector_results.append(det_result)

            out = (detector_results,) + real_returns

            if return_circuits:
                out = out + (raw_stim_circ, tqecd_circ, annotated)

            return out

        return wrapper

    # Support both @find_detectors and @find_detectors(return_circuits=True)
    if func is not None:
        return _decorator(func)
    return _decorator
