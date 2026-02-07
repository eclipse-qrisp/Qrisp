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
`tqecd <https://github.com/tqec/tqecd>`_ to automatically identify
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
import jax
from jax.tree_util import tree_flatten

from qrisp import QuantumArray, QuantumBool
from qrisp.jasp import make_jaspr
from qrisp.jasp.primitives.parity_primitive import parity
from qrisp.circuit import QuantumCircuit as QC

# ───────────────────── stim instruction categories ─────────────────────
_RESET_OPS = frozenset({"R", "RX", "RY"})
_MEASUREMENT_OPS = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY"})
_TWO_QUBIT_OPS = frozenset({"CX", "CY", "CZ", "XCZ", "YCZ", "SWAP", "ISWAP"})


# ───────────────────── internal helpers ────────────────────────────────


def _tvals(inst):
    """Return plain target-value list for a stim instruction."""
    return [t.value for t in inst.targets_copy()]


def _build_analysis_args(jaspr, n_qubits, quantum_arrays):
    """Build concrete args for calling ``to_qc`` on *jaspr*.

    Iterates over the jaspr's invars and replaces ``QubitArray`` types with
    concrete qubit lists from an analysis circuit.  Static parameters have
    been bound via closure before tracing, so only QuantumArray-derived
    invars remain.
    """
    qubits = list(QC(n_qubits).qubits)

    # Partition qubits into slices corresponding to each QuantumArray arg
    off, slices = 0, []
    for a in quantum_arrays:
        n = a.size * a.qtype.size
        slices.append(qubits[off:off + n])
        off += n
    it = iter(slices)

    args = []
    for v in jaspr.invars[:-1]:                          # exclude trailing QC
        s = str(v.aval)
        if s == "QubitArray":
            args.append(next(it))
        elif "int" in s and getattr(v.aval, "shape", ()):
            args.append(np.arange(n_qubits))             # ind_array for qubit indexing
        elif "int" in s or "bool" in s:
            args.append(1)                               # qtype_size (always 1 for QuantumBool)
        else:
            raise ValueError(f"Unexpected invar: {s}")
    return args


def _prepare_for_tqecd(stim_circ):
    """Re-structure *stim_circ* into the fragment format expected by tqecd.

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
    all_qubits = {t.value for i in stim_circ.flattened()
                  for t in i.targets_copy()
                  if not t.is_measurement_record_target}

    # Split instructions into rounds.
    # A round boundary occurs when we've accumulated measurements and the
    # next instruction is no longer a measurement.
    rounds, cur = [], {"R": [], "G": [], "M": []}
    for inst in stim_circ.flattened():
        if inst.name == "TICK":
            continue
        if cur["M"] and inst.name not in _MEASUREMENT_OPS:
            rounds.append(cur)
            cur = {"R": [], "G": [], "M": []}
        key = "R" if inst.name in _RESET_OPS else \
              "M" if inst.name in _MEASUREMENT_OPS else "G"
        cur[key].append(inst)
    if cur["M"]:
        rounds.append(cur)

    out = stim.Circuit()
    for q in sorted(all_qubits):
        out.append("QUBIT_COORDS", [q], [float(q), 0.0])

    for ri, rnd in enumerate(rounds):
        # First round: implicit R for un-reset qubits (tqecd requires it)
        reset_qbs = {t.value for i in rnd["R"] for t in i.targets_copy()}
        if ri == 0 and (unreset := sorted(all_qubits - reset_qbs)):
            out.append("R", unreset)
        for i in rnd["R"]:
            out.append(i.name, _tvals(i), i.gate_args_copy())
        out.append("TICK")

        # Gate moments — two-qubit gates each get their own moment

        for i in rnd["G"]:
            tgts, ga = i.targets_copy(), i.gate_args_copy()
            if i.name in _TWO_QUBIT_OPS:
                for j in range(0, len(tgts), 2):
                    out.append(i.name, [tgts[j].value, tgts[j+1].value], ga)
                    out.append("TICK")
            else:
                out.append(i.name, _tvals(i), ga)
                out.append("TICK")

        # Measurements
        for i in rnd["M"]:
            out.append(i.name, _tvals(i), i.gate_args_copy())
        if ri < len(rounds) - 1:
            out.append("TICK")
    return out


def _extract_detectors(annotated_circ, total_meas, inv_meas_map):
    """Extract detector definitions from an annotated stim circuit.

    Returns a list of lists of :class:`~qrisp.circuit.Clbit`, one list per
    ``DETECTOR`` instruction.
    """
    return [
        [inv_meas_map[total_meas + t.value]
         for t in inst.targets_copy() if t.is_measurement_record_target]
        for inst in annotated_circ.flattened() if inst.name == "DETECTOR"
    ]


# ───────────────────── public API ──────────────────────────────────────

def find_detectors(func=None, *, return_circuits=False):
    r"""Decorator that automatically identifies stim detectors and returns them.

    ``find_detectors`` leverages `tqecd <https://github.com/tqec/tqecd>`_
    to automatically discover detector parity checks in quantum error correction
    circuits. The detected parities are returned as an additional value alongside
    the decorated function's original return values.

    .. note::

        This feature requires the optional ``tqecd`` package.  Install it
        as described `here <https://tqec.github.io/tqecd/index.html>`_.

    Pitfalls & Limitations
    ^^^^^^^^^^^^^^^^^^^^^^

    The ``find_detectors`` feature comes with some constraints:
    
    **QuantumArray arguments must be QuantumBool**
        All ``QuantumArray`` arguments to the decorated function must have
        ``qtype=QuantumBool()``.  Other quantum types will raise a
        ``TypeError``.  Static parameters (integers, booleans, etc.) are
        allowed and will be passed through to the function unchanged.

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
        
        This is critical: in a two-round repetition code without resets
        between rounds, tqecd will only find first-round detectors and
        miss the temporal detectors that compare round 1 vs round 2.

    **Detectors referencing non-returned measurements are discarded**
        If a detector involves a measurement whose result is *not* part of
        the decorated function's return value, the detector is silently
        dropped.  Make sure all relevant measurements are returned.
        
        For example, if you measure both syndrome qubits but only return
        one measurement, detectors involving the unreturned measurement
        will be filtered out.

    **Multiple QuantumArray arguments receive disjoint qubit slices**
        When the decorated function takes more than one QuantumArray
        argument, it is assumed that each argument indeed represents
        a distinct set of qubits. This can not be guaranteed at tracing
        time and needs to be ensured from user side.
        Each array is mapped to its own contiguous slice of
        qubits during analysis.  The slices are allocated in positional
        order.
    
    **May discover more detectors than expected**
        In some circuits, ``find_detectors`` may identify *more* valid
        detectors than a minimal set.  For example, in 2-round codes,
        tqecd can discover boundary detectors from data qubit measurements
        that are mathematically valid but redundant with syndrome-based
        detectors.  All discovered detectors are correct (they will sample
        to zero in noiseless circuits); the extra ones represent additional
        stabilizer flows through the circuit.

    Parameters
    ----------
    func : callable, optional
        The function to decorate.  When using keyword arguments (e.g.
        ``@find_detectors(return_circuits=True)``), *func* is ``None`` and
        the decorator returns a wrapper that accepts *func*.

        The decorated function must accept at least one ``QuantumArray``
        argument (with ``qtype=QuantumBool()``).  Additional non-QuantumArray
        arguments (integers, booleans, strings, etc.) are treated as static
        parameters: they are captured by closure before JAX tracing so they
        behave as compile-time constants inside the function body.
    return_circuits : bool, default ``False``
        When ``True``, three additional items are appended to the return
        value (after the detector list and the original returns):

        * **raw_stim** — the stim circuit obtained directly from ``to_qc``
          ``to_stim``, before any restructuring.
        * **tqecd_input** — the circuit after ``_prepare_for_tqecd`` has
          re-arranged it into proper fragments (the input fed to tqecd).
        * **annotated** — the circuit that comes back from
          ``tqecd.annotate_detectors_automatically``, containing the
          ``DETECTOR`` instructions.

        This is useful for debugging or understanding why detectors were
        or were not discovered.

    Returns
    -------
    The decorated function returns:
        (detector_bools, \*original_returns)
    or, when ``return_circuits=True``:
        (detector_bools, \*original_returns, raw_stim, tqecd_input, annotated)
        where *detector_bools* is a ``list`` of Jasp-traced boolean values, one
        per discovered detector, each representing the ``parity()`` of the
        measurements that constitute that detector. The original return values
        from the decorated function are unpacked and appended after the detector list.

    Examples
    --------
    **Example 1: Single-round syndrome extraction**
    
    Three data qubits with two ancilla parity checks::
    
        from qrisp import *
        from qrisp.jasp.evaluation_tools.stim_extraction import extract_stim
        from qrisp.misc.stim_tools import find_detectors
        
        @find_detectors
        def syndrome(data, ancilla):
            # Reset ancilla qubits
            reset(ancilla[0])
            reset(ancilla[1])
            
            # Syndrome extraction
            cx(data[0], ancilla[0])
            cx(data[1], ancilla[0])
            cx(data[1], ancilla[1])
            cx(data[2], ancilla[1])
            
            # Measure syndromes
            return measure(ancilla[0]), measure(ancilla[1])
        
        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(3,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(2,))
            detectors, m0, m1 = syndrome(data, ancilla)
            return detectors, m0, m1
        
        # Run and extract stim circuit
        result = main()
        stim_circ = result[-1]
        print(f"Found {stim_circ.num_detectors} detectors")
    
    **Example 2: Two-round repetition code**
    
    Distance-3 repetition code with temporal detectors comparing rounds::
    
        @find_detectors
        def rep_code_2rounds(data, ancilla):
            measurements = []
            
            for round_num in range(2):
                # Reset ancillas
                reset(ancilla[0])
                reset(ancilla[1])
                
                # Parity checks
                cx(data[0], ancilla[0])
                cx(data[1], ancilla[0])
                cx(data[1], ancilla[1])
                cx(data[2], ancilla[1])
                
                # Measure syndromes
                measurements.append(measure(ancilla[0]))
                measurements.append(measure(ancilla[1]))
            
            # Final data readout
            for i in range(3):
                measurements.append(measure(data[i]))
            
            return tuple(measurements)
        
        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(3,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(2,))
            detectors, *measurements = rep_code_2rounds(data, ancilla)
            return (detectors,) + tuple(measurements)
        
        result = main()
        stim_circ = result[-1]
        # Will find >= 6 detectors (may discover more valid boundary detectors):
        # - 2 detectors in round 1 (vs initial reset)
        # - 2 detectors in round 2 (vs round 1)
        # - 2+ data-boundary detectors (tqecd may find additional valid ones)
        print(f"Found {stim_circ.num_detectors} detectors")
    
    **Example 3: Debug mode with circuit inspection**
    
    Use ``return_circuits=True`` to inspect intermediate circuits::
    
        @find_detectors(return_circuits=True)
        def debug_syndrome(data, ancilla):
            reset(ancilla[0])
            cx(data[0], ancilla[0])
            cx(data[1], ancilla[0])
            return measure(ancilla[0])
        
        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(2,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(1,))
            detectors, m, raw_stim, tqecd_input, annotated = debug_syndrome(data, ancilla)
            
            print("Raw stim circuit:")
            print(raw_stim)
            print("\nAfter tqecd restructuring:")
            print(tqecd_input)
            print("\nWith detectors annotated:")
            print(annotated)
            
            return detectors, m
        
        main()
    
    **Example 4: Multi-array arguments**
    
    Separate data and ancilla qubits::
    
        @find_detectors
        def surface_check(data_qubits, ancilla_qubits):
            # Reset ancilla
            reset(ancilla_qubits[0])
            
            # X-stabilizer (Hadamard + CNOTs)
            h(ancilla_qubits[0])
            cx(ancilla_qubits[0], data_qubits[0])
            cx(ancilla_qubits[0], data_qubits[1])
            cx(ancilla_qubits[0], data_qubits[2])
            cx(ancilla_qubits[0], data_qubits[3])
            h(ancilla_qubits[0])
            
            return measure(ancilla_qubits[0])
        
        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(4,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(1,))
            detectors, m = surface_check(data, ancilla)
            return detectors, m
        
        result = main()
        # Note: This finds 0 detectors because the X-stabilizer measurement
        # on Z-basis |0⟩ initialized qubits is not deterministic.
        # For detectors, would need X-basis initialization (RX) on data qubits.
    
    **Example 5: Three-round code with explicit resets**
    
    Shows importance of reset between rounds::
    
        @find_detectors
        def three_rounds(data, ancilla):
            results = []
            
            for _ in range(3):
                # CRITICAL: Reset between rounds
                reset(ancilla[0])
                
                # Syndrome extraction
                cx(data[0], ancilla[0])
                cx(data[1], ancilla[0])
                
                results.append(measure(ancilla[0]))
            
            return tuple(results)
        
        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(2,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(1,))
            detectors, m1, m2, m3 = three_rounds(data, ancilla)
            return detectors, m1, m2, m3
        
        result = main()
        stim_circ = result[-1]
        # Will find >= 3 detectors (one per round plus temporal)
        print(f"Found {stim_circ.num_detectors} detectors")
        
        # Verify all detectors are valid (noiseless samples = 0)
        sampler = stim_circ.compile_detector_sampler()
        samples = sampler.sample(shots=1000)
        assert samples.sum() == 0, "All detectors should be deterministic!"
    
    **Example 6: Partial measurement return**
    
    Only detectors involving returned measurements are kept::
    
        @find_detectors
        def partial_return(data, ancilla):
            reset(ancilla[0])
            reset(ancilla[1])
            
            cx(data[0], ancilla[0])
            cx(data[1], ancilla[0])
            cx(data[1], ancilla[1])
            cx(data[2], ancilla[1])
            
            m0 = measure(ancilla[0])
            m1 = measure(ancilla[1])
            
            # Only return m0, discard m1
            return m0
        
        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(3,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(2,))
            detectors, m = partial_return(data, ancilla)
            return detectors, m
        
        result = main()
        stim_circ = result[-1]
        # Only detectors involving m0 (the returned measurement) are kept.
        # Detectors that depend on m1 (unreturned) are filtered out.
        # Typically finds 1 detector from the m0 measurement vs initial reset.
        print(f"Found {stim_circ.num_detectors} detectors")

    **Example 7: Static parameters (integers, booleans)**

    The decorated function can accept non-QuantumArray arguments such as
    integers or booleans.  These are bound via closure before tracing so
    that JAX treats them as compile-time constants::

        @find_detectors
        def configurable_syndrome(data, ancilla, num_rounds, use_extra_gate):
            results = []
            for _ in range(num_rounds):
                reset(ancilla[0])
                cx(data[0], ancilla[0])
                cx(data[1], ancilla[0])
                if use_extra_gate:
                    cx(data[1], ancilla[0])
                    cx(data[1], ancilla[0])  # cancels out
                results.append(measure(ancilla[0]))
            return tuple(results)

        @extract_stim
        def main():
            data = QuantumArray(qtype=QuantumBool(), shape=(2,))
            ancilla = QuantumArray(qtype=QuantumBool(), shape=(1,))
            detectors, *ms = configurable_syndrome(data, ancilla, 3, True)
            return (detectors,) + tuple(ms)

        result = main()
        stim_circ = result[-1]
        print(f"Found {stim_circ.num_detectors} detectors")
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
            # --- Validate & classify arguments ---
            quantum_arrays = []
            for a in args:
                if isinstance(a, QuantumArray):
                    if not isinstance(a.qtype, QuantumBool):
                        raise TypeError(
                            f"find_detectors: QuantumArray args must have "
                            f"qtype QuantumBool, got {a.qtype}")
                    quantum_arrays.append(a)
                elif isinstance(a, jax.core.Tracer):
                    raise TypeError(
                        "find_detectors: traced (non-static) non-QuantumArray "
                        "arguments are not supported. All non-QuantumArray "
                        "parameters must be concrete Python values (int, bool, "
                        "etc.), not JAX tracers.")
            if not quantum_arrays:
                raise TypeError(
                    "find_detectors: at least one QuantumArray argument is required")

            n_qubits = sum(a.size * a.qtype.size for a in quantum_arrays)

            # --- 1. Trace inner function to get sub-jaspr ---
            # Bind non-QuantumArray args via closure so JAX only traces
            # the quantum arguments.  Static values (ints, bools, …)
            # become constants inside the jaspr.
            qa_idx = [i for i, a in enumerate(args) if isinstance(a, QuantumArray)]
            static = {i: a for i, a in enumerate(args) if not isinstance(a, QuantumArray)}

            def fn_qa_only(*qa_args, _kw=kwargs):
                full = [None] * len(args)
                for i, q in zip(qa_idx, qa_args): full[i] = q
                for i, v in static.items():       full[i] = v
                return fn(*full, **_kw)

            jaspr = make_jaspr(fn_qa_only)(*(args[i] for i in qa_idx))

            # --- 2. Analysis: to_qc → stim → tqecd ---
            result = jaspr.to_qc(*_build_analysis_args(jaspr, n_qubits, quantum_arrays))

            # Flatten analysis Clbits (consistent ordering via tree_flatten)
            analysis_clbits, _ = tree_flatten(result[:-1])

            # Convert to stim
            raw_stim, meas_map = result[-1].to_stim(return_measurement_map=True)
            inv_map = {v: k for k, v in meas_map.items()}

            # Prepare for tqecd and annotate
            tqecd_circ = _prepare_for_tqecd(raw_stim)
            annotated = tqecd.annotate_detectors_automatically(tqecd_circ)

            # Extract detector Clbit sets
            all_dets = _extract_detectors(
                annotated, tqecd_circ.num_measurements, inv_map)

            # Keep only detectors whose Clbits all appear in returned measurements
            returned = set(analysis_clbits)
            relevant = [(i, d) for i, d in enumerate(all_dets)
                        if all(cb in returned for cb in d)]

            # --- 2b. Simulate noiseless circuit to determine detector expectations ---
            # Use stim's built-in method to remove all noise processes
            expectations = annotated.without_noise() \
                .compile_detector_sampler().sample(shots=1)[0].tolist()

            # --- 3. Call real function & map analysis Clbits to traced booleans ---
            real_returns = fn(*args, **kwargs)
            if not isinstance(real_returns, tuple):
                real_returns = (real_returns,)
            traced_flat, _ = tree_flatten(real_returns)
            cb_to_traced = dict(zip(analysis_clbits, traced_flat))

            # --- 4. Emit parity calls for each detector ---
            det_results = [
                parity(*[cb_to_traced[cb] for cb in det],
                       expectation=bool(expectations[idx]))
                for idx, det in relevant
            ]

            out = (det_results,) + real_returns
            if return_circuits:
                out += (raw_stim, tqecd_circ, annotated)
            return out

        return wrapper

    # Support both @find_detectors and @find_detectors(return_circuits=True)
    if func is not None:
        return _decorator(func)
    return _decorator
