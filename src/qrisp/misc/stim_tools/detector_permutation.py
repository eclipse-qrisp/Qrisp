import stim
import numpy as np


def permute_detectors(circuit: stim.Circuit, permutation) -> stim.Circuit:
    """
    Permutes the detectors in a Stim circuit according to the given permutation.

    The function moves all detectors to the end of the circuit and applies the permutation.
    It preserves the semantic meaning of each detector (which measurements it targets)
    and its spatial coordinates (compensating for ``SHIFT_COORDS`` instructions).

    Parameters
    ----------
    circuit : stim.Circuit
        The input Stim circuit. Assumed to have no ``REPEAT`` blocks.
    permutation : iter
        An iterable of integers representing the desired order of detectors.
        permutation[i] = k means the i-th detector in the output should be
        the k-th detector from the input. E.g. (2, 1, 0) means
        Output_D0 = Input_D2, Output_D1 = Input_D1, Output_D2 = Input_D0.

    Returns
    -------
    new_circuit : stim.Circuit
        A new Stim circuit with permuted detectors.

    Examples
    --------

    We create a circuit with three detectors. The first and third detectors are triggered by deterministic errors.

    ::

        from qrisp import QuantumVariable, measure
        from qrisp.jasp import extract_stim, parity
        from qrisp.misc.stim_tools import stim_noise, permute_detectors
        import numpy as np

        @extract_stim
        def detector_circuit():
            qv = QuantumVariable(3)
            # Apply deterministic noise to qv[0] and qv[2]
            stim_noise("X_ERROR", 1.0, qv[0])
            stim_noise("X_ERROR", 1.0, qv[2])

            m0 = measure(qv[0])
            m1 = measure(qv[1])
            m2 = measure(qv[2])

            # Detector 0 checks m0 (fires)
            d0 = parity(m0, expectation=0)
            # Detector 1 checks m1 (silent)
            d1 = parity(m1, expectation=0)
            # Detector 2 checks m2 (fires)
            d2 = parity(m2, expectation=0)

            return d0, d1, d2

        d0_idx, d1_idx, d2_idx, stim_circ = detector_circuit()

    We sample from the original circuit:

    ::

        # Sample original
        sampler = stim_circ.compile_detector_sampler()
        print(sampler.sample(1))
        # Yields: [[True, False, True]]

    Now we permute the detectors using the permutation (1, 2, 0).

    ::

        # Permute: (1, 2, 0)
        # New D0 comes from Old D1 (False)
        # New D1 comes from Old D2 (True)
        # New D2 comes from Old D0 (True)
        permuted_circ = permute_detectors(stim_circ, (1, 2, 0))

        sampler_perm = permuted_circ.compile_detector_sampler()
        print(sampler_perm.sample(1))
        # Yields: [[False, True, True]]

    """

    # Check for REPEAT blocks
    for instr in circuit:
        if instr.name == "REPEAT":
            raise ValueError("Circuits with REPEAT blocks are not supported.")

    detector_list = []
    other_instructions_reversed = []

    accum_shift = np.zeros(0)
    accum_meas = 0

    # Iterate backwards is safer for accumulating "future" offsets that the detector is being moved past
    # Convert to list first as stim.Circuit doesn't support reversed() directly in older versions or might be slow
    instructions = list(circuit)

    for instr in reversed(instructions):
        if instr.name == "DETECTOR":
            # Modify the detector to be valid at the CURRENT accumulated end context

            # 1. Update measurement targets
            new_targets = []
            for t in instr.targets_copy():
                if t.is_measurement_record_target:
                    # Original: rec[-k]
                    # New: rec[-(k + accum_meas)]
                    # t.value is negative. e.g. -1.
                    # new value = -1 - accum_meas
                    new_val = t.value - accum_meas
                    new_targets.append(stim.target_rec(new_val))
                else:
                    new_targets.append(t)

            # 2. Update coordinate arguments
            # We want: (args_new) + (total_shift_at_end) = (args_old) + (shift_at_loc)
            # args_new = args_old + shift_at_loc - total_shift_at_end
            #          = args_old - (total_shift_at_end - shift_at_loc)
            #          = args_old - (accumulated_shift_seen_so_far)

            old_args = np.array(instr.gate_args_copy())

            # Ensure size matches for subtraction
            max_len = max(len(old_args), len(accum_shift))

            p_old = np.zeros(max_len)
            p_old[: len(old_args)] = old_args

            p_accum = np.zeros(max_len)
            p_accum[: len(accum_shift)] = accum_shift

            new_args = p_old - p_accum

            new_args_list = list(new_args)

            # Create updated instruction
            new_instr = stim.CircuitInstruction("DETECTOR", new_targets, new_args_list)
            detector_list.append(new_instr)

        else:
            other_instructions_reversed.append(instr)

            if instr.name == "SHIFT_COORDS":
                # Update accumulator with these shifts
                args = instr.gate_args_copy()
                if len(args) > len(accum_shift):
                    new_accum = np.zeros(len(args))
                    new_accum[: len(accum_shift)] = accum_shift
                    accum_shift = new_accum

                accum_shift[: len(args)] += args

            # Check for measurements
            # Creating a tiny circuit is the most robust way to check measurement count of an unknown instruction
            # (e.g. M, MPP, MX all produce measurements)
            temp = stim.Circuit()
            temp.append(instr)
            m_count = temp.num_measurements
            if m_count > 0:
                accum_meas += m_count

    # Re-order lists

    # detector_list was collected backwards (last detector in circuit is at index 0)
    # We need to reverse it to restore original order 0..N
    detector_list.reverse()

    # other_instructions was collected backwards
    other_instructions = list(reversed(other_instructions_reversed))

    # Validate Permutation
    n_detectors = len(detector_list)
    perm_list = list(permutation)

    if len(perm_list) != n_detectors:
        # Check if the user perhaps provided a permutation for a subset?
        # The spec says "numbers 0...N". Assuming N is total count.
        raise ValueError(
            f"Circuit contains {n_detectors} detectors, but permutation has length {len(perm_list)}."
        )

    if sorted(perm_list) != list(range(n_detectors)):
        raise ValueError("Permutation must contain numbers 0 to N-1 exactly once.")

    # Apply Permutation
    # permitation[i] = k means output[i] comes from input[k]
    permuted_detectors = [detector_list[k] for k in perm_list]

    # Construct result
    new_circuit = stim.Circuit()
    for instr in other_instructions:
        new_circuit.append(instr)
    for instr in permuted_detectors:
        new_circuit.append(instr)

    return new_circuit
