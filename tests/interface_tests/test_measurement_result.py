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

"""Unit tests for measurement_result.py."""

from collections.abc import Mapping

import pytest

from qrisp.interface.measurement_result import (
    DecodedMeasurementResult,
    LazyDict,
    MeasurementResult,
    MultiMeasurementResult,
    _IntKeyedResult,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _Ready(LazyDict):
    """Concrete LazyDict that populates immediately with a fixed dict."""

    def __init__(self, data: dict):
        """Store the source data to be used on first population."""
        super().__init__()
        self._source = data

    def _populate(self) -> None:
        """Copy the source dict into ``_data``."""
        self._data = dict(self._source)


class _Broken(LazyDict):
    """Concrete LazyDict whose _populate always raises."""

    def _populate(self) -> None:
        """Always raise to simulate a permanently failing backend."""
        raise RuntimeError("always fails")


def _populated_mr(data: dict) -> MeasurementResult:
    """Return a pre-populated MeasurementResult containing *data*."""
    mr = MeasurementResult()
    mr._inject(data)
    return mr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLazyDict:
    """Tests for the LazyDict abstract base class."""

    def test_initial_state(self):
        """A freshly created LazyDict must not be populated and must have no data."""
        ld = _Ready({"a": 1})
        assert not ld._populated
        assert ld._data is None

    def test_populate_called_exactly_once(self):
        """_populate must be invoked on first access and never again after that."""
        count = 0

        class _Counting(LazyDict):
            """LazyDict that counts how many times _populate is called."""

            def _populate(self):
                """Increment counter and set empty data."""
                nonlocal count
                count += 1
                self._data = {}

        ld = _Counting()
        len(ld)
        len(ld)
        assert count == 1

    def test_populate_not_retried_after_success(self):
        """After a successful population _populated must be True."""
        ld = _Ready({"k": 1})
        len(ld)
        assert ld._populated

    def test_populate_retried_after_failure(self):
        """If _populate raises, _populated stays False and the next access retries."""
        attempts = 0

        class _FailOnce(LazyDict):
            """LazyDict whose first _populate call raises; the second succeeds."""

            def _populate(self):
                """Fail on the first call, succeed on the second."""
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("first attempt fails")
                self._data = {"ok": True}

        ld = _FailOnce()
        with pytest.raises(RuntimeError):
            len(ld)
        assert not ld._populated
        assert len(ld) == 1  # second attempt succeeds
        assert attempts == 2

    def test_getitem_triggers_population(self):
        """Subscripting an unpopulated LazyDict must trigger population."""
        ld = _Ready({"x": 42})
        assert ld["x"] == 42
        assert ld._populated

    def test_getitem_raises_key_error_for_missing_key(self):
        """Accessing a key that does not exist must raise KeyError."""
        ld = _Ready({"x": 1})
        with pytest.raises(KeyError):
            _ = ld["missing"]

    def test_iter_triggers_population(self):
        """Iterating over an unpopulated LazyDict must trigger population."""
        ld = _Ready({"a": 1, "b": 2})
        assert set(ld) == {"a", "b"}

    def test_len_triggers_population(self):
        """Calling len() on an unpopulated LazyDict must trigger population."""
        ld = _Ready({"a": 1, "b": 2})
        assert len(ld) == 2

    def test_repr_when_already_populated(self):
        """repr() of an already-populated LazyDict must match the underlying dict repr."""
        ld = _Ready({"a": 1})
        len(ld)  # force population
        assert repr(ld) == repr({"a": 1})

    def test_repr_populates_when_not_yet_populated(self):
        """repr() must trigger population and return the dict repr."""
        ld = _Ready({"a": 1})
        r = repr(ld)
        assert ld._populated
        assert r == repr({"a": 1})

    def test_repr_returns_pending_when_populate_raises(self):
        """repr() must return a '<ClassName pending>' string when population fails."""
        ld = _Broken()
        r = repr(ld)
        assert "pending" in r.lower()
        assert type(ld).__name__ in r

    def test_eq_with_plain_dict(self):
        """A LazyDict must compare equal to a plain dict with the same contents."""
        ld = _Ready({"a": 1})
        assert ld == {"a": 1}

    def test_eq_reflexive_with_plain_dict(self):
        """Equality must hold in both directions (plain dict == LazyDict)."""
        ld = _Ready({"a": 1})
        assert {"a": 1} == ld

    def test_eq_with_other_lazy_dict_same_content(self):
        """Two LazyDicts with the same underlying data must be equal."""
        a = _Ready({"x": 1})
        b = _Ready({"x": 1})
        assert a == b

    def test_eq_with_other_lazy_dict_different_content(self):
        """Two LazyDicts with different data must not be equal."""
        a = _Ready({"x": 1})
        b = _Ready({"x": 2})
        assert a != b

    def test_eq_non_mapping_returns_not_implemented(self):
        """__eq__ must return NotImplemented for non-Mapping operands."""
        ld = _Ready({"a": 1})
        # Must call __eq__ directly to observe NotImplemented (== swallows it).
        assert ld.__eq__(42) is NotImplemented
        assert ld.__eq__("string") is NotImplemented

    def test_is_unhashable(self):
        """LazyDict instances must not be hashable (Mapping convention)."""
        ld = _Ready({})
        with pytest.raises(TypeError):
            hash(ld)

    def test_is_mapping(self):
        """LazyDict must be recognised as a collections.abc.Mapping."""
        assert isinstance(_Ready({}), Mapping)


class TestMeasurementResult:
    """Tests for MeasurementResult — the raw bitstring-count result type."""

    def test_initial_state(self):
        """A freshly created MeasurementResult must be unpopulated with no stored error."""
        mr = MeasurementResult()
        assert not mr._populated
        assert mr._error is None

    def test_raises_before_inject(self):
        """Accessing data before _inject must raise RuntimeError mentioning dispatch."""
        mr = MeasurementResult()
        with pytest.raises(RuntimeError, match="dispatch"):
            len(mr)

    def test_raises_on_every_access_before_inject(self):
        """Every data access before _inject must raise, not just the first one."""
        mr = MeasurementResult()
        for _ in range(3):
            with pytest.raises(RuntimeError):
                iter(mr)

    def test_inject_populates(self):
        """_inject must set _populated to True and store the provided counts."""
        mr = MeasurementResult()
        mr._inject({"00": 512, "11": 512})
        assert mr._populated
        assert mr._data == {"00": 512, "11": 512}

    def test_inject_makes_getitem_work(self):
        """After _inject, subscript access must return the correct count."""
        mr = _populated_mr({"00": 512, "11": 512})
        assert mr["00"] == 512

    def test_inject_makes_iter_work(self):
        """After _inject, iteration must yield all bitstring keys."""
        mr = _populated_mr({"00": 512, "11": 512})
        assert set(mr) == {"00", "11"}

    def test_inject_makes_len_work(self):
        """After _inject, len() must return the number of distinct bitstrings."""
        mr = _populated_mr({"00": 512, "11": 512})
        assert len(mr) == 2

    def test_inject_makes_eq_work(self):
        """After _inject, equality comparison against a plain dict must work."""
        mr = _populated_mr({"00": 512, "11": 512})
        assert mr == {"00": 512, "11": 512}

    def test_inject_error_stores_exception(self):
        """_inject_error must cause the stored exception to be raised on access."""
        mr = MeasurementResult()
        mr._inject_error(ValueError("hardware fault"))
        with pytest.raises(ValueError, match="hardware fault"):
            len(mr)

    def test_inject_error_raises_on_every_access(self):
        """The stored error must be re-raised on every subsequent access, not just once."""
        mr = MeasurementResult()
        mr._inject_error(RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError, match="fail"):
                iter(mr)

    def test_inject_error_does_not_set_populated(self):
        """_inject_error must leave _populated as False so the error is raised each time."""
        mr = MeasurementResult()
        mr._inject_error(RuntimeError("err"))
        assert not mr._populated

    def test_inject_after_failed_access(self):
        """_inject can still succeed after a failed access attempt."""
        mr = MeasurementResult()
        with pytest.raises(RuntimeError):
            len(mr)
        mr._inject({"0": 1.0})
        assert len(mr) == 1


class TestDecodedMeasurementResult:
    """Tests for DecodedMeasurementResult — the human-readable result type."""

    def test_not_populated_on_construction(self):
        """Construction must not trigger decoding; the result must start unpopulated."""
        dmr = DecodedMeasurementResult(_populated_mr({"0": 1.0}), lambda k: k)
        assert not dmr._populated

    def test_decodes_keys(self):
        """The decoder must be applied to every key in the raw result."""
        dmr = DecodedMeasurementResult(
            _populated_mr({"0": 0.5, "1": 0.5}),
            lambda k: int(k, 2),
        )
        assert dmr == {0: 0.5, 1: 0.5}

    def test_duplicate_decoded_keys_are_summed(self):
        """Two raw keys decoding to the same label must have their counts merged."""
        dmr = DecodedMeasurementResult(
            _populated_mr({"00": 0.3, "01": 0.7}),
            lambda k: k[0],  # both "00" and "01" decode to "0"
        )
        assert dmr == {"0": pytest.approx(1.0)}

    def test_sorted_descending_by_probability(self):
        """Decoded keys must be ordered from highest to lowest probability."""
        dmr = DecodedMeasurementResult(
            _populated_mr({"0": 0.2, "1": 0.8}),
            lambda k: int(k, 2),
        )
        assert list(dmr.keys()) == [1, 0]

    def test_sort_fallback_when_values_cannot_be_negated(self):
        """When -value raises TypeError the result falls back to insertion order."""

        class _UnnegableFloat(float):
            """Float subclass whose negation always raises TypeError."""

            def __neg__(self):
                """Raise TypeError to simulate an un-negatable value."""
                raise TypeError("cannot negate")

            def __add__(self, other):
                """Add two _UnnegableFloat values."""
                return _UnnegableFloat(float.__add__(self, other))

            def __radd__(self, other):
                """Right-add supporting sum()."""
                return _UnnegableFloat(float.__radd__(self, other))

        dmr = DecodedMeasurementResult(
            _populated_mr({"0": _UnnegableFloat(0.5), "1": _UnnegableFloat(0.3)}),
            lambda k: k,
        )
        assert set(dmr) == {"0", "1"}  # both keys present; order not guaranteed

    def test_repr_pending_when_raw_not_yet_populated(self):
        """repr() must return a pending string when the raw result is not yet populated."""
        raw = MeasurementResult()
        dmr = DecodedMeasurementResult(raw, lambda k: k)
        assert "pending" in repr(dmr).lower()

    def test_eq_with_plain_dict(self):
        """Equality comparison against a plain dict must work in both directions."""
        dmr = DecodedMeasurementResult(
            _populated_mr({"0": 1.0}),
            lambda k: int(k, 2),
        )
        assert dmr == {0: 1.0}
        assert {0: 1.0} == dmr

    def test_lazy_decode_on_first_access(self):
        """Decoding must not happen until the result is actually accessed."""
        raw = MeasurementResult()
        dmr = DecodedMeasurementResult(raw, lambda k: int(k, 2))
        assert not dmr._populated
        raw._inject({"0": 1.0})
        assert dmr == {0: 1.0}
        assert dmr._populated

    def test_propagates_error_from_raw(self):
        """If the raw result carries an error, accessing the decoded result re-raises it."""
        raw = MeasurementResult()
        raw._inject_error(RuntimeError("upstream error"))
        dmr = DecodedMeasurementResult(raw, lambda k: k)
        with pytest.raises(RuntimeError, match="upstream error"):
            len(dmr)


# ---------------------------------------------------------------------------
# _IntKeyedResult
# ---------------------------------------------------------------------------


class TestIntKeyedResult:
    """Tests for _IntKeyedResult — the internal bitstring-to-int conversion layer."""

    def test_not_populated_on_construction(self):
        """Construction must not trigger conversion; the result must start unpopulated."""
        r = _IntKeyedResult(_populated_mr({"0000": 1.0}), num_bits=4)
        assert not r._populated

    def test_converts_bitstring_to_int(self):
        """Bitstrings must be converted to their integer values."""
        r = _IntKeyedResult(_populated_mr({"0011": 1.0}), num_bits=4)
        assert r[3] == pytest.approx(1.0)  # 0011 = 3

    def test_truncates_to_num_bits(self):
        """Bitstrings longer than num_bits must be truncated before conversion."""
        r = _IntKeyedResult(_populated_mr({"0011": 1.0}), num_bits=2)
        assert r[0] == pytest.approx(1.0)  # "0011"[:2] = "00" = 0

    def test_strips_spaces_from_bitstrings(self):
        """Spaces embedded in bitstrings must be removed before processing."""
        r = _IntKeyedResult(_populated_mr({"0 1": 1.0}), num_bits=2)
        # "0 1" -> strip spaces -> "01" -> [:2] = "01" = 1
        assert r[1] == pytest.approx(1.0)

    def test_normalises_shot_counts(self):
        """Counts far from 1 (i.e. raw shot counts) must be normalised to probabilities."""
        r = _IntKeyedResult(_populated_mr({"00": 512, "11": 512}), num_bits=2)
        assert r[0] == pytest.approx(0.5)
        assert r[3] == pytest.approx(0.5)

    def test_no_normalisation_when_already_probabilities(self):
        """Values summing to 1.0 must be left unchanged."""
        r = _IntKeyedResult(_populated_mr({"00": 0.5, "11": 0.5}), num_bits=2)
        assert r[0] == pytest.approx(0.5)
        assert r[3] == pytest.approx(0.5)

    def test_no_normalisation_within_tolerance(self):
        """Total within 1e-3 of 1.0 must not be normalised."""
        # total = 1.0 exactly → no normalisation
        r = _IntKeyedResult(_populated_mr({"00": 0.9995, "11": 0.0005}), num_bits=2)
        assert r[0] == pytest.approx(0.9995)
        assert r[3] == pytest.approx(0.0005)
        # abs(1 - 1.0005) = 0.0005 < 1e-3 → no normalisation
        r2 = _IntKeyedResult(_populated_mr({"0": 1.0005}), num_bits=1)
        assert r2[0] == pytest.approx(1.0005)

    def test_no_normalisation_when_total_is_zero(self):
        """Total of zero must not cause a division by zero."""
        r = _IntKeyedResult(_populated_mr({"00": 0, "11": 0}), num_bits=2)
        assert r[0] == 0
        assert r[3] == 0

    def test_colliding_keys_are_summed(self):
        """Bitstrings that truncate to the same integer must have their counts merged."""
        # "000"[:2] = "00" = 0, "001"[:2] = "00" = 0 — both map to 0
        r = _IntKeyedResult(_populated_mr({"000": 0.4, "001": 0.6}), num_bits=2)
        assert r[0] == pytest.approx(1.0)

    def test_lazy_before_raw_populated(self):
        """Conversion must not happen until the raw MeasurementResult is populated."""
        raw = MeasurementResult()
        r = _IntKeyedResult(raw, num_bits=4)
        assert not r._populated
        raw._inject({"0000": 1.0})
        assert r[0] == pytest.approx(1.0)
        assert r._populated

    def test_propagates_error_from_raw(self):
        """If the raw result carries an error, accessing the int-keyed result re-raises it."""
        raw = MeasurementResult()
        raw._inject_error(RuntimeError("hardware error"))
        r = _IntKeyedResult(raw, num_bits=4)
        with pytest.raises(RuntimeError, match="hardware error"):
            len(r)


class TestMultiMeasurementResult:
    """Unit tests for MultiMeasurementResult."""

    def test_lazy_before_dispatch(self):
        """Result must be unpopulated until the raw MeasurementResult is injected."""
        from qrisp import QuantumFloat, multi_measurement
        from qrisp.default_backend import QrispSimulatorBackend

        qf_0 = QuantumFloat(4)
        qf_1 = QuantumFloat(4)
        qf_0[:] = 3
        qf_1[:] = 5

        bb = QrispSimulatorBackend().batched()
        res = multi_measurement([qf_0, qf_1], backend=bb)

        assert isinstance(res, MultiMeasurementResult)
        assert isinstance(res, LazyDict)
        assert not res._populated

        bb.dispatch()

        assert res == {(3, 5): 1.0}

    def test_correct_results_after_dispatch(self):
        """Decoded labels and probabilities must match a direct (non-batched) execution."""
        from qrisp import QuantumFloat, h, multi_measurement
        from qrisp.default_backend import QrispSimulatorBackend

        qf_0 = QuantumFloat(4)
        qf_1 = QuantumFloat(4)
        qf_0[:] = 3
        qf_1[:] = 2
        h(qf_1[0])
        qf_sum = qf_0 + qf_1

        bb = QrispSimulatorBackend().batched()
        res = multi_measurement([qf_0, qf_1, qf_sum], backend=bb)
        bb.dispatch()

        assert set(res.keys()) == {(3, 2, 5), (3, 3, 6)}
        assert abs(res[(3, 2, 5)] - 0.5) < 1e-6
        assert abs(res[(3, 3, 6)] - 0.5) < 1e-6

    def test_propagates_error_from_raw(self):
        """If the raw result carries an error, accessing the decoded result re-raises it."""
        raw = MeasurementResult()
        raw._inject_error(RuntimeError("backend fault"))
        r = MultiMeasurementResult(raw, qv_list=[], cl_reg_list=[])
        with pytest.raises(RuntimeError, match="backend fault"):
            len(r)
