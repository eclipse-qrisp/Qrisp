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

import pytest

from qrisp import QuantumEnvironment, QuantumVariable, cx, h, s, t, x, y, z


class TestQuantumEnvironmentBasicUsage:
    """Test basic usage of QuantumEnvironment."""

    def test_enter_returns_self_or_none(self):
        """Test that __enter__ returns a value without crashing."""
        q_env = QuantumEnvironment()
        with q_env as result:
            assert result is q_env or result is None

    def test_sequential_fresh_instances(self):
        """Test that multiple fresh instances can be used sequentially."""
        for _ in range(3):
            with QuantumEnvironment():
                pass

    def test_deeply_nested_different_instances(self):
        """Test that deeply nested different instances work correctly."""
        with QuantumEnvironment():
            with QuantumEnvironment():
                with QuantumEnvironment():
                    pass

    def test_deepest_environment_restored_after_exit(self):
        """Test that the ContextVar is restored to its previous value after exiting."""
        outer = QuantumEnvironment()
        inner = QuantumEnvironment()

        with outer:
            before_inner = QuantumEnvironment._deepest_environment.get()
            with inner:
                pass
            after_inner = QuantumEnvironment._deepest_environment.get()
            assert before_inner is after_inner

    def test_deepest_environment_restored_after_exception(self):
        """Test that the ContextVar is restored even if an exception occurs inside."""

        before = QuantumEnvironment._deepest_environment.get()
        try:
            with QuantumEnvironment():
                raise ValueError("oops")
        except ValueError:
            pass
        after = QuantumEnvironment._deepest_environment.get()
        assert before is after


class TestQuantumEnvironmentDocstringExamples:
    """Test that the examples in the docstring of QuantumEnvironment work correctly."""

    def test_docstring_example_qv_operations(self):
        """Test that qv operations are correctly recorded in the QuantumEnvironment."""
        qv = QuantumVariable(5)

        with QuantumEnvironment():
            x(qv)

        assert len(qv.qs.data) == 10

        for i in range(5):
            assert "qb_alloc" in str(qv.qs.data[i])
            assert f"qv.{i}" in str(qv.qs.data[i])

        for i in range(5):
            assert "x" in str(qv.qs.data[5 + i]).lower()
            assert f"qv.{i}" in str(qv.qs.data[5 + i])

    def test_docstring_example_nested_environments(self):
        """Test that nested QuantumEnvironments in the docstring example work correctly."""

        a = QuantumVariable(1)
        with QuantumEnvironment():
            b = QuantumVariable(1)
            cx(a, b)
            with QuantumEnvironment():
                c = QuantumVariable(1)
                cx(b, c)

            c.uncompute()  # works because c was created in a sub-environment
            b.uncompute()  # works because b was created in the same environment

            # doesn't work because a was created outside this environment
            with pytest.raises(
                RuntimeError,
                match="Could not uncompute QuantumVariables \\['a'\\] "
                "because they were not created within this QuantumEnvironment",
            ):
                a.uncompute()

    def test_custom_quantum_environment_compile(self):
        """Test that the compile example in the docstring works correctly."""

        class ExampleEnvironment(QuantumEnvironment):
            """Example of a custom QuantumEnvironment that compiles its instructions in a specific way."""

            def compile(self):
                """Compile the instructions by skipping every second instruction."""

                for idx, instruction in enumerate(self.env_data):

                    if idx % 2:
                        continue

                    if isinstance(instruction, QuantumEnvironment):
                        instruction.compile()

                    else:
                        self.env_qs.append(instruction)

        qv = QuantumVariable(6)

        with ExampleEnvironment():
            x(qv[0])
            y(qv[1])
            with ExampleEnvironment():
                z(qv[2])
                t(qv[3])
            with ExampleEnvironment():
                s(qv[4])
            h(qv[5])

        assert len(qv.qs.data) == 9

        for i in range(6):
            assert "qb_alloc" in str(qv.qs.data[i])
            assert f"qv.{i}" in str(qv.qs.data[i])

        assert "x" in str(qv.qs.data[6]).lower()
        assert "qv.0" in str(qv.qs.data[6])

        assert "z" in str(qv.qs.data[7]).lower()
        assert "qv.2" in str(qv.qs.data[7])

        assert "h" in str(qv.qs.data[8]).lower()
        assert "qv.5" in str(qv.qs.data[8])

    def test_quantum_environment_print_compile(self):
        """Test that printing a QuantumSession inside a QuantumEnvironment displays only instructions from that environment."""

        qv = QuantumVariable(3)
        x(qv[0])
        with QuantumEnvironment():
            y(qv[1])
            with QuantumEnvironment():
                z(qv[2])

                assert len(qv.qs.data) == 1
                assert "z" in str(qv.qs.data[0]).lower()
                assert "qv.2" in str(qv.qs.data[0])

            assert len(qv.qs.data) == 2
            assert "y" in str(qv.qs.data[0]).lower()
            assert "qv.1" in str(qv.qs.data[0])
            assert "jasp.q_env" in str(qv.qs.data[1])
            # NOTE: this print statement is part of the test
            print(qv.qs)
            assert len(qv.qs.data) == 2
            assert "y" in str(qv.qs.data[0]).lower()
            assert "qv.1" in str(qv.qs.data[0])
            assert "z" in str(qv.qs.data[1]).lower()
            assert "qv.2" in str(qv.qs.data[1])

        assert len(qv.qs.data) == 6
        assert "qb_alloc" in str(qv.qs.data[0])
        assert "qv.0" in str(qv.qs.data[0])
        assert "qb_alloc" in str(qv.qs.data[1])
        assert "qv.1" in str(qv.qs.data[1])
        assert "qb_alloc" in str(qv.qs.data[2])
        assert "qv.2" in str(qv.qs.data[2])
        assert "x" in str(qv.qs.data[3]).lower()
        assert "qv.0" in str(qv.qs.data[3])
        assert "y" in str(qv.qs.data[4]).lower()
        assert "qv.1" in str(qv.qs.data[4])
        assert "z" in str(qv.qs.data[5]).lower()
        assert "qv.2" in str(qv.qs.data[5])


class TestQuantumEnvErrors:
    """Test error conditions for QuantumEnvironment."""

    def test_error_quantum_environment_used_twice(self):
        """Test that a QuantumEnvironment cannot be entered twice."""

        q_env = QuantumEnvironment()

        with q_env:
            pass

        with pytest.raises(
            RuntimeError,
            match="QuantumEnvironment has already been entered. "
            "QuantumEnvironments cannot be reused",
        ):
            with q_env:
                pass

    def test_error_quantum_environment_nested(self):
        """Test that a QuantumEnvironment raises RuntimeError when nested."""

        q_env = QuantumEnvironment()

        with pytest.raises(
            RuntimeError,
            match="QuantumEnvironment has already been entered. "
            "QuantumEnvironments cannot be reused",
        ):
            with q_env:
                with q_env:
                    pass
