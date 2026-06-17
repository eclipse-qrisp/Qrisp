# Remaud Adder CI Failure Analysis

## Symptom

`test_l2_ladder` and `test_remaud_adder` fail on CI with:

```
Exception: Found jrange with external carry value
```

at `src/qrisp/environments/jiteration_environment.py:115` when the full test suite runs, but **pass when run in isolation**.

## Root Cause

The exception fires when a `jrange` body's second-iteration flattened jaspr produces **>1 output variable** (the loop index + a QuantumState from nested `with invert():` / `with control():` environments). This is meant to catch unhandled "external carry values."

The failure is **test-order dependent** — it only happens when other tests run before the remaud adder tests, specifically when `gidney_venting_adder` is imported.

## Trigger Chain

1. `from qrisp.alg_primitives.arithmetic.adders import gidney_venting_adder` (via `adders/__init__.py`)
2. This triggers `from qrisp.jasp import check_for_tracing_mode` at module level, which loads the entire `jasp` package eagerly
3. The `jasp` package init runs all submodule `*` imports, registering JAX primitives and initializing environment classes
4. When `test_gidney_venting_adder.py` tests execute, they call `@custom_control`-decorated functions (`gidney_cq_venting_adder`) in tracing mode
5. `custom_control` traces the controlled version via `make_jaspr(ammended_func)`, which invokes the full environment flattening pipeline
6. That pipeline runs `jiteration_environment.jcompile`, which sets `eqn.primitive.iteration_1_eqn` on the equation's primitive
7. If JAX internally reuses `Primitive` objects by name during jaxpr construction, the stale `iteration_1_eqn` attribute persists to the next jrange encountered — the remaud adder tests see a phantom "second iteration" and the external-carry check fires

## Confirmed

- Removing the `gidney_venting_adder` import from `adders/__init__.py` makes the remaud tests pass on CI
- Running just the remaud tests in isolation passes on all Python versions (3.11, 3.12, 3.13)
- No test function name conflicts exist between `test_remaud_adder.py` and `test_gidney_venting_adder.py`
- `QuantumPrimitive` instances are not singletons (`Primitive("name")` creates a new object each time), but JAX may still reuse them internally

## Suspects

| Component | Why |
|-----------|-----|
| `@custom_control` → `make_jaspr(ammended_func)` | Traces a controlled version, triggering full environment flattening including `jiteration_environment` |
| `@terminal_sampling` tests | Use `make_jaspr` at first call, which flattens environments including `jrange` bodies |
| `eqn.primitive.iteration_1_eqn` | Stored as a mutable attribute on the primitive — if the primitive object is reused across equations, this leaks |
| `jasp` eager loading | On `main`, `jasp` is loaded lazily (first `from qrisp.jasp import *` inside a test function). On this branch, `gidney_venting_adder` forces eager loading during package init, changing initialization order |

## To Debug Further

1. **Verify primitive reuse**: Check if JAX's `make_jaxpr`/`eval_jaxpr` reuses `Primitive` objects by name when constructing equations
2. **Bisect tests**: Run the full suite but skip `test_gidney_venting_adder.py` to confirm the venting adder tests are the trigger
3. **Instrument `iteration_env_evaluator`**: Log `id(eqn.primitive)` to see if the same primitive object appears across different `jrange` loops
4. **Test `terminal_sampling` isolation**: Run `test_terminal_add` (from the stash) before the remaud tests and check if it alone triggers the failure
