# JASP MLIR Dialect Definition

This directory contains the TableGen definitions for the JASP (Qrisp) MLIR dialect.

## Files

- **`JaspDialect.td`** - Dialect and quantum types (QuantumState, Qubit, QubitArray)
- **`JaspOps.td`** - Operation definitions (11 operations)
- **`JaspPythonOps.td`** - Python binding specification

## Generating Python Bindings

### Prerequisites

- **MLIR installation** with `mlir-tblgen` in PATH
- Install MLIR: https://mlir.llvm.org/getting_started/

### Generation Command

From this directory, run:

```bash
mlir-tblgen JaspPythonOps.td \
    -gen-python-op-bindings \
    -bind-dialect=jasp \
    -I /path/to/llvm-project/mlir/include \
    -I . \
    -o ../dialect_implementation/_jasp_ops_gen.py
```

Replace `/path/to/llvm-project/mlir/include` with your MLIR include directory.