include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Jasp dialect
//===----------------------------------------------------------------------===//

def JaspDialect : Dialect {
    let summary = "A dialect for real-time hybrid quantum computation.";
    let description = [{
        This dialect provides simple data structures that enable expression and
		manipulation of quantum computations.
    }];

    let name = "jasp";
    let cppNamespace = "::jasp";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Jasp types
//===----------------------------------------------------------------------===//

class JaspType<string name, string typeMnemonic>
    : TypeDef<JaspDialect, name> {
    let mnemonic = typeMnemonic;
}

def QuantumStateType : JaspType<"QuantumState", "QuantumState"> {
    let summary = "An opaque type describing the quantum state of the machine";
    let description = "This object is passed around the program to capture the computations.";
}

def QubitType : JaspType<"Qubit", "Qubit"> {
    let summary = "A type describing an individual qubit";
    let description = "Qubit objects are semantically identical to integers as the simply index the QuantumState. This especially implies that it is semantically well defined to copy a qubit.";
}

def QubitArrayType : JaspType<"QubitArray", "QubitArray"> {
    let summary = "A type describing a dynamically sized collection of Qubits";
    let description = "QubitArrays enable expression of dynamically sized programs. They are semantically equivalent to immutable arrays of integers.";
}