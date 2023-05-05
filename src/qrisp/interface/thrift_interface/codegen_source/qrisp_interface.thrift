namespace py codegen

//Definition of a portable Quantum Circuit structure

struct Qubit
{
	1: string identifier,
}

struct Clbit
{
	1: string identifier,
}

struct Operation
{
	1: string name,
	2: i32 num_qubits,
	3: i32 num_clbits,
	4: list<double> params,
	5: optional QuantumCircuit definition,
}

struct Instruction
{
	1: Operation op,
	2: list<Qubit> qubits,
	3: list<Clbit> clbits,
}

struct QuantumCircuit
{
	1: list<Qubit> qubits,
	2: list<Clbit> clbits,
	3: list<Instruction> data,
	4: bool init = true,
}



// Backend Status information
struct ConnectivityEdge
{
	1: Qubit qb1,
	2: Qubit qb2,
}

struct BackendStatus
{
	1: string name,
	2: bool online,
	3: optional list<Qubit> qubit_list,
	4: optional list<Operation> elementary_ops,
	5: optional list<ConnectivityEdge> connectivity_map,
}



//Defintion of the Backend Service
service BackendService
{	
	map<string,i32> run(1: QuantumCircuit qc, 2: i32 shots, 3: string token),
	BackendStatus ping(),
}
