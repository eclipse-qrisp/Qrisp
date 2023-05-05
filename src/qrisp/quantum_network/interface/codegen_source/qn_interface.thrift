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


struct Message
{
   1: list<Qubit> qubits,
   2: string annotation,
}


//Defintion of the Backend Service
service QuantumNetworkService
{	

   void register_client(1: string name),
   
   QuantumCircuit get_clear_qc(1: string name),
   
   list<Qubit> request_qubits(1: i32 amount, 2: string name),
   
   void send_qubits(1: string sender, 2: string recipient, 3: Message msg),
   
	map<string,i32> run(1: QuantumCircuit qc, 2: string name),
   
   list<Message> inbox(1: string name),
   
   QuantumCircuit get_overall_qc(),
   
   void reset_network_state(),
   
}