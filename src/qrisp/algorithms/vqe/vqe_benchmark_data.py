"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

import matplotlib.pyplot as plt
import dill as pickle

class VQEBenchmark:
    """
    This class is a wrapper for representing and evaluating the data collected in the :meth:`.benchmark <qrisp.qaoa.QAOAProblem.benchmark>` method.
    
    Attributes
    ----------
    layer_depth : list[int]
        The amount of VQE layers for each run.
    circuit_depth : list[int]
        The depth of the compiled circuit of each run.
    qubit_amount : list[int]
        The amount of qubits of the compiled circuit of each run.
    shots : list[int]
        The amount of shots per backend call of each run.
    iterations : list[int]
        The amount of backend calls of each run.
    energy : list[dict]
        The energy of the problem Hamiltonian for the optimized ciruits for each run.
    runtime : list[float]
        The amount of time passed (in seconds) of each run.
    optimal_energy : float
        The exact ground state energy of the problem Hamiltonian.
    hamiltonian : :ref:`QubitOperator`
        The problem Hamiltonian.
    
    """
    
    def __init__(self, benchmark_data, optimal_energy, hamiltonian):
        
        self.layer_depth = benchmark_data["layer_depth"]
        self.circuit_depth = benchmark_data["circuit_depth"]
        self.qubit_amount = benchmark_data["qubit_amount"]
        self.shots = benchmark_data["shots"]
        self.iterations = benchmark_data["iterations"]
        self.energy= benchmark_data["energy"]
        self.runtime = benchmark_data["runtime"]
        self.optimal_energy= optimal_energy
        self.hamiltonian = hamiltonian
        
    def evaluate(self, cost_metric = "oqv", gain_metric = "approx_ratio"):
        r"""
        Evaluates the data in terms of a cost and a gain metric.
        
        **Cost metric**
        
        The default cost metric is overall quantum volume
        
        .. math::
            
            \text{OQV} = \text{circuit_depth} \times \text{qubits} \times \text{shots} \times \text{iterations}
        
        **Gain metric**
        
        By default, two gain metrics are avialable.
        
        The `approximation ratio <https://en.wikipedia.org/wiki/Approximation_algorithm>`_ 
        is a standard quantity in approximation algorithms and can be selected by
        setting ``gain_metric = "approx_ratio"``.
        
        Users can implement their own cost/gain metric by calling ``.evaluate`` with a suited function. 
        For more information check the examples.
        

        Parameters
        ----------
        cost_metric : str or callable, optional
            The method to evaluate the cost of each run. The default is "oqv".
        gain_metric : str or callable, optional
            The method to evaluate the gain of each run. The default is "approx_ratio".

        Returns
        -------
        cost_data : list[float]
            A list containing the cost values of each run.
        gain_data : list[float]
            A list containing the gain of each run.
            
        Examples
        --------
        
        We set up a Heisenberg problem instance and perform some benchmarking.
        
        ::

            from networkx import Graph

            G =Graph()
            G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
            from qrisp.vqe.problems.heisenberg import *

            vqe = heisenberg_problem(G,1,0)
            H = create_heisenberg_hamiltonian(G,1,0)

            benchmark_data = vqe.benchmark(qarg = QuantumVariable(5),
                                depth_range = [1,2,3],
                                shot_range = [5000,10000],
                                iter_range = [25,50],
                                optimal_energy = H.ground_state_energy(),
                                repetitions = 2
                                )
        
        We now evaluate the cost using the default metrics.
        
        ::
            
            cost_data, gain_data = benchmark_data.evaluate()
            
            print(cost_data[:10])
            #Yields: [15625000, 15625000, 31250000, 31250000, 31250000, 31250000, 62500000, 62500000, 29375000, 29375000]
            print(gain_data[:10])
            #Yields: [0.8580900440328681, 0.8543553838641942, 0.8510356859364849, 0.8539404216232306, 0.8587643576744335, 0.8635364234455172, 0.8497389289334728, 0.8600092443973252, 0.884232665213583, 0.8656112346503356]

        To set up a user specified cost metric we create a customized function
        
        ::
            
            def runtime(run_data):
                return run_data["runtime"]
            
            cost_data, gain_data = benchmark_data.evaluate(cost_metric = runtime)
            
        This function extracts the runtime (in seconds) and uses that as a cost metric. 
        The ``run_data`` dictionary contains the following entries:
            
        * ``layer_depth``: The amount of layers
        * ``circuit_depth``: The depth of the compiled circuit as returned by :meth:`.depth <qrisp.QuantumCircuit.depth>` method.
        * ``qubit_amount``: The amount of qubits of the compiled circuit.
        * ``shots``: The amount of shots that have been performed in this run.
        * ``iterations``: The amount of backend calls, that the optimizer was allowed to do.
        * ``energy``: The energy of the problem Hamiltonian for the optimized ciruits for each run.
        * ``runtime``: The time (in seconds) that the ``run`` method of :ref:`VQEProblem` took.
        * ``optimal_energy``: The exact ground state energy of the problem Hamiltonian.
        
        """
        
        if isinstance(cost_metric, str):
            
            if cost_metric == "oqv":
                cost_metric = overall_quantum_volume
            else:
                raise Exception(f"Cost metric {cost_metric} is unknown")
                
            
        if isinstance(gain_metric, str):
            
            if gain_metric == "approx_ratio":
                gain_metric = lambda x : approximation_ratio(x["energy"], self.optimal_energy)
            else:
                raise Exception(f"Gain metric {gain_metric} is unknown")
                
        
        cost_data = []
        gain_data = []
        
        for i in range(len(self.layer_depth)):
            
            run_data = {"layer_depth" : self.layer_depth[i],
                         "circuit_depth" : self.circuit_depth[i],
                         "qubit_amount" : self.qubit_amount[i],
                         "shots" : self.shots[i],
                         "iterations" : self.iterations[i],
                         "energy" : self.energy[i],
                         "runtime" : self.runtime[i],
                         "optimal_energy" : self.optimal_energy
                         }
            
            cost_data.append(cost_metric(run_data))
            gain_data.append(gain_metric(run_data))
        
        return cost_data, gain_data
    
    def visualize(self, cost_metric = "oqv", gain_metric = "approx_ratio"):
        """
        Plots the results of :meth:`.evaluate <qrisp.vqe.VQEBenchmark.evaluate>`.

        Parameters
        ----------
        cost_metric : str or callable, optional
            The method to evaluate the cost of each run. The default is "oqv".
        gain_metric : str or callable, optional
            The method to evaluate the gain of each run. The default is "approx_ratio".

        Examples
        --------
        
        We create a Heisenberg problem instance and benchmark several parameters:
        
        ::
            
            from networkx import Graph
            G =Graph()
            G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])

            from qrisp.vqe.problems.heisenberg import *

            vqe = heisenberg_problem(G,1,0)
            H = create_heisenberg_hamiltonian(G,1,0)

            benchmark_data = vqe.benchmark(qarg = QuantumVariable(5),
                                depth_range = [1,2,3],
                                shot_range = [5000,10000],
                                iter_range = [25,50],
                                optimal_energy = H.ground_state_energy(),
                                repetitions = 2
                                )
            
        To visualize the results, we call the corresponding method.
        
        ::
            
            benchmark_data.visualize()
            
        .. image:: vqe_benchmark_plot.png
            
            
        """
        
        cost_data, gain_data = self.evaluate(cost_metric, gain_metric)
        
        plt.plot(cost_data, gain_data, "x")
        
        if isinstance(cost_metric, str):
            if cost_metric == "oqv":
                cost_name = "Overall quantum volume"
        else:
            cost_name = cost_metric.__name__
            
        if isinstance(gain_metric, str):           
            if gain_metric == "approx_ratio":
                gain_name = "Approximation ratio"
        else:
            gain_name = gain_metric.__name__
            
        plt.xlabel(cost_name)
        plt.ylabel(gain_name)
        plt.grid()
        plt.show()
    
    def rank(self, metric = "approx_ratio", print_res = False, average_repetitions = False):
        """
        Ranks the runs of the benchmark according to a given metric.
        
        The default metric is approximation ratio. Similar to :meth:`.evaluate <qrisp.vqe.VQEBenchmark.evaluate>`,
        the metric can be user specified.

        Parameters
        ----------
        metric : str or callable, optional
            The metric according to which should be ranked. The default is "approx_ratio".

        Returns
        -------
        list[dict]
            List of dictionaries, where the first element has the highest rank.

        Examples
        --------
        
        We create a Heisenberg problem instance and benchmark several parameters:
        
        ::
            
            from networkx import Graph
            G =Graph()
            G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])

            from qrisp.vqe.problems.heisenberg import *

            vqe = heisenberg_problem(G,1,0)
            H = create_heisenberg_hamiltonian(G,1,0)

            benchmark_data = vqe.benchmark(qarg = QuantumVariable(5),
                                depth_range = [1,2,3],
                                shot_range = [5000,10000],
                                iter_range = [25,50],
                                optimal_energy = H.ground_state_energy(),
                                repetitions = 2
                                )
            
        To rank the results, we call the according method:
        
        ::
            
            print(benchmark_data.rank()[0])
            #Yields: {'layer_depth': 3, 'circuit_depth': 69, 'qubit_amount': 5, 'shots': 10000, 'iterations': 50, 'runtime': 2.202655076980591, 'optimal_energy': -7.711545013271984, 'energy': -7.465600000000004, 'metric': 0.9681069081683767}

        """
        
        if isinstance(metric, str):
    
            if metric == "approx_ratio":
                def approx_ratio(x):
                    return approximation_ratio(x["energy"], self.optimal_energy)
    
                metric = approx_ratio
    
        run_data_list = []
    
        if average_repetitions:
            # Create a dictionary to store aggregated averages
            average_dict = {}
    
        for i in range(len(self.layer_depth)):
    
            run_data = {"layer_depth": self.layer_depth[i],
                        "circuit_depth": self.circuit_depth[i],
                        "qubit_amount": self.qubit_amount[i],
                        "shots": self.shots[i],
                        "iterations": self.iterations[i],
                        "runtime": self.runtime[i],
                        "optimal_energy": self.optimal_energy,
                        "energy" : self.energy[i]
                        }
    
            run_data["metric"] = metric(run_data)
            if average_repetitions:
                # Create a unique key based on the parameters
                key = (run_data['layer_depth'], run_data['shots'], run_data['iterations'])
    
                # Add the result to the corresponding key in the dictionary
                if key not in average_dict:
                    average_dict[key] = {
                        'total_metric': 0,
                        'count': 0
                    }
                average_dict[key]['total_metric'] += metric(run_data)
                average_dict[key]['count'] += 1
    
            run_data_list.append(run_data)
    
        if average_repetitions:
            # Calculate the average for each unique parameter combination
            
            temp = list(run_data_list)
            run_data_list = []
            for run_data in temp:
                key = (run_data['layer_depth'], run_data['shots'], run_data['iterations'])
                
                if not key in average_dict:
                    continue
                
                run_data['metric'] = average_dict[key]['total_metric'] / average_dict[key]['count']
                del run_data["energy"]
                del run_data["runtime"]
                run_data_list.append(run_data)
                
                del average_dict[key]
    
        run_data_list.sort(key=lambda x: x["metric"], reverse=True)
        
        if print_res:
            self.print_rank_table(run_data_list, metric.__name__)
    
        return run_data_list

    
    def print_rank_table(self, run_data_list, metric_name):
        """
        Prints a nicely formatted table of the ranked runs.
    
        Parameters
        ----------
        run_data_list : list[dict]
            List of dictionaries containing run data.
        metric : function
            Function to rank the run data
    
        """
        header = ["Rank", metric_name, "Overall QV", "p", "QC depth", "QB count", "Shots", "Iterations"]
        
        
        # Print the header row
        print("{:<5} {:<12} {:<12} {:<4} {:<10} {:<9} {:<7} {:<10}".format(*header))
        print("============================================================================")
        
        for i, run_data in enumerate(run_data_list):
            
            oqv = sci_notation(overall_quantum_volume(run_data), 4)
            metric_value = sci_notation(run_data["metric"], 3)
            
            
            row = [i, metric_value, oqv, run_data["layer_depth"], run_data["circuit_depth"], run_data["qubit_amount"],
                   run_data["shots"], run_data["iterations"]]
            
            # Print each row
            print("{:<5} {:<12} {:<12} {:<4} {:<10} {:<9} {:<7} {:<10}".format(*row))
        
    def save(self, filename):
        """
        Saves the data to the harddrive for later use.

        Parameters
        ----------
        filename : string
            The filename where to save the data.

        Examples
        --------
        
        We create a Heisenberg problem and benchmark several parameters:
        
        ::
            
            from networkx import Graph
            G =Graph()
            G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])

            from qrisp.vqe.problems.heisenberg import *

            vqe = heisenberg_problem(G,1,0)
            H = create_heisenberg_hamiltonian(G,1,0)

            benchmark_data = vqe.benchmark(qarg = QuantumVariable(5),
                                depth_range = [1,2,3],
                                shot_range = [5000,10000],
                                iter_range = [25,50],
                                optimal_energy = H.ground_state_energy(),
                                repetitions = 2
                                )
            
        To save the results, we call the according method.
        
        ::
            
            benchmark_data.save("example.vqe")
            

        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Benchmark data saved to {filename}")
        except Exception as e:
            print(f"Error saving benchmark data: {e}")

    @classmethod
    def load(cls, filename):
        """
        Loads benchmark data from the harddrive that has been saved by 
        :meth:`.save <qrisp.vqe.VQEBenchmark.save>`.

        Parameters
        ----------
        filename : string
            The filename to load from.

        Returns
        -------
        obj : VQEBenchmark
            The loaded data.

        Examples
        --------
        
        We assume that the code from the example in :meth:`.save <qrisp.vqe.VQEBenchmark.save>`
        has been executed and load the corresponding data:
            
        ::
            
            from qrisp.vqe import VQEBenchmark
            
            benchmark_data = VQEBenchmark.load("example.vqe")
            
            
        """
        try:
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
            return obj
        except Exception as e:
            print(f"Error loading benchmark data: {e}")
            return None
    

# create qScore        

def overall_quantum_volume(run_data):
    return run_data["circuit_depth"]*run_data["qubit_amount"]*run_data["shots"]*run_data["iterations"]

def approximation_ratio(energy, optimal_energy):
    """
    Parameters
    ----------
    energy : float
        The energy of the problem Hamiltonian for the optimized ciruit.
    optimal_energy: float
        The optimal energy of the problem Hamiltonian.

    Returns
    -------
    float
        The approximation ratio. 

    """
    return energy/optimal_energy

def ilog(n, base):
    """
    Find the integer log of n with respect to the base.

    >>> import math
    >>> for base in range(2, 16 + 1):
    ...     for n in range(1, 1000):
    ...         assert ilog(n, base) == int(math.log(n, base) + 1e-10), '%s %s' % (n, base)
    """
    
    if abs(n) < 1:
        n = 1/n
    
    count = 0
    while n >= base:
        count += 1
        n //= base
    return count

def sci_notation(n, prec=3):
    """
    Represent n in scientific notation, with the specified precision.

    >>> sci_notation(1234 * 10**1000)
    '1.234e+1003'
    >>> sci_notation(10**1000 // 2, prec=1)
    '5.0e+999'
    """
    base = 10
    exponent = ilog(n, base)
    
    if abs(n) < 1:
        exponent = -exponent
    
    mantissa = n / base**exponent
    return '{0:.{1}f}e{2:+d}'.format(mantissa, prec, exponent)