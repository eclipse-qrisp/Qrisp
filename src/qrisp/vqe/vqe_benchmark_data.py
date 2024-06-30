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
        The amount of QAOA layers for each run.
    circuit_depth : list[int]
        The depth of the compiled circuit of each run.
    qubit_amount : list[int]
        The amount of qubits of the compiled circuit of each run.
    shots : list[int]
        The amount of shots per backend call of each run.
    iterations : list[int]
        The amount of backend calls of each run.
    energy : list[dict]
        The iptimal energy of the problem.
    runtime : list[float]
        The amount of time passed (in seconds) of each run.
    optimal_solution : -
        The optimal solution of the problem.
    cost_function : callable
        The classical cost function of the benchmarked problem.
    
    """
    
    def __init__(self, benchmark_data, optimal_solution, cost_function):
        
        self.layer_depth = benchmark_data["layer_depth"]
        self.circuit_depth = benchmark_data["circuit_depth"]
        self.qubit_amount = benchmark_data["qubit_amount"]
        self.shots = benchmark_data["shots"]
        self.iterations = benchmark_data["iterations"]
        self.counts = benchmark_data["counts"]
        self.runtime = benchmark_data["runtime"]
        
        #self.optimal_solution = optimal_solution
        #self.cost_function = cost_function
        
    def evaluate(self, cost_metric = "oqv", gain_metric = "approx_ratio"):
        r"""
        Evaluates the data in terms of a cost and a gain metric.
        
        **Cost metric**
        
        The default cost metric is overall quantum volume
        
        .. math::
            
            \text{OQV} = \text{circuit\_depth} \times \text{qubits} \times \text{shots} \times \text{iterations}
        
        **Gain metric**
        
        By default, two gain metrics are avialable.
        
        The `approximation ratio <https://en.wikipedia.org/wiki/Approximation_algorithm>`_ 
        is a standard quantity in approximation algorithms and can be selected by
        setting ``gain_metric = "approx_ration"``.
        
        The time to solution metric as used in `this paper <http://arxiv.org/abs/2308.02342>`_
        can be selected with ``gain_metric = "tts"``.
        
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
        
        We set up a MaxCut instance and perform some benchmarking.
        
        ::
            
            from qrisp import *
            from networkx import Graph
            G = Graph()

            G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

            from qrisp.qaoa import maxcut_problem

            max_cut_instance = maxcut_problem(G)

            benchmark_data = max_cut_instance.benchmark(qarg = QuantumVariable(5),
                                       depth_range = [3,4,5],
                                       shot_range = [5000, 10000],
                                       iter_range = [25, 50],
                                       optimal_solution = "11100",
                                       repetitions = 2
                                       )
        
        We now evaluate the cost using the default metrics.
        
        ::
            
            cost_data, gain_data = benchmark_data.evaluate()
            
            print(cost_data[:10])
            #Yields: [17500000, 17500000, 35000000, 35000000, 35000000, 35000000, 70000000, 70000000, 22500000, 22500000]
            print(gain_data[:10])
            #Yields: [0.8425333333333328, 0.9379999999999996, 0.9256666666666667, 0.8816999999999998, 0.764399999999999, 0.6228000000000001, 0.8136000000000001, 0.9213999999999997, 0.8541333333333333, 0.6424333333333333]
        
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
        * ``counts``: The measurement results as returned by ``qarg.get_measurement()``.
        * ``runtime``: The time (in seconds) that the ``run`` method of :ref:`QAOAProblem` took.
        * ``optimal_solution``: The optimal solution of the problem

        """
        
        if isinstance(cost_metric, str):
            
            if cost_metric == "oqv":
                cost_metric = overall_quantum_volume
            else:
                raise Exception(f"Cost metric {cost_metric} is unknown")
                
            
        if isinstance(gain_metric, str):
            
            if gain_metric == "approx_ratio":
                gain_metric = lambda x : approximation_ratio(x["counts"], self.optimal_solution, self.cost_function)
            
            elif gain_metric == "tts":
                gain_metric = lambda x : time_to_solution(x["counts"], self.optimal_solution, self.cost_function)
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
                         "counts" : self.counts[i],
                         "runtime" : self.runtime[i],
                         "optimal_solution" : self.optimal_solution
                         }
            
            cost_data.append(cost_metric(run_data))
            gain_data.append(gain_metric(run_data))
        
        return cost_data, gain_data
    
    def visualize(self, cost_metric = "oqv", gain_metric = "approx_ratio"):
        """
        Plots the results of :meth:`.evaluate <qrisp.qaoa.QAOABenchmark.evaluate>`.

        Parameters
        ----------
        cost_metric : str or callable, optional
            The method to evaluate the cost of each run. The default is "oqv".
        gain_metric : str or callable, optional
            The method to evaluate the gain of each run. The default is "approx_ratio".

        Examples
        --------
        
        We create a MaxCut instance and benchmark several parameters
        
        ::
            
            from qrisp import *
            from networkx import Graph
            G = Graph()

            G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

            from qrisp.qaoa import maxcut_problem

            max_cut_instance = maxcut_problem(G)

            benchmark_data = max_cut_instance.benchmark(qarg = QuantumVariable(5),
                                       depth_range = [3,4,5],
                                       shot_range = [5000, 10000],
                                       iter_range = [25, 50],
                                       optimal_solution = "11100",
                                       repetitions = 2
                                       )
            
        To visualize the results, we call the corresponding method.
        
        ::
            
            benchmark_data.visualize()
            
        .. image:: benchmark_plot.png
            
            
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
            elif gain_metric == "tts":
                gain_name = "Time to solution"
        else:
            gain_name = gain_metric.__name__
            
        plt.xlabel(cost_name)
        plt.ylabel(gain_name)
        plt.grid()
        plt.show()
    
    def rank(self, metric = "approx_ratio", print_res = False, average_repetitions = False):
        """
        Ranks the runs of the benchmark according to a given metric.
        
        The default metric is approximation ratio. Similar to :meth:`.evaluate <qrisp.qaoa.QAOABenchmark.evaluate>`,
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
        
        We create a MaxCut instance and benchmark several parameters
        
        ::
            
            from qrisp import *
            from networkx import Graph
            G = Graph()

            G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

            from qrisp.qaoa import maxcut_problem

            max_cut_instance = maxcut_problem(G)

            benchmark_data = max_cut_instance.benchmark(qarg = QuantumVariable(5),
                                       depth_range = [3,4,5],
                                       shot_range = [5000, 10000],
                                       iter_range = [25, 50],
                                       optimal_solution = "11100",
                                       repetitions = 2
                                       )
            
        To rank the results, we call the according method:
        
        ::
            
            print(benchmark_data.rank()[0])
            #Yields: {'layer_depth': 5, 'circuit_depth': 44, 'qubit_amount': 5, 'shots': 10000, 'iterations': 50, 'counts': {'11100': 0.4909, '00011': 0.4909, '00010': 0.002, '11110': 0.002, '00001': 0.002, '11101': 0.002, '10000': 0.0015, '01000': 0.0015, '00100': 0.0015, '11011': 0.0015, '10111': 0.0015, '01111': 0.0015, '00000': 0.0001, '10010': 0.0001, '01010': 0.0001, '11010': 0.0001, '00110': 0.0001, '10110': 0.0001, '01110': 0.0001, '10001': 0.0001, '01001': 0.0001, '11001': 0.0001, '00101': 0.0001, '10101': 0.0001, '01101': 0.0001, '11111': 0.0001, '11000': 0.0, '10100': 0.0, '01100': 0.0, '10011': 0.0, '01011': 0.0, '00111': 0.0}, 'runtime': 1.4269020557403564, 'optimal_solution': '11100'}
            
        """
        
        if isinstance(metric, str):
    
            if metric == "approx_ratio":
                def approx_ratio(x):
                    return approximation_ratio(x["counts"], self.optimal_solution, self.cost_function)
    
                metric = approx_ratio
    
            elif metric == "time_to_sol":
                metric = lambda x: time_to_solution(x["counts"], self.optimal_solution, self.cost_function)
    
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
                        "optimal_solution": self.optimal_solution,
                        "counts" : self.counts[i]
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
                del run_data["counts"]
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
        
        We create a MaxCut instance and benchmark several parameters
        
        ::
            
            from qrisp import *
            from networkx import Graph
            G = Graph()

            G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

            from qrisp.qaoa import maxcut_problem

            max_cut_instance = maxcut_problem(G)

            benchmark_data = max_cut_instance.benchmark(qarg = QuantumVariable(5),
                                       depth_range = [3,4,5],
                                       shot_range = [5000, 10000],
                                       iter_range = [25, 50],
                                       optimal_solution = "11100",
                                       repetitions = 2
                                       )
            
        To save the results, we call the according method.
        
        ::
            
            benchmark_data.save("example.qaoa")
            

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
        :meth:`.save <qrisp.qaoa.QAOABenchmark.save>`.

        Parameters
        ----------
        filename : string
            The filename to load from.

        Returns
        -------
        obj : QAOABenchmark
            The loaded data.

        Examples
        --------
        
        We assume that the code from the example in :meth:`.save <qrisp.qaoa.QAOABenchmark.save>`
        has been executed and load the corresponding data:
            
        ::
            
            from qrisp.qaoa import QAOABenchmark
            
            benchmark_data = QAOABenchmark.load("example.qaoa")
            
            
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

def max_five_metric(metric_dict):
    counts = metric_dict["counts"].copy()
    maxfive = sorted(counts , key=counts.get, reverse=True)[:5]
    fivesol = []
    for name, age in counts.items():  
        if name in maxfive:
            fivesol.append((name, age))
    return fivesol


def time_to_solution(run_data, optimal_solution, cost_function):
    """
    Parameters
    ----------
    obj_function : objective function of the problem
                    (i.e. the "maxcut_obj" method for the MaxCut Problem).

    counts : the result dictionary from the QAOA method, contaning the 
                    bitstrings as keys and the counts divided by the number
                    of shots as values.

    optimal_solution: the optimal solution of the problem

    G : the Graph related to the problem

    Returns
    -------
    time to solution measure from http://arxiv.org/abs/2308.02342.
    
    It corresponds to 1/p_opt, where p_opt is the sum of the squared 
    amplitudes associated to the binary strings encoding the optimal solution.

    """
    obj_function = lambda x : cost_function({x : 1})
    optimal_solution_cost = obj_function(optimal_solution)
    
    return 1/sum([v for k,v in run_data["counts"].items() if obj_function(k)==optimal_solution_cost])


def approximation_ratio(counts, optimal_solution, cost_function):
    """
    Parameters
    ----------
    counts : the result dictionary from the QAOA method, contaning the 
                    bitstrings as keys and the counts divided by the number
                    of shots as values.
    optimal_solution: the optimal solution of the problem
    cost_function : Cost Function used to evaluate the optimization

    Returns
    -------
    approximation ratio measure, commonly used to evaluate the MaxCut Problem

    """
    return cost_function(counts)/cost_function({optimal_solution: 1})

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