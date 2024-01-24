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



from qrisp.qaoa.qaoa_problem import QAOAProblem
from qrisp.qaoa.problems.maxCut import create_maxcut_cl_cost_function, create_maxcut_cost_operator
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
import networkx as nx
import matplotlib.pyplot as plt

"""
This class entails example usage of Fourier heuristic for QAOA Problems, which are used for finding 
good starting parameters for regular graph MaxCut QAOA problems. For further details, see the implementation, 
mainly the class `fourier_optimization_wrapper`.

The general idea is to exploit the energy landscape of regular graph QAOA optimizations w.r.t. the parameterization.
The optimization curve for these parameters in such problem instances is "rather smooth". Therefore, a discrete sine/cosine 
transformation of the parameters is effective. This also lowers the number of parameters to be optimized, 
if fourier_depth < (original) depth. For more details, see the paper below. 

The original implementation stems from the paper 
https://arxiv.org/pdf/1812.01041.pdf 
by L. Zheo et al.
"""


giraf = nx.random_regular_graph(3, 16, seed=133)


#instanciate QAOAProblem
QAOAinstance = QAOAProblem(create_maxcut_cost_operator(giraf), RX_mixer, create_maxcut_cl_cost_function(giraf))
qarg = QuantumVariable(giraf.number_of_nodes())
#set the fourier_depth
QAOAinstance.set_fourier_depth(4)
res = QAOAinstance.run( qarg=qarg, depth=5 )

#print the ideal solutions
aClcostFct = create_maxcut_cl_cost_function(giraf)
print("5 most likely Solutions") 
maxfive = sorted(res , key = res.get, reverse=True)[:5]
for name, age in res .items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:
        print((name, age))
        print(aClcostFct(name, giraf))

#draw graph
nx.draw(giraf,with_labels = True)
plt.show() 




