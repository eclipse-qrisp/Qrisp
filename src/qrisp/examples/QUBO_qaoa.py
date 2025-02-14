"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp.qaoa import *
import numpy as np

# Pulser tutorial example from QAOA and QAA to solve a QUBO problem
#        available here: https://pulser.readthedocs.io/en/stable/tutorials/qubo.html
Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)

# The following tutorials are taken from A tutorial on Formulating and Using QUBO models from 
#                                        Fred Glover, Gary Kochenberger, and YU Du;
#                         available here: https://arxiv.org/ftp/arxiv/papers/1811/1811.11538.pdf

# QUBO tutorial no. 1
Q = np. array(
    [
        [-5,2,4,0],
        [2,-3,1,0],
        [4,1,-8,5],
        [0,0,5,-6],
    ]
)

# QUBO tutorial: number partitioning
Q = np.array(
    [
        [-3525,175,325,775,1050,425,525,250],
        [175,-1113,91,217,294,119,147,70],
        [325,91,-1989,403,546,221,273,130],
        [775,217,403,-4185,1302,527,651,310],
        [1050,294,546,1302,-5208,714,882,420],
        [425,119,221,527,714,-2533,357,170],
        [525,147,273,651,882,357,-3045,210],
        [250,70,130,310,420,170,210,-1560],
    ]
)

# QUBO tutorial: maxcut - here one has to find maximum of QUBO_obj aka max y=x.T@Q@x
Q = np.array(
    [
        [2,-1,-1,0,0],
        [-1,2,0,-1,0],
        [-1,0,3,-1,-1],
        [0,-1,-1,3,-1],
        [0,0,-1,-1,2],
    ]
)

# QUBO tutorial: minimum vertex cover - here one has to find maximum of QUBO_obj aka max y=x.T@Q@x
Q = np.array(
    [
        [-15,4,4,0,0],
        [4,-15,0,4,0],
        [4,0,-23,4,4],
        [0,4,4,-23,4],
        [0,0,4,4,-15],
    ]
)

# QUBO tutorial: max-2-sat problem
Q = np.array(
    [
        [1,0,0,0],
        [0,0,-0.5,0.5],
        [0,-0.5,0,1],
        [0,0.5,1,-2],
    ]
)

# QUBO tutorial: set packing - - here one has to find maximum of QUBO_obj aka max y=x.T@Q@x
Q = np.array(
    [
        [1,-3,-3,-3],
        [-3,1,0,0],
        [-3,0,1,-3],
        [-3,0,-3,1],
    ]
)

# QUBO tutorial: set partitioning
Q = np.array(
    [
        [-17,10,10,10,0,20],
        [10,-18,10,10,10,20],
        [10,10,-29,10,20,20],
        [10,10,10,-19,10,10],
        [0,10,20,10,-17,10],
        [20,20,20,10,10,-28],
    ]
)

# QUBO tutorial: graph coloring
Q = np.array(
    [
        [-4,4,4,2,0,0,0,0,0,0,0,0,2,0,0],
        [4,-4,4,0,2,0,0,0,0,0,0,0,0,2,0],
        [4,4,-4,0,0,2,0,0,0,0,0,0,0,0,2],
        [2,0,0,-4,4,4,2,0,0,2,0,0,2,0,0],
        [0,2,0,4,-4,4,0,2,0,0,2,0,0,2,0],
        [0,0,2,4,4,-4,0,0,2,0,0,2,0,0,2],
        [0,0,0,2,0,0,-4,4,4,2,0,0,0,0,0],
        [0,0,0,0,2,0,4,-4,4,0,2,0,0,0,0],
        [0,0,0,0,0,2,4,4,-4,0,0,2,0,0,0],
        [0,0,0,2,0,0,2,0,0,-4,4,4,2,0,0],
        [0,0,0,0,2,0,0,2,0,4,-4,4,0,2,0],
        [0,0,0,0,0,2,0,0,2,4,4,-4,0,0,2],
        [2,0,0,2,0,0,0,0,0,2,0,0,-4,4,4],
        [0,2,0,0,2,0,0,0,0,0,2,0,4,-4,4],
        [0,0,2,0,0,2,0,0,0,0,0,2,4,4,-4],
    ]
)

# QUBO tutorial: general 0/1 programming - here one has to find maximum of QUBO_obj aka max y=x.T@Q@x
Q = np.array(
    [
        [526,-150,-160,-190,-180,-20,-40,30,60,120],
        [-150,574,-180,-200,-200,-20,-40,30,60,120],
        [-160,-180,688,-220,-200,-40,-80,20,40,80],
        [-190,-200,-220,645,-240,-30,-60,40,80,160],
        [-180,-200,-200,-240,605,-20,-40,40,80,160],
        [-20,-20,-40,-30,-20,130,-20,0,0,0],
        [-40,-40,-80,-60,-40,-20,240,0,0,0],
        [30,30,20,40,40,0,0,-110,-20,-40],
        [60,60,40,80,80,0,0,-20,-240,-80],
        [120,120,80,160,160,0,0,-40,-80,-560]
    ]
)

# QUBO tutorial: quadratic assignment
Q = np.array(
    [
        [-400,200,200,200,40,75,200,16,30],
        [200,-400,200,40,200,65,16,200,26],
        [200,200,-400,75,65,200,30,26,200],
        [200,40,75,-400,200,200,200,24,45],
        [40,200,65,200,-400,20,24,200,39],
        [75,65,200,200,200,-400,45,39,200],
        [200,16,30,200,24,45,-400,200,200],
        [16,200,26,24,200,39,200,-400,200],
        [30,26,200,45,39,200,200,200,-400],
    ]
)

# QUBO tutorial: quadratic knapsack - here one has to find maximum of QUBO_obj aka max y=x.T@Q@x
Q = np.array(
    [
        [1922,-476,-397,-235,-80,-160],
        [-476,1565,-299,-177,-60,-120],
        [-397,-299,1352,-148,-50,-100],
        [-235,-177,-148,874,-30,-60],
        [-80,-60,-50,-30,310,-20],
        [-160,-120,-100,-60,-20,600],
    ]
)

solution = solve_QUBO(Q, depth = 1, backend = def_backend, n_solutions=7, print_res=True)
