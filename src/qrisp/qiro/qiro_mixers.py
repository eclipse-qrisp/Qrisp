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


from qrisp import x, rx

def qiro_RXMixer(solutions = [],exclusions = []):
    union = solutions + exclusions
    def RX_mixer(qv, beta):

        for i in range(len(qv)):
            if not i in union:
                rx(2 * beta, qv[i])
    return RX_mixer


def qiro_init_function(solutions = [], exclusions = []):
    union = solutions + exclusions
    def init_state(qv):
        from qrisp import h
        #for i in problem.nodes:
        for i in range(len(qv)):
            if not i in union:
                h(qv[i])
        for i in solutions:
            x(qv[i])
    return init_state
