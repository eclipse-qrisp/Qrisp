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

class JRangeIterator:
    
    def __init__(self, stop):
        
        self.stop = stop
        
    def __iter__(self):
        self.iteration = 0
        return self
    
    def __next__(self):
        
        self.iteration += 1
        if self.iteration == 1:
            from qrisp.environments import JIterationEnvironment
            self.iter_env = JIterationEnvironment()
            self.iter_env.__enter__()
            return self.stop
        elif self.iteration == 2:
            self.stop += 1
            self.iter_env.__exit__(None, None, None)
            self.iter_env.__enter__()
            return self.stop
        elif self.iteration == 3:
            self.stop += 1
            self.iter_env.__exit__(None, None, None)
            raise StopIteration

def jrange(stop):
    if isinstance(stop, int):
        return range(stop)
    else:
        return JRangeIterator(stop)
    
