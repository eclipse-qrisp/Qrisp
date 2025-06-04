"""
********************************************************************************
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
********************************************************************************
"""

import numpy as xp
import os

try:
    float_thresh = os.environ["QRISP_SIMULATOR_FLOAT_THRESH"]
except KeyError:
    float_thresh = 1e-5
float_thresh = xp.float32(float_thresh)

try:
    cutoff_ratio = os.environ["QRISP_SIMULATOR_CUTOFF_RATIO"]
except KeyError:
    cutoff_ratio = 2e-4
cutoff_ratio = xp.float32(cutoff_ratio)

try:
    sparsification_rate = os.environ["QRISP_SIMULATOR_SPARSIFICATION_RATE"]
except KeyError:
    sparsification_rate = 0.4
sparsification_rate = xp.float32(sparsification_rate)