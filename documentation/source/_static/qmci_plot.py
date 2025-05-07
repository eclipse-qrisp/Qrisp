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

import matplotlib.pyplot as plt
import numpy as np

# Define the interval width
width = 0.125
# Left edges of each bar: 0, 0.125, 0.25, etc.
left_edges = np.arange(0, 1, width)

# Midpoints (to evaluate y = x^2 at each bar's center)
midpoints = left_edges + width / 2

# Compute the function values at midpoints
heights = left_edges**2

# Plot bars
plt.bar(left_edges+width/2, heights, width=width, color="#444444", label="Bar plot of x^2")

# Also plot the smooth curve
x = np.linspace(0, 1, 100)
y = x**2
plt.plot(x, y, color="#20306f", linewidth=5, zorder=2)

plt.xlabel(r'$x$', fontsize=18, color="#444444", fontname="Segoe UI")
plt.ylabel(r'$f(x)$', fontsize=18, color="#444444", fontname="Segoe UI")
plt.tick_params(axis='both',color="#444444", labelsize=18)
plt.grid()

# Hide the top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#plt.tight_layout()
#plt.show()
#plt.savefig("qmci.svg", format = "svg", dpi = 80, bbox_inches = "tight")