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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Function to generate the torus
def generate_torus(R=6, r=2, n_major=15, n_minor=10, flatten_z=1.5):
    u = np.linspace(0, 2 * np.pi, n_major)
    v = np.linspace(0, 2 * np.pi, n_minor)
    u, v = np.meshgrid(u, v)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = flatten_z * r * np.sin(v)

    return x, y, z, u, v

# Equal axis squale
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Custom colormap: blue to violet
colors = ["#101e6d", "#9a00d3"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_bv", colors)

# Plot setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()

# Generate torus geometry
x, y, z, u, v = generate_torus()

# Color data based on angular position
color_data = (np.sin(u) + 1) / 2

# Plot torus surface
ax.plot_surface(x, y, z, facecolors=cmap(color_data),
                rstride=1, cstride=1,
                linewidth=0, antialiased=True, alpha=0.9)

# Grid lines 
#for i in range(x.shape[0]):
 #   ax.plot(x[i, :], y[i, :], z[i, :], color='black', lw=2)
#for j in range(x.shape[1]):
#    ax.plot(x[:, j], y[:, j], z[:, j], color='black', lw=2)

# Black dotes
#ax.scatter(x, y, z, color='black', s=35)

# Angle view and equal scaling
ax.view_init(elev=25, azim=35)
set_axes_equal(ax)

plt.tight_layout()
plt.show()

#plt.savefig("torus.pdf", bbox_inches = "tight")
#plt.savefig("torus.svg", format = "svg", dpi = 80, bbox_inches = "tight")

