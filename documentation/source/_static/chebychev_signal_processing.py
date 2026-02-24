import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection
from numpy.polynomial import Chebyshev
from numpy.polynomial.chebyshev import chebval

# ── Qrisp palette ──────────────────────────────────────────────────────
QRISP_NAVY   = "#20306F"
QRISP_PURPLE = "#7D2BFF"

# ── Target: narrow Gaussian on [-1, 1] ─────────────────────────────────
mu    = 0
sigma = 0.05

def gaussian(x):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Chebyshev fit (high degree for reference)
x_nodes    = np.cos(np.pi * np.arange(201) / 200)
f_nodes    = gaussian(x_nodes)
cheb_fit   = Chebyshev.fit(x_nodes, f_nodes, deg=100)
cheb_coeffs = cheb_fit.coef

x_plot     = np.linspace(-1, 1, 2000)
f_gaussian = gaussian(x_plot)

# Polynomial approximations at several degrees
degrees = [6, 10, 16, 24, 40]
palette = ["#A0B4F0",   # lightest – low degree
           "#7B8FDC",
           "#5E78D2",
           "#3F5CB8",
           QRISP_NAVY] # strongest – high degree
lws     = 1.1*np.array([2.3, 2.5, 2.7, 3, 3.6])[::-1]

# ── Figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.8, 4.), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Gradient fill helper
def gradient_fill(ax, x, y, color, n_bands=50, alpha_base=0.28):
    """Fill under curve with vertical alpha gradient (strong at base, fading up)."""
    rgb = to_rgb(color)
    for i in range(n_bands):
        lo = i / n_bands
        hi = (i + 1) / n_bands
        alpha = alpha_base * (1 - lo)**1.01
        ax.fill_between(x, y * lo, y * hi,
                        color=(*rgb, alpha), linewidth=0)


def variable_width_line(
    ax,
    x,
    y,
    color,
    lw,
    alpha=1.0,
    x_focus=0.0,
    focus_width=0.20,
    edge_scale=0.72,
    center_boost=0.62,
    ripple_amp=0.06,
    zorder=3,
):
    """Draw a curve with linewidth varying along x (thicker near center)."""
    rgb = to_rgb(color)
    mid_x = 0.5 * (x[:-1] + x[1:])
    focus = np.exp(-((mid_x - x_focus) / focus_width) ** 2)
    phase = (mid_x - mid_x.min()) / (mid_x.max() - mid_x.min() + 1e-12)
    ripple = 1.0 + ripple_amp * np.sin(18.0 * np.pi * phase)
    widths = lw * (edge_scale + center_boost * focus) * ripple

    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        linewidths=widths,
        colors=[(*rgb, alpha)],
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lc)


def subtle_glow_line(ax, x, y, color, lw):
    """Variable-width stroke with a tiny, matching glow."""
    variable_width_line(
        ax,
        x,
        y,
        color,
        lw + 1.0,
        alpha=0.08,
        focus_width=0.24,
        edge_scale=0.76,
        center_boost=0.54,
        ripple_amp=0.03,
        zorder=2,
    )
    variable_width_line(
        ax,
        x,
        y,
        color,
        lw + 0.55,
        alpha=0.14,
        focus_width=0.22,
        edge_scale=0.74,
        center_boost=0.58,
        ripple_amp=0.04,
        zorder=3,
    )
    variable_width_line(
        ax,
        x,
        y,
        color,
        lw,
        alpha=0.98,
        focus_width=0.20,
        edge_scale=0.72,
        center_boost=0.62,
        ripple_amp=0.05,
        zorder=4,
    )

# Gradient fill under target Gaussian (subtle)
gradient_fill(ax, x_plot, f_gaussian, QRISP_NAVY, n_bands=50, alpha_base=0.5)
# Target Gaussian on top
subtle_glow_line(ax, x_plot, f_gaussian, "#05034B", lw=3)

i = 0
# Polynomial curves with gradient fill (low → high degree, back-to-front)
for deg, col, lw in zip(degrees, palette, lws):
    if i == 4:
        continue
    i+=1
    y = chebval(x_plot, cheb_coeffs[:deg + 1])
    gradient_fill(ax, x_plot, y, QRISP_PURPLE, n_bands=40, alpha_base=0.2)
    subtle_glow_line(ax, x_plot, y, col, lw=lw)




# ── Axis styling (match other panes) ───────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.26, 1.05)
ax.set_xticks(np.arange(-0.5, 0.51, 0.1))
ax.set_yticks(np.arange(-0.2, 1.01, 0.2))

plt.tight_layout(pad=0.1)
ax.grid(True, which="major", color=("#C3C3C3", 0.006), linewidth=0.4)
#ax.axvline(0.0, color=("black", 0.7), linewidth=1.5, zorder=5)
ax.axhline(0.0, color=("black", 0.7), linewidth=1.5, zorder=2)
plt.savefig("signal_processing.png", dpi=220, facecolor="white")
plt.show()
print("saved → signal_processing.png")