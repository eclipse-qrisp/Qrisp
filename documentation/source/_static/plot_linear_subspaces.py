import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from itertools import combinations


def _plane_normal_and_d(points):
    """Compute plane normal and offset d from 3+ coplanar points (ax+by+cz=d)."""
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)
    d = np.dot(n, points[0])
    return n, d


def _intersect_line_plane(p0, p1, n, d):
    """Return t parameter where line p0->p1 intersects plane n·x=d, or None."""
    denom = np.dot(n, p1 - p0)
    if abs(denom) < 1e-12:
        return None
    t = (d - np.dot(n, p0)) / denom
    if 1e-9 < t < 1 - 1e-9:  # strictly interior to segment
        return t
    return None


def _point_in_polygon_3d(point, poly_pts):
    """Check if a point lies inside a convex 3D polygon (assumed coplanar)."""
    n = len(poly_pts)
    v1 = poly_pts[1] - poly_pts[0]
    v2 = poly_pts[2] - poly_pts[0]
    normal = np.cross(v1, v2)
    for i in range(n):
        edge = poly_pts[(i + 1) % n] - poly_pts[i]
        to_point = point - poly_pts[i]
        cross = np.cross(edge, to_point)
        if np.dot(cross, normal) < -1e-9:
            return False
    return True


def _split_line_by_planes(p0, p1, plane_configs):
    """Split a line segment at its intersections with the infinite planes.
    We split at the infinite plane (not just the polygon) because depth sorting
    requires segments to be on one side of each plane."""
    t_values = [0.0, 1.0]
    for cfg in plane_configs:
        pts = cfg['points']
        n, d = _plane_normal_and_d(pts)
        t = _intersect_line_plane(p0, p1, n, d)
        if t is not None:
            t_values.append(t)
    t_values = sorted(set(t_values))
    return t_values


def _split_polygon_by_plane(poly_pts, n, d):
    """Split a polygon into two halves along plane n·x=d. Returns (pos, neg) point lists."""
    signs = np.dot(poly_pts, n) - d
    pos_pts, neg_pts = [], []
    n_pts = len(poly_pts)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        pi, pj = poly_pts[i], poly_pts[j]
        si, sj = signs[i], signs[j]
        if si >= -1e-9:
            pos_pts.append(pi)
        if si <= 1e-9:
            neg_pts.append(pi)
        # If edge crosses the plane, add intersection to both sides
        if (si > 1e-9 and sj < -1e-9) or (si < -1e-9 and sj > 1e-9):
            denom = np.dot(n, pj - pi)
            if abs(denom) > 1e-12:
                t = (d - np.dot(n, pi)) / denom
                ix = pi + np.clip(t, 0, 1) * (pj - pi)
                pos_pts.append(ix)
                neg_pts.append(ix)
    return pos_pts, neg_pts


def _split_planes_for_depth_sorting(plane_configs):
    """Split every plane by every other plane so matplotlib can depth-sort the pieces."""
    # Start with each plane as a list of sub-polygons
    all_polys = [[(cfg['points'], cfg['color'], cfg['alpha'])] for cfg in plane_configs]

    for i in range(len(plane_configs)):
        for j in range(len(plane_configs)):
            if i == j:
                continue
            n, d = _plane_normal_and_d(plane_configs[j]['points'])
            new_polys = []
            for (pts, color, alpha) in all_polys[i]:
                pos, neg = _split_polygon_by_plane(pts, n, d)
                if len(pos) >= 3:
                    new_polys.append((np.array(pos), color, alpha))
                if len(neg) >= 3:
                    new_polys.append((np.array(neg), color, alpha))
                if len(pos) < 3 and len(neg) < 3:
                    new_polys.append((pts, color, alpha))  # keep original
            all_polys[i] = new_polys

    # Flatten
    result = []
    for polys in all_polys:
        for (pts, color, alpha) in polys:
            result.append({'points': pts, 'color': color, 'alpha': alpha})
    return result

def plot_linear_subspaces(plane_configs=None, line_configs=None, 
                         view_angle=(20, 45), figsize=(12, 10)):
    """
    Plot intersecting planes and lines to visualize linear algebra concepts.

    Parameters:
    -----------
    plane_configs : list of dict
        Each dict should contain:
        - 'points': 4 corner points of the plane as (4, 3) array
        - 'color': color string or RGB tuple
        - 'alpha': transparency value (0-1)

    line_configs : list of dict
        Each dict should contain:
        - 'points': 2 endpoints of the line as (2, 3) array
        - 'color': color string or RGB tuple
        - 'linewidth': width of the line

    view_angle : tuple
        (elevation, azimuth) for 3D view

    figsize : tuple
        Figure size
    """

    # Default configurations if none provided
    if plane_configs is None:
        # Qrisp brand colors
        qrisp_navy = (32/255, 48/255, 111/255)    # RGB(32, 48, 111)
        qrisp_purple = (99/255, 102/255, 241/255)  # RGB(99, 102, 241)

        plane_configs = [
            {'points': np.array([[-2, -2, 0], [3, -2, 0], [3, 3, 0], [-2, 3, 0]]),
             'color': 'gray', 'alpha': 0.4},
            # Green plane 1
            # Navy blue plane (tilted off-axis)
            {'points': np.array([[-1, -1.5, -1.5], [2, -2.5, 1.5], [2, 1.5, 2.5], [-1, 2.5, -0.5]]),
             'color': qrisp_navy, 'alpha': 0.55},
            # Purple plane
            {'points': np.array([[-1.5, -1.5, -1], [-1.5, -1.5, 3], [1.5, 2, 3], [1.5, 2, -1]]),
             'color': qrisp_purple, 'alpha': 0.45},
        ]

    if line_configs is None:
        line_configs = [
            # Black lines (axes or intersection lines)
            {'points': np.array([[-2.5, 0, 0], [3, 0, 0]]),
             'color': 'black', 'linewidth': 2},
            {'points': np.array([[0, -2.5, 0], [0, 3, 0]]),
             'color': 'black', 'linewidth': 2},
            {'points': np.array([[0, 0, -2], [0, 0, 3.5]]),
             'color': 'black', 'linewidth': 2},
            # Intersection line in white for contrast
            {'points': np.array([[0, -2.5, -2], [0, 3, 3.5]]),
             'color': 'white', 'linewidth': 4},
        ]

    # Split planes along each other's intersection for correct depth sorting
    orig_plane_configs = list(plane_configs)
    plane_configs = _split_planes_for_depth_sorting(plane_configs)

    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Collect ALL faces (plane sub-polygons + line ribbon segments) into a
    # single Poly3DCollection. Matplotlib only depth-sorts faces correctly
    # WITHIN a single collection.
    all_faces = []
    all_facecolors = []
    all_edgecolors = []
    all_alphas = []

    # Add plane faces
    for config in plane_configs:
        points = config['points']
        color = config['color']
        alpha = config['alpha']
        all_faces.append(points)
        all_facecolors.append(color)
        all_edgecolors.append('none')
        all_alphas.append(alpha)

    # Add line ribbon faces — split at plane intersections
    for config in line_configs:
        points = config['points']
        color = config['color']
        linewidth = config.get('linewidth', 2)

        p0, p1 = points[0], points[1]
        direction = p1 - p0
        t_splits = _split_line_by_planes(p0, p1, orig_plane_configs)

        # Build a perpendicular offset for ribbon width
        arbitrary = np.array([0, 0, 1]) if abs(direction[2] / np.linalg.norm(direction)) < 0.9 else np.array([1, 0, 0])
        perp = np.cross(direction, arbitrary)
        perp = perp / np.linalg.norm(perp) * 0.012 * linewidth

        for k in range(len(t_splits) - 1):
            s0 = p0 + t_splits[k] * direction
            s1 = p0 + t_splits[k + 1] * direction
            quad = np.array([s0 - perp, s0 + perp, s1 + perp, s1 - perp])
            all_faces.append(quad)
            all_facecolors.append(color)
            all_edgecolors.append(color)
            all_alphas.append(1.0)

    # Add everything as one collection so faces are depth-sorted together
    combined = Poly3DCollection(all_faces, linewidths=0)
    combined.set_facecolors([(*mcolors.to_rgba(c)[:3], a)
                             for c, a in zip(all_facecolors, all_alphas)])
    combined.set_edgecolors([(*mcolors.to_rgba(c)[:3], a)
                             for c, a in zip(all_edgecolors, all_alphas)])
    ax.add_collection3d(combined)

    # Set equal aspect ratio and limits
    ax.set_xlim([-2.5, 3])
    ax.set_ylim([-2.5, 3])
    ax.set_zlim([-2, 3.5])

    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Hide everything axis-related
    ax.axis('off')

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example 1: Default configuration
    # fig, ax = plot_linear_subspaces()
    # plt.show()

    # Example 2: Custom configuration
    custom_planes = [
        {'points': np.array([[-2, -2, 0], [3, -2, 0], [3, 3, 0], [-2, 3, 0]]),
         'color': 'red', 'alpha': 0.3},
        {'points': np.array([[-1, -2, -1], [2, -2, 2], [2, 2, 2], [-1, 2, -1]]),
         'color': 'blue', 'alpha': 0.5},
    ]

    custom_lines = [
        {'points': np.array([[0, 0, -2], [0, 0, 3]]),
         'color': 'purple', 'linewidth': 3},
    ]

    qrisp_navy = (32/255, 48/255, 111/255)    # RGB(32, 48, 111)
    qrisp_purple = (99/255, 102/255, 241/255)  # RGB(99, 102, 241)

    coord_system_scaling = 0.88
    custom_planes = [
        {'points': coord_system_scaling*np.array([[-2, -2, 0], [3, -2, 0], [3, 3, 0], [-2, 3, 0]]),
            'color': 'gray', 'alpha': 0.4},
        # Green plane 1
        # Navy blue plane (tilted off-axis)
        {'points': 0.85*np.array([[-1, -1.5, -1.5], [2, -2.5, 1.5], [2, 1.5, 2.5], [-1, 2.5, -0.5]]),
            'color': qrisp_navy, 'alpha': 0.88},
        # Purple plane
        # {'points': np.array([[-1.5, -1.5, -1], [-1.5, -1.5, 3], [1.5, 2, 3], [1.5, 2, -1]]),
        #     'color': qrisp_purple, 'alpha': 0.7},
    ]

    custom_lines = [
        # Black lines (axes or intersection lines)
        {'points': coord_system_scaling*np.array([[-2.5, 0, 0], [3.5, 0, 0]]),
            'color': 'gray', 'linewidth': 1},
        {'points': coord_system_scaling*np.array([[0, -3.5, 0], [0, 4.5, 0]]),
            'color': 'gray', 'linewidth': 1},
        {'points': 0.8*coord_system_scaling*np.array([[0, 0, -2], [0, 0, 3.5]]),
            'color': 'gray', 'linewidth': 1},
        # Intersection line
        # {'points': 0.8*np.array([[0, -2.5, -2], [0, 3, 3.5]]),
        #     'color': 'red', 'linewidth': 4},
    #     {'points': 0.4*np.array([[3, 3, -2], [-2.5, -2.5, 5]]),
    # 'color': "black", 'linewidth': 4},
    #         {'points': 0.6*np.array([[3, 3, -2], [-2.5, -2.5, 5]]),
    # 'color': "black", 'linewidth': 4},
                {'points': 0.6*np.array([[4, 4, -3], [-2, -2, 4]]),
    'color': "black", 'linewidth': 4},
        
    ]

    # Uncomment to see custom example:
    fig, ax = plot_linear_subspaces(plane_configs=custom_planes, 
                                   line_configs=custom_lines,
                                   view_angle=(11, -104))
    plt.show()
