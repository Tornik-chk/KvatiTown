"""
Collision Checker — circles and rectangles (axis-aligned and rotated).
Algorithms used:
  - Circle vs Circle      : distance between centres vs sum of radii
  - Circle vs Rectangle   : SAT with circle-special axis (closest point on rect to circle centre)
  - Rectangle vs Rectangle: Separating Axis Theorem (SAT) over all 4 candidate axes
"""

from __future__ import annotations
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class CircleObj:
    def __init__(self, name: str, cx: float, cy: float, r: float):
        self.name = name
        self.cx   = cx
        self.cy   = cy
        self.r    = r

class RectObj:
    def __init__(self, name: str, x: float, y: float, w: float, h: float, angle_deg: float):
        """
        x, y   : left-top corner (before rotation)
        w, h   : width, height
        angle  : counterclockwise rotation in degrees around (x, y)
        """
        self.name      = name
        self.x         = x
        self.y         = y
        self.w         = w
        self.h         = h
        self.angle_deg = angle_deg
        self.corners   = self._compute_corners()

    def _compute_corners(self) -> np.ndarray:
        """Return 4x2 array of corner coordinates after rotation."""
        ox, oy = self.x, self.y
        rad = math.radians(self.angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Corners relative to top-left before rotation
        local = np.array([
            [0,      0     ],
            [self.w, 0     ],
            [self.w, self.h],
            [0,      self.h],
        ], dtype=float)

        # Rotate around origin then translate
        rot = np.array([[cos_a, -sin_a],
                        [sin_a,  cos_a]])
        corners = (rot @ local.T).T + np.array([ox, oy])
        return corners

    def axes(self) -> List[np.ndarray]:
        """Return the 2 unique SAT axes (edge normals) for this rectangle."""
        result = []
        for i in range(2):          # only 2 unique normals for a rectangle
            edge = self.corners[i+1] - self.corners[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                result.append(normal / norm)
        return result


# ---------------------------------------------------------------------------
# Collision checkers
# ---------------------------------------------------------------------------

def _project_polygon(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project all corners onto axis, return (min, max)."""
    dots = corners @ axis
    return dots.min(), dots.max()


def _intervals_overlap(a_min, a_max, b_min, b_max) -> bool:
    return a_max >= b_min and b_max >= a_min


def check_circle_circle(a: CircleObj, b: CircleObj) -> bool:
    """Collide if distance between centres <= sum of radii (includes containment)."""
    dx = a.cx - b.cx
    dy = a.cy - b.cy
    dist_sq = dx*dx + dy*dy
    r_sum = a.r + b.r
    return dist_sq <= r_sum * r_sum


def check_circle_rect(c: CircleObj, r: RectObj) -> bool:
    """
    SAT for circle vs convex polygon.
    Axes tested: the rectangle's 2 edge normals + the axis from each corner to circle centre.
    The critical extra axis is the one from the closest corner to the circle centre —
    this handles the corner case (literally) that pure edge-normal SAT misses.
    """
    centre = np.array([c.cx, c.cy])
    corners = r.corners

    # Rectangle edge-normal axes
    axes = r.axes()

    # Find closest corner to circle centre and add that axis
    dists = np.linalg.norm(corners - centre, axis=1)
    closest_corner = corners[np.argmin(dists)]
    corner_axis = centre - closest_corner
    norm = np.linalg.norm(corner_axis)
    if norm > 1e-12:
        axes.append(corner_axis / norm)

    for axis in axes:
        # Project rectangle
        r_min, r_max = _project_polygon(corners, axis)
        # Project circle: centre ± radius
        c_proj = centre @ axis
        c_min, c_max = c_proj - c.r, c_proj + c.r

        if not _intervals_overlap(r_min, r_max, c_min, c_max):
            return False   # separating axis found

    return True


def check_rect_rect(a: RectObj, b: RectObj) -> bool:
    """
    SAT for two convex polygons (rectangles).
    Test all 4 axes: 2 from each rectangle's edge normals.
    """
    axes = a.axes() + b.axes()

    for axis in axes:
        a_min, a_max = _project_polygon(a.corners, axis)
        b_min, b_max = _project_polygon(b.corners, axis)
        if not _intervals_overlap(a_min, a_max, b_min, b_max):
            return False

    return True


def check_collision(obj_a, obj_b) -> bool:
    if isinstance(obj_a, CircleObj) and isinstance(obj_b, CircleObj):
        return check_circle_circle(obj_a, obj_b)
    if isinstance(obj_a, CircleObj) and isinstance(obj_b, RectObj):
        return check_circle_rect(obj_a, obj_b)
    if isinstance(obj_a, RectObj) and isinstance(obj_b, CircleObj):
        return check_circle_rect(obj_b, obj_a)
    if isinstance(obj_a, RectObj) and isinstance(obj_b, RectObj):
        return check_rect_rect(obj_a, obj_b)
    raise TypeError(f"Unknown object types: {type(obj_a)}, {type(obj_b)}")


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------

def parse_file(path: str) -> List:
    objects = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            tokens = line.split()
            kind = tokens[0].lower()
            if kind == 'circle':
                name = tokens[1]
                cx, cy, r = float(tokens[2]), float(tokens[3]), float(tokens[4])
                objects.append(CircleObj(name, cx, cy, r))
            elif kind == 'rectangle':
                name = tokens[1]
                x, y    = float(tokens[2]), float(tokens[3])
                w, h    = float(tokens[4]), float(tokens[5])
                angle   = float(tokens[6])
                objects.append(RectObj(name, x, y, w, h, angle))
            else:
                print(f"Warning: unknown object type '{kind}', skipping.")
    return objects


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_objects(objects: List, colliding_names: set):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_title('Collision Checker', fontsize=13)

    for obj in objects:
        hit   = obj.name in colliding_names
        color = '#e74c3c' if hit else '#2ecc71'
        edge  = '#922b21' if hit else '#1a8a4a'
        alpha = 0.55

        if isinstance(obj, CircleObj):
            patch = Circle((obj.cx, obj.cy), obj.r,
                           facecolor=color, edgecolor=edge,
                           linewidth=1.8, alpha=alpha)
            ax.add_patch(patch)
            ax.text(obj.cx, obj.cy, obj.name,
                    ha='center', va='center', fontsize=8, fontweight='bold')

        elif isinstance(obj, RectObj):
            patch = Polygon(obj.corners, closed=True,
                            facecolor=color, edgecolor=edge,
                            linewidth=1.8, alpha=alpha)
            ax.add_patch(patch)
            cx = obj.corners[:, 0].mean()
            cy = obj.corners[:, 1].mean()
            ax.text(cx, cy, obj.name,
                    ha='center', va='center', fontsize=8, fontweight='bold')

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='#1a8a4a', label='No collision'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='#922b21', label='Collision'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    # Auto-scale axes with margin
    ax.autoscale_view()
    margin = 20
    xl = ax.get_xlim(); yl = ax.get_ylim()
    ax.set_xlim(xl[0] - margin, xl[1] + margin)
    ax.set_ylim(yl[0] - margin, yl[1] + margin)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python collision_checker.py <objects_file>")
        sys.exit(1)

    objects = parse_file(sys.argv[1])
    n = len(objects)
    print(f"Loaded {n} objects.\n")

    colliding_names: set = set()
    collisions: List[Tuple[str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if check_collision(objects[i], objects[j]):
                a_name = objects[i].name
                b_name = objects[j].name
                collisions.append((a_name, b_name))
                colliding_names.add(a_name)
                colliding_names.add(b_name)

    if collisions:
        print("Collisions detected:")
        for a, b in collisions:
            print(f"  {a} collides with {b}")
    else:
        print("No collisions detected.")

    draw_objects(objects, colliding_names)


if __name__ == '__main__':
    main()