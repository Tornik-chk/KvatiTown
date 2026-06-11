
import os
import sys
import math
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

class Circle:
    """Represents a 2D circle by its center and radius."""
    def __init__(self, name: str, cx: float, cy: float, radius: float):
        self.name   = name
        self.cx     = cx        # x-coordinate of center
        self.cy     = cy        # y-coordinate of center
        self.radius = radius

    def __repr__(self):
        return f"Circle({self.name}, center=({self.cx},{self.cy}), r={self.radius})"


class Rectangle:
    """
    Represents a 2D rectangle defined by its top-left corner, width, height,
    and a rotation angle (degrees, counterclockwise positive) applied around
    the top-left corner.
    """
    def __init__(self, name: str, tlx: float, tly: float,
                 width: float, height: float, angle_deg: float):
        self.name      = name
        self.tlx       = tlx        # top-left x (before rotation)
        self.tly       = tly        # top-left y (before rotation)
        self.width     = width
        self.height    = height
        self.angle_deg = angle_deg  # counterclockwise rotation in degrees

    def get_corners(self):
        """
        Compute the four rotated corner coordinates in world space.

        The rectangle is defined in local space as:
            TL=(0,0)  TR=(w,0)  BR=(w,h)  BL=(0,h)

        Each corner is rotated by `angle_deg` around the top-left corner (origin),
        then translated by (tlx, tly).

        Returns a list of four (x, y) tuples: [TL, TR, BR, BL]
        """
        angle_rad = math.radians(self.angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        w, h  = self.width, self.height

        # Local corner offsets (before rotation)
        local_corners = [
            (0,  0),    # top-left
            (w,  0),    # top-right
            (w,  h),    # bottom-right
            (0,  h),    # bottom-left
        ]

        world_corners = []
        for (lx, ly) in local_corners:
            # Rotate around origin (top-left = pivot)
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            # Translate to world space
            world_corners.append((self.tlx + rx, self.tly + ry))

        return world_corners

    def get_axes(self):
        """
        Return the two unique edge-normal axes for this rectangle (used by SAT).
        Because a rectangle has two pairs of parallel edges, only two axes are needed.
        Each axis is a unit vector perpendicular to one edge.
        """
        corners = self.get_corners()
        axes = []
        # We only need axes for edges 0→1 (horizontal in local space)
        # and 1→2 (vertical in local space) — the other two are parallel duplicates.
        for i in range(2):
            ax = corners[i + 1][0] - corners[i][0]
            ay = corners[i + 1][1] - corners[i][1]
            # Perpendicular (normal) to this edge
            nx, ny = -ay, ax
            # Normalize
            length = math.hypot(nx, ny)
            if length == 0:
                continue
            axes.append((nx / length, ny / length))
        return axes

    def __repr__(self):
        return (f"Rectangle({self.name}, tl=({self.tlx},{self.tly}), "
                f"w={self.width}, h={self.height}, angle={self.angle_deg}°)")


# =============================================================================
# SECTION 2 — PARSING
# =============================================================================

def parse_shapes(filepath: str):
    """
    Read shape definitions from a text file.

    Rules:
      - Lines starting with '!' are comments and are ignored.
      - Blank / whitespace-only lines are ignored.
      - Circle format:    circle    [name] [cx] [cy] [radius]
      - Rectangle format: rectangle [name] [tlx] [tly] [w] [h] [angle_deg]

    Returns a list of Circle and Rectangle objects.
    """
    shapes = []
    with open(filepath, 'r') as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Skip blank lines and comment lines
            if not line or line.startswith('!'):
                continue

            parts = line.split()
            kind  = parts[0].lower()

            try:
                if kind == 'circle':
                    # circle [name] [cx] [cy] [radius]
                    name, cx, cy, r = parts[1], float(parts[2]), float(parts[3]), float(parts[4])
                    shapes.append(Circle(name, cx, cy, r))

                elif kind == 'rectangle':
                    # rectangle [name] [tlx] [tly] [w] [h] [angle_deg]
                    name  = parts[1]
                    tlx   = float(parts[2])
                    tly   = float(parts[3])
                    w     = float(parts[4])
                    h     = float(parts[5])
                    angle = float(parts[6])
                    shapes.append(Rectangle(name, tlx, tly, w, h, angle))

                else:
                    print(f"[Warning] Line {line_num}: Unknown shape type '{kind}' — skipped.")

            except (IndexError, ValueError) as e:
                print(f"[Warning] Line {line_num}: Parse error ({e}) — skipped.")

    return shapes


# =============================================================================
# SECTION 3 — COLLISION MATH (from scratch, no libraries)
# =============================================================================

# ---------------------------------------------------------------------------
# 3a. Circle vs Circle
# ---------------------------------------------------------------------------

def circle_circle_collision(c1: Circle, c2: Circle) -> bool:
    """
    Two circles collide if the distance between their centers is less than or
    equal to the sum of their radii.

    Math:
        d = sqrt((x2-x1)² + (y2-y1)²)
        colliding  ⟺  d <= r1 + r2

    We compare d² to (r1+r2)² to avoid an unnecessary sqrt.
    """
    dx = c2.cx - c1.cx
    dy = c2.cy - c1.cy
    dist_sq          = dx * dx + dy * dy
    radii_sum_sq     = (c1.radius + c2.radius) ** 2
    return dist_sq <= radii_sum_sq


# ---------------------------------------------------------------------------
# 3b. Rectangle vs Rectangle — Separating Axis Theorem (SAT)
# ---------------------------------------------------------------------------

def project_polygon_onto_axis(corners, axis):
    """
    Project all corners of a convex polygon onto a given axis vector and return
    the [min, max] scalar interval of the projection.

    The projection of a point P onto unit axis A is the dot product: P · A.
    The projection of the entire polygon is the interval [min(P·A), max(P·A)]
    over all vertices P.
    """
    dots = [corner[0] * axis[0] + corner[1] * axis[1] for corner in corners]
    return min(dots), max(dots)


def intervals_overlap(min_a, max_a, min_b, max_b) -> bool:
    """
    Two 1-D intervals [min_a, max_a] and [min_b, max_b] overlap if and only if
    neither one is completely to the left of the other.

    Overlap  ⟺  max_a >= min_b  AND  max_b >= min_a
    """
    return max_a >= min_b and max_b >= min_a


def rect_rect_collision(r1: Rectangle, r2: Rectangle) -> bool:
    """
    Separating Axis Theorem (SAT) for two convex polygons (rectangles).

    THEORY:
    Two convex shapes do NOT collide if and only if there exists at least one
    axis (the "separating axis") along which the projections of the two shapes
    do not overlap.

    For two rectangles, we only need to test axes that are perpendicular to the
    edges of each rectangle.  A rectangle has 2 unique edge directions, so we
    test 2 + 2 = 4 axes in total.

    ALGORITHM:
        For each candidate axis A:
            1. Project both rectangles onto A → get intervals [min1,max1], [min2,max2]
            2. If the intervals do NOT overlap → shapes are SEPARATED → no collision
        If ALL axes show overlap → shapes are COLLIDING (no separating axis found)
    """
    corners1 = r1.get_corners()
    corners2 = r2.get_corners()

    # Gather all candidate separating axes (4 total for two rectangles)
    axes = r1.get_axes() + r2.get_axes()

    for axis in axes:
        min1, max1 = project_polygon_onto_axis(corners1, axis)
        min2, max2 = project_polygon_onto_axis(corners2, axis)

        if not intervals_overlap(min1, max1, min2, max2):
            # A separating axis was found → definitely no collision
            return False

    # No separating axis found on any of the 4 axes → shapes overlap
    return True


# ---------------------------------------------------------------------------
# 3c. Circle vs Rectangle — Closest Point + SAT hybrid
# ---------------------------------------------------------------------------

def closest_point_on_segment(px, py, ax, ay, bx, by):
    """
    Find the point on line segment AB that is closest to point P.

    Method: project P onto the line through A and B, then clamp the parameter
    t to [0,1] so the result stays on the segment.

        t = dot(AP, AB) / dot(AB, AB)
        t_clamped = clamp(t, 0, 1)
        closest = A + t_clamped * AB
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        # Degenerate segment (A == B): closest point is A itself
        return ax, ay

    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))   # clamp to segment

    return ax + t * abx, ay + t * aby


def circle_rect_collision(c: Circle, r: Rectangle) -> bool:
    """
    Detect collision between a circle and a (possibly rotated) rectangle.

    STRATEGY — transform to rectangle's local coordinate system:

    Instead of rotating the rectangle's corners into world space and then doing
    complex math in world space, we do the opposite: we transform the circle's
    center into the rectangle's LOCAL (unrotated) coordinate system where the
    rectangle is always axis-aligned.  This makes the closest-point clamp
    trivial.

    Steps:
      1. Translate the circle center by subtracting the rectangle's top-left pivot.
      2. Rotate the translated center by NEGATIVE angle_deg (undo the rectangle's
         rotation) to enter local space.
      3. In local space, the rectangle occupies [0, width] × [0, height].
         Clamp the transformed circle center to that box to find the closest point.
      4. Compute the distance from the circle center to the closest point.
      5. If distance <= radius → collision.

    EDGE CASES:
      - If the circle center is INSIDE the rectangle, the closest point IS the
        center itself, distance = 0 → always collides (correct).
    """
    angle_rad = math.radians(r.angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Step 1: Translate so the rectangle's top-left becomes the origin
    tx = c.cx - r.tlx
    ty = c.cy - r.tly

    # Step 2: Rotate by NEGATIVE angle to enter the rectangle's local frame
    # (Rotating by -θ undoes the rectangle's rotation)
    local_x = tx * cos_a + ty * sin_a
    local_y = -tx * sin_a + ty * cos_a

    # Step 3: Find the closest point inside the local-space AABB [0,w] × [0,h]
    closest_x = max(0.0, min(local_x, r.width))
    closest_y = max(0.0, min(local_y, r.height))

    # Step 4: Distance from circle center (in local space) to closest point
    dx = local_x - closest_x
    dy = local_y - closest_y
    dist_sq = dx * dx + dy * dy

    # Step 5: Compare to radius²
    return dist_sq <= c.radius * c.radius


# ---------------------------------------------------------------------------
# 3d. Dispatcher — check any two shapes
# ---------------------------------------------------------------------------

def check_collision(shape_a, shape_b) -> bool:
    """
    Route to the correct collision function based on the types of both shapes.
    """
    if isinstance(shape_a, Circle) and isinstance(shape_b, Circle):
        return circle_circle_collision(shape_a, shape_b)

    elif isinstance(shape_a, Rectangle) and isinstance(shape_b, Rectangle):
        return rect_rect_collision(shape_a, shape_b)

    elif isinstance(shape_a, Circle) and isinstance(shape_b, Rectangle):
        return circle_rect_collision(shape_a, shape_b)

    elif isinstance(shape_a, Rectangle) and isinstance(shape_b, Circle):
        return circle_rect_collision(shape_b, shape_a)   # symmetric

    else:
        raise TypeError(f"Unsupported shape types: {type(shape_a)}, {type(shape_b)}")


# =============================================================================
# SECTION 4 — COLLISION DETECTION DRIVER
# =============================================================================

def find_all_collisions(shapes):
    """
    Check every unique pair of shapes (O(n²) brute force — fine for small n).
    Returns:
      - colliding_pairs : list of (name_a, name_b) strings
      - colliding_names : set of shape names involved in at least one collision
    """
    colliding_pairs = []
    colliding_names = set()

    n = len(shapes)
    for i in range(n):
        for j in range(i + 1, n):    # i < j ensures no duplicates
            if check_collision(shapes[i], shapes[j]):
                colliding_pairs.append((shapes[i].name, shapes[j].name))
                colliding_names.add(shapes[i].name)
                colliding_names.add(shapes[j].name)

    return colliding_pairs, colliding_names


# =============================================================================
# SECTION 5 — VISUALIZATION (matplotlib only)
# =============================================================================

# Color palette
COLOR_COLLIDING   = '#E53935'   # vivid red
COLOR_CIRCLE_OK   = '#1E88E5'   # strong blue
COLOR_RECT_OK     = '#FDD835'   # golden yellow
COLOR_EDGE        = '#212121'   # near-black outline
COLOR_TEXT        = '#FAFAFA'   # off-white text on shapes
COLOR_TEXT_DARK   = '#212121'   # dark text for yellow rects
BG_COLOR          = '#1A1A2E'   # dark navy background
GRID_COLOR        = '#2A2A4A'   # subtle grid


def draw_circle_patch(ax, circle: Circle, color: str):
    """Draw a filled circle using matplotlib's Circle patch."""
    patch = plt.Circle(
        (circle.cx, circle.cy),
        circle.radius,
        color=color,
        ec=COLOR_EDGE,
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(patch)

    # Label: place text at center
    text_color = COLOR_TEXT if color == COLOR_COLLIDING else COLOR_TEXT
    ax.text(
        circle.cx, circle.cy, circle.name,
        ha='center', va='center',
        fontsize=7, fontweight='bold',
        color=text_color,
        zorder=4
    )


def draw_rectangle_patch(ax, rect: Rectangle, color: str):
    """
    Draw a filled, rotated rectangle using matplotlib's Polygon patch
    built from the pre-computed world-space corners.
    """
    corners = rect.get_corners()   # list of 4 (x,y) tuples
    patch = MplPolygon(
        corners,
        closed=True,
        color=color,
        ec=COLOR_EDGE,
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(patch)

    # Label: centroid = average of all 4 corners
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    text_color = COLOR_TEXT_DARK if color == COLOR_RECT_OK else COLOR_TEXT
    ax.text(
        cx, cy, rect.name,
        ha='center', va='center',
        fontsize=7, fontweight='bold',
        color=text_color,
        zorder=4
    )


def visualize(shapes, colliding_names: set, colliding_pairs):
    """
    Render all shapes with matplotlib.

    Color rules:
      - Any shape involved in a collision → red
      - Non-colliding circles             → blue
      - Non-colliding rectangles          → yellow
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw each shape
    for shape in shapes:
        is_colliding = shape.name in colliding_names

        if isinstance(shape, Circle):
            color = COLOR_COLLIDING if is_colliding else COLOR_CIRCLE_OK
            draw_circle_patch(ax, shape, color)

        elif isinstance(shape, Rectangle):
            color = COLOR_COLLIDING if is_colliding else COLOR_RECT_OK
            draw_rectangle_patch(ax, shape, color)

    # Build a legend
    legend_handles = [
        mpatches.Patch(color=COLOR_COLLIDING,  label='Colliding shape'),
        mpatches.Patch(color=COLOR_CIRCLE_OK,  label='Circle (no collision)'),
        mpatches.Patch(color=COLOR_RECT_OK,    label='Rectangle (no collision)'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=9,
        facecolor='#2A2A4A',
        edgecolor='#555',
        labelcolor='white'
    )

    # Annotate detected collisions in a text box
    if colliding_pairs:
        pair_text = '\n'.join(f"{a} ↔ {b}" for a, b in colliding_pairs)
        ax.text(
            0.01, 0.99, f"Collisions detected:\n{pair_text}",
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2A2A4A',
                      edgecolor='#E53935', alpha=0.9),
            zorder=5
        )

    # Axis formatting
    ax.set_xlim(-50, 850)
    ax.set_ylim(-50, 850)
    ax.set_aspect('equal')
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle='--')
    ax.tick_params(colors='#888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    ax.set_title(
        '2D Collision Checker  •  SAT + Distance Formula',
        fontsize=14, fontweight='bold',
        color='white', pad=14
    )
    ax.set_xlabel('X', color='#888', fontsize=9)
    ax.set_ylabel('Y', color='#888', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collision_output.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("\n[Visualization saved to collision_output.png]")
    plt.show()


# =============================================================================
# SECTION 6 — MAIN ENTRY POINT
# =============================================================================

def main():
    # Determine input file (command-line arg or default)
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'test2.txt'

    print(f"=== 2D Collision Checker ===")
    print(f"Reading shapes from: {filepath}\n")

    # --- Parse ---
    shapes = parse_shapes(filepath)
    print(f"Loaded {len(shapes)} shapes:")
    for s in shapes:
        print(f"  {s}")

    # --- Detect collisions ---
    print("\n--- Collision Results ---")
    colliding_pairs, colliding_names = find_all_collisions(shapes)

    if not colliding_pairs:
        print("No collisions detected.")
    else:
        for name_a, name_b in colliding_pairs:
            print(f"{name_a} collides with {name_b}")

    print(f"\nTotal collisions: {len(colliding_pairs)}")
    print(f"Shapes involved:  {sorted(colliding_names)}")

    # --- Visualize ---
    visualize(shapes, colliding_names, colliding_pairs)


if __name__ == '__main__':
    main()