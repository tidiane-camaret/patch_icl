"""
Tubular / vessel morphology via SPACE COLONIZATION (build-time only; expensive).

Pipeline (spec ss10.3):
  1. grow_tree: space-colonization centerline -> branching node graph. Tortuosity
     perturbs growth direction; branching_density sets attractor density + reach.
  2. caliber per node TAPERS toward the leaves (Murray-law-ish r_parent^3 = sum r_child^3);
     deepest generation hits `thinness` min radius.
  3. rasterize each (node, parent) edge as a capsule (vectorized point-to-segment
     distance threshold) -> uint8 mask. No pixel loops.

Final foreground area is fixed later by area.enforce_area_fraction, but calibers
are scaled here toward region_size so the tree starts in the right ballpark.
"""

import numpy as np

from ..config import map_caliber_px, map_region_size


def grow_tree(image_size, params, rng, max_nodes=160):
    """Space-colonization tree. Returns (nodes [N,2] yx, parent [N] int).

    Kept deliberately SPARSE (few attractors, long steps, wide kill radius) so the
    result reads as a branching vessel, not a dense bush.
    """
    tortuosity = float(params.get("tortuosity", 0.5))
    branching_density = float(params.get("branching_density", 0.5))

    # Attractors: more branches with branching_density, but sparse overall.
    n_attractors = int(18 + 50 * branching_density)
    cx = image_size * rng.uniform(0.40, 0.60)
    cy = image_size * rng.uniform(0.30, 0.55)
    spread = image_size * (0.22 + 0.16 * branching_density)
    attractors = rng.normal([cy, cx], spread, size=(n_attractors, 2))
    attractors = np.clip(attractors, 1, image_size - 2)

    step = image_size * 0.06
    infl = step * (4.0 + 6.0 * branching_density)   # attraction radius
    kill = step * 2.4                               # consume radius (wide -> sparse)

    nodes = [np.array([float(image_size - 2), float(cx)])]  # root near bottom
    parent = [-1]
    last_dir = np.array([-1.0, 0.0])                # grow upward initially

    alive = np.ones(len(attractors), bool)
    for _ in range(max_nodes):
        if not alive.any():
            break
        pts = np.array(nodes)
        A = attractors[alive]
        # nearest node for each live attractor
        d = np.linalg.norm(A[:, None, :] - pts[None, :, :], axis=2)  # [nA, nNodes]
        in_reach = d.min(axis=1) <= infl
        if not in_reach.any():
            break
        nearest = d.argmin(axis=1)
        grown_any = False
        # each node grows toward the mean direction of attractors it owns
        for ni in np.unique(nearest[in_reach]):
            owners = A[in_reach & (nearest == ni)]
            if len(owners) == 0:
                continue
            node = pts[ni]
            direction = (owners - node).mean(axis=0)
            nrm = np.linalg.norm(direction)
            if nrm < 1e-6:
                continue
            direction = direction / nrm
            # tortuosity: blend in a random angular perturbation
            jitter = rng.normal(0, 0.6 * tortuosity, size=2)
            direction = direction + jitter
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            new = node + step * direction
            new = np.clip(new, 1, image_size - 2)
            nodes.append(new)
            parent.append(int(ni))
            grown_any = True
            last_dir = direction
        if not grown_any:
            break
        # consume attractors near any node
        pts = np.array(nodes)
        dist_all = np.linalg.norm(attractors[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        alive &= dist_all > kill
        if len(nodes) >= max_nodes:
            break

    return np.array(nodes), np.array(parent)


def _caliber(nodes, parent, leaf_radius, exponent=2.3, max_radius=None):
    """Radius per node: leaves -> leaf_radius; parents via r^p = sum child r^p.

    A gentler exponent (2.3 vs Murray's 3) and a cap keep the trunk from
    dominating a small frame while preserving taper toward the leaves.
    """
    if max_radius is None:
        max_radius = leaf_radius * 3.0
    n = len(nodes)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    radius = np.zeros(n)
    order = sorted(range(n), key=lambda i: -_depth(i, parent))  # deepest first
    for i in order:
        if not children[i]:
            radius[i] = leaf_radius
        else:
            r = (sum(radius[c] ** exponent for c in children[i])) ** (1 / exponent)
            radius[i] = min(r, max_radius)
    return radius


def _depth(i, parent, _cache={}):
    d, p = 0, parent[i]
    while p >= 0:
        d += 1
        p = parent[p]
    return d


def _rasterize(image_size, nodes, parent, radius):
    """OR of capsules (point-to-segment distance <= local radius) for every edge."""
    H = W = image_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    mask = np.zeros((H, W), bool)
    for i, p in enumerate(parent):
        if p < 0:
            continue
        p0, p1 = nodes[p], nodes[i]
        r = max(radius[i], 0.6)
        seg = p1 - p0
        L2 = float(seg @ seg)
        if L2 < 1e-6:
            dist = np.sqrt((yy - p0[0]) ** 2 + (xx - p0[1]) ** 2)
        else:
            t = ((yy - p0[0]) * seg[0] + (xx - p0[1]) * seg[1]) / L2
            t = np.clip(t, 0.0, 1.0)
            projy = p0[0] + t * seg[0]
            projx = p0[1] + t * seg[1]
            dist = np.sqrt((yy - projy) ** 2 + (xx - projx) ** 2)
        mask |= dist <= r
    return mask.astype(np.uint8)


def make_vessel_tree(image_size, params, rng):
    """Branching tubular structure. (mask uint8 [H,W], realized_meta)."""
    nodes, parent = grow_tree(image_size, params, rng)
    if len(nodes) < 2:
        # Degenerate growth -> fall back to a short stub so fg is never empty.
        nodes = np.array([[image_size * 0.5, image_size * 0.4],
                          [image_size * 0.5, image_size * 0.6]])
        parent = np.array([-1, 0])

    leaf_r = map_caliber_px(params.get("thinness", 0.5), image_size)
    radius = _caliber(nodes, parent, leaf_r)

    # Globally scale calibers to approach region_size, then rasterize once at the
    # scaled radii (spec ss10.3 "globally scale calibers"). Area ~ linear in radius
    # (centerline length fixed), so radius_scale ~ target_area / area_at_unit.
    mask0 = _rasterize(image_size, nodes, parent, radius)
    area0 = max(int(mask0.sum()), 1)
    target = map_region_size(params.get("region_size", 0.15)) * mask0.size
    scale = float(np.clip(target / area0, 0.4, 3.0))
    radius = np.maximum(radius * scale, leaf_r)       # never thinner than the leaf min
    mask = _rasterize(image_size, nodes, parent, radius)

    realized_area = float(mask.sum()) / mask.size
    return mask, {"morphology": "tubular", "n_nodes": int(len(nodes)),
                  "min_caliber_px": float(leaf_r), "realized_area": realized_area}
