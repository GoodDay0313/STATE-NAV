"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np
import os
import sys
from pathlib import Path

import torch


def get_project_root():
    """
    Find the project root directory by looking for marker files.

    Returns:
        Path: Path to the project root directory (where setup.py, package.xml, or CMakeLists.txt exists)
    """
    # Start from this file's directory
    current = Path(__file__).resolve()

    # Walk up the directory tree looking for project root markers
    for parent in [current] + list(current.parents):
        # Check for common project root markers
        if (parent / "setup.py").exists() or \
           (parent / "package.xml").exists() or \
           (parent / "CMakeLists.txt").exists():
            return parent

    # Fallback: if no marker found, assume project root is 3 levels up from this file
    # (statenav_global/utility/utils.py -> statenav_global -> state_nav)
    return current.parents[2]


def get_source_inference_dir():
    """
    Return the path to statenav_global/inference/ in the SOURCE tree.

    When running from an installed colcon package, __file__ points to the
    install tree which lacks heavy assets (model weights, checkpoints/).
    This helper detects that case and resolves to the source tree instead.
    """
    # Direct path: statenav_global/inference/ relative to this file
    # (this file is statenav_global/utility/utils.py)
    local_inference = Path(__file__).resolve().parent.parent / "inference"

    # If the assets exist locally, we're running from source — use it directly
    if (local_inference / "checkpoints").exists():
        return str(local_inference)

    # We're in the install tree. Resolve to source via colcon workspace layout:
    #   install/<pkg>/lib/python3.x/site-packages/statenav_global/utility/utils.py
    #   -> <ws_root>/src/state_nav/statenav_global/inference/
    parts = Path(__file__).resolve().parts
    try:
        idx = parts.index("install")
        ws_root = Path(*parts[:idx])
        src_inference = ws_root / "src" / "state_nav" / "statenav_global" / "inference"
        if src_inference.exists():
            return str(src_inference)
    except ValueError:
        pass

    # Fallback: return the local path (caller will get FileNotFoundError with a clear path)
    return str(local_inference)


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    print(f"Set random seed to {seed} in numpy and torch.")
    return rng

















# Math

def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))





# Geometry

# Module-level utility functions that can be imported directly
def get_ray(start, end):
    """Calculate ray origin and direction from start to end point."""
    orig = [start.x, start.y]
    direc = [end.x - start.x, end.y - start.y]
    return orig, direc


def get_dist(start, end):
    """Calculate Euclidean distance between two nodes."""
    return math.hypot(end.x - start.x, end.y - start.y)


def is_intersect_rec(start, end, o, d, a, b):
    v1 = [o[0] - a[0], o[1] - a[1]]
    v2 = [b[0] - a[0], b[1] - a[1]]
    v3 = [-d[1], d[0]]

    div = np.dot(v2, v3)

    if div == 0:
        return False

    t1 = np.linalg.norm(np.cross(v2, v1)) / div
    t2 = np.dot(v1, v3) / div

    if t1 >= 0 and 0 <= t2 <= 1:
        shot = BasicNode((o[0] + t1 * d[0], o[1] + t1 * d[1]))
        dist_obs = get_dist(start, shot)
        dist_seg = get_dist(start, end)
        if dist_obs <= dist_seg:
            return True

    return False

def is_intersect_circle(o, d, a, r, delta = 0):
    d2 = np.dot(d, d)

    if d2 == 0:
        return False

    t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

    if 0 <= t <= 1:
        shot = BasicNode((o[0] + t * d[0], o[1] + t * d[1]))
        if get_dist(shot, BasicNode(a)) <= r + delta:
            return True

    return False



def is_inside_circle(node, circle, delta=None):

    """
    
    Args:
        node: BasicNode object with x and y attributes
        circle: Tuple, containing (x, y, r) representing circle
        delta: Tolerance value for point-in-circle check (default: None)
    
    Returns:
        bool: True if point is inside circle, False otherwise
    """
    
    if delta is None:
        delta = 0.00

    x, y, r = circle
    if math.hypot(node.x - x, node.y - y) <= r + delta:
        return True
    return False


def is_inside_circles(node, circles, delta=None):
    if delta is None:
        delta = 0.00

    for circle in circles:
        if is_inside_circle(node, circle, delta):
            return True
    return False


class BasicNode:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Utils:
    def __init__(self):
        self.delta = 0.01#0.5
        self.obs_circle = []
        self.obs_rectangle = []
        self.obs_boundary = []

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)
            
        delta = 0.01
        for (ox, oy, w, h) in self.obs_boundary:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list


    def is_collision(self, start, end, delta=None):
        
        if self.is_inside_obs(start, delta) or self.is_inside_obs(end, delta):
            return True

        o, d = get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, (x, y), r):
                return True

        return False
    
                

    def is_inside_obs(self, node, delta=None):
        
        if delta is None:
            delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x) <= w \
                    and 0 <= node.y - (y) <= h:
                return True

        return False


# Expose free functions as class-level utilities for convenient access
Utils.set_random_seed = set_random_seed
Utils.wrap_to_pi = wrap_to_pi
Utils.get_ray = get_ray
Utils.get_dist = get_dist
Utils.is_intersect_rec = is_intersect_rec
Utils.is_intersect_circle = is_intersect_circle
Utils.is_inside_circle = is_inside_circle
