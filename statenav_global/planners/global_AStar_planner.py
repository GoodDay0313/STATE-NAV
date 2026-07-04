import heapq
import math
import time
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

from . import BasePlanner
from statenav_global.utility import plotting


_PKG_ROOT = Path(__file__).resolve().parent.parent
cfg = OmegaConf.load(_PKG_ROOT / "configs/planning_config.yaml")

step_T = cfg.step_T


GridState = Tuple[int, int, int]


class GlobalAStar(BasePlanner):
    """Heading-aware A* over the global traversability map.

    A node is (row, col, theta_layer).  The edge cost follows the same
    interpretation as the existing RRT* cost:

        distance / cmd_v + heading_change / (cmd_w / step_T)
    """

    MOVES = (
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    )

    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        x_start: tuple,
        x_goal: tuple,
        heading_start: float,
        obstacle_cmd_v_threshold: float = 0.01,
        obstacle_cmd_w_threshold: float = 0.01,
        unknown_cmd_v: float = 0.3,
        unknown_cmd_w: float = 0.2,
        unknown_cost_multiplier: float = 1.5,
        unknown_neighbor_radius: int = 2,
    ) -> None:
        super().__init__(task_extent, rng)
        self.x_range = (task_extent[0], task_extent[1])
        self.y_range = (task_extent[2], task_extent[3])

        self.x_start = tuple(x_start)
        self.x_goal = tuple(x_goal)
        self.heading_start = heading_start
        self.global_map = None

        self.obstacle_cmd_v_threshold = obstacle_cmd_v_threshold
        self.obstacle_cmd_w_threshold = obstacle_cmd_w_threshold
        self.unknown_cmd_v = unknown_cmd_v
        self.unknown_cmd_w = unknown_cmd_w
        self.unknown_cost_multiplier = unknown_cost_multiplier
        self.unknown_neighbor_radius = max(1, int(unknown_neighbor_radius))

        self.path = []
        self.grid_path = []
        self.plan_cost = math.inf
        self.expanded_count = 0

        self.plotting = plotting.Plotting(x_start, x_goal, task_extent=task_extent)

    def plan(self, heading_c=0):
        return self.plan_nominal(heading_c=heading_c)

    def plan_nominal(self, heading_c=0):
        if self.global_map is None:
            raise ValueError("GlobalAStar.global_map must be assigned before planning.")
        if not self.global_map.is_TraversabilityMap_built:
            self.path = []
            self.grid_path = []
            self.plan_cost = math.inf
            return False

        start_row, start_col = self.global_map.xy2grid(self.x_start[0], self.x_start[1])
        goal_row, goal_col = self.global_map.xy2grid(self.x_goal[0], self.x_goal[1])
        if start_row is None or start_col is None or goal_row is None or goal_col is None:
            self.path = []
            self.grid_path = []
            self.plan_cost = math.inf
            return False

        start_layer = self.global_map.get_theta_layer(self.heading_start)
        start = (start_row, start_col, start_layer)
        goal = (goal_row, goal_col)

        linear_cost, angular_cost, direction_blocked = self._build_cost_maps()
        result = self._search(start, goal, linear_cost, angular_cost, direction_blocked)
        if result is None:
            self.path = []
            self.grid_path = []
            self.plan_cost = math.inf
            return False

        self.grid_path, self.plan_cost = result
        self.path = [list(self.global_map.grid2xy(row, col)) for row, col, _ in self.grid_path]
        return True

    def replan(self, initial_start=None, total_lookahead_time=4.5, plot_map=False):
        if not self.global_map.is_TraversabilityMap_built:
            return

        start_time = time.time()
        print("\nReplanning Global Path with A*!")
        print(" Previous global start was: ", np.round(self.x_start[0], 3), np.round(self.x_start[1], 3))
        print(" Current Robot Position: ", np.round(self.global_map.robot_x, 3), np.round(self.global_map.robot_y, 3), np.round(np.rad2deg(self.global_map.robot_heading)))

        self.set_plan_start(initial_start, total_lookahead_time)
        print(" The start is: ", np.round(self.x_start[0], 3), np.round(self.x_start[1], 3), np.round(np.rad2deg(self.heading_start), 3))
        print(" The goal is: ", np.round(self.x_goal[0], 3), np.round(self.x_goal[1], 3))

        reached = self.plan(heading_c=self.global_map.robot_heading)
        print("  A* Computation Time took:", np.round(time.time() - start_time, 3), "s")
        if reached:
            print(f" Total Navigation Cost (Estimated Traversal Time): {np.round(self.plan_cost, 3)} s")
            print(f" Expanded A* states: {self.expanded_count}")
            print("Global Path Replanned with A*!\n")
        else:
            print(" A* did not find a path.\n")

        if plot_map:
            self.plot_map()

    def set_plan_start(self, initial_start, total_lookahead_time) -> bool:
        start_x, start_y, start_heading = self._compute_start(initial_start, total_lookahead_time)
        self.x_start = (start_x, start_y)
        self.heading_start = start_heading
        self.plotting.xI = self.x_start
        self.plotting.xG = self.x_goal
        return True

    def _compute_start(self, initial_start, total_lookahead_time):
        if self.path:
            cmd_v_limit, cmd_w_limit = self.global_map.get_cmd_limits()
            waypoint = self.global_map.get_waypoint(
                self.global_map.robot_x,
                self.global_map.robot_y,
                self.global_map.robot_heading,
                total_lookahead_time,
                cmd_v_limit,
                cmd_w_limit,
                self.path,
            )
            print(" Next Waypoint: ", waypoint)
            if waypoint is not False and not np.allclose(waypoint, self.x_goal):
                heading = math.atan2(waypoint[1] - self.global_map.robot_y, waypoint[0] - self.global_map.robot_x)
                return float(waypoint[0]), float(waypoint[1]), float(heading)

        return float(self.global_map.robot_x), float(self.global_map.robot_y), float(self.global_map.robot_heading)

    def _build_cost_maps(self):
        trav = np.array(self.global_map.TraversabilityMap, copy=True)
        v = trav[:, :, :, 0]
        w = trav[:, :, :, 1]

        known = np.isfinite(v) & np.isfinite(w)
        if hasattr(self.global_map, "TravMap_MetaInfo") and self.global_map.TravMap_MetaInfo is not None:
            trav_meta = np.array(self.global_map.TravMap_MetaInfo, copy=True)
            estimated_value = self.global_map.TravmapInfo_IntEnum.TRAV_Estimated.value
            if trav_meta.shape == known.shape:
                known &= trav_meta == estimated_value

        direction_blocked = known & (v <= self.obstacle_cmd_v_threshold) & (w <= self.obstacle_cmd_w_threshold)
        valid = known & ~direction_blocked & (v > self.obstacle_cmd_v_threshold) & (w > self.obstacle_cmd_w_threshold)

        theta_res = self.global_map.TraversabilityMap_theta_resolution
        linear_cost = np.full(v.shape, np.inf, dtype=np.float64)
        angular_cost = np.full(w.shape, np.inf, dtype=np.float64)

        linear_cost[valid] = 1.0 / v[valid]
        angular_cost[valid] = theta_res / (w[valid] / step_T)

        unknown = ~known
        if np.any(unknown):
            unknown_linear = 1.0 / max(self.unknown_cmd_v, self.obstacle_cmd_v_threshold)
            unknown_angular = theta_res / (max(self.unknown_cmd_w, self.obstacle_cmd_w_threshold) / step_T)
            self._fill_unknown_costs(linear_cost, valid, unknown, unknown_linear)
            self._fill_unknown_costs(angular_cost, valid, unknown, unknown_angular)

        return linear_cost, angular_cost, direction_blocked

    def _fill_unknown_costs(self, cost_map, valid, unknown, default_cost):
        rows, cols, layers = cost_map.shape
        radius = self.unknown_neighbor_radius

        for layer in range(layers):
            known_layer = valid[:, :, layer]
            unknown_layer = unknown[:, :, layer]
            if not np.any(unknown_layer):
                continue

            padded_cost = np.pad(
                np.where(known_layer, cost_map[:, :, layer], np.nan),
                radius,
                mode="constant",
                constant_values=np.nan,
            )

            unknown_rows, unknown_cols = np.where(unknown_layer)
            for row, col in zip(unknown_rows, unknown_cols):
                window = padded_cost[row:row + 2 * radius + 1, col:col + 2 * radius + 1]
                if np.all(np.isnan(window)):
                    inferred = default_cost
                else:
                    local_mean = float(np.nanmean(window))
                    local_max = float(np.nanmax(window))
                    inferred = max(default_cost, local_mean * self.unknown_cost_multiplier, 0.5 * local_max)
                cost_map[row, col, layer] = inferred

    def _search(self, start: GridState, goal: Tuple[int, int], linear_cost, angular_cost, direction_blocked):
        rows = self.global_map.map_size_rows
        cols = self.global_map.map_size_cols
        layers = self.global_map.TraversabilityMap_size_layers

        g_score = np.full((rows, cols, layers), np.inf, dtype=np.float64)
        came_from: Dict[GridState, GridState] = {}
        g_score[start] = 0.0

        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (self._heuristic(start[0], start[1], goal[0], goal[1]), counter, start))
        closed = np.zeros((rows, cols, layers), dtype=bool)
        self.expanded_count = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            row, col, layer = current
            if closed[current]:
                continue
            closed[current] = True
            self.expanded_count += 1

            if row == goal[0] and col == goal[1]:
                return self._reconstruct_path(came_from, current), float(g_score[current])

            for next_state, edge_cost in self._neighbors(current, linear_cost, angular_cost, direction_blocked):
                if closed[next_state]:
                    continue
                tentative = g_score[current] + edge_cost
                if tentative < g_score[next_state]:
                    came_from[next_state] = current
                    g_score[next_state] = tentative
                    counter += 1
                    priority = tentative + self._heuristic(next_state[0], next_state[1], goal[0], goal[1])
                    heapq.heappush(open_heap, (priority, counter, next_state))

        return None

    def _neighbors(self, current: GridState, linear_cost, angular_cost, direction_blocked):
        row, col, current_layer = current
        rows = self.global_map.map_size_rows
        cols = self.global_map.map_size_cols

        for dr, dc in self.MOVES:
            next_row = row + dr
            next_col = col + dc
            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue

            move_layer = self._move_to_layer(dr, dc)
            if direction_blocked[row, col, move_layer] or direction_blocked[next_row, next_col, move_layer]:
                continue

            edge_cost = self._edge_cost(
                linear_cost,
                angular_cost,
                row,
                col,
                current_layer,
                next_row,
                next_col,
                move_layer,
            )
            if not np.isfinite(edge_cost):
                continue

            yield (next_row, next_col, move_layer), edge_cost

    def _edge_cost(self, linear_cost, angular_cost, row1, col1, current_layer, row2, col2, move_layer):
        dr = row2 - row1
        dc = col2 - col1
        if abs(dr) + abs(dc) == 1:
            move_dist = self.global_map.map_resolution
        else:
            move_dist = self.global_map.map_resolution * math.sqrt(2.0)

        lc1 = linear_cost[row1, col1, move_layer]
        lc2 = linear_cost[row2, col2, move_layer]
        ac1 = angular_cost[row1, col1, move_layer]
        ac2 = angular_cost[row2, col2, move_layer]
        if not np.isfinite(lc1 + lc2 + ac1 + ac2):
            return math.inf

        diff = self._layer_diff(current_layer, move_layer)
        linear_part = move_dist * 0.5 * (lc1 + lc2)
        angular_part = diff * 0.5 * (ac1 + ac2)
        return linear_part + angular_part

    def _move_to_layer(self, dr, dc):
        res = self.global_map.map_resolution
        dx = -dr * res
        dy = -dc * res
        theta = math.atan2(dy, dx)
        return self.global_map.get_theta_layer(theta)

    def _layer_diff(self, layer_a, layer_b):
        num_layers = self.global_map.TraversabilityMap_size_layers
        diff = abs(int(layer_a) - int(layer_b))
        if diff > num_layers / 2:
            diff = num_layers - diff
        return diff

    def _heuristic(self, row, col, goal_row, goal_col):
        drow = abs(goal_row - row)
        dcol = abs(goal_col - col)
        diagonal = min(drow, dcol)
        straight = max(drow, dcol) - diagonal
        distance = self.global_map.map_resolution * (straight + math.sqrt(2.0) * diagonal)
        return distance / max(float(cfg.maximum_cmd_v), self.obstacle_cmd_v_threshold)

    @staticmethod
    def _reconstruct_path(came_from: Dict[GridState, GridState], current: GridState):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plot_map(self):
        if self.path:
            self.plotting.animation_interactive(
                [],
                self.path,
                "A*, Expanded States = " + str(self.expanded_count) + ", ETA = " + str(np.round(self.plan_cost, 1)),
            )
        else:
            self.plotting.plot_grid("A* Plain Environment")
