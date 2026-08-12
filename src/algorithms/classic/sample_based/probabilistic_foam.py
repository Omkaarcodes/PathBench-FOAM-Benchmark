"""
Radius-Based Probabilistic Foam path planning algorithm.

Based on: Gonzalez-Sieira et al., "Probabilistic Foam: A Comparison with RRT
for Path Planning", Sensors 2021, 21(12), 4156.
https://www.mdpi.com/1424-8220/21/12/4156

The algorithm grows a tree of "bubbles" (spheres of free space) from the start
position toward the goal. Each bubble is centred at a sampled point with a
radius equal to the distance to the nearest obstacle (capped at a configurable
maximum). New bubbles are created by sampling points on the surface of existing
bubbles. When a bubble contains or reaches the goal, the path is extracted by
traversing parent pointers.
"""

from typing import List, Optional

import numpy as np
import torch

from algorithms.classic.sample_based.core.sample_based_algorithm import SampleBasedAlgorithm
from algorithms.basic_testing import BasicTesting
from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest
from algorithms.configuration.maps.dense_map import DenseMap
from algorithms.configuration.maps.map import Map
from algorithms.configuration.maps.sparse_map import SparseMap
from algorithms.configuration.entities.obstacle import Obstacle

from simulator.services.services import Services
from simulator.views.map.display.map_display import MapDisplay

from structures import Point


class ProbabilisticFoam(SampleBasedAlgorithm):
    """
    Radius-Based Probabilistic Foam planner.

    Grows a tree of overlapping bubbles (spheres of free space) from the start
    position.  Each bubble's radius equals the distance from its centre to the
    nearest obstacle, capped by *max_bubble_radius*.

    Parameters
    ----------
    max_bubble_radius : float
        Upper bound on any single bubble's radius.  A smaller value forces
        denser sampling in open areas; a larger value lets the foam expand
        quickly through corridors.  Default 15.
    max_iterations : int
        Maximum number of sampling iterations before giving up.  Default 15000.
    goal_bias : float
        Probability [0, 1] of biasing the next sample toward the goal instead
        of picking a random bubble surface point.  Default 0.05.
    """

    _graph: Forest
    _bubbles: List[dict]          # [{vertex, radius}, ...]
    _obstacle_positions: Optional[np.ndarray]  # Nx2 (or NxD) array of obstacle coords

    def __init__(self, services: Services, testing: BasicTesting = None,
                 max_bubble_radius: float = 15.0,
                 max_iterations: int = 15000,
                 goal_bias: float = 0.05) -> None:
        super().__init__(services, testing)

        self._max_bubble_radius = max_bubble_radius
        self._max_iterations = max_iterations
        self._goal_bias = goal_bias

        self._graph = gen_forest(
            self._services,
            Vertex(self._get_grid().agent.position),
            Vertex(self._get_grid().goal.position),
            []
        )
        self._graph.edges_removable = False

        # Pre-compute obstacle positions for fast clearance queries
        self._obstacle_positions = self._precompute_obstacles()

        # Create the initial bubble at the agent's start position
        start_vertex = self._graph.root_vertex_start
        start_radius = self._bubble_radius(start_vertex.position)
        self._bubbles = [{"vertex": start_vertex, "radius": start_radius}]

        self._init_displays()

    # ------------------------------------------------------------------
    # Obstacle / clearance helpers
    # ------------------------------------------------------------------

    def _precompute_obstacles(self) -> np.ndarray:
        """Return an (N, D) array of all obstacle cell coordinates."""
        grid = self._get_grid()
        if isinstance(grid, DenseMap):
            obstacles = np.transpose(np.where(grid.grid == grid.WALL_ID))
        elif isinstance(grid, SparseMap):
            tmp = np.full(grid.size, grid.CLEAR_ID, dtype=np.uint8)
            for o in grid.obstacles:
                if isinstance(o, Obstacle):
                    tmp[o.position.values] = grid.WALL_ID
            obstacles = np.transpose(np.where(tmp == grid.WALL_ID))
        else:
            obstacles = np.empty((0, grid.size.n_dim))
        return obstacles

    def _clearance(self, point: Point) -> float:
        """Euclidean distance from *point* to the nearest obstacle cell."""
        if self._obstacle_positions.shape[0] == 0:
            # No obstacles – return a large clearance
            return float(np.max(np.array(self._get_grid().size)))
        pos = np.array(point.values, dtype=float)
        diffs = self._obstacle_positions - pos
        dists = np.linalg.norm(diffs, axis=1)
        return float(np.min(dists))

    def _bubble_radius(self, center: Point) -> float:
        """Bubble radius = min(clearance, max_bubble_radius), >= 1."""
        return max(1.0, min(self._clearance(center), self._max_bubble_radius))

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_on_bubble_surface(self, center: Point, radius: float) -> Point:
        """
        Return a uniformly random point on the surface of the bubble (sphere)
        defined by *center* and *radius*.
        """
        n_dim = center.n_dim
        # Random direction (uniform on unit sphere)
        direction = np.random.randn(n_dim)
        direction /= (np.linalg.norm(direction) + 1e-12)
        # Point on the surface
        surface_pt = np.array(center.values, dtype=float) + radius * direction
        # Round to integer grid coordinates
        surface_pt = np.round(surface_pt).astype(int)
        return Point(*surface_pt.tolist())

    def _select_bubble_for_expansion(self) -> dict:
        """
        Choose a bubble to expand.  Bubbles closer to the goal or with larger
        radius (more frontier) are slightly favoured via a weighted random pick.
        """
        goal = np.array(self._get_grid().goal.position.values, dtype=float)
        weights = []
        for b in self._bubbles:
            center = np.array(b["vertex"].position.values, dtype=float)
            dist_to_goal = np.linalg.norm(goal - center) + 1e-6
            # Weight: prefer larger bubbles closer to the goal
            w = b["radius"] / dist_to_goal
            weights.append(w)
        weights = np.array(weights)
        weights /= weights.sum()
        idx = np.random.choice(len(self._bubbles), p=weights)
        return self._bubbles[idx]

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_path(self, q_new: Vertex) -> None:
        """Trace back from *q_new* to start through parent pointers and move
        the agent along the resulting path."""
        goal_v = Vertex(self._get_grid().goal.position)
        self._graph.add_edge(q_new, goal_v)

        path: List[Vertex] = [goal_v]
        while len(path[-1].parents) != 0:
            for parent in path[-1].parents:
                path.append(parent)
                break
        # Remove the start vertex itself (agent is already there)
        del path[-1]
        path.reverse()

        for p in path:
            self.move_agent(p.position)
            self.key_frame(ignore_key_frame_skip=True)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _find_path_internal(self) -> None:
        grid: Map = self._get_grid()

        for _ in range(self._max_iterations):
            # ----- Optionally bias toward goal -----
            if np.random.random() < self._goal_bias:
                q_sample = grid.goal.position
            else:
                # Pick a bubble and sample on its surface
                bubble = self._select_bubble_for_expansion()
                q_sample = self._sample_on_bubble_surface(
                    bubble["vertex"].position, bubble["radius"]
                )

            # Discard out-of-bounds or obstacle samples
            if grid.is_out_of_bounds_pos(q_sample):
                continue
            if not grid.is_agent_valid_pos(q_sample):
                continue

            # Find nearest existing bubble centre to q_sample
            q_near: Vertex = self._graph.get_nearest_vertex(
                [self._graph.root_vertex_start], q_sample
            )
            if q_near.position == q_sample:
                continue

            # Check line-of-sight between q_near and q_sample
            line = grid.get_line_sequence(q_near.position, q_sample)
            if not grid.is_valid_line_sequence(line):
                continue

            # Create new vertex / bubble
            q_new = Vertex(q_sample)
            self._graph.add_edge(q_near, q_new)

            new_radius = self._bubble_radius(q_sample)
            self._bubbles.append({"vertex": q_new, "radius": new_radius})

            self.key_frame()

            # Check if the new bubble contains or is near the goal
            dist_to_goal = float(np.linalg.norm(
                np.array(q_new.position.values, dtype=float) -
                np.array(grid.goal.position.values, dtype=float)
            ))
            if dist_to_goal <= new_radius or grid.is_agent_in_goal_radius(agent_pos=q_new.position):
                self._extract_path(q_new)
                return
