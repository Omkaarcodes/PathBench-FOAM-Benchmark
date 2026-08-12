"""
Radius-Biased Probabilistic Foam (RBPF) path planning algorithm.

Based on: Nascimento et al., "Safe Path Planning Algorithms for Mobile Robots
Based on Probabilistic Foam", Sensors 2021, 21(12), 4156.
https://www.mdpi.com/1424-8220/21/12/4156

RBPF is a variant of PFM where parent bubbles are selected using Roulette
Wheel selection with probability proportional to their radius.  This biases
the foam toward large-clearance passages, yielding safer (higher-clearance)
paths at the potential cost of longer path length.
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


class ProbabilisticFoamRBPF(SampleBasedAlgorithm):
    """
    Radius-Biased Probabilistic Foam (RBPF).

    Like PFM, but parent bubbles are selected from an open list using
    Roulette-Wheel selection: the probability of selecting bubble i is
    p_i = r_i / sum(r_j) (Eq. 6 in the paper).  This biases propagation
    through wide passages, producing safer paths with high clearance.

    Parameters
    ----------
    r_min : float
        Minimum acceptable bubble radius.  Default 1.0.
    K : int
        Maximum child constant (for 2-D, paper suggests K = 4).  Default 4.
    max_iterations : int
        Safety cap on total sampling attempts.  Default 50000.
    """

    _graph: Forest
    _bubbles: List[dict]
    _obstacle_positions: Optional[np.ndarray]

    def __init__(self, services: Services, testing: BasicTesting = None,
                 r_min: float = 1.0,
                 K: int = 4,
                 max_iterations: int = 50000) -> None:
        super().__init__(services, testing)

        self._r_min = r_min
        self._K = K
        self._max_iterations = max_iterations

        self._graph = gen_forest(
            self._services,
            Vertex(self._get_grid().agent.position),
            Vertex(self._get_grid().goal.position),
            []
        )
        self._graph.edges_removable = False

        self._obstacle_positions = self._precompute_obstacles()

        start_vertex = self._graph.root_vertex_start
        start_radius = self._bubble_radius(start_vertex.position)
        self._bubbles = [{"vertex": start_vertex, "radius": start_radius}]

        self._init_displays()

    # ------------------------------------------------------------------
    # Obstacle / clearance helpers
    # ------------------------------------------------------------------

    def _precompute_obstacles(self) -> np.ndarray:
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
        if self._obstacle_positions.shape[0] == 0:
            return float(np.max(np.array(self._get_grid().size)))
        pos = np.array(point.values, dtype=float)
        diffs = self._obstacle_positions - pos
        dists = np.linalg.norm(diffs, axis=1)
        return float(np.min(dists))

    def _bubble_radius(self, center: Point) -> float:
        return max(0.0, self._clearance(center))

    # ------------------------------------------------------------------
    # Sampling / selection helpers
    # ------------------------------------------------------------------

    def _max_children(self, parent_radius: float, n_dim: int) -> int:
        """N = K * (r / r_min)^(n-1)."""
        if parent_radius < self._r_min:
            return 0
        return max(1, int(self._K * (parent_radius / self._r_min) ** (n_dim - 1)))

    def _sample_on_bubble_surface(self, center: Point, radius: float) -> Point:
        n_dim = center.n_dim
        direction = np.random.randn(n_dim)
        direction /= (np.linalg.norm(direction) + 1e-12)
        surface_pt = np.array(center.values, dtype=float) + radius * direction
        surface_pt = np.round(surface_pt).astype(int)
        return Point(*surface_pt.tolist())

    def _is_inside_foam(self, point: Point) -> bool:
        pt = np.array(point.values, dtype=float)
        for b in self._bubbles:
            c = np.array(b["vertex"].position.values, dtype=float)
            if np.linalg.norm(pt - c) < b["radius"]:
                return True
        return False

    def _roulette_select(self, open_list: List[dict]) -> int:
        """
        Roulette-Wheel selection: p_i = r_i / sum(r_j)  (Eq. 6).
        Returns the *index* into open_list.
        """
        radii = np.array([b["radius"] for b in open_list], dtype=float)
        probs = radii / radii.sum()
        return int(np.random.choice(len(open_list), p=probs))

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_path(self, q_new: Vertex) -> None:
        goal_v = Vertex(self._get_grid().goal.position)
        self._graph.add_edge(q_new, goal_v)

        path: List[Vertex] = [goal_v]
        while len(path[-1].parents) != 0:
            for parent in path[-1].parents:
                path.append(parent)
                break
        del path[-1]
        path.reverse()

        for p in path:
            self.move_agent(p.position)
            self.key_frame(ignore_key_frame_skip=True)

    # ------------------------------------------------------------------
    # Core algorithm  (Roulette-Wheel parent selection)
    # ------------------------------------------------------------------

    def _find_path_internal(self) -> None:
        grid: Map = self._get_grid()
        n_dim = grid.agent.position.n_dim

        # Open list of candidate parent bubbles
        open_list: List[dict] = list(self._bubbles)
        total_samples = 0

        while open_list and total_samples < self._max_iterations:
            # Select parent by roulette (proportional to radius)
            idx = self._roulette_select(open_list)
            parent = open_list[idx]
            p_vertex = parent["vertex"]
            p_radius = parent["radius"]
            N = self._max_children(p_radius, n_dim)

            children_expanded = 0
            attempts = 0
            max_attempts = N * 5

            while children_expanded < N and attempts < max_attempts and total_samples < self._max_iterations:
                attempts += 1
                total_samples += 1

                q_sample = self._sample_on_bubble_surface(
                    p_vertex.position, p_radius
                )

                if grid.is_out_of_bounds_pos(q_sample):
                    continue
                if not grid.is_agent_valid_pos(q_sample):
                    continue
                if self._is_inside_foam(q_sample):
                    continue

                line = grid.get_line_sequence(p_vertex.position, q_sample)
                if not grid.is_valid_line_sequence(line):
                    continue

                child_radius = self._bubble_radius(q_sample)
                if child_radius < self._r_min:
                    continue

                q_new = Vertex(q_sample)
                self._graph.add_edge(p_vertex, q_new)

                bubble = {"vertex": q_new, "radius": child_radius}
                self._bubbles.append(bubble)
                open_list.append(bubble)
                children_expanded += 1

                self.key_frame()

                dist_to_goal = float(np.linalg.norm(
                    np.array(q_new.position.values, dtype=float) -
                    np.array(grid.goal.position.values, dtype=float)
                ))
                if dist_to_goal <= child_radius or grid.is_agent_in_goal_radius(agent_pos=q_new.position):
                    self._extract_path(q_new)
                    return

            # Remove exhausted parent from open list
            open_list.pop(idx)
