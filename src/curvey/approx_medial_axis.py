from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
import scipy
from numpy import (
    arccos,
    arcsin,
    argmax,
    array,
    ceil,
    clip,
    concatenate,
    diff,
    linspace,
    ndarray,
    newaxis,
    pi,
    roll,
    sqrt,
    stack,
    where,
    zeros,
)
from numpy.linalg import norm
from sortedcontainers import SortedList

from curvey.edges import Edges
from curvey.util import angle_to_points


@dataclass
class SampledPoints:
    """Points uniformly sampled on the border of a maximal disk on the medial axis"""

    points: ndarray
    """`(n, 2)` array of points"""

    distance: ndarray
    """`(n,)` vector of distance to boundary"""

    direction: ndarray
    """Unit vectors pointing to the closest point on the boundary"""

    def __len__(self) -> int:
        """Number of points"""
        return len(self.points)

    def get_disk(self, i: int) -> Disk:
        """Return the `i`th disk"""
        return Disk(
            pt=self.points[i],
            r=self.distance[i],
            vec=self.direction[i],
        )

    def iter_disks(self) -> Iterator[Disk]:
        """Iterate over each point's inscribed disk"""
        for pt, r, direction in zip(self.points, self.distance, self.direction):
            yield Disk(pt=pt, r=r, vec=direction)

    def biggest_disk(self) -> Disk:
        """The inscribed disk with the largest radius"""
        i = cast(int, argmax(self.distance))
        return self.get_disk(i)

    @staticmethod
    def empty() -> SampledPoints:
        """A `SampledPoints` with no points"""
        return _EMPTY_SAMPLED_PTS


_EMPTY_SAMPLED_PTS = SampledPoints(
    points=zeros((0, 2)),
    distance=zeros(0),
    direction=zeros((0, 2)),
)


class Disk(NamedTuple):
    """A maximally inscribed disk on the medial axis"""

    pt: ndarray
    """Center of the disk"""

    r: float
    """Radius of the disk"""

    vec: ndarray
    """Unit vector pointing towards the boundary"""


class Vertex(NamedTuple):
    """A vertex in the medial axis"""

    disk: Disk
    """Inscribed disk corresponding to the vertex"""

    idx: int
    """Index of this vertex in the MA"""


class VisitedDisks:
    """Keep track of set of all disks in the interior we've already visited while building the MA"""

    def __init__(self):
        # Preallocate storage for disks
        n0 = 100
        self._pts = zeros((n0, 2))
        self._sq_radii = zeros(n0)
        self._i = 0

    def append(self, pt: ndarray, r: float) -> None:
        # Check if we need to increase storage capacity
        n = len(self._pts)
        if self._i == n:
            # Need to allocate more space
            self._pts = concatenate([self._pts, np.zeros_like(self._pts)], axis=0)
            self._sq_radii = concatenate([self._sq_radii, np.zeros_like(self._sq_radii)])

        self._pts[self._i] = pt
        self._sq_radii[self._i] = r**2
        self._i += 1

    def has_visited(self, pts: ndarray) -> ndarray:
        if self._i == 0:
            return zeros(len(pts), dtype=bool)

        dist2 = scipy.spatial.distance.cdist(pts, self._pts[: self._i], "sqeuclidean")
        return (dist2 <= self._sq_radii[newaxis, : self._i]).any(axis=1)

    def discard_visited(self, pts: ndarray) -> ndarray:
        if self._i == 0:
            return pts

        return pts[~self.has_visited(pts)]


class FailedToFindMedialAxis(Exception):
    pass


class ApproxMedialAxisBuilder:
    def __init__(
        self,
        boundary: Edges,
        dist_thresh: float,
        angle_thresh: float,
        abs_err: float,
        pt0: ndarray,
        min_edge_length: float | None = None,
    ):
        self.boundary: Edges = boundary
        self.dist_thresh: float = dist_thresh
        self.angle_thresh: float = angle_thresh
        self.abs_err: float = abs_err
        self.min_edge_length = min_edge_length
        self.verts: list[Vertex] = []
        self.edges: list[tuple[int, int]] = []
        self.visited_disks: VisitedDisks = VisitedDisks()
        self.queue: SortedList[Vertex] = SortedList(key=lambda v: v.disk.r)

        # Initialize a maximal disk on the medial axis
        self.pt0: ndarray = pt0
        disk = self.find_initial_medial_axis_point(pt0)
        v0 = self.add_vertex(disk, prnt=None)
        self.queue.add(v0)

    def find_interior_point(self) -> ndarray:
        tris = self.boundary.triangulate()
        tri = max(tris.shapely.geoms, key=lambda t: t.area)
        return array(tri.centroid.coords[0])

    def find_initial_medial_axis_point(self, pt0: ndarray) -> Disk:
        # pt0 is in the interior
        disk = self.maximal_disk_at(pt0)

        while True:
            sampled = self.sample_disk_boundary(disk)
            if len(sampled) == 0:
                msg = "Failed to find medial axis. The abs_thresh or dist_thresh may be too large."
                raise FailedToFindMedialAxis(msg)

            if ma_pts := self.medial_axis_points(sampled):
                # Found a point on the medial axis
                return ma_pts.biggest_disk()

            # This should converge on the medial axis after a couple of steps
            disk = sampled.biggest_disk()

    def distance_to_boundary(self, points: ndarray) -> SampledPoints:
        _, _, boundary_points = self.boundary.closest_point(points)
        direction = boundary_points - points
        distance = norm(direction, axis=1)
        _distance = distance[:, newaxis]
        # Avoid dividing by zero when the point is exactly on the boundary
        np.divide(direction, _distance, where=_distance > 0, out=direction)
        return SampledPoints(points, distance, direction)

    def maximal_disk_at(self, pt: ndarray) -> Disk:
        return self.distance_to_boundary(pt[newaxis]).get_disk(0)

    def medial_axis_points(self, sampled: SampledPoints) -> SampledPoints:
        """Given a set of points sampled on the boundary of an MA disk, find points on the MA"""

        # Points or their adjacent points that haven't been visited yet
        unvisited = ~self.visited_disks.has_visited(sampled.points)
        unvisited = unvisited | roll(unvisited, -1)

        if not unvisited.any():
            return SampledPoints.empty()

        (i0,) = where(unvisited)
        i1 = (i0 + 1) % len(sampled)

        d0 = sampled.direction[i0]
        d1 = sampled.direction[i1]
        theta = arccos(clip((d0 * d1).sum(axis=1), -1, 1))
        above_angle_thresh = theta > self.angle_thresh
        if not above_angle_thresh.any():
            return SampledPoints.empty()

        d0, d1 = d0[above_angle_thresh], d1[above_angle_thresh]
        p0 = sampled.points[i0[above_angle_thresh]]
        p1 = sampled.points[i1[above_angle_thresh]]

        # Divergence test - heads of the vectors should point in different directions
        dtail2 = ((p1 - p0) ** 2).sum(axis=1)
        dtail = sqrt(dtail2)[:, newaxis]
        head1 = p1 + d1 * dtail
        head0 = p0 + d0 * dtail
        dhead2 = ((head1 - head0) ** 2).sum(axis=1)
        diverges = dhead2 > dtail2

        # noinspection PyUnresolvedReferences
        if not diverges.any():
            return SampledPoints.empty()

        # Return midpoints of adjacent sampled.pts where theta > theta_thresh AND adjacent
        # sampled.direction vectors point in different directions
        ma_pts = (p0[diverges] + p1[diverges]) / 2

        if self.min_edge_length is not None and self.queue:
            # Discard points if they're too close to vertices already in the queue
            queue_points = stack([v.disk.pt for v in self.queue], axis=0)
            dist2 = scipy.spatial.distance.cdist(ma_pts, queue_points, metric="sqeuclidean")
            ma_pts = ma_pts[~(dist2 < self.min_edge_length**2).any(axis=1)]

        if len(ma_pts) == 0:
            return SampledPoints.empty()

        return self.distance_to_boundary(ma_pts)

    def sample_disk_boundary(self, disk: Disk) -> SampledPoints:
        if disk.r < self.abs_err:
            return SampledPoints.empty()

        n = int(ceil(pi / arcsin(self.abs_err / disk.r)))
        n = max(n, 4)
        theta = linspace(0, 2 * pi, n, endpoint=False)
        pts = disk.pt[newaxis] + disk.r * angle_to_points(theta)
        return self.distance_to_boundary(pts)

    def visit(self, pt: ndarray, r: float):
        self.visited_disks.append(pt=pt, r=r)

    def discard_visited(self, pts: ndarray) -> ndarray:
        return self.visited_disks.discard_visited(pts)

    def add_vertex(self, disk: Disk, prnt: int | None) -> Vertex:
        idx = len(self.verts)
        v = Vertex(disk=disk, idx=idx)
        self.verts.append(v)
        if prnt is not None:
            self.edges.append((prnt, idx))
        return v

    def step(self, v0: Vertex):
        # Generate uniformly distributed samples on the disk boundary
        sampled = self.sample_disk_boundary(v0.disk)

        # Neighboring points whose direction vectors to their closest boundary points diverge
        ma_pts = self.medial_axis_points(sampled)

        for disk1 in ma_pts.iter_disks():
            if disk1.r > self.dist_thresh:
                v1 = self.add_vertex(disk1, prnt=v0.idx)
                self.queue.add(v1)
            else:
                self.visit(disk1.pt, disk1.r)

        self.visit(v0.disk.pt, v0.disk.r)

    def run(self, max_step: int | None = None):
        step = 0
        while self.queue:
            step += 1
            vertex = self.queue.pop()
            self.step(vertex)

            if max_step is not None and step == max_step:
                break

    def finalize(self, close_loops=True) -> Edges:
        """Convert the accumulated vertices into a standalone Edges object"""
        if not self.verts:
            return Edges.empty()

        ama = Edges(
            points=stack([v.disk.pt for v in self.verts], axis=0),
            edges=stack(self.edges, axis=0) if self.edges else zeros((0, 2), dtype="int"),
            point_data={"distance": array([v.disk.r for v in self.verts])},
        )

        if close_loops and len(ama.edges):
            ama = self.close_loops(ama)

        return ama

    def close_loops(self, ama: Edges) -> Edges:
        # Look for leaf verts near eachother
        distance = ama.point_data["distance"]
        adj = ama.to_csgraph(weighted=False, directed=False)
        (leaf_verts,) = where(adj.sum(axis=1) == 1)
        if len(leaf_verts) == 0:
            return ama

        leaf_points = ama.points[leaf_verts]
        tree = scipy.spatial.KDTree(leaf_points)
        pairs = tree.query_pairs(distance[leaf_verts].max(), output_type="ndarray")

        # Discard pairs that aren't in eachothers' disks
        dp = diff(leaf_points[pairs], axis=1).squeeze(axis=1)
        dist = norm(dp, axis=1)
        r = distance[pairs].min(axis=1)
        i = dist <= r

        if self.min_edge_length is not None:
            i &= dist >= self.min_edge_length

        leaf_vert_pairs = leaf_verts[pairs[i]]
        edges = concatenate([ama.edges, leaf_vert_pairs], axis=0)
        return ama.with_(edges=edges)
