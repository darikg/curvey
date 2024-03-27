# curvey.curve

## ::: curvey.curve.Curve
       options:
         members: []

## Constructors
### ::: curvey.curve.Curve.circle
### ::: curvey.curve.Curve.dumbbell
### ::: curvey.curve.Curve.ellipse
### ::: curvey.curve.Curve.from_curvature
### ::: curvey.curve.Curve.star

## Curve-valued properties
### ::: curvey.curve.Curve.area
### ::: curvey.curve.Curve.center
### ::: curvey.curve.Curve.centroid
### ::: curvey.curve.Curve.data
### ::: curvey.curve.Curve.is_simple
### ::: curvey.curve.Curve.laplacian
### ::: curvey.curve.Curve.length
### ::: curvey.curve.Curve.n
### ::: curvey.curve.Curve.orientation
### ::: curvey.curve.Curve.signed_area

## Vertex-valued properties
The following properties have length `n`:

### ::: curvey.curve.Curve.arclength
### ::: curvey.curve.Curve.curvature
### ::: curvey.curve.Curve.dual_edge_length
### ::: curvey.curve.Curve.normal
### ::: curvey.curve.Curve.points
### ::: curvey.curve.Curve.tangent
### ::: curvey.curve.Curve.turning_angle
### ::: curvey.curve.Curve.x
### ::: curvey.curve.Curve.y

## Vertex + 1 -valued properties

These are special cases and have length `n + 1`
### ::: curvey.curve.Curve.closed_arclength
### ::: curvey.curve.Curve.closed_points

## Edge-valued properties
The following properties have length `n`:
### ::: curvey.curve.Curve.edge
### ::: curvey.curve.Curve.edge_length
### ::: curvey.curve.Curve.unit_edge
### ::: curvey.curve.Curve.edge_normal
### ::: curvey.curve.Curve.cum_edge_length

## Transformations
All transformations return a new `Curve`; nothing modifies a `Curve` inplace.

## Basic transformations
### ::: curvey.curve.Curve.with_points
### ::: curvey.curve.Curve.with_data
### ::: curvey.curve.Curve.drop_data
### ::: curvey.curve.Curve.reverse
### ::: curvey.curve.Curve.scale
### ::: curvey.curve.Curve.translate
### ::: curvey.curve.Curve.roll
### ::: curvey.curve.Curve.rotate
### ::: curvey.curve.Curve.reflect
### ::: curvey.curve.Curve.to_cw
### ::: curvey.curve.Curve.to_ccw
### ::: curvey.curve.Curve.transform
### ::: curvey.curve.Curve.to_area
### ::: curvey.curve.Curve.to_edge_midpoints
### ::: curvey.curve.Curve.to_length
### ::: curvey.curve.Curve.to_orientation

## Sampling transformations
### ::: curvey.curve.Curve.collapse_shortest_edges
### ::: curvey.curve.Curve.interpolate
### ::: curvey.curve.Curve.split_edges
### ::: curvey.curve.Curve.split_longest_edges
### ::: curvey.curve.Curve.subdivide

## Target transformations
These transformations accept an additional target curve.
### ::: curvey.curve.Curve.align_to
### ::: curvey.curve.Curve.optimize_edge_lengths_to
### ::: curvey.curve.Curve.orient_to
### ::: curvey.curve.Curve.register_to
### ::: curvey.curve.Curve.roll_to

## Plotting
### ::: curvey.curve.Curve.plot
### ::: curvey.curve.Curve.plot_edges
### ::: curvey.curve.Curve.plot_points
### ::: curvey.curve.Curve.plot_vectors

## Special
### ::: curvey.curve.Curve.check_same_n_vertices
### ::: curvey.curve.Curve.deriv
### ::: curvey.curve.Curve.edge_intersections
### ::: curvey.curve.Curve.interpolator

## Library interface
### ::: curvey.curve.Curve.to_shapely
### ::: curvey.curve.Curve.triangulate
