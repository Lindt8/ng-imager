from ngimager.geometry.plane import Plane
import numpy as np

def test_plane_grid_counts():
    pl = Plane.from_cfg([0,0,0], [0,0,1], -1, 1, 0.5, -1, 1, 0.25)
    assert pl.nu == 5   # -1,-0.5,0,0.5,1
    assert pl.nv == 9
    X = pl.plane_to_world(0.5, -0.75)
    u,v = pl.world_to_plane(X)
    assert abs(u-0.5)<1e-9 and abs(v+0.75)<1e-9
