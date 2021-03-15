import os
import sys


def test_shallow_water_demo():
    import mpi4jax

    rootpath = os.path.dirname(mpi4jax.__file__)
    demopath = os.path.join(rootpath, "..", "docs", "_static")
    sys.path.append(demopath)

    try:
        from demo import solve_shallow_water

        sol = solve_shallow_water(86_400)
        assert len(sol) > 100
    finally:
        removed_path = sys.path.pop()
        assert removed_path == demopath
