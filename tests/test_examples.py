import pytest


@pytest.fixture(autouse=True)
def add_examples_to_path():
    import os
    import sys

    rootpath = os.path.dirname(__file__)
    demopath = os.path.join(rootpath, "..", "examples")
    sys.path.append(demopath)

    try:
        yield
    finally:
        removed_path = sys.path.pop()
        assert removed_path == demopath


def test_shallow_water_demo():
    from shallow_water import solve_shallow_water

    sol = solve_shallow_water(86_400)
    assert len(sol) > 100
