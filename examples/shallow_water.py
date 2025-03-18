"""mpi4jax demo application -- Shallow water

A non-linear shallow water solver, adapted from:

https://github.com/dionhaefner/shallow-water

Usage examples:

    # runs demo on 4 processes
    $ mpirun -n 4 python shallow_water.py

    # saves output animation as shallow-water.mp4
    $ mpirun -n 4 python shallow_water.py --save-animation

    # runs demo as a benchmark (no output)
    $ mpirun -n 4 python shallow_water.py --benchmark

"""

import os
import sys
import math
import time
import warnings
from contextlib import ExitStack
from collections import namedtuple
from functools import partial

import numpy as np
from mpi4py import MPI

try:
    import tqdm
except ImportError:
    warnings.warn("Could not import tqdm, can't show progress bar")
    HAS_TQDM = False
else:
    HAS_TQDM = True

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# on GPU, put each process on its own device
os.environ["CUDA_VISIBLE_DEVICES"] = str(mpi_rank)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import mpi4jax  # noqa: E402


#
# MPI setup
#

supported_nproc = (1, 2, 4, 6, 8, 16)
if mpi_size not in supported_nproc:
    raise RuntimeError(
        f"Got invalid number of MPI processes: {mpi_size}. "
        f"Please choose one of these: {supported_nproc}."
    )

nproc_y = min(mpi_size, 2)
nproc_x = mpi_size // nproc_y

proc_idx = np.unravel_index(mpi_rank, (nproc_y, nproc_x))

#
# Grid setup
#

# we use 1 cell overlap on each side of the domain
nx_global = 360 + 2
ny_global = 180 + 2

# grid spacing in metres
dx = 5e3
dy = 5e3

# make sure processes divide the domain evenly
assert (nx_global - 2) % nproc_x == 0
assert (ny_global - 2) % nproc_y == 0

nx_local = (nx_global - 2) // nproc_x + 2
ny_local = (ny_global - 2) // nproc_y + 2

x_global, y_global = (
    np.arange(-1, nx_global - 1) * dx,
    np.arange(-1, ny_global - 1) * dy,
)
yy_global, xx_global = np.meshgrid(y_global, x_global, indexing="ij")

length_x = x_global[-2] - x_global[1]
length_y = y_global[-2] - y_global[1]

# this extracts the processor-local domain from a global array
local_slice = (
    slice((ny_local - 2) * proc_idx[0], (ny_local - 2) * proc_idx[0] + ny_local),
    slice((nx_local - 2) * proc_idx[1], (nx_local - 2) * proc_idx[1] + nx_local),
)

x_local = x_global[local_slice[1]]
y_local = y_global[local_slice[0]]

xx_local = xx_global[local_slice]
yy_local = yy_global[local_slice]

#
# Model parameters
#

# physical parameters
GRAVITY = 9.81
DEPTH = 100.0
CORIOLIS_F = 2e-4
CORIOLIS_BETA = 2e-11
CORIOLIS_PARAM = CORIOLIS_F + yy_local * CORIOLIS_BETA
LATERAL_VISCOSITY = 1e-3 * CORIOLIS_F * dx**2

# other parameters
DAY_IN_SECONDS = 86_400
PERIODIC_BOUNDARY_X = True

ADAMS_BASHFORTH_A = 1.5 + 0.1
ADAMS_BASHFORTH_B = -(0.5 + 0.1)

# output parameters
PLOT_ETA_RANGE = 10
PLOT_EVERY = 100
MAX_QUIVERS = (25, 50)


# set time step based on CFL condition
dt = 0.125 * min(dx, dy) / np.sqrt(GRAVITY * DEPTH)


@jax.jit
def get_initial_conditions():
    """For the initial conditions, we use a horizontal jet in geostrophic balance."""
    # global initial conditions
    u0_global = 10 * jnp.exp(
        -((yy_global - 0.5 * length_y) ** 2) / (0.02 * length_x) ** 2
    )
    v0_global = jnp.zeros_like(u0_global)

    # approximate balance h_y = -(f/g)u
    coriolis_global = CORIOLIS_F + yy_global * CORIOLIS_BETA
    h_geostrophy = np.cumsum(-dy * u0_global * coriolis_global / GRAVITY, axis=0)
    h0_global = (
        DEPTH
        + h_geostrophy
        # make sure h0 is centered around depth
        - h_geostrophy.mean()
        # small perturbation to break symmetry
        + 0.2
        * np.sin(xx_global / length_x * 10 * np.pi)
        * np.cos(yy_global / length_y * 8 * np.pi)
    )

    h0_local = h0_global[local_slice]
    u0_local = u0_global[local_slice]
    v0_local = v0_global[local_slice]

    h0_local = enforce_boundaries(h0_local, "h")
    u0_local = enforce_boundaries(u0_local, "u")
    v0_local = enforce_boundaries(v0_local, "v")

    return h0_local, u0_local, v0_local


@partial(jax.jit, static_argnums=(1,))
def enforce_boundaries(arr, grid):
    """Handle boundary exchange between processors.

    This is where mpi4jax comes in!
    """
    assert grid in ("h", "u", "v")

    # start sending west, go clockwise
    send_order = (
        "west",
        "north",
        "east",
        "south",
    )

    # start receiving east, go clockwise
    recv_order = (
        "east",
        "south",
        "west",
        "north",
    )

    overlap_slices_send = dict(
        south=(1, slice(None), Ellipsis),
        west=(slice(None), 1, Ellipsis),
        north=(-2, slice(None), Ellipsis),
        east=(slice(None), -2, Ellipsis),
    )

    overlap_slices_recv = dict(
        south=(0, slice(None), Ellipsis),
        west=(slice(None), 0, Ellipsis),
        north=(-1, slice(None), Ellipsis),
        east=(slice(None), -1, Ellipsis),
    )

    proc_neighbors = {
        "south": (proc_idx[0] - 1, proc_idx[1]) if proc_idx[0] > 0 else None,
        "west": (proc_idx[0], proc_idx[1] - 1) if proc_idx[1] > 0 else None,
        "north": (proc_idx[0] + 1, proc_idx[1]) if proc_idx[0] < nproc_y - 1 else None,
        "east": (proc_idx[0], proc_idx[1] + 1) if proc_idx[1] < nproc_x - 1 else None,
    }

    if PERIODIC_BOUNDARY_X:
        if proc_idx[1] == 0:
            proc_neighbors["west"] = (proc_idx[0], nproc_x - 1)

        if proc_idx[1] == nproc_x - 1:
            proc_neighbors["east"] = (proc_idx[0], 0)

    for send_dir, recv_dir in zip(send_order, recv_order):
        send_proc = proc_neighbors[send_dir]
        recv_proc = proc_neighbors[recv_dir]

        if send_proc is None and recv_proc is None:
            continue

        if send_proc is not None:
            send_proc = np.ravel_multi_index(send_proc, (nproc_y, nproc_x))

        if recv_proc is not None:
            recv_proc = np.ravel_multi_index(recv_proc, (nproc_y, nproc_x))

        recv_idx = overlap_slices_recv[recv_dir]
        recv_arr = jnp.empty_like(arr[recv_idx])

        send_idx = overlap_slices_send[send_dir]
        send_arr = arr[send_idx]

        if send_proc is None:
            recv_arr = mpi4jax.recv(recv_arr, source=recv_proc, comm=mpi_comm)
            arr = arr.at[recv_idx].set(recv_arr)
        elif recv_proc is None:
            mpi4jax.send(send_arr, dest=send_proc, comm=mpi_comm)
        else:
            recv_arr = mpi4jax.sendrecv(
                send_arr,
                recv_arr,
                source=recv_proc,
                dest=send_proc,
                comm=mpi_comm,
            )
            arr = arr.at[recv_idx].set(recv_arr)

    if not PERIODIC_BOUNDARY_X and grid == "u" and proc_idx[1] == nproc_x - 1:
        arr = arr.at[:, -2].set(0.0)

    if grid == "v" and proc_idx[0] == nproc_y - 1:
        arr = arr.at[-2, :].set(0.0)

    return arr


ModelState = namedtuple("ModelState", "h, u, v, dh, du, dv")


@partial(jax.jit, static_argnums=(1,))
def shallow_water_step(state, is_first_step):
    """Perform one step of the shallow-water model.

    Returns modified model state.
    """
    h, u, v, dh, du, dv = state

    hc = jnp.pad(h[1:-1, 1:-1], 1, "edge")
    hc = enforce_boundaries(hc, "h")

    fe = jnp.empty_like(u)
    fn = jnp.empty_like(u)

    fe = fe.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[1:-1, 2:]) * u[1:-1, 1:-1])
    fn = fn.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[2:, 1:-1]) * v[1:-1, 1:-1])
    fe = enforce_boundaries(fe, "u")
    fn = enforce_boundaries(fn, "v")

    dh_new = dh.at[1:-1, 1:-1].set(
        -(fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx - (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
    )

    # nonlinear momentum equation
    q = jnp.empty_like(u)
    ke = jnp.empty_like(u)

    # planetary and relative vorticity
    q = q.at[1:-1, 1:-1].set(
        CORIOLIS_PARAM[1:-1, 1:-1]
        + ((v[1:-1, 2:] - v[1:-1, 1:-1]) / dx - (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy)
    )
    # potential vorticity
    q = q.at[1:-1, 1:-1].mul(
        1.0 / (0.25 * (hc[1:-1, 1:-1] + hc[1:-1, 2:] + hc[2:, 1:-1] + hc[2:, 2:]))
    )
    q = enforce_boundaries(q, "h")

    du_new = du.at[1:-1, 1:-1].set(
        -GRAVITY * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
        + 0.5
        * (
            q[1:-1, 1:-1] * 0.5 * (fn[1:-1, 1:-1] + fn[1:-1, 2:])
            + q[:-2, 1:-1] * 0.5 * (fn[:-2, 1:-1] + fn[:-2, 2:])
        )
    )
    dv_new = dv.at[1:-1, 1:-1].set(
        -GRAVITY * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
        - 0.5
        * (
            q[1:-1, 1:-1] * 0.5 * (fe[1:-1, 1:-1] + fe[2:, 1:-1])
            + q[1:-1, :-2] * 0.5 * (fe[1:-1, :-2] + fe[2:, :-2])
        )
    )
    ke = ke.at[1:-1, 1:-1].set(
        0.5
        * (
            0.5 * (u[1:-1, 1:-1] ** 2 + u[1:-1, :-2] ** 2)
            + 0.5 * (v[1:-1, 1:-1] ** 2 + v[:-2, 1:-1] ** 2)
        )
    )
    ke = enforce_boundaries(ke, "h")

    du_new = du_new.at[1:-1, 1:-1].add(-(ke[1:-1, 2:] - ke[1:-1, 1:-1]) / dx)
    dv_new = dv_new.at[1:-1, 1:-1].add(-(ke[2:, 1:-1] - ke[1:-1, 1:-1]) / dy)

    if is_first_step:
        u = u.at[1:-1, 1:-1].add(dt * du_new[1:-1, 1:-1])
        v = v.at[1:-1, 1:-1].add(dt * dv_new[1:-1, 1:-1])
        h = h.at[1:-1, 1:-1].add(dt * dh_new[1:-1, 1:-1])
    else:
        u = u.at[1:-1, 1:-1].add(
            dt
            * (
                ADAMS_BASHFORTH_A * du_new[1:-1, 1:-1]
                + ADAMS_BASHFORTH_B * du[1:-1, 1:-1]
            )
        )
        v = v.at[1:-1, 1:-1].add(
            dt
            * (
                ADAMS_BASHFORTH_A * dv_new[1:-1, 1:-1]
                + ADAMS_BASHFORTH_B * dv[1:-1, 1:-1]
            )
        )
        h = h.at[1:-1, 1:-1].add(
            dt
            * (
                ADAMS_BASHFORTH_A * dh_new[1:-1, 1:-1]
                + ADAMS_BASHFORTH_B * dh[1:-1, 1:-1]
            )
        )

    h = enforce_boundaries(h, "h")
    u = enforce_boundaries(u, "u")
    v = enforce_boundaries(v, "v")

    if LATERAL_VISCOSITY > 0:
        # lateral friction
        fe = fe.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx
        )
        fn = fn.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
        )
        fe = enforce_boundaries(fe, "u")
        fn = enforce_boundaries(fn, "v")

        u = u.at[1:-1, 1:-1].add(
            dt
            * (
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
            )
        )

        fe = fe.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (v[1:-1, 2:] - u[1:-1, 1:-1]) / dx
        )
        fn = fn.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (v[2:, 1:-1] - u[1:-1, 1:-1]) / dy
        )
        fe = enforce_boundaries(fe, "u")
        fn = enforce_boundaries(fn, "v")

        v = v.at[1:-1, 1:-1].add(
            dt
            * (
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
            )
        )

    return ModelState(h, u, v, dh_new, du_new, dv_new)


@partial(jax.jit, static_argnums=(1,))
def do_multistep(state, num_steps):
    """Perform multiple model steps back-to-back."""
    return jax.lax.fori_loop(
        0, num_steps, lambda _, s: shallow_water_step(s, False), state
    )


def solve_shallow_water(t1, num_multisteps=10):
    """Iterate the model forward in time."""
    # initial conditions
    h, u, v = get_initial_conditions()
    du, dv, dh = (jnp.zeros((ny_local, nx_local)) for _ in range(3))

    state = ModelState(h, u, v, dh, du, dv)
    sol = [state]

    state = shallow_water_step(state, True)
    sol.append(state)
    t = dt

    if HAS_TQDM:
        pbar = tqdm.tqdm(
            total=math.ceil(t1 / dt),
            disable=mpi_rank != 0,
            unit="model day",
            initial=t / DAY_IN_SECONDS,
            unit_scale=dt / DAY_IN_SECONDS,
            bar_format=(
                "{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, "
                "{rate_fmt}{postfix}]"
            ),
        )

    # pre-compile JAX kernel
    do_multistep(state, num_multisteps)

    start = time.perf_counter()
    with ExitStack() as es:
        if HAS_TQDM:
            es.enter_context(pbar)

        while t < t1:
            state = do_multistep(state, num_multisteps)
            state[0].block_until_ready()
            sol.append(state)

            t += dt * num_multisteps

            if t < t1 and HAS_TQDM:
                pbar.update(num_multisteps)

    end = time.perf_counter()

    if mpi_rank == 0:
        print(f"\nSolution took {end - start:.2f}s")

    return sol


@jax.vmap
@jax.jit
def reassemble_array(arr):
    """This converts an array containing the solution from each processor as
    first axis to the full solution.

    Shape (mpi_size, ny_local, nx_local) -> (ny_global, nx_global)
    """
    out = jnp.empty((ny_global, nx_global), dtype=arr.dtype)
    for i in range(mpi_size):
        proc_idx_i = np.unravel_index(i, (nproc_y, nproc_x))
        local_slice_i = (
            slice(
                (ny_local - 2) * proc_idx_i[0],
                (ny_local - 2) * proc_idx_i[0] + ny_local,
            ),
            slice(
                (nx_local - 2) * proc_idx_i[1],
                (nx_local - 2) * proc_idx_i[1] + nx_local,
            ),
        )
        out = out.at[local_slice_i].set(arr[i])

    return out


def animate_shallow_water(sol):
    """Create a matplotlib animation of the result."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    quiver_stride = (
        slice(1, -1, ny_global // MAX_QUIVERS[0]),
        slice(1, -1, nx_global // MAX_QUIVERS[1]),
    )

    # set up figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # surface plot of height anomaly
    cs = ax.pcolormesh(
        0.5 * (x_global[:-1] + x_global[1:]) / 1e3,
        0.5 * (y_global[:-1] + y_global[1:]) / 1e3,
        sol[0].h[1:-1, 1:-1] - DEPTH,
        vmin=-PLOT_ETA_RANGE,
        vmax=PLOT_ETA_RANGE,
        cmap="RdBu_r",
    )

    # quiver plot of velocity
    cq = ax.quiver(
        xx_global[quiver_stride] / 1e3,
        yy_global[quiver_stride] / 1e3,
        sol[0].u[quiver_stride],
        sol[0].v[quiver_stride],
        clip_on=True,
    )

    # time indicator
    t = ax.text(
        s="",
        x=0.05,
        y=0.95,
        ha="left",
        va="top",
        backgroundcolor=(1, 1, 1, 0.8),
        transform=ax.transAxes,
    )

    ax.set(
        aspect="equal",
        xlim=(x_global[1] / 1e3, x_global[-2] / 1e3),
        ylim=(y_global[1] / 1e3, y_global[-2] / 1e3),
        xlabel="$x$ (km)",
        ylabel="$y$ (km)",
    )

    plt.colorbar(
        cs,
        orientation="horizontal",
        label="Surface height anomaly (m)",
        pad=0.2,
        shrink=0.8,
    )
    fig.tight_layout()

    def animate(i):
        state = sol[i]

        eta = state.h - DEPTH
        cs.set_array(eta[1:-1, 1:-1].flatten())
        cq.set_UVC(state.u[quiver_stride], state.v[quiver_stride])

        current_time = PLOT_EVERY * dt * i
        t.set_text(f"t = {current_time / DAY_IN_SECONDS:.2f} days")
        return (cs, cq, t)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(sol), interval=50, blit=True, repeat_delay=3_000
    )
    return anim


if __name__ == "__main__":
    benchmark_mode = "--benchmark" in sys.argv

    sol = solve_shallow_water(t1=10 * DAY_IN_SECONDS, num_multisteps=PLOT_EVERY)

    if benchmark_mode:
        sys.exit(0)

    # copy solution to mpi_rank 0
    full_sol_arr, _ = mpi4jax.gather(jnp.asarray(sol), root=0, comm=mpi_comm)

    if mpi_rank == 0:
        # full_sol_arr has shape (nproc, time, nvars, ny, nx)
        full_sol_arr = jnp.moveaxis(full_sol_arr, 0, 2)
        full_sol = [ModelState(*reassemble_array(x)) for x in full_sol_arr]

        anim = animate_shallow_water(full_sol)

        if "--save-animation" in sys.argv:
            # save animation as MP4 video (requires ffmpeg)
            anim.save("shallow-water.mp4", writer="ffmpeg", dpi=100)
        else:
            import matplotlib.pyplot as plt

            plt.show()
